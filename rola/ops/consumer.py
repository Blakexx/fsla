"""Python driver for the V3 consumer CUDA kernel (``ROLA_V3_SPEC.md`` §3/§5).

Consumes exactly the Stage 3 schedule (``routing/schedule_v3.py``) and the same
routing-input contract as the canonical fp64 oracle ``naive_rola``, so the two
cannot drift apart: any test may hand the identical arguments to both.

This module contains no numerics. Everything it computes is the §3 owner-rectangle
arithmetic -- the per-level ``(stride, modulus)`` triple that turns a local leaf
index into a digit, and the owner-id mixed radix -- lowered to the flat integer
arrays the kernel's ``V3ConsumerParams`` expects. The derivations are:

```text
fixed level  l < j : stride = BC,                modulus = 1,        owner_stride = prod(widths[l+1:j])
boundary level j   : stride = prod(widths[j+1:]), modulus = c,        owner_stride = 1
spanned level l > j: stride = prod(widths[l+1:]), modulus = width_l,  owner_stride = 1
```

A fixed level's ``stride = BC, modulus = 1`` makes ``(i / stride) % modulus`` zero
for every local leaf ``i < BC``, so its digit is the owner's fixed digit -- the
same expression the spanned levels use, which is what makes the kernel
topology-agnostic (§3) rather than branching on a level's role.
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache
from math import prod

import torch

from rola.planner.schedule import (
    DEFAULT_SM_SHARED_BYTES,
    PANEL_MAX_EXPONENT,
    DecayStream,
    Plan,
    Schedule,
    build_clock_stream,
    build_schedule,
    leaf_rate_dials,
    panel_exponent_bound,
)
from rola.routing.types import Topology


@_lru_cache(maxsize=1)
def sm_shared_bytes() -> int:
    """The device's per-SM shared-memory budget, for §6's residency rule.

    Device-queried once (`cudaDevAttrMaxSharedMemoryPerMultiprocessor`) and RECORDED
    IN THE PLAN by `select_plan`, per the §10 ruling: a plan then states the
    machine it reasoned for instead of silently depending on the machine it is
    replayed on. Falls back to `DEFAULT_SM_SHARED_BYTES` when there is no device or
    no extension -- the CPU-only path §6.2/§6.3 selection must keep working on, and
    the fallback is VISIBLE in the plan because the number is a field.
    """
    import torch as _torch
    if not _torch.cuda.is_available():
        return DEFAULT_SM_SHARED_BYTES
    try:
        from rola.ops._ext import extension
        return int(extension().rola_sm_shared_bytes())
    except Exception:
        # A build failure must not silently change plan SELECTION -- it must fail at
        # the launch, where it is diagnosable. Selection falls back to the declared
        # constant, which is the Ampere value this repo targets.
        return DEFAULT_SM_SHARED_BYTES


def level_descriptors(topology: Topology, plan: Plan) -> dict[str, list[int]]:
    """Lower the §3 owner rectangle to the kernel's flat per-level integer arrays."""
    widths = topology.widths
    geometry = plan.geometry
    j = geometry.first_spanned_level
    c = geometry.slice_digits
    stride, modulus, owner_stride, row_offset = [], [], [], []
    offset = 0
    for level, width in enumerate(widths):
        row_offset.append(offset)
        offset += width
        if level < j:
            stride.append(plan.BC)
            modulus.append(1)
            owner_stride.append(max(1, prod(widths[level + 1:j])))
        elif level == j:
            stride.append(max(1, prod(widths[j + 1:])))
            modulus.append(c)
            owner_stride.append(1)
        else:
            stride.append(max(1, prod(widths[level + 1:])))
            modulus.append(width)
            owner_stride.append(1)
    return {
        "level_stride": stride,
        "level_modulus": modulus,
        "level_row_offset": row_offset,
        "level_width": list(widths),
        "owner_stride": owner_stride,
    }


def _consumer_forward(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    plan: Plan,
    *,
    initial_state: torch.Tensor | None = None,
    paging=None,
    continue_paging: bool = False,
    decay=None,
    write_levels_stored: tuple[torch.Tensor, ...] | None = None,
    eps: float = 1e-5,
    schedule: Schedule | None = None,
    sequence_parallel=None,
    sequence_parallel_tensors=None,
) -> tuple[torch.Tensor, torch.Tensor, Schedule]:
    """Run the V3 consumer. Signature mirrors ``naive_rola``'s routing inputs.

    ``v``: ``[B,T,H,d_v]``. Returns ``(y, final_state, schedule)`` with
    ``y: [B,T,H,d_v]`` and ``final_state: [B,H,N,d_v(+1)]`` -- the same shapes the
    oracle returns, so a test compares them directly.

    ``paging``: an optional
    :class:`~rola.ops.paging.PageArena` (spec §7.1). When given, the
    state is the arena's page array rather than a dense ``[BH, N, cols]`` tensor
    and the returned ``final_state`` is the ARENA, not a tensor -- materializing
    ``[BH, N, cols]`` on the way out is exactly the allocation paging exists to
    avoid. Paging is memory management and is BITWISE INVISIBLE: the same routing
    produces bit-identical ``y`` paged or not.

    ``continue_paging``: this call is CALL-SCOPED BY DEFAULT (``False``) -- the
    arena is reset before this call's routing is planned, so a reused arena
    computes the same thing a fresh one would, mirroring the unpaged path's
    "allocate zeros every call" semantics and the layer's own default (a fresh
    `RoLA.forward` resets the arena unless a prior `recurrent_state` is being
    continued). Pass ``True`` only to deliberately continue an arena's already-
    resident pages across this call boundary without going through
    :meth:`~rola.ops.paging.PageArena.adopt_dense` -- e.g. successive
    chunks of one sequence. Meaningless (and rejected) without ``paging``. Fix pass
    (BLOCKER B1, `workflows/20260730-wiring/paging-review.md`): an earlier revision
    reused the arena across calls with no reset and no flag, so a second,
    independent forward silently computed a continuation of the first.

    ``decay``: optional ``LeafMassDecay``. ``write_levels_stored`` is the
    (optionally separately quantized) representation the clock reads (spec §2);
    it defaults to ``write_levels``. The clock stream is staged by
    ``build_clock_stream``, a *separate* builder from the schedule, so the stored
    tensor never enters the bitmap builder's scope -- P5.20.5 finding 2 (tile
    liveness keys on the live deposit ``W``, never on the clock) is structural
    here rather than a convention.

    ``sequence_parallel``: an optional
    :class:`~rola.routing.sequence_parallel.SequenceParallelPlan`
    (spec §6.4). ``None`` -- and equally a plan whose ``split`` is ``False`` -- is
    ``P = 1``: the single-launch path, with every sequence-parallel pointer null in
    the kernel, bit-identical to the pre-``P`` kernel. ``P = 1`` is the DEFAULT and
    the primary benchmarked configuration everywhere (§6.4); any headline number
    quoted at ``P > 1`` is a reporting defect.

    Under a split plan the launch becomes §6.4's three passes with two launch
    boundaries -- summaries, ordered fold, reads -- sequenced inside the extension
    entry so a caller cannot run them out of order, and the returned
    ``final_state`` is published by pass 3's LAST segment. That state is
    RE-ASSOCIATION-EQUIVALENT to the ``P = 1`` state, not bitwise identical to it
    (§6.4, amended 2026-07-30): it is graded against the fp64 oracle at the same
    tolerance ``P = 1`` is held to, and the unwritten-leaf carve-out is bitwise.
    """
    from rola.ops._ext import extension

    topology = plan.topology
    B, T, H, d_v = v.shape
    if d_v != topology.d_v:
        raise ValueError(f"v's last dim {d_v} must match topology.d_v={topology.d_v}")
    if not v.is_cuda:
        raise ValueError("the RoLA consumer is a CUDA kernel; v must be on a CUDA device")
    # §4.6's `kerneled` column, enforced where the launch happens and not only where
    # the dispatch is decided. `rola_func` routes such a config to the reference path;
    # this refusal is what makes that routing a guarantee rather than a convention
    # that a direct caller of the private path gets to ignore.
    unratified = topology.unratified_activations
    if unratified:
        raise ValueError(
            f"the configured activations {unratified} have no ratified kernel row "
            "(`kerneled=False` in rola/routing/activations.py). This is NOT the kernel "
            "declining work it was asked to do: nothing was ever measured for these, so "
            "there is nothing to launch. Call `rola.rola_func`, which runs an unratified "
            "activation on the reference path carrying an explicit `not_kerneled` flag "
            "-- a labeled research surface, which a silent kernel path would not be")
    global_norm = plan.normalization == "global"
    if global_norm and g_read is not None:
        raise ValueError("normalization='global' fixes g_read := 1 (spec §1); pass g_read=None")
    if not global_norm and g_read is None:
        raise ValueError("normalization='raw' requires an explicit g_read")

    # §5's warp tiling, as REALIZED. One thread per channel makes the CTA `d_v`
    # threads, i.e. `W = d_v/32` warps partitioned purely along the channel axis:
    # `(a, b) = (1, d_v/32)`. `a > 1` needs more threads than channels, which
    # decouples the thread -> channel map the deposit and the LeafMassDecay walk are
    # written in -- a separate increment. Rejected loudly rather than ignored, for
    # the same reason `plan.decay` is (review F3): a
    # stamped plan parameter the launch does not honour is silently wrong.
    expected_warp_tile = (1, max(1, d_v // 32))
    if tuple(plan.warp_tile) != expected_warp_tile:
        raise NotImplementedError(
            f"plan.warp_tile={tuple(plan.warp_tile)} is not realized; the consumer runs "
            f"(a, b) = {expected_warp_tile} at d_v={d_v} (one thread per channel, so a=1 and "
            "b = d_v/32 warps, a pure channel-group partition). a > 1 requires decoupling the "
            "thread -> channel map from the state registers and is its own increment.")

    if schedule is None:
        schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan)
    if (decay is not None) != plan.decay:
        raise ValueError(
            f"plan.decay={plan.decay} disagrees with decay={'a LeafMassDecay' if decay is not None else 'None'}; "
            "the decay path is a compile-time template axis (spec §6), so the plan and the "
            "argument must agree rather than the plan being inferred")

    if decay is None:
        clock_amps = None
        dials = None
        if write_levels_stored is not None:
            raise ValueError("write_levels_stored is meaningless without a LeafMassDecay")
    else:
        decay.validate_against(topology)
        stored_src = tuple(write_levels if write_levels_stored is None
                           else write_levels_stored)
        write_src = tuple(write_levels)

        # ---- THE DECAY ARM'S PER-CALL PREAMBLE, BUILT AT MOST ONCE PER SCHEDULE --
        # `build_clock_stream` + `leaf_rate_dials` + `panel_exponent_bound` are pure
        # functions of `(stored, write, decay, BT)`, and this used to rebuild all
        # three on EVERY call -- including calls handed an already-built `schedule=`.
        # MEASURED at 0.78-1.40 ms per call on
        # `benchmarks/ops/benchmark_rola_v3_new_consumer.py`'s cells, i.e. a pure
        # waste term on every decay launch that reuses a schedule, and large enough
        # to dominate a small kernel outright (it made a decay-vs-none paired cell
        # read +346 % where the KERNEL's own cost was +44 %).
        #
        # The memo lives on the SCHEDULE because that is the object a caller already
        # reuses to say "same routing", and its validity key is identity-plus-version
        # over strongly-held references rather than `_source_key` -- see
        # `DecayStream` for why the cheaper key is unsound across calls.
        def _build_decay_stream():
            ca = build_clock_stream(stored_src, write_src, plan)
            dl = leaf_rate_dials(decay, topology).to(v.device)
            pb = panel_exponent_bound(ca, decay, plan.BT)
            return DecayStream(
                clock_amps=ca, dials=dl, panel_bound=pb,
                src_stored=stored_src, src_write=write_src,
                src_versions=DecayStream.versions(stored_src, write_src),
                decay=decay, BT=plan.BT)

        stream = schedule.decay_artifacts(
            stored_src, write_src, decay, plan.BT, build=_build_decay_stream)
        clock_amps = stream.clock_amps
        dials = stream.dials
        # ---- THE PANEL DOMAIN IS NOW EVERY DECAY LAUNCH'S DOMAIN ------------
        # This check used to be gated on `plan.intra_lowering == "panel"`. The
        # `intra_lowering` axis is gone (item 6b) and the surviving consumer applies
        # `panel_read_factor`/`panel_write_factor` on EVERY decay tile -- it has no
        # token-sequential walk to fall back to -- so the admission bound is a
        # property of the decay arm itself. Widening the guard to match what the
        # kernel does is the correction; leaving it keyed on a deleted field would
        # have silently disarmed it.
        if plan.decay:
            # The panel DOMAIN, checked rather than documented (`intra_panel.cuh`,
            # §12 rules 1 and 3). The lowering exponentiates `keep^{+P}` and
            # `keep^{-P}` separately; past the fp32 exp range the write factor
            # saturates at the kernel's clamp while the read factor underflows,
            # and their product -- which is O(1) at `t ~ j` -- comes out as ~0.
            # A wrong answer, not a rounding one, so it is refused here rather
            # than clamped into silence. The bound is exact and conservative and
            # costs O(T * sum_l width_l); inside it the kernel's clamp is
            # provably unreachable.
            # Memoized with the stream, and NOT skipped: the admission decision is
            # made on every call, from a bound that is a pure function of the same
            # inputs the memo is keyed on.
            bound = stream.panel_bound
            if not bound <= PANEL_MAX_EXPONENT:
                raise ValueError(
                    f"the decay arm's panel lowering is not admissible on this routing: "
                    f"the per-leaf "
                    f"reference exponent is bounded by {bound:.3g}, above the "
                    f"PANEL_MAX_EXPONENT={PANEL_MAX_EXPONENT} inside which the two separately "
                    "exponentiated survival panels are an exact rewrite of the recurrence. "
                    "This is a survival DYNAMIC RANGE limit (a near-unit leaf rate met by a "
                    "near-one-hot write clock over a whole tile), not a density or a size "
                    "limit. There is no second decay lowering to fall back to (item 6b "
                    "retired the token-sequential walk with the old consumer), so this is a "
                    "hard refusal: the routing itself is outside the lowering's exact domain, "
                    "and running it anyway would clamp one panel while the other underflows "
                    "and return a wrong answer rather than a rounded one.")

    BH = B * H
    N = topology.N
    cols = d_v + (1 if global_norm else 0)
    if continue_paging and paging is None:
        raise ValueError("continue_paging=True is meaningless without paging=an arena")
    page_table = None
    if paging is None:
        if initial_state is None:
            state = torch.zeros((BH, N, cols), device=v.device, dtype=torch.float32)
        else:
            expected = (B, H, N, cols)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}")
            state = initial_state.reshape(BH, N, cols).to(torch.float32).contiguous().clone()
    else:
        if initial_state is not None:
            # The arena is the carrier of prior state under paging. Loading a dense
            # one is `PageArena.adopt_dense`, an EXPLICIT bridge the caller calls
            # -- doing it implicitly here would provision the dense limit behind the
            # caller's back, which is the opposite of what §7.1 is for.
            raise ValueError(
                "under paging the arena carries prior state; call arena.adopt_dense(...) "
                "explicitly instead of passing initial_state")
        paging.plan(schedule, fresh=not continue_paging)
        state = paging.state
        page_table = paging.page_table

    v_f = v.permute(0, 2, 1, 3).reshape(BH, T, d_v).to(torch.float32).contiguous()
    desc = level_descriptors(topology, plan)
    geometry = plan.geometry

    # §6.4. An unsplit plan is indistinguishable from no plan at all here, on
    # purpose: `P = 1` must reach the kernel through exactly the arguments it
    # reached it through before `P` existed, or the primary benchmarked
    # configuration is not the configuration that was benchmarked.
    owners_arg = schedule.owners
    sp_args: tuple = (None,) * 14
    if sequence_parallel is not None and sequence_parallel.split:
        from rola.routing.sequence_parallel import build_sequence_parallel_tensors
        if tuple(sequence_parallel.P) and len(sequence_parallel.P) != schedule.owners.shape[0]:
            raise ValueError(
                f"the sequence-parallel plan covers {len(sequence_parallel.P)} owners but the "
                f"schedule has {schedule.owners.shape[0]}; `P` is keyed on `schedule.owners` "
                "row for row (spec §6.4), so a mismatch is a plan built against other routing")
        if sequence_parallel.BC != plan.BC or sequence_parallel.cols != cols:
            raise ValueError(
                f"the sequence-parallel workspace is shaped (BC={sequence_parallel.BC}, "
                f"cols={sequence_parallel.cols}) but this launch is (BC={plan.BC}, cols={cols})")
        # `sequence_parallel_tensors` lets a caller supply the realized work lists and
        # workspace instead of having them built here. The one in-tree use is the gate
        # that POISONS the workspace with NaN to prove every byte of it is written
        # before it is read (the workspace is `torch.empty`, so that is a claim the
        # kernel makes and something has to check it). It is not a performance hook:
        # the tensors are launch-scoped either way.
        sp_tensors = (sequence_parallel_tensors
                      if sequence_parallel_tensors is not None
                      else build_sequence_parallel_tensors(sequence_parallel, schedule))
        owners_arg = sp_tensors.owners
        sp_args = sp_tensors.as_kernel_arguments()

    if paging is not None:
        # §4.5's ONE EVENT-WAIT, and it is a CORRECTNESS requirement rather than a
        # tidiness one. `PageArena._admit_pages` publishes the page table and zeroes
        # the admitted slots on the arena's SIDE STREAM; without this join the
        # consumer launch below reads a page table that is still being written and a
        # value pool that is still being zeroed. It is a stream-to-stream wait, not a
        # host synchronization -- the host never learns when the zeroing finished,
        # only that the consumer will not start before it did -- so the overlap the
        # side stream buys is preserved.
        #
        # MEASURED, not hypothesized: with the layer cut-over choosing `BC` per launch
        # the missing join fired as `index_copy_(): index out of bounds` inside
        # `_admit_pages` itself and poisoned the CUDA context (56 failures in
        # `tests/layers`, all of them green under `CUDA_LAUNCH_BLOCKING=1` -- the
        # signature of a stream race, since blocking serializes the streams the join
        # was supposed to order).
        paging.wait()

    ext = extension()
    y_f, state_out, _num, _den = ext.rola_consumer_forward(
        [t.to(torch.float32).contiguous() for t in schedule.read_amps],
        [t.to(torch.float32).contiguous() for t in schedule.write_amps],
        None if clock_amps is None else [t.to(torch.float32).contiguous() for t in clock_amps],
        dials,
        schedule.read_tile_bits.contiguous(),
        schedule.write_tile_bits.contiguous(),
        schedule.read_token_bits.contiguous(),
        schedule.write_token_bits.contiguous(),
        schedule.read_slice_union.contiguous(),
        schedule.write_slice_union.contiguous(),
        owners_arg.contiguous(),
        None if global_norm else schedule.g_read.to(torch.float32).contiguous(),
        schedule.g_write.to(torch.float32).contiguous(),
        v_f,
        state,
        page_table,
        desc["level_stride"],
        desc["level_modulus"],
        desc["level_row_offset"],
        desc["level_width"],
        desc["owner_stride"],
        geometry.first_spanned_level,
        geometry.slice_digits,
        geometry.n_slices,
        geometry.owners_total,
        H,
        plan.BT,
        plan.BC,
        global_norm,
        plan.mma_precision,
        plan.compact,
        plan.ring_slots,
        float(eps),
        *sp_args,
    )
    y = y_f.view(B, H, T, d_v).permute(0, 2, 1, 3).contiguous().to(v.dtype)
    if paging is not None:
        return y, paging, schedule
    return y, state_out.view(B, H, N, cols), schedule


__all__ = ["_consumer_forward", "level_descriptors"]
