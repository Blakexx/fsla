"""§6.4 sequence parallelism: the per-owner `P` policy, its gates and its workspace.

``ROLA_V3_SPEC.md`` §6.4 (contract NORMATIVE, policy NORMATIVE, amended 2026-07-30).
Everything here is PLAN-TIME. Device code never sees a `P`, a wave count or a
byte budget; it sees a flat ``(owner, segment)`` work list and a workspace pointer,
exactly as it sees ``BT`` and ``compact`` as stamped template arguments rather
than as things to decide (§7.4.1's sanctioned pattern).

**`P = 1` is the default and the primary benchmarked configuration.** When the
policy admits no split, :func:`select_sequence_parallelism` returns a plan whose
``split`` is ``False`` and which carries no tensors at all, so the consumer takes
the single-launch path with every sequence-parallel pointer null -- bit-identical
to the pre-`P` kernel, which is what makes the baseline a baseline.

Symbols (defined at first use): ``BC`` leafs per owner block, ``BT`` token tile,
``d_v`` per-head value dim, ``cols = d_v + 1`` under ``global`` normalization and
``d_v`` under ``raw``, ``P_o`` the number of sequence segments owner ``o`` is split
into, ``n_slots`` the total workspace slots (``sum_o P_o`` over SPLIT owners --
``P_o - 1`` segment boundaries plus one entry-state stash; see :func:`slot_bytes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor

import torch

# ---------------------------------------------------------------------------
# Measured constants
# ---------------------------------------------------------------------------
#
# Transcribed from `development/measurement/measured_constants.json`
# (RTX 3080 Ti / GA102 / sm86, driver 595.95, CUDA 12.4.99, medians of 7 trials,
# generated 2026-07-28) so that the planner has no dependency on a path outside
# this repository. Two of the three inputs the gates need are NOT vendored and are
# read per-plan from the machine instead, because they can be:
#
#   * `sms`            -- `torch.cuda.get_device_properties().multi_processor_count`
#   * `blocks_per_sm`  -- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the
#                         EXACT instantiation the plan will launch, at its real
#                         dynamic SMEM size (`rola_consumer_occupancy`).
#
# What remains vendored is the two bandwidths, which are benchmark results and
# cannot be queried. THE CAVEAT from the source file travels with them.

#: Clean-HBM streaming bandwidth (`l2_scatter.contig_HBM`, working set 134 MB, far
#: above the 6 MiB L2). This is the value `measured_constants.json` itself
#: designates as the trustworthy HBM number and feeds to `Machine.hbm_bytes_per_s`.
MEASURED_HBM_BYTES_PER_S = 737.6e9

#: `bench_state_roundtrip`: per-CTA state store+load, 80 CTAs at once, at two
#: footprints. **These are NOT HBM numbers and the source file says so explicitly**
#: -- at 64 KiB/CTA the 80-CTA working set is 5.12 MB, inside the 6 MiB L2, so the
#: figure is mostly L2-served; at 128 KiB/CTA it is 10.24 MB and only partly so.
#: They are exactly the right prices for §6.4's split roundtrip anyway, because that
#: roundtrip IS a per-CTA state store+load at a per-CTA footprint in this range --
#: which is the whole reason the spec names this bench rather than a bandwidth
#: constant. They will NOT transfer to a datacenter L2 or to the `nc ~ 65K` regime.
MEASURED_STATE_ROUNDTRIP_BYTES_PER_S = {
    64 << 10: 3710.14e9,
    128 << 10: 1585.96e9,
}


def state_roundtrip_bytes_per_s(bytes_per_cta: int) -> float:
    """Price a split roundtrip at the measured point that matches its footprint.

    A two-point measured curve, used as a STEP rather than interpolated: the two
    points differ by which memory actually serves them (L2 vs an L2/HBM blend), so
    a line drawn between them would be a line through a regime change. Below the
    smaller footprint the smaller footprint's rate is the closest measurement that
    exists, and it is used unextrapolated.
    """
    for size in sorted(MEASURED_STATE_ROUNDTRIP_BYTES_PER_S):
        if bytes_per_cta <= size:
            return MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[size]
    return MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[max(MEASURED_STATE_ROUNDTRIP_BYTES_PER_S)]


#: The three §6.4 admission gates, as recorded in the plan when one clamps `P_o`.
GATES = ("occupancy", "bandwidth", "capacity")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SequenceParallelPolicy:
    """§6.4's tunables. Every one is a POLICY number, not a numeric tolerance."""

    #: §6.4: "raises total segments only to the smallest count clearing the waves/SM
    #: target (~1-2)". The target is the number of waves the launch should present;
    #: at or above it the occupancy gate is CLOSED and `P` cannot help.
    waves_target: float = 1.0

    #: Hard ceiling on `P_o`, independent of the wave arithmetic. `P` is an
    #: occupancy dial for the low-owner corner, never the throughput story (§6.4),
    #: so an owner is never split further than this however starved the grid is.
    #:
    #: **There is deliberately no matching FLOOR, and that is a recorded caveat
    #: rather than an oversight (spec §6.4 measured status, 2026-07-30).** The
    #: apportionment can hand an owner exactly one extra segment, and `P_o = 2`
    #: measured SLOWER than `P_o = 1` at two of three benchmarked cells while
    #: `P_o >= 3` paid. The cause is a fixed per-launch HOST cost -- building the
    #: work lists and workspace, ~0.4-0.5 ms, flat in `P` -- that none of the three
    #: gates prices, not a property of the algorithm. A `min_split` floor would bake
    #: a constant measured on one 64-owner cell of one GPU into the policy, which is
    #: the small-regime conclusion that inverts at `nc ~ 65K` (where the grid
    #: saturates and the occupancy gate closes `P` entirely). The right fix is
    #: removing the host cost; the floor is a policy decision, left to be ratified.
    max_p: int = 8

    #: §6.4 gate 2's "with margin": the split's roundtrip traffic TIME may be at most
    #: this fraction of the launch's existing traffic time. See
    #: :func:`select_sequence_parallelism` for why the gate is expressed as a
    #: relative traffic budget and what that does and does not price.
    bandwidth_margin: float = 0.10

    #: §6.4's `workspace_cap: Optional[bytes]`. A user-supplied value WINS. `None`
    #: resolves to :attr:`cap_fraction` of TOTAL device memory -- deterministic
    #: across runs, unlike a free-memory query, and safe against activations
    #: allocated later in the step.
    workspace_cap: int | None = None

    #: The `None` resolution's fraction of total device memory. A few percent, the
    #: cuBLAS/FlashAttention workspace convention.
    cap_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.waves_target <= 0:
            raise ValueError(f"waves_target must be positive, got {self.waves_target}")
        if self.max_p < 1:
            raise ValueError(f"max_p must be at least 1, got {self.max_p}")
        if not 0.0 <= self.bandwidth_margin:
            raise ValueError(f"bandwidth_margin must be non-negative, got {self.bandwidth_margin}")
        if self.workspace_cap is not None and self.workspace_cap < 0:
            raise ValueError(f"workspace_cap must be non-negative, got {self.workspace_cap}")
        if not 0.0 < self.cap_fraction <= 1.0:
            raise ValueError(f"cap_fraction must lie in (0, 1], got {self.cap_fraction}")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerLedger:
    """§6.4: "What is unified is the LEDGER, not the allocator."

    The §7.1 arena owns persistent cross-chunk state pages and nothing else; the
    sequence-parallel workspace is a launch-scoped torch tensor owned by the
    framework's caching allocator. They are deliberately different allocators with
    different lifetimes. What they share is this account: ONE cap, against which the
    planner projects the peak of everything it is responsible for.

    **State pages are never the degradable item** (§6.4 gate 3). When the projection
    does not fit, `P` is what gives.
    """

    #: The §7.1 arena's committed page bytes (0 when unpaged).
    arena_page_bytes: int
    #: The sequence-parallel workspace this plan asks for.
    workspace_bytes: int
    #: The planner's other materializations -- the §4 schedule bitmaps and the
    #: token-major route stream the consumer loads.
    materialization_bytes: int
    #: The resolved cap, and how it was resolved.
    cap_bytes: int
    cap_source: str   # "user" | "fraction_of_total"

    @property
    def projected_peak_bytes(self) -> int:
        return self.arena_page_bytes + self.workspace_bytes + self.materialization_bytes

    @property
    def fits(self) -> bool:
        return self.projected_peak_bytes <= self.cap_bytes

    @property
    def workspace_headroom_bytes(self) -> int:
        """What is left for the workspace once the non-degradable terms are paid."""
        return max(0, self.cap_bytes - self.arena_page_bytes - self.materialization_bytes)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SequenceParallelPlan:
    """The `P` half of one plan object (§6.4), alongside `BT`/`compact`/lowering.

    ``P[o]`` is the split of the owner at row ``o`` of ``schedule.owners``, so this
    composes with the existing plan axes by construction: it is keyed on the SAME
    owner list the §4 schedule already emits, in the same order.
    """

    #: Per row of `schedule.owners`.
    P: tuple[int, ...]
    #: The gate that clamped `P_o`, per owner; `None` where nothing clamped it.
    #: §6.4: "record the limiting gate in the plan whenever `P_o` is clamped".
    limiting_gate: tuple[str | None, ...]
    #: The gate that bound the plan AS A WHOLE (the first of occupancy/bandwidth/
    #: capacity to bite), or `None` when the requested split was admitted entire.
    binding_gate: str | None

    #: Occupancy evidence, both terms measured on this machine for this kernel.
    sms: int
    blocks_per_sm: int
    waves_before: float
    waves_after: float

    #: Bandwidth evidence.
    base_traffic_bytes: int
    roundtrip_bytes: int
    roundtrip_seconds: float
    base_traffic_seconds: float

    ledger: PlannerLedger
    n_slots: int
    BC: int
    cols: int

    @property
    def split(self) -> bool:
        """`True` iff any owner is split. `False` is `P = 1` and takes the
        single-launch path with no workspace and no work-list tensors."""
        return self.n_slots > 0

    @property
    def total_segments(self) -> int:
        return sum(self.P)

    def slot_offsets(self) -> tuple[int, ...]:
        """Exclusive prefix sum of :func:`owner_slot_counts`: owner `o`'s first slot.

        Pass 3's segment `p` reads slot `slot_offsets()[o] + p` -- the entry-state
        stash at `p = 0`, the accumulated boundary after segment `p - 1` otherwise.
        """
        out, acc = [], 0
        for n in owner_slot_counts(self.P):
            out.append(acc)
            acc += n
        return tuple(out)


@dataclass(frozen=True, slots=True)
class SequenceParallelTensors:
    """The device-side realization of a split plan: work lists plus workspace.

    LAUNCH-SCOPED. Every tensor here is an ordinary torch allocation whose lifetime
    is this forward; none of it is the §7.1 arena, deliberately (§6.4: mixing a
    transient lifetime into the arena would put launch-scoped garbage under the
    compactor and erode the page cache's semantics).
    """

    owners: torch.Tensor        # [n_work, 2]  pass-3 grid: (bh, owner)
    work_seg: torch.Tensor      # [n_work]
    work_nseg: torch.Tensor     # [n_work]
    work_slot: torch.Tensor     # [n_work]
    summary_owners: torch.Tensor  # [n_summary, 2]  pass-1 grid
    summary_seg: torch.Tensor
    summary_nseg: torch.Tensor
    summary_slot: torch.Tensor
    scan_owners: torch.Tensor   # [n_split, 2]  pass-2 grid
    scan_nseg: torch.Tensor
    scan_slot: torch.Tensor
    ws_state: torch.Tensor      # [n_slots, BC, cols] float32
    ws_scale: torch.Tensor      # [n_slots, BC] float32
    ws_leaf: torch.Tensor       # [n_slots, BC] int32
    ws_count: torch.Tensor      # [n_slots] int32

    def as_kernel_arguments(self) -> tuple:
        """The 14 arguments the consumer entry takes, in its declared order."""
        return (self.work_seg, self.work_nseg, self.work_slot,
                self.summary_owners, self.summary_seg, self.summary_nseg, self.summary_slot,
                self.scan_owners, self.scan_nseg, self.scan_slot,
                self.ws_state, self.ws_scale, self.ws_leaf, self.ws_count)


#: Bytes one workspace slot costs. `BC * cols` floats of state (§6.4's amended
#: `P * BC * d_v` term: a boundary slot carries the sparse summary's `B` and then the
#: dense accumulated boundary, in the same bytes at different points of the slot's
#: life), plus the sparse summary's own metadata: a per-leaf `A`, a leaf-id list, a
#: count. Charged uniformly, including to the entry-state stash, which uses only the
#: state block -- accounting a slot at less than a slot would be optimism.
def slot_bytes(BC: int, cols: int) -> int:
    return BC * cols * 4 + BC * 4 + BC * 4 + 4


def owner_slot_counts(P) -> tuple[int, ...]:
    """Workspace slots owner `o` needs: `P_o` when split, `0` when not.

    **`P_o`, not `P_o - 1` (fix pass R1, `workflows/20260730-wiring/
    pincrement-review.md`; spec §6.4's bound amended to match).** A split owner's
    `P_o - 1` segment boundaries are joined by ONE entry-state slot holding `M_0`,
    which pass 2 stashes and pass 3's segment 0 reads. That slot is what makes pass
    3's reads disjoint from pass 3's writes by construction: without it segment 0
    read the `p.state` rows that the same launch's last segment writes from another
    CTA, and CUDA orders nothing between CTAs of one launch. The hazard was not
    theoretical -- a permitted permutation of the pass-3 work list produced `y` with
    relative error 1.0, silently, with the state untouched.

    An UNSPLIT owner still costs nothing at all: it reads and writes its own
    `p.state` rows from its single CTA, which no other CTA touches.
    """
    return tuple(p if p > 1 else 0 for p in P)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _largest_remainder(weights: list[float], total: int, ceilings: list[int]) -> list[int]:
    """Apportion ``total`` units over ``weights``, capped per entry by ``ceilings``.

    Largest-remainder (Hamilton) apportionment, with ties broken by descending
    weight and then by ascending index -- so the result is a deterministic function
    of the inputs and does not depend on sort stability or on floating-point
    comparison order. Units freed by a ceiling are NOT redistributed: §6.4 says
    "over-splitting pays fold overhead for nothing", so an owner that cannot take
    its share does not push that share onto an owner the policy did not choose.
    """
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n
    mass = sum(weights)
    if mass <= 0:
        return [0] * n
    exact = [total * w / mass for w in weights]
    out = [min(int(floor(e)), c) for e, c in zip(exact, ceilings)]
    remainder = total - sum(out)
    if remainder > 0:
        order = sorted(range(n), key=lambda i: (-(exact[i] - floor(exact[i])), -weights[i], i))
        for i in order:
            if remainder == 0:
                break
            if out[i] < ceilings[i]:
                out[i] += 1
                remainder -= 1
    return out


def _shrink_to(extra: list[int], allowed_total: int, weights: list[float]) -> list[int]:
    """Scale an apportionment down to ``allowed_total`` units, deterministically."""
    if sum(extra) <= allowed_total:
        return list(extra)
    if allowed_total <= 0:
        return [0] * len(extra)
    return _largest_remainder(weights, allowed_total, ceilings=list(extra))


def consumer_blocks_per_sm(plan) -> int:
    """CTAs per SM for the instantiation ``plan`` will launch, asked of the driver.

    Not estimated: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is called on the
    real templated kernel at its real dynamic SMEM size. An occupancy number
    computed for a differently shaped kernel is not a number about this one, and the
    §6.4 occupancy gate is the one place a wrong occupancy silently produces a plan
    that is worse than `P = 1`.
    """
    from rola.ops._ext import extension

    # The consumer's shared memory is entirely COMPILE-TIME
    # (`consumer::consumer_smem_bytes<...>`), so `gram_cols` is not a query axis: the
    # instantiation identity is `(decay, compact, BC, BT, d_v)` and nothing else
    # moves the footprint. `_spanned_moduli`, which existed only to build the old
    # consumer's runtime `gram_cols` argument, went with it (item 6b).
    return int(extension().rola_consumer_occupancy(
        plan.decay, plan.compact, plan.BC, plan.BT, plan.topology.d_v))


def select_sequence_parallelism(
    plan,
    schedule,
    *,
    policy: SequenceParallelPolicy | None = None,
    arena_page_bytes: int = 0,
    blocks_per_sm: int | None = None,
    sms: int | None = None,
) -> SequenceParallelPlan:
    """Assign `P_o` per §6.4's policy, through the three conjunctive gates.

    *Selection is plan-time and TARGETED.* `P_o` is apportioned in proportion to the
    owner's live-tile count -- `schedule.owner_costs`, which IS that count (§4's
    builder forms it as read-candidate tiles plus `write_cost_weight` times
    write-candidate tiles, over the realized schedule), and total segments are
    raised only to the smallest count clearing the waves/SM target. This is the same
    sanctioned pattern the lowering selection uses: the planner reads schedule
    statistics, device code branches on nothing.

    *The three gates, all priced from measured constants, all conjunctive.*

    1. **Occupancy.** `waves = ctas / (sms * blocks_per_sm)`, with `blocks_per_sm`
       read off the driver for this exact instantiation and `sms` off the device.
       At or above `waves_target` the gate is CLOSED: the grid already saturates and
       `P` cannot help, so `P = 1` everywhere and the plan records `occupancy`.

    2. **Bandwidth.** The split's roundtrip is `(P_o - 1) * BC * cols * 4 B * 2` per
       split owner (§6.4's own formula: one store and one load of the boundary),
       priced by `bench_state_roundtrip` at the per-CTA footprint that matches it.
       It is admitted only while that TIME stays within `bandwidth_margin` of the
       launch's existing traffic time (the §4 route stream, the state roundtrip the
       launch already pays, and the split-K output).

       *This is a RELATIVE traffic budget, and the reading is stated rather than
       implied.* §6.4 asks that traffic "stay under the measured achievable-
       bandwidth ceiling with margin"; a bytes total is only under a bytes/second
       ceiling relative to some duration, and the only measured rate that could
       supply a projected duration prices the readout contraction alone -- which
       would make the gate's strictness an artifact of how much of the kernel the
       flop model happens to cover. Pricing the ADDED traffic against the traffic
       the launch already moves needs no such projection, is monotone in `P`, and
       captures exactly the failure §6.4 names: on a saturated bus extra roundtrips
       worsen wall-clock while "improving" occupancy. What it does NOT capture is a
       launch that is bandwidth-bound in absolute terms while moving few bytes; no
       such launch exists here, because the base term already counts every byte the
       launch moves.

       *What the priced formula omits, stated rather than absorbed.* Since the R1
       fix the split also stores one DENSE entry-state stash per split owner
       (`BC * cols * 4` bytes, once, in pass 2) whose read replaces a `p.state` read
       segment 0 already paid. §6.4's gate-2 formula is `(P_o - 1) * BC * cols * 4 * 2`
       verbatim and is priced verbatim here; the stash store is one added
       half-roundtrip per split owner on top of it, i.e. the gate under-prices the
       split's traffic by at most `1 / (2*(P_o - 1))` of what it charges. Changing a
       NORMATIVE formula is a spec ratification, not an implementation choice, so
       this is recorded rather than quietly folded in.

    3. **Capacity.** The workspace must fit the resolved cap, accounted in ONE
       ledger against the arena's pages and the planner's other materializations.
       `P_o` degrades to fit; **state pages are never the degradable item.** The
       budget is in workspace SLOTS, and since the R1 fix a split owner costs `P_o`
       of them rather than `P_o - 1` (:func:`owner_slot_counts`).

    Every clamp is recorded per owner in `limiting_gate`, and the first gate to bite
    is recorded in `binding_gate`.
    """
    policy = policy or SequenceParallelPolicy()
    n_owners = int(schedule.owners.shape[0])
    BC = plan.BC
    cols = plan.topology.d_v + (1 if plan.normalization == "global" else 0)
    n_tiles = schedule.n_tiles

    if sms is None:
        sms = torch.cuda.get_device_properties(schedule.owners.device).multi_processor_count
    if blocks_per_sm is None:
        blocks_per_sm = consumer_blocks_per_sm(plan)
    capacity = max(1, sms * blocks_per_sm)
    waves_before = n_owners / capacity

    # Traffic the P = 1 launch already moves, in bytes. Every term is a real
    # transfer this launch performs, so the base is a measurement of the plan
    # rather than a model of it.
    BH, T, d_v = schedule.BH, schedule.T, plan.topology.d_v
    base_traffic = (schedule.stream_bytes()
                    + n_owners * BC * cols * 4 * 2        # state load + store, per CTA
                    + BH * T * cols * 4)                  # the split-K output
    base_seconds = base_traffic / MEASURED_HBM_BYTES_PER_S

    cap_bytes, cap_source = _resolve_cap(policy, schedule.owners.device)
    materialization_bytes = schedule.schedule_bytes() + schedule.stream_bytes()

    def _finish(P: list[int], gates: list[str | None], binding: str | None):
        n_slots = sum(owner_slot_counts(P))
        ws_bytes = n_slots * slot_bytes(BC, cols)
        roundtrip = sum((p - 1) * BC * cols * 4 * 2 for p in P)
        rt_seconds = roundtrip / state_roundtrip_bytes_per_s(BC * cols * 4)
        ledger = PlannerLedger(
            arena_page_bytes=arena_page_bytes,
            workspace_bytes=ws_bytes,
            materialization_bytes=materialization_bytes,
            cap_bytes=cap_bytes,
            cap_source=cap_source)
        return SequenceParallelPlan(
            P=tuple(P), limiting_gate=tuple(gates), binding_gate=binding,
            sms=sms, blocks_per_sm=blocks_per_sm,
            waves_before=waves_before, waves_after=sum(P) / capacity,
            base_traffic_bytes=base_traffic, roundtrip_bytes=roundtrip,
            roundtrip_seconds=rt_seconds, base_traffic_seconds=base_seconds,
            ledger=ledger, n_slots=n_slots, BC=BC, cols=cols)

    if n_owners == 0:
        return _finish([], [], None)

    # ---- gate 1: occupancy -------------------------------------------------
    if waves_before >= policy.waves_target:
        # The grid already presents at least the target waves. §6.4: at design scale
        # the liveness-filtered grid saturates naturally and `P` is not the
        # throughput story -- so this is the COMMON outcome, not a failure.
        return _finish([1] * n_owners, ["occupancy"] * n_owners, "occupancy")

    target_segments = int(ceil(policy.waves_target * capacity))
    extra_wanted = max(0, target_segments - n_owners)
    weights = [float(c) for c in schedule.owner_costs.tolist()]
    # An owner cannot have more segments than it has tiles, and `max_p` caps it
    # regardless. Both are structural bounds, not thresholds on a measured density.
    ceilings = [max(0, min(policy.max_p, n_tiles) - 1)] * n_owners
    extra = _largest_remainder(weights, extra_wanted, ceilings)

    # `limiting_gate[o]` is the gate that LAST reduced owner `o`'s `P_o`, and
    # `binding_gate` the gate that last reduced the plan's total -- i.e. in both
    # cases the gate that actually determined the number, not merely one that had an
    # opinion about it. The gates are conjunctive and applied in §6.4's order, so a
    # later gate overwriting an earlier one is the later gate being the tighter of
    # the two, which is exactly what "the limiting gate" names.
    binding: str | None = None
    gates: list[str | None] = [None] * n_owners
    if sum(extra) < extra_wanted:
        # The wave target is unreachable before the other two gates are even
        # consulted, because an owner cannot be split past its own tile count (or
        # past `max_p`). That is an occupancy-side limit and is recorded as one.
        binding = "occupancy"
        for i, (e, c) in enumerate(zip(extra, ceilings)):
            if e >= c and c > 0:
                gates[i] = "occupancy"

    def _apply(allowed: int, gate: str) -> None:
        nonlocal extra, binding
        if sum(extra) <= allowed:
            return
        shrunk = _shrink_to(extra, allowed, weights)
        for i, (old, new) in enumerate(zip(extra, shrunk)):
            if new < old:
                gates[i] = gate
        extra = shrunk
        binding = gate

    def _slots_for(extra_v: list[int]) -> int:
        return sum(owner_slot_counts([1 + e for e in extra_v]))

    def _apply_slots(allowed_slots: int, gate: str) -> None:
        """Clamp the apportionment to a budget expressed in SLOTS, by bisection.

        The bisection only ever ACCEPTS a total whose realized slot count it has
        checked against the budget, so the result cannot overrun the cap even where
        the slot count is not strictly monotone in the extra-segment total (a
        largest-remainder reapportionment can move which owners are split, and the
        stash makes the slot count depend on that count, not only on the total).
        The cost of such a corner is at most a slot of unused headroom.
        """
        if _slots_for(extra) <= allowed_slots:
            return
        lo, hi = 0, sum(extra)   # `_slots_for` at `lo` is 0, at `hi` is over budget
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _slots_for(_shrink_to(extra, mid, weights)) <= allowed_slots:
                lo = mid
            else:
                hi = mid - 1
        _apply(lo, gate)

    # ---- gate 2: bandwidth -------------------------------------------------
    per_slot_roundtrip = BC * cols * 4 * 2
    _apply(int(floor(policy.bandwidth_margin * base_seconds
                     * state_roundtrip_bytes_per_s(BC * cols * 4) / per_slot_roundtrip)),
           "bandwidth")

    # ---- gate 3: capacity --------------------------------------------------
    # `headroom` subtracts the NON-DEGRADABLE terms first: §6.4 gate 3 degrades
    # `P_o` to fit and state pages are never the degradable item, so the arena's
    # pages and the planner's materializations are charged before the workspace is
    # offered anything at all. A cap smaller than those alone leaves zero headroom
    # and `P` collapses to 1 -- it does not evict a page.
    #
    # The budget is in SLOTS and the apportionment is in EXTRA SEGMENTS, and since
    # the R1 fix those are no longer the same number: a split owner costs its
    # `extra + 1` slots (`owner_slot_counts`). Slot count is monotone non-decreasing
    # in the extra-segment total under `_shrink_to`, so the largest affordable total
    # is found by bisection over it rather than by dividing -- dividing would
    # over-admit by exactly one slot per split owner, which is the accounting error
    # the extra slot would otherwise introduce.
    headroom = max(0, cap_bytes - arena_page_bytes - materialization_bytes)
    _apply_slots(headroom // slot_bytes(BC, cols), "capacity")

    return _finish([1 + e for e in extra], gates, binding)


def _resolve_cap(policy: SequenceParallelPolicy, device) -> tuple[int, str]:
    """§6.4's cap resolution. A user value wins; `None` is a fixed fraction of TOTAL
    device memory -- deterministic across runs, unlike a free-memory query, and safe
    against activations allocated later in the step."""
    if policy.workspace_cap is not None:
        return int(policy.workspace_cap), "user"
    total = torch.cuda.get_device_properties(device).total_memory
    return int(policy.cap_fraction * total), "fraction_of_total"


# ---------------------------------------------------------------------------
# Realization
# ---------------------------------------------------------------------------


def build_sequence_parallel_tensors(sp: SequenceParallelPlan,
                                    schedule) -> SequenceParallelTensors:
    """Lower a split plan to the flat work lists and the launch-scoped workspace.

    Three grids come out of one apportionment:

      * **pass 3** runs every `(owner, segment)` -- the reads, and the LAST segment
        publishes the owner's final state, running serially from the accumulated
        boundary.
      * **pass 1** runs every `(owner, segment)` EXCEPT the last. That segment's
        outcome IS the final state, which pass 3 produces directly, so summarizing
        it would be a `P`th of the pass-1 work for nothing.
      * **pass 2** runs once per SPLIT owner: it stashes the owner's entry state
        `M_0` into slot `slot0[o]` and folds its `P - 1` summaries in order into
        slots `slot0[o] + 1 ...`.

    **The order of the pass-3 work list is NOT load-bearing, and that is a property
    of the kernel rather than of this function (fix pass R1).** Every pass-3 read is
    a workspace slot published by an earlier launch, or a `p.state` row of an
    unsplit owner that no other CTA touches; nothing a pass-3 CTA reads is written
    by another CTA of the same launch. Before the fix, a split owner's segment 0
    read the rows its own last segment writes, and the answer depended on the block
    scheduler -- gated now by
    ``test_the_pass_three_work_list_order_is_not_load_bearing``.

    Owner order is `schedule.owners`' cost-descending order, and segments are
    ascending within an owner, so at `P_o = 1` for every owner the pass-3 grid is
    `schedule.owners` itself, row for row.
    """
    if not sp.split:
        raise ValueError(
            "build_sequence_parallel_tensors was called on an unsplit plan; `P = 1` "
            "takes the single-launch path with no workspace and no work list at all, "
            "which is what makes it bit-identical to the pre-P kernel")
    device = schedule.owners.device
    P = torch.tensor(sp.P, dtype=torch.int64)
    owners_cpu = schedule.owners.to("cpu", torch.int64)
    n_owners = P.numel()

    # `P_o` slots per SPLIT owner (the `P_o - 1` boundaries plus the entry-state
    # stash), none at all for an unsplit one -- `owner_slot_counts`' rule, as a
    # tensor.
    per_owner_slots = torch.where(P > 1, P, torch.zeros_like(P))
    slot0 = torch.zeros(n_owners, dtype=torch.int64)
    slot0[1:] = torch.cumsum(per_owner_slots, 0)[:-1]
    assert int(per_owner_slots.sum()) == sp.n_slots, (
        "the plan's n_slots disagrees with the realized slot layout; the workspace "
        "would be sized for a different addressing than the kernel performs")

    def _expand(counts: torch.Tensor):
        """(owner index, segment index) for a per-owner segment count."""
        idx = torch.repeat_interleave(torch.arange(n_owners, dtype=torch.int64), counts)
        starts = torch.zeros(n_owners, dtype=torch.int64)
        starts[1:] = torch.cumsum(counts, 0)[:-1]
        seg = torch.arange(int(counts.sum()), dtype=torch.int64) - starts[idx]
        return idx, seg

    work_idx, work_seg = _expand(P)
    sum_idx, sum_seg = _expand(P - 1)
    split_idx = (P > 1).nonzero(as_tuple=True)[0]

    # ONE host->device transfer for every integer array, not eleven. This is the
    # planner increment's own lesson applied at a second site (`_sort_owners`:
    # "ONE host->device transfer for the whole index block"), and it is not
    # speculative here: the first benchmark of this path measured the work-list
    # build at 0.72-0.83 ms, essentially independent of `P` and about half of the
    # whole `P = 1` consumer wall-clock, which was enough to make `P = 2` the only
    # forced point SLOWER than `P = 1`. The cost was launch and transfer overhead
    # around a few kilobytes of indices, so it is packed into one contiguous CPU
    # block, transferred once, and sliced on the device.
    blocks = [
        owners_cpu[work_idx].reshape(-1),          # [2 * n_work]
        work_seg, P[work_idx], slot0[work_idx],
        owners_cpu[sum_idx].reshape(-1),           # [2 * n_summary]
        sum_seg, P[sum_idx], slot0[sum_idx],
        owners_cpu[split_idx].reshape(-1),         # [2 * n_split]
        P[split_idx], slot0[split_idx],
    ]
    sizes = [int(b.numel()) for b in blocks]
    flat = torch.cat([b.to(torch.int32) for b in blocks]).contiguous().to(device)
    cut, parts = 0, []
    for size in sizes:
        parts.append(flat[cut:cut + size])
        cut += size
    (owners_w, seg_w, nseg_w, slot_w,
     owners_s, seg_s, nseg_s, slot_s,
     owners_c, nseg_c, slot_c) = parts

    ws_shape_state = (sp.n_slots, sp.BC, sp.cols)
    # `torch.empty`, not `zeros`: pass 1 writes every boundary slot's count, leaf
    # list, `A` and `B` before pass 2 reads any of them; pass 2 overwrites the whole
    # dense `BC x cols` block of every slot it will be read from, INCLUDING the
    # entry-state stash, before pass 3 reads it -- so a zero-fill would be a memset of
    # bytes that are all written first. (The stash slot's `ws_scale`/`ws_leaf`/
    # `ws_count` entries are the one exception, and they are never read either: the
    # fold indexes `slot0 + 1 + s`.) That the workspace is genuinely fully written
    # is not left as an argument: `test_a_dirty_workspace_changes_nothing` poisons it
    # with NaN and asserts the outputs are bit-identical.
    return SequenceParallelTensors(
        owners=owners_w.view(-1, 2),
        work_seg=seg_w,
        work_nseg=nseg_w,
        work_slot=slot_w,
        summary_owners=owners_s.view(-1, 2),
        summary_seg=seg_s,
        summary_nseg=nseg_s,
        summary_slot=slot_s,
        scan_owners=owners_c.view(-1, 2),
        scan_nseg=nseg_c,
        scan_slot=slot_c,
        ws_state=torch.empty(ws_shape_state, dtype=torch.float32, device=device),
        ws_scale=torch.empty((sp.n_slots, sp.BC), dtype=torch.float32, device=device),
        ws_leaf=torch.empty((sp.n_slots, sp.BC), dtype=torch.int32, device=device),
        ws_count=torch.empty((sp.n_slots,), dtype=torch.int32, device=device),
    )


__all__ = [
    "GATES",
    "MEASURED_HBM_BYTES_PER_S",
    "MEASURED_STATE_ROUNDTRIP_BYTES_PER_S",
    "PlannerLedger",
    "SequenceParallelPlan",
    "SequenceParallelPolicy",
    "SequenceParallelTensors",
    "build_sequence_parallel_tensors",
    "consumer_blocks_per_sm",
    "owner_slot_counts",
    "select_sequence_parallelism",
    "slot_bytes",
    "state_roundtrip_bytes_per_s",
]
