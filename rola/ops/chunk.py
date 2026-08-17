"""The CHUNK consumer's execution path (operator adoption; docs/internals/chunk/) -- the kernel arm.

`rola_op`'s facade is untouched: this module is the private seam's kernel-arm
implementation, opinionated about what it supports and LOUD about what it
does not (the facade stays generic; capabilities return through the paging
and backward work, not
through silent degradation).

Launch arguments arrive as a :class:`~rola.facts.plan.ChunkPlan` that the chunk DAG
built (:mod:`rola.facts.chunk_dag`), and the reverse pass's as the
:class:`~rola.facts.plan.BackwardPlan` that holds it by reference
(:mod:`rola.facts.backward_dag`). What lives here is the family's primitive BODIES --
the arm, the envelope, the packing, the state plane, the seeds, the two scans and the
un-packing -- and the launches that read a plan; the DAGs declare which of them a call
class runs and in what order.

The packing happens ONCE per call, so every entry reads the same bytes. On the
shipped path it is a VIEW: the producer's solve writes the `[B, H, T, sumW]` bf16
side directly and `_side_plane` reconstructs it from the levels' strides
(docs/internals/chunk/chunk.md#the-side-plane). The copy survives for the sides that
are not already that layout -- a hand-built bundle, a foreign producer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rola.ops._ext import extension
from rola.ops.paging import MMA_K_QUANTUM

if TYPE_CHECKING:
    from rola.facts.plan import BackwardPlan, ChunkPlan
    from rola.facts.types import Seeds

#: THE CHUNK LENGTH -- the operator's token quantum. `T` must be a multiple of it,
#: and shaping a sequence to it is the CALLER's job, which is why this is public
#: rather than only reachable through a refusal message.
CHUNK_TOKENS = 32

#: The pinned plan: a fixed preference order over MEASURED arms, never a tuned
#: heuristic. It generalizes only by falling back to the largest built BC when the
#: pinned one is not in this topology's manifest.
_PIN_C, _PIN_BC = CHUNK_TOKENS, 128


def packed_copy(view: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """`view` as a contiguous `dtype` tensor, in ONE copy.

    `.to(dtype)` preserves a permuted view's strides, so `.to(dtype).contiguous()`
    walks the tensor twice; an already-contiguous destination fuses the transpose
    and the narrowing into the one copy that is actually needed.
    """
    return torch.empty(view.shape, dtype=dtype, device=view.device).copy_(view)


def _arm_for(D: int, B: int):
    """The manifest arm this topology runs, or None."""
    arms = [tuple(a[:4]) for a in extension().chunk_arms()]
    bcs = [bc for (c, bc, d, b) in arms if d == D and b == B and c == _PIN_C]
    if not bcs:
        return None
    return (_PIN_C, _PIN_BC if _PIN_BC in bcs else max(bcs))


def arm(routes) -> tuple[int, int]:
    """`(C, BC)` -- the built arm this routing runs on. Refused topologies have none.

    The facade needs `BC` before the launch, because the page arena's extent policy
    is stated in owners; it is the one launch parameter that leaves this module.
    """
    widths = tuple(l.width for l in routes.topology.levels)
    return _arm_for(len(widths), widths[0])


def unsupported(routes, decay, *, d_v: int) -> str | None:
    """Why the chunk arm refuses this call, or None if it runs.

    `d_v` is the value stream's per-head width, which is a fact about `v` and not
    about the routing -- the caller reads it off the operand and states it here.
    """
    top = routes.topology
    widths = tuple(l.width for l in top.levels)
    if d_v != 64:
        return f"d_v={d_v}; the chunk matrix is built at d_v=64"
    if routes.normalization != "global":
        return f"normalization={routes.normalization!r}; the chunk consumer is global-only"
    if decay is not None:
        return "decay is not implemented in the operator (the decay retirement; docs/internals/chunk/chunk.md); use the reference arm"
    if len(set(widths)) != 1:
        return f"widths={widths}; the built manifest is uniform-width"
    if _arm_for(len(widths), widths[0]) is None:
        return f"topology (D={len(widths)}, B={widths[0]}) has no built arm in the manifest"
    if routes.tokens % _PIN_C:
        return f"T={routes.tokens} is not a multiple of the chunk C={_PIN_C}"
    return None


def scan_arms() -> set[tuple[int, int, int, int]]:
    """The `(C, BC, D, B)` arms the two SCANS are built for -- read off the census,
    not restated, so this cannot disagree with what shipped."""
    return {tuple(row[:4]) for row in extension().chunk_scan_b_census()}


def backward_unsupported(routes, decay, *, d_v: int) -> str | None:
    """Why a GRAD-BEARING call has no arm, or None if it trains.

    The forward's arm matrix is wider than the scans': the reverse pass carries the
    TRAINABLE topologies (`tools/gen_shards.py`'s `SCAN_ARMS`), not every arm the
    forward can infer at. A forward that runs and a backward that does not is
    exactly the known-broken path this refuses at the boundary instead of
    discovering three launches later.
    """
    reason = unsupported(routes, decay, d_v=d_v)
    if reason is not None:
        return reason
    top = routes.topology
    c, bc = arm(routes)
    want = (c, bc, len(top.levels), top.levels[0].width)
    if want not in scan_arms():
        return (f"the reverse pass is not built for arm C={want[0]} BC={want[1]} "
                f"depth={want[2]} width={want[3]}; it carries "
                f"{sorted(scan_arms())} (docs/internals/chunk/backward.md)")
    return None


def _side_plane(levels) -> torch.Tensor | None:
    """The `[B, H, T, sumW]` bf16 plane these levels are already views OF, or None.

    A STRIDE fact, not a provenance one: the levels are the plane iff each is the
    window of one bf16 allocation whose layout is exactly this side's, which is
    checkable and is what a hand-built bundle either satisfies or does not
    (docs/internals/chunk/chunk.md#the-side-plane).
    """
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    total = sum(widths)
    want = (heads * tokens * total, total, tokens * total, 1)
    base = levels[0]
    offset = 0
    for level, width in zip(levels, widths):
        if (level.dtype is not torch.bfloat16
                or level.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
                or level.stride() != want
                or level.storage_offset() != base.storage_offset() + offset):
            return None
        offset += width
    span = base.storage_offset() + batch * heads * tokens * total
    if span * base.element_size() > base.untyped_storage().nbytes():
        return None
    return base.as_strided((batch, heads, tokens, total),
                           (heads * tokens * total, tokens * total, total, 1),
                           base.storage_offset())


def pack_side(levels) -> torch.Tensor:
    """The per-level `[B, T, H, width_l]` tuple as one `[B, H, T, sumW]` bf16 tensor.

    ONE destination and one pass per level: concatenating in the public layout
    materializes an intermediate no kernel reads, and the transpose and the
    narrowing are then a second pass over it. A side that is already the plane skips
    all of it -- the producer emits this layout.
    """
    plane = _side_plane(levels)
    if plane is not None:
        return plane
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    packed = torch.empty((batch, heads, tokens, sum(widths)),
                         dtype=torch.bfloat16, device=levels[0].device)
    offset = 0
    for level, width in zip(levels, widths):
        packed[:, :, :, offset:offset + width].copy_(level.permute(0, 2, 1, 3))
        offset += width
    return packed


def backward_seeds(plan: BackwardPlan, dy: torch.Tensor, y: torch.Tensor):
    """`(dnum, dden, dm)` -- the readout divide's own reverse, over one retained plan.

    `dm = eps * dden` exists exactly when the bundle carries a read mass. The eps
    floor is ADDITIVE, so all three are exact and none of them branches on `den`.
    """
    fwd = plan.forward
    dyk = dy if dy.is_contiguous() else dy.contiguous()
    out = extension().chunk_backward_seeds(dyk, y, plan.den, fwd.read_mass, fwd.eps)
    return out[0], out[1], (out[2] if len(out) > 2 else None)


def _scan_operands(plan: BackwardPlan, seeds: Seeds) -> tuple:
    """THE SCANS' SHARED ABI, spelled ONCE and read off the plan.

    Both scans take the same ten arguments because they walk the same schedule from
    the same tables in opposite directions (docs/internals/chunk/backward.md). What
    made that a hazard was the tuple being hand-built at the call site, where the two
    could drift apart by an edit to one of them.
    """
    fwd = plan.forward
    return (fwd.read_plane, fwd.write_plane, fwd.g_write, fwd.v, seeds.dnum,
            seeds.dden, fwd.union_table, fwd.arm.c, fwd.arm.bc, list(fwd.widths))


def scan_a(plan: BackwardPlan, seeds: Seeds) -> torch.Tensor:
    """SCAN A: the read side's digit gradient, fp32 in the kernel's layout.

    It rebuilds the forward state from zero over the RETAINED union table -- no
    per-chunk snapshot is read, because none was kept -- and is independent of
    :func:`scan_b`, which the backward DAG declares.
    """
    (d_read,) = extension().chunk_scan_a_probe(*_scan_operands(plan, seeds))
    return d_read


def scan_b(plan: BackwardPlan, seeds: Seeds):
    """SCAN B: `(dv, d_gain, d_write)`, fp32 in the kernels' layouts.

    It walks the same tokens downward carrying the adjoint state. Its slot-sum
    output is a probe byproduct that the operand gradients do not read.

    The two scans associate their sums differently by construction -- each walks its
    own partition -- so nothing either returns is bit-reproducible, and neither is
    the fan-in that assembles the digit gradients (`docs/api.md`).
    """
    dv, _slot_sum, d_gain, d_write = extension().chunk_scan_b_probe(
        *_scan_operands(plan, seeds))
    return dv, d_gain, d_write


def unpack_grads(plan: BackwardPlan, dv, d_read, d_write, d_gain, d_mass,
                 *, v_dtype: torch.dtype) -> list[torch.Tensor]:
    """The scans' fp32 kernel layouts back to the OPERANDS', in `tensors()`' order.

    That order is load-bearing rather than incidental: the autograd node receives the
    operands flat and in exactly that order, so a gradient list ordered any other way
    lands every gradient on the wrong tensor SILENTLY -- the shapes agree
    (`tests/unit/test_interface.py`). Each gradient is narrowed to its operand's dtype
    and un-permuted to its public layout because autograd requires
    `grad.dtype == input.dtype`, and the packing that produced the kernel layout was
    the node's own.
    """
    routes = plan.forward.routes

    def levels(packed_side, side):
        out, off = [], 0
        for level in side:
            w = level.shape[-1]
            out.append(packed_side[..., off:off + w].permute(0, 2, 1, 3)
                       .to(level.dtype).contiguous())
            off += w
        return out

    grads = [dv.permute(0, 2, 1, 3).to(v_dtype).contiguous()]
    grads += levels(d_read, routes.read)
    grads += levels(d_write, routes.write)
    #: `g_read` does not exist on this surface, so the gains are the write gain
    #: alone -- the same one clause `tensors()` builds.
    grads.append(d_gain.permute(0, 2, 1).to(routes.g_write.dtype).contiguous())
    if routes.read_mass is not None:
        #: SURPRISE 2: the m-carry is forward-exact but NOT gradient-neutral.
        grads.append(d_mass.permute(0, 2, 1).to(routes.read_mass.dtype).contiguous())
    return grads


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """An empty `[BH, N/16, 16, d_v+1]` fp32 state plane for this routing.

    The ATOM-MAJOR view of `[BH, N, d_v+1]`, byte for byte: leaves are stored in
    canonical order, so the reshape between the two is free and the kernel's
    per-atom addressing and a reader's per-leaf one describe the same bytes.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in. It is uninitialized because the
    dense plane is written WHOLE: the owner blocks tile the leaf space exactly,
    so every atom is stored by exactly one CTA.
    """
    widths = tuple(l.width for l in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, MMA_K_QUANTUM, d_v + 1)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.float32)


def atom_bits(write_plane: torch.Tensor, widths) -> torch.Tensor:
    """`[BH, N/16]` uint8: the atoms this routing's WRITE side touches, exactly.

    The page plan's input (docs/internals/paging/stats.md). It is keyed to the
    16-leaf quantum and not to `BC`, so it is a property of the routing alone and is
    legitimately taken BEFORE an arm is chosen -- which is what lets residency be an
    OUTPUT of the plan rather than a consequence of it. Device-side: no host
    synchronization happens here.
    """
    return extension().chunk_stats(write_plane, list(widths))


def execute(plan: ChunkPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """`(y [B,T,H,d_v], den)` -- the consumer launch and the divide, over one plan.

    Every launch argument is a plan field: nothing is derived, chosen or re-packed
    here, which is the whole of what the facts engine bought.

    A block-grain liveness bitmap was reduced over this call's union table, and a
    block whose byte says it contributes nothing exits before it touches anything
    (docs/internals/chunk/chunk_kernel.md#liveness-exit).
    """
    m = extension()
    num, den = m.chunk_forward_raw(
        plan.read_plane, plan.write_plane, plan.g_write, plan.v, plan.union_table,
        plan.arm.c, plan.arm.bc, list(plan.widths), plan.state_in, plan.state_out,
        plan.page_table, plan.block_bits)
    #: THE READ MASS RIDES INTO THE FLOOR. `read_mass` is the per-token scale the read
    #: levels still carry, so numerator and denominator are both `m` times the
    #: normalized ones; passing `eps * m` makes the divide `m*num / (m*(den + eps))`,
    #: which is the defined readout exactly rather than to within a scaled floor.
    y = m.chunk_divide_pass(num, den, plan.read_mass, plan.eps, plan.y_dtype)
    return y, den

