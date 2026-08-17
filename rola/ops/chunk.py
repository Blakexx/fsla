"""The CHUNK consumer's execution path (operator adoption; docs/internals/chunk/) -- the kernel arm.

`rola_op`'s facade is untouched: this module is the private seam's kernel-arm
implementation, opinionated about what it supports and LOUD about what it
does not (the facade stays generic; capabilities return through the paging
and backward work, not
through silent degradation).

Launch arguments arrive as a :class:`~rola.facts.plan.ChunkPlan` that the chunk DAG
built (:mod:`rola.facts.chunk_dag`). What lives here is the family's primitive
BODIES -- the arm, the envelope, the packing, the state plane -- and the launch that
reads the plan; the DAG declares which of them a call class runs and in what order.

The packing happens ONCE per call, so every entry reads the same bytes. On the
shipped path it is a VIEW: the producer's solve writes the `[B, H, T, sumW]` bf16
side directly and `_side_plane` reconstructs it from the levels' strides
(docs/internals/chunk/chunk.md#the-side-plane). The copy survives for the sides that
are not already that layout -- a hand-built bundle, a foreign producer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rola.ops._ext import extension
from rola.ops.paging import MMA_K_QUANTUM

if TYPE_CHECKING:
    from rola.facts.plan import ChunkPlan

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


@dataclass(slots=True)
class SavedForward:
    """THE CTX CONTRACT: what the forward hands the backward, and nothing else.

    Everything here is `O(L)` with a per-topology constant, and that is the claim
    the whole two-pass design rests on: the backward REBUILDS the state and
    REPLAYS the schedule, so no per-chunk snapshot and nothing of size `O(N * L)`
    is retained. The state, the per-chunk compaction lists, the causal thresholds
    and the numerator are absent by construction. The union table is the ONE
    forward byproduct inherited by save rather than by replay: rebuilding it would
    be a second launch over the whole amplitude plane for a bit-identical answer,
    and it is the seam the schedule is replayed from
    (docs/internals/chunk/backward.md).

    Populated only when the call carries a gradient; on an inference call these
    tensors are freed with the launch exactly as they were before this existed.
    """

    pr: torch.Tensor | None = None
    pw: torch.Tensor | None = None
    gw: torch.Tensor | None = None
    rm: torch.Tensor | None = None
    tbl: torch.Tensor | None = None
    den: torch.Tensor | None = None
    vk: torch.Tensor | None = None
    #: the plan's non-tensor half: the arm, the topology, the readout constant and
    #: V's row stride, which the backward's launch needs and cannot re-derive.
    c: int = 0
    bc: int = 0
    widths: tuple[int, ...] = ()
    eps: float = 0.0
    v_row_bytes: int = 0

    @staticmethod
    def of(plan: ChunkPlan, den: torch.Tensor) -> SavedForward:
        """The ctx a grad-bearing call keeps, off the plan it just ran.

        Every field is a reference to something the call already built, so a saving
        call and an inference call issue the same launches in the same order; what
        differs is only what stays alive.
        """
        return SavedForward(
            pr=plan.read_plane, pw=plan.write_plane, gw=plan.g_write,
            rm=plan.read_mass, tbl=plan.union_table, den=den, vk=plan.v,
            c=plan.arm.c, bc=plan.arm.bc, widths=plan.widths, eps=plan.eps,
            v_row_bytes=plan.v_row_bytes)

    def retained_bytes(self) -> int:
        """What a grad-bearing call costs OVER an inference one. `y` and `v` are the
        op's own output and operand, alive either way, so neither is counted here
        and neither is stored here."""
        return sum(t.numel() * t.element_size()
                   for t in (self.pr, self.pw, self.gw, self.rm, self.tbl, self.den)
                   if t is not None)

    def kernel_v_is_the_callers(self, v: torch.Tensor) -> bool:
        """Whether `vk` aliases the caller's `v` rather than being a repack of it --
        the one field whose cost depends on the caller's dtype and contiguity."""
        return self.vk is not None and self.vk.data_ptr() == v.data_ptr()


def backward(saved: SavedForward, dy: torch.Tensor, y: torch.Tensor):
    """The reverse pass over one saved forward: `(dv, d_read, d_write, d_gw, d_m)`.

    Three launches and no allocation of anything `O(N * L)`:

    1. the SEEDS (`chunk_backward_seeds`) -- `dnum`, `dden`, and `dm = eps * dden`
       when a read mass exists. The eps floor is additive, so all three are exact
       and none of them branches on `den`;
    2. SCAN A, which rebuilds the forward state from zero over the SAVED table and
       emits the read side's digit gradient;
    3. SCAN B, which walks the same tokens downward carrying the adjoint state and
       emits `dv`, `d gw` and the write side's digit gradient.

    The two scans associate their sums differently by construction -- each walks
    its own partition -- so nothing here is bit-reproducible, and neither is the
    fan-in that assembles the digit gradients (`docs/api.md`).

    Everything returned is fp32, in the KERNEL's layouts. Narrowing to the
    operands' dtypes and un-permuting to their public layouts is the caller's, and
    :class:`rola.interface.ChunkConsumer` is the one caller.
    """
    if saved.tbl is None or saved.vk is None:
        raise RuntimeError(
            "rola.ops.chunk.backward was handed a ctx the forward never filled. The "
            "forward populates it only when the call carries a gradient, so this is a "
            "backward reached from an inference forward -- which cannot happen through "
            "ChunkConsumer and means the node was bypassed.")
    m = extension()
    dyk = dy if dy.is_contiguous() else dy.contiguous()
    seeds = m.chunk_backward_seeds(dyk, y, saved.den, saved.rm, saved.eps)
    dnum, dden = seeds[0], seeds[1]
    dm = seeds[2] if len(seeds) > 2 else None
    args = (saved.pr, saved.pw, saved.gw, saved.vk, dnum, dden, saved.tbl,
            saved.c, saved.bc, list(saved.widths))
    (dpr,) = m.chunk_scan_a_probe(*args)
    dv, _dw_slotsum, dgw, dpw = m.chunk_scan_b_probe(*args)
    return dv, dpr, dpw, dgw, dm


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

