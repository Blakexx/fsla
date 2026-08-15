"""The CHUNK consumer's execution path (operator adoption; docs/internals/chunk/) -- the kernel arm.

`rola_op`'s facade is untouched: this module is the private seam's kernel-arm
implementation, opinionated about what it supports and LOUD about what it
does not (the facade stays generic; capabilities return through the paging
and backward work, not
through silent degradation).

The packing performed here is torch glue, and it happens ONCE per call
(`pack` -> `PackedRoutes`), so every entry a call makes reads the same bytes.
It is still scheduled for deletion -- the producer's epilogue is where a
`[B, H, T, sumW]` bf16 side is eventually emitted directly -- and the seam's
shape does not change when it goes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.paging import MMA_K_QUANTUM

#: THE CHUNK LENGTH -- the operator's token quantum. `T` must be a multiple of it,
#: and shaping a sequence to it is the CALLER's job, which is why this is public
#: rather than only reachable through a refusal message.
CHUNK_TOKENS = 32

#: The pinned plan: a fixed preference order over MEASURED arms, never a tuned
#: heuristic. It generalizes only by falling back to the largest built BC when the
#: pinned one is not in this topology's manifest.
_PIN_C, _PIN_BC = CHUNK_TOKENS, 128


def _pack(view: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
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


def _pack_side(levels) -> torch.Tensor:
    """The per-level `[B, T, H, width_l]` tuple as one `[B, H, T, sumW]` bf16 tensor.

    ONE destination and one pass per level: concatenating in the public layout
    materializes an intermediate no kernel reads, and the transpose and the
    narrowing are then a second pass over it.
    """
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    packed = torch.empty((batch, heads, tokens, sum(widths)),
                         dtype=torch.bfloat16, device=levels[0].device)
    offset = 0
    for level, width in zip(levels, widths):
        packed[:, :, :, offset:offset + width].copy_(level.permute(0, 2, 1, 3))
        offset += width
    return packed


@dataclass(frozen=True, slots=True)
class PackedRoutes:
    """One call's routing bundle in the KERNEL's layout, materialized ONCE.

    The bundle's public layout is `[B, T, H, width_l]` per level; every chunk entry
    reads `[B, H, T, sumW]` bf16 per side. Both the stats pass and the launch need
    exactly that, so it is built once per call and carried, rather than derived at
    each entry from the bundle that outlives them.

    `routes` rides along because the launch's non-tensor facts -- the topology, the
    device, `T` -- are the bundle's and are not recoverable from the packed tensors.
    """

    routes: object
    pr: torch.Tensor
    pw: torch.Tensor
    gw: torch.Tensor
    rm: torch.Tensor | None


def pack(routes) -> PackedRoutes:
    """THE packing, and the only one: `routes` in the layout the kernels read."""
    mass = routes.read_mass
    return PackedRoutes(
        routes=routes,
        pr=_pack_side(routes.read),
        pw=_pack_side(routes.write),
        gw=_pack(routes.g_write.permute(0, 2, 1)),
        #: fp32, because it scales the readout's floor and `eps` is an fp32 definition
        #: constant; bf16 here would quantize the floor, not the amplitudes.
        rm=None if mass is None else _pack(mass.permute(0, 2, 1), torch.float32))


def stats(packed: PackedRoutes) -> torch.Tensor:
    """`[BH, N/16]` uint8: the atoms this routing's WRITE side touches, exactly.

    The page plan's input (docs/internals/paging/stats.md).  It is keyed to the
    16-leaf quantum and not to `BC`, so it is a property of the routing alone and
    is legitimately taken BEFORE an arm is chosen -- which is what lets residency
    be an OUTPUT of the plan rather than a consequence of it.  Device-side: no
    host synchronization happens here.
    """
    widths = [l.width for l in packed.routes.topology.levels]
    return extension().chunk_stats(packed.pw, widths)


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


def run(packed: PackedRoutes, v: torch.Tensor, state_in: torch.Tensor | None = None,
        want_state: bool = False, arena=None,
        sparse_grid: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`(y [B,T,H,d_v], final state or None)` through table pass, kernel, divide.

    `packed` is this call's :func:`pack` -- the routing already in the kernels'
    layout, so a stateful call's stats pass and its launch read the same bytes
    instead of each building their own copy of them.

    `state_in` is a plane of the call's backing, or `None` for a fresh sequence;
    `want_state` asks for the carried state back. A call that asks for neither
    runs the kernel variant with no state I/O compiled into it at all
    (docs/internals/chunk/chunk_kernel.md), so it pays nothing for the feature.

    `arena` is a `PageArena` already planned over this call's write set: it makes
    the backing PAGED, so the plane is its `[slots, 16, cols]` storage and its
    table resolves each atom to a slot. The two backings are ONE kernel and one
    base translation -- an atom with no slot is skipped, which is exact rather
    than defensive because the plan's bitmap is the write set exactly
    (docs/internals/paging/paging.md). Dense is `arena=None`, where the slot IS
    the atom.

    `sparse_grid` runs the launch over LIVE BLOCKS ONLY: a block-grain liveness
    reduction over this call's union table, then a fixed persistent grid whose
    work loop skips the dead ones. It is the PLAN's choice and never a density
    heuristic inside the kernel -- the reduction costs a launch of its own, so it
    pays where the dead-block count is large and not where it is not
    (docs/internals/chunk/chunk_kernel.md#work-loop).
    """
    m = extension()
    routes = packed.routes
    widths = tuple(l.width for l in routes.topology.levels)
    c, bc = _arm_for(len(widths), widths[0])
    pr, pw, gw = packed.pr, packed.pw, packed.gw
    vk = _pack(v.permute(0, 2, 1, 3))
    if arena is None:
        page_tbl = None
        s_out = state_plane(routes, d_v=v.shape[-1]) if want_state else None
    else:
        #: the admission's table write and slot zeroing ran on the arena's side
        #: stream; this is the one wait that puts them before the launch.
        arena.wait()
        page_tbl = arena.page_table
        s_out = arena.state if want_state else None
    tbl = m.chunk_union_table(pr, pw, vk, c, bc, list(widths))
    bits = (m.chunk_block_bits(tbl, bc, list(widths), routes.tokens, page_tbl)
            if sparse_grid else None)
    num, den = m.chunk_forward_raw(pr, pw, gw, vk, tbl, c, bc, list(widths), state_in, s_out,
                                   page_tbl, bits)
    #: THE READ MASS RIDES INTO THE FLOOR. `packed.rm` is the per-token scale the read
    #: levels still carry, so numerator and denominator are both `m` times the
    #: normalized ones; passing `eps * m` makes the divide `m*num / (m*(den + eps))`,
    #: which is the defined readout exactly rather than to within a scaled floor.
    y = m.chunk_divide_pass(num, den, packed.rm, READOUT_EPS, v.dtype)
    return y, s_out
