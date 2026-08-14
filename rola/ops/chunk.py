"""The CHUNK consumer's execution path (operator adoption; docs/internals/chunk/) -- the kernel arm.

`rola_op`'s facade is untouched: this module is the private seam's kernel-arm
implementation, opinionated about what it supports and LOUD about what it
does not (the facade stays generic; capabilities return through the paging
and backward work, not
through silent degradation).

The packing performed here (level concat + [B,T,H,*] -> [B,H,T,*] permutes)
is torch glue, measured and scheduled for deletion by the e2e cleanup step
(fused into the producer epilogue); the seam's shape does not change when it
goes.
"""
from __future__ import annotations

import torch

from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.paging import MMA_K_QUANTUM

#: the pinned plan (measured: BC at the dense optimum, C at the flagship,
#: swizzle off) -- generalized only by falling back to the largest built BC
#: when the pinned one is not in the manifest for this topology.  A fixed
#: preference order over MEASURED arms, never a tuned heuristic.
_PIN_C, _PIN_BC = 32, 128


def _arm_for(D: int, B: int):
    """The manifest arm this topology runs, or None."""
    arms = [tuple(a[:4]) for a in extension().chunk_arms()]
    bcs = [bc for (c, bc, d, b) in arms if d == D and b == B and c == _PIN_C]
    if not bcs:
        return None
    return (_PIN_C, _PIN_BC if _PIN_BC in bcs else max(bcs))


def unsupported(routes, decay, initial_state, output_final_state) -> str | None:
    """Why the chunk arm refuses this call, or None if it runs."""
    top = routes.topology
    widths = tuple(l.width for l in top.levels)
    if top.d_v != 64:
        return f"d_v={top.d_v}; the chunk matrix is built at d_v=64"
    if routes.normalization != "global":
        return f"normalization={routes.normalization!r}; the chunk consumer is global-only"
    if decay is not None:
        return "decay is not implemented in the operator (the decay retirement; docs/internals/chunk/chunk.md); use the reference arm"
    if initial_state is not None or output_final_state:
        return "state chaining arrives with the paged state I/O work (docs/internals/paging/); not yet in the chunk consumer"
    if len(set(widths)) != 1:
        return f"widths={widths}; the built manifest is uniform-width"
    if _arm_for(len(widths), widths[0]) is None:
        return f"topology (D={len(widths)}, B={widths[0]}) has no built arm in the manifest"
    if routes.tokens % _PIN_C:
        return f"T={routes.tokens} is not a multiple of the chunk C={_PIN_C}"
    return None


def stats(routes) -> torch.Tensor:
    """`[BH, N/16]` uint8: the atoms this routing's WRITE side touches, exactly.

    The page plan's input (docs/internals/paging/stats.md).  It is keyed to the
    16-leaf quantum and not to `BC`, so it is a property of the routing alone and
    is legitimately taken BEFORE an arm is chosen -- which is what lets residency
    be an OUTPUT of the plan rather than a consequence of it.  Device-side: no
    host synchronization happens here.
    """
    widths = tuple(l.width for l in routes.topology.levels)
    pw = torch.cat(routes.write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    return extension().chunk_stats(pw, list(widths))


def state_plane(routes, *, zeros: bool = False) -> torch.Tensor:
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
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, MMA_K_QUANTUM,
             routes.topology.d_v + 1)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.float32)


def run(routes, v: torch.Tensor, state_in: torch.Tensor | None = None,
        want_state: bool = False, arena=None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`(y [B,T,H,d_v], final state or None)` through table pass, kernel, divide.

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
    """
    m = extension()
    widths = tuple(l.width for l in routes.topology.levels)
    c, bc = _arm_for(len(widths), widths[0])
    pr = torch.cat(routes.read, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    pw = torch.cat(routes.write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    gw = routes.g_write.permute(0, 2, 1).to(torch.bfloat16).contiguous()
    vk = v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    if arena is None:
        page_tbl = None
        s_out = state_plane(routes) if want_state else None
    else:
        #: the admission's table write and slot zeroing ran on the arena's side
        #: stream; this is the one wait that puts them before the launch.
        arena.wait()
        page_tbl = arena.page_table
        s_out = arena.state if want_state else None
    y = m.chunk_forward(pr, pw, gw, vk, c, bc, list(widths), READOUT_EPS, state_in, s_out,
                        page_tbl)
    #: kernel arm hands back the operand dtype (the facade's documented
    #: contract); the fp32 accumulator is the kernel's, not the caller's.
    return y.permute(0, 2, 1, 3).to(v.dtype), s_out
