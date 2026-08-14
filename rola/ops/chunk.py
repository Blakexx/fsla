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


def run(routes, v: torch.Tensor) -> torch.Tensor:
    """y [B,T,H,d_v] through table pass + chunk kernel + divide."""
    m = extension()
    widths = tuple(l.width for l in routes.topology.levels)
    c, bc = _arm_for(len(widths), widths[0])
    pr = torch.cat(routes.read, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    pw = torch.cat(routes.write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    gw = routes.g_write.permute(0, 2, 1).to(torch.bfloat16).contiguous()
    vk = v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    y = m.chunk_forward(pr, pw, gw, vk, c, bc, list(widths), READOUT_EPS)
    #: kernel arm hands back the operand dtype (the facade's documented
    #: contract); the fp32 accumulator is the kernel's, not the caller's.
    return y.permute(0, 2, 1, 3).to(v.dtype)
