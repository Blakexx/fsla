# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The K31 intra kernel's call surface — a side kernel, outside the shipped dispatch.

The kernel REDUCES into ``o`` and ``den`` (``fan_in_add`` semantics) rather than
returning them, so the caller owns their allocation and their initial contents;
:func:`intra_forward` allocates zeros when none are handed in, which is what a
standalone conformance run wants and what a concurrent run with the carry kernel
must never do.
"""

from __future__ import annotations

import torch

from rola.ops._ext import extension

DENSE_BOTH = 0
READ_SPARSE = 1
WRITE_SPARSE = 2
SHARED_SPARSE = 3

#: the built windows -- the carry family's `W` arms, mirrored (integration rung
#: I1(b)).  ``WINDOW`` names the flagship because the architecture doc's two
#: constraints (the C* basin and the intra's ``W/4`` flop/byte intensity) settle
#: it there; ``WINDOW_SMALL`` is the conformance cell's.
WINDOW = 384
WINDOW_WIDE = 512
WINDOW_SMALL = 64
LEVEL_WIDTH = 256
VALUE_WIDTH = 64


def pack_modes(modes: tuple[int, ...]) -> int:
    """Two bits per level, level 0 in the low pair."""
    packed = 0
    for level, mode in enumerate(modes):
        if mode not in (DENSE_BOTH, READ_SPARSE, WRITE_SPARSE, SHARED_SPARSE):
            raise ValueError(f"level {level}: unknown support mode {mode!r}")
        packed |= mode << (2 * level)
    return packed


def intra_forward(pread, pwrite, gwrite, v, modes, *, o=None, den=None, window=WINDOW):
    """RED the within-window causal intra term into ``(o, den)`` and return them.

    ``pread``/``pwrite`` are ``[BH, L, D*256]`` bf16 operand planes, ``gwrite`` is
    ``[BH, L]`` bf16, ``v`` is token-major ``[B, T, H, 64]`` bf16. ``modes`` is one
    per-level support mode, and ``window`` names the built arm -- the SAME window
    the carry call is given, since the two kernels share one grid.
    """
    bh, length, width = pread.shape
    levels = len(modes)
    if width != levels * LEVEL_WIDTH:
        raise ValueError(f"the plane is {width} wide; {levels} levels want {levels * LEVEL_WIDTH}")
    if o is None:
        o = torch.zeros((bh, length, VALUE_WIDTH), dtype=torch.float32, device=pread.device)
    if den is None:
        den = torch.zeros((bh, length), dtype=torch.float32, device=pread.device)
    extension().intra_forward(pread, pwrite, gwrite, v, o, den, levels, pack_modes(modes),
                              int(window))
    return o, den


def arms():
    """The built ``(levels, window, smem_bytes)`` rows."""
    return extension().intra_arms()


def build_stamp() -> int:
    """A fact only the loaded binary can produce — the extension-trap check."""
    return extension().intra_build_stamp()


__all__ = [
    "DENSE_BOTH",
    "READ_SPARSE",
    "SHARED_SPARSE",
    "WINDOW",
    "WINDOW_SMALL",
    "WINDOW_WIDE",
    "WRITE_SPARSE",
    "arms",
    "backward_arms",
    "backward_stamp",
    "build_stamp",
    "intra_backward",
    "intra_forward",
    "pack_modes",
]


def intra_backward(pread, pwrite, gwrite, v, dnum, dden, modes, *, window=WINDOW,
                   d_pr=None, d_pw=None, d_gw=None, d_v=None):
    """RED the WITHIN-WINDOW half of the four cotangents into the gradient planes.

    ``dnum`` is ``[BH, L, 64]`` bf16 and ``dden`` ``[BH, L]`` fp32 -- the seeds the op
    seam forms from the division's backward. The planes are the carry's reverse scans'
    too, so a standalone conformance run allocates its own zeros and a run beside the
    scans must not.
    """
    bh, length, width = pread.shape
    levels = len(modes)
    if width != levels * LEVEL_WIDTH:
        raise ValueError(f"the plane is {width} wide; {levels} levels want {levels * LEVEL_WIDTH}")
    opt = {"dtype": torch.float32, "device": pread.device}
    if d_pr is None:
        d_pr = torch.zeros((bh, length, width), **opt)
    if d_pw is None:
        d_pw = torch.zeros((bh, length, width), **opt)
    if d_gw is None:
        d_gw = torch.zeros((bh, length), **opt)
    if d_v is None:
        d_v = torch.zeros((bh, length, VALUE_WIDTH), **opt)
    extension().intra_backward(pread, pwrite, gwrite, v, dnum, dden, d_pr, d_pw, d_gw, d_v,
                               levels, pack_modes(modes), int(window))
    return d_pr, d_pw, d_gw, d_v


def backward_arms():
    """The built reverse ``(levels, window, smem, regs, local, threads)`` rows."""
    return extension().intra_backward_arms()


def backward_stamp() -> int:
    """A fact only the loaded binary can produce -- the reverse pass's extension check."""
    return extension().intra_backward_stamp()
