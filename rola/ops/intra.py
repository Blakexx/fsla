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

WINDOW = 256
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


def intra_forward(pread, pwrite, gwrite, v, modes, *, o=None, den=None):
    """RED the within-window causal intra term into ``(o, den)`` and return them.

    ``pread``/``pwrite`` are ``[BH, L, D*256]`` bf16 operand planes, ``gwrite`` is
    ``[BH, L]`` bf16, ``v`` is token-major ``[B, T, H, 64]`` bf16. ``modes`` is one
    per-level support mode.
    """
    bh, length, width = pread.shape
    levels = len(modes)
    if width != levels * LEVEL_WIDTH:
        raise ValueError(f"the plane is {width} wide; {levels} levels want {levels * LEVEL_WIDTH}")
    if o is None:
        o = torch.zeros((bh, length, VALUE_WIDTH), dtype=torch.float32, device=pread.device)
    if den is None:
        den = torch.zeros((bh, length), dtype=torch.float32, device=pread.device)
    extension().intra_forward(pread, pwrite, gwrite, v, o, den, levels, pack_modes(modes))
    return o, den


def build_stamp() -> int:
    """A fact only the loaded binary can produce — the extension-trap check."""
    return extension().intra_build_stamp()


__all__ = [
    "DENSE_BOTH",
    "READ_SPARSE",
    "SHARED_SPARSE",
    "WRITE_SPARSE",
    "build_stamp",
    "intra_forward",
    "pack_modes",
]
