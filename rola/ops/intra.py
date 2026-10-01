# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTRA, as a caller reaches it: every pair inside one window, over the bundle, reduced into the output planes.

One launch over the bundle's rows and values (docs/internals/intra/intra_kernel.md): a window's Gram over each level's
u8 digits, the value product in u8 weights against the s8 values, and the rows REDUCED into the caller's ``num``
``[BH, L, 64]`` and ``den`` ``[BH, L]`` fp32 -- the planes the carry reduces into beside it, zeroed by their owner. The
reader's per-token factor is not applied: it cancels in ``num / den`` (the output contract, carry and intra alike). The
slab gate reads the facts pass's tile unions (`rola.ops.facts`), and a side's dense levels -- the same statics that
pass took -- are never gated.

The per-level support modes are the carry's geometry input (its carve order reads them), kept here until the carry's
int8 port (docs/internals/intra/intra_api.md).
"""
from __future__ import annotations

import torch

from rola.engine.facts.liveness import LivenessLayout, SideStatics
from rola.ops._ext import extension
from rola.ops.carry import WINDOW

#: THE PER-LEVEL MODE, two bits per level: bit 0 the READ side is value-sparse there, bit 1 the WRITE side. The carry's
#: geometry block orders its carve by it.
DENSE_BOTH = 0
READ_SPARSE = 1
WRITE_SPARSE = 2
BOTH_SPARSE = 3

#: The value width the kernel reads.
VALUE_WIDTH = 64
#: THE ROWS' LEVELS: u8 rows serve depth <= 2, each level a whole number of 16-digit copy granules up to 256.
MAX_LEVELS = 2
WIDTH_GRANULE = 16
MAX_LEVEL_WIDTH = 256
#: The value product's weights: two u8 digits unless a caller declares one (ruled 2026-09-29).
DEFAULT_WEIGHT_DIGITS = 2
#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `IntraPhase` order, booked over every CTA.
LEDGER_PHASES = ("reader", "gate", "wait", "gram", "epilogue", "value", "deposit")
LEDGER_WARPS = 8

__all__ = ["BOTH_SPARSE", "DEFAULT_WEIGHT_DIGITS", "DENSE_BOTH", "LEDGER_PHASES", "LEDGER_WARPS", "MAX_LEVELS",
           "MAX_LEVEL_WIDTH", "READ_SPARSE", "VALUE_WIDTH", "WIDTH_GRANULE", "WINDOW", "WRITE_SPARSE", "admits",
           "build_stamp", "footprint", "intra_forward", "ledger_bind", "pack_modes", "reads"]


def pack_modes(modes: tuple[int, ...]) -> int:
    """Two bits per level, level 0 in the low pair."""
    packed = 0
    for level, mode in enumerate(modes):
        if mode not in (DENSE_BOTH, READ_SPARSE, WRITE_SPARSE, BOTH_SPARSE):
            raise ValueError(f"level {level}: unknown support mode {mode!r}")
        packed |= mode << (2 * level)
    return packed


def _dense_bits(statics: SideStatics) -> int:
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def intra_forward(routes, values, unions: torch.Tensor, layout: LivenessLayout, read_statics: SideStatics,
                  write_statics: SideStatics, num: torch.Tensor, den: torch.Tensor, *,
                  weight_digits: int = DEFAULT_WEIGHT_DIGITS) -> None:
    """REDUCE the within-window pairs of the bundle (``routes``, ``values``) into ``num`` and ``den``.

    ``unions`` is the facts pass's ``[BH, 2, groups, L / 64]`` tile-union table in ``layout``; the statics are the
    ones it was made with. Each level's rows are read at their own width (:func:`admits`), at most ``layout``'s, and
    the call is a whole number of `WINDOW`-token windows.
    """
    extension().rola_intra_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.v_hat,
        values.v_lo, values.scale, unions, num, den, list(layout.B), _dense_bits(read_statics),
        _dense_bits(write_statics), int(weight_digits), WINDOW)


def footprint(widths) -> int:
    """The kernel's shared-memory bytes at these level widths (the seam refuses widths it does not read)."""
    return extension().intra_footprint([int(width) for width in widths])


def reads(widths) -> bool:
    """Whether the kernel reads levels of these widths: one or two, each a whole number of `WIDTH_GRANULE`-digit
    granules up to `MAX_LEVEL_WIDTH` (no device asked)."""
    widths = tuple(int(width) for width in widths)
    return 1 <= len(widths) <= MAX_LEVELS and all(
        width % WIDTH_GRANULE == 0 and WIDTH_GRANULE <= width <= MAX_LEVEL_WIDTH for width in widths)


def admits(widths, d_v: int, device: torch.device | int | None = None) -> bool:
    """Whether the intra runs a call at these level widths and value width: widths it :func:`reads`, ``d_v`` the
    `VALUE_WIDTH`, and the footprint within the device's shared-memory opt-in."""
    if d_v != VALUE_WIDTH or not reads(widths):
        return False
    return footprint(widths) <= torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def build_stamp() -> int:
    """A fact only the loaded binary can produce -- the extension-trap check."""
    return extension().intra_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/intra/intra_kernel.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[8, 7]`` int64 cycles, or ``None``."""
    extension().rola_intra_ledger_bind(ledger)
