# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS, as a caller reaches it: one launch over the bundle's rows for both sides' class-1 words and the
write lists' exponents.

The words' LAYOUT, their reader, the folds over them and the pure-torch MODEL the kernel is graded against live in
:mod:`rola.engine.facts.liveness`; the list exponents' definition is the fp16 fold's reference
(``tests/oracle/fold_reference.py``, ``list_exponents``). What is here is the call: the layout and the two sides' static operands a bundle's topology implies
(:func:`call_layout`, :func:`call_statics`), packed into the bits R13 requires so no logical width crosses the seam,
and the launch (docs/internals/facts/facts_api.md).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts.liveness import LivenessLayout, SideStatics, mask_words, side_statics
from rola.ops._ext import extension
from rola.ops.carry import WINDOW

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "WINDOW", "Facts", "call_layout", "call_statics", "facts", "ledger_bind",
           "list_level"]

#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `FactsPhase` order, booked over every block.
LEDGER_PHASES = ("columns", "vote", "store")
LEDGER_WARPS = 8


class Facts(NamedTuple):
    """One call's facts: ``words`` the ``[BH, 2, rows, ceil(L / 32)]`` int32 class-1 table (:class:`LivenessLayout`),
    ``unions`` the ``[BH, 2, groups, ceil(L / 64)]`` int32 tile unions (``tile_unions``), and ``list_exp`` the
    ``[BH, windows, B_list, 2]`` int16 list exponents -- per window and digit of the list level (:func:`list_level`)
    the smallest ``K`` with every token's ``b m <= 2^K`` (column 0) and ``b m s <= 2^K`` (column 1), ``b`` its write
    byte (a dense level's zero at the floor), ``m`` the write gain times the write levels' scales and ``s`` the value
    scale; ``-126`` for a list no token writes. ``B_list`` is 0 when the call has no list level."""

    words: torch.Tensor
    unions: torch.Tensor
    list_exp: torch.Tensor


def call_layout(routes) -> LivenessLayout:
    """The layout a bundle's words are read in: every level at its padded width, the call's tokens."""
    widths = routes.topology.padded_widths
    return LivenessLayout(D=len(widths), B=widths, L=routes.tokens)


def call_statics(routes, layout: LivenessLayout) -> tuple[SideStatics, SideStatics]:
    """``(read, write)``: each side's static operands as the bundle declares them -- a level not flagged ``sparse`` is
    DENSE (not read; its words are the width mask), and the rows' own widths are the logical ones."""
    return tuple(side_statics(layout, tuple(not flag for flag in flags), routes.topology.widths)
                 for flags in routes.sparse)


def list_level(read_statics: SideStatics) -> int:
    """THE WRITE LISTS' LEVEL, the carry's level R: at depth one the level, at depth two the level that is not the
    read-sparse one (level S: level 1 when it is read-sparse or level 0 is not, else level 0); ``-1`` deeper, where
    the carry is not built (docs/internals/owner/owner.md)."""
    dense = read_statics.dense_levels
    if len(dense) == 1:
        return 0
    if len(dense) == 2:
        level_s = 1 if not dense[1] or dense[0] else 0
        return 1 - level_s
    return -1


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the kernel skips reads on."""
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def facts(routes, values, layout: LivenessLayout, read_statics: SideStatics, write_statics: SideStatics) -> Facts:
    """THE PASS: both sides' words, their tile unions and the list exponents from the bundle (``routes``,
    ``values``) in ONE launch.

    Same words, bit for bit, as :func:`rola.engine.facts.liveness.liveness_words` over the bundle's rows, and their
    unions as :func:`rola.engine.facts.liveness.tile_unions`; the list exponents those of ``list_exponents`` within
    fp32 rounding (``tests/integration/test_facts_pass.py``). The rows are read where the
    bundle holds them, any width, offset or dtype of the format, each level's ``[B, T, H, w_l]`` view by its strides.
    """
    BH = routes.batch * routes.heads
    window_words = WINDOW // 32
    words = torch.empty((BH, 2, layout.rows, layout.words), dtype=torch.int32, device=routes.device)
    unions = torch.empty((BH, 2, layout.groups, layout.tiles), dtype=torch.int32, device=routes.device)
    level = list_level(read_statics)
    width = layout.B[level] if level >= 0 else 0
    list_exp = torch.empty((BH, -(-layout.words // window_words), width, 2), dtype=torch.int16, device=routes.device)
    count = mask_words(layout.rows)
    extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
        words, unions, list_exp, list(layout.B),
        _dense_bits(read_statics), [read_statics.mask_word(i) for i in range(count)],
        _dense_bits(write_statics), [write_statics.mask_word(i) for i in range(count)], level, WINDOW)
    return Facts(words=words, unions=unions, list_exp=list_exp)


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/facts/facts.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[8, 3]`` int64 cycles (``LEDGER_PHASES``), or ``None``."""
    extension().rola_facts_ledger_bind(ledger)
