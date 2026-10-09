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

from rola.engine.facts import lists as list_model
from rola.engine.facts.liveness import LivenessLayout, SideStatics, mask_words, side_statics
from rola.ops._ext import extension
from rola.ops.carry import WINDOW
from rola.routing.factors import carry_level_s

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "WINDOW", "Facts", "call_layout", "call_statics", "facts", "ledger_bind",
           "list_level", "read_list_level"]

#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `FactsPhase` order, booked over every block.
LEDGER_PHASES = ("columns", "vote", "lists", "store")
LEDGER_WARPS = 16


class Facts(NamedTuple):
    """One call's facts: ``words`` the ``[BH, 2, rows, ceil(L / 32)]`` int32 class-1 table (:class:`LivenessLayout`),
    ``unions`` the ``[BH, 2, groups, ceil(L / 64)]`` int32 tile unions (``tile_unions``), and ``list_exp`` the
    ``[BH, windows, B_list, 2]`` int16 list exponents -- per window and digit of the list level (:func:`list_level`)
    the smallest ``K`` with every token's ``b m <= 2^K`` (column 0) and ``b m s <= 2^K`` (column 1), ``b`` its write
    byte (a dense level's zero at the floor), ``m`` the write gain times the write levels' scales and ``s`` the value
    scale; ``-126`` for a list no token writes. ``B_list`` is 0 when the call has no list level. ``fm`` the ``[BH, L,
    2]`` fp32 factor products, each token's ``(F, m)`` = ``(m s, m)`` in that order (the fold's; written where the call
    has a list level). A DENSE level's rows and groups are left unwritten: they are its width mask over the live tokens,
    which a reader takes from the side's statics. ``write`` and ``read`` each side's lists at its listed level
    (:class:`rola.engine.facts.lists.ListSet`; empty where the level is dense or absent), ``gains16`` ``[BH, L, 2]`` f16
    each token's ``(m, m s)`` at its window's powers of two and ``gains_exp`` ``[BH, windows, 2]`` their exponents
    (docs/internals/facts/facts.md#lists-out, #gains)."""

    words: torch.Tensor
    unions: torch.Tensor
    list_exp: torch.Tensor
    fm: torch.Tensor
    write: list_model.ListSet
    read: list_model.ListSet
    gains16: torch.Tensor
    gains_exp: torch.Tensor


def call_layout(routes) -> LivenessLayout:
    """The layout a bundle's words are read in: every level at its padded width, the call's tokens."""
    widths = routes.topology.padded_widths
    return LivenessLayout(D=len(widths), B=widths, L=routes.tokens)


def call_statics(routes, layout: LivenessLayout) -> tuple[SideStatics, SideStatics]:
    """``(read, write)``: each side's static operands as the bundle declares them -- a level not flagged ``sparse`` is
    DENSE (neither read nor written: its words are the width mask), and the rows' own widths are the logical ones."""
    return tuple(side_statics(layout, tuple(not flag for flag in flags), routes.topology.widths)
                 for flags in routes.sparse)


def list_level(read_statics: SideStatics) -> int:
    """THE WRITE LISTS' LEVEL, the carry's level R: at depth one the level, at depth two the level that is not the
    read-sparse one (level S: level 1 when it is read-sparse or level 0 is not, else level 0); ``-1`` deeper, where
    the carry is not built (docs/internals/owner/owner.md)."""
    dense = read_statics.dense_levels
    if len(dense) == 1:
        return 0
    level_s = carry_level_s(tuple(not d for d in dense))
    return 1 - level_s if level_s >= 0 else -1


def read_list_level(read_statics: SideStatics) -> int:
    """THE READ LISTS' LEVEL, the carry's level S: at depth one the level, at depth two the read-sparse one
    (:func:`rola.routing.factors.carry_level_s`); ``-1`` deeper. Its lists exist where it is sparse on the read side."""
    dense = read_statics.dense_levels
    if len(dense) == 1:
        return 0
    return carry_level_s(tuple(not d for d in dense))


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the kernel skips reads on."""
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def facts(routes, values, layout: LivenessLayout, read_statics: SideStatics, write_statics: SideStatics) -> Facts:
    """THE PASS: both sides' words, their tile unions and the list exponents from the bundle (``routes``,
    ``values``) in ONE launch.

    Same words at every sparse level, bit for bit, as :func:`rola.engine.facts.liveness.liveness_words` over the
    bundle's rows, and their unions as :func:`rola.engine.facts.liveness.tile_unions`; the list exponents those of ``list_exponents`` within
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
    fm = torch.empty((BH, layout.L, 2), dtype=torch.float32, device=routes.device)
    count = mask_words(layout.rows)
    windows = -(-layout.words // window_words)
    read_level = read_list_level(read_statics)
    sets = {}
    for side, lvl, statics in ((1, level, write_statics), (0, read_level, read_statics)):
        listed = lvl >= 0 and not statics.dense_levels[lvl]
        width = layout.B[lvl] if listed else 0
        groups = -(-width // 32)

        def u16(*shape, dtype=torch.int16):
            return torch.empty(shape, dtype=dtype, device=routes.device)

        entries = u16(BH, windows, width, list_model.LIST_CAP, 4) if side else u16(BH, windows, width,
                                                                                    list_model.LIST_CAP)
        sets[side] = (entries, list_model.ListSet(
            slot=entries[..., 0] if side else entries, at=u16(BH, windows, width, list_model.LIST_AT),
            union=u16(BH, windows, groups, WINDOW), count=u16(BH, windows, groups),
            cv=entries[..., 2].view(torch.float16) if side else None,
            cm=entries[..., 3].view(torch.float16) if side else None))
    with_gains = level >= 0
    gains16 = torch.empty((BH, layout.L if with_gains else 0, 2), dtype=torch.float16, device=routes.device)
    gains_exp = torch.empty((BH, windows if with_gains else 0, 2), dtype=torch.int16, device=routes.device)
    (w_entries, w), (r_entries, r) = sets[1], sets[0]
    outs = [w_entries, w.at, w.union, w.count, r_entries, r.at, r.union, r.count, gains16, gains_exp]
    extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
        words, unions, list_exp, fm, outs, list(layout.B),
        _dense_bits(read_statics), [read_statics.mask_word(i) for i in range(count)],
        _dense_bits(write_statics), [write_statics.mask_word(i) for i in range(count)], level, read_level, WINDOW)
    return Facts(words=words, unions=unions, list_exp=list_exp, fm=fm, write=w, read=r, gains16=gains16,
                 gains_exp=gains_exp)


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/facts/facts.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[LEDGER_WARPS, 4]`` int64 cycles (``LEDGER_PHASES``), or
    ``None``."""
    extension().rola_facts_ledger_bind(ledger)
