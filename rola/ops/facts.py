# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FACTS PASS, as a caller reaches it: one launch over the bundle's rows for both sides' class-1 words, the
window's columns, the write list level's exponents and each side's lists in the carry's final form.

The words' LAYOUT, their reader, the folds over them and the pure-torch MODEL the kernel is graded against live in
:mod:`rola.engine.facts.liveness`; the lists' model in :mod:`rola.engine.facts.lists`; the list exponents' definition
is the fp16 fold's reference (``tests/oracle/fold_reference.py``, ``list_exponents``). What is here is the call: the
layout and the two sides' static operands a bundle's topology implies (:func:`call_layout`, :func:`call_statics`),
packed into the bits R13 requires so no logical width crosses the seam, and the launch
(docs/internals/facts/facts_api.md). The lists carry no chunk size of the carry's; a call names only the arm's read
pool row (``rola.ops.carry.form``'s ``read_row``), the read entries' row offsets being in its bytes.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import lists as list_model
from rola.engine.facts.liveness import GROUP_DIGITS, LivenessLayout, SideStatics, mask_words, side_statics
from rola.ops._ext import extension

__all__ = ["COMPOSITE", "LEDGER_PHASES", "LEDGER_WARPS", "WINDOW", "Facts", "call_layout", "call_statics", "facts",
           "ledger_bind", "list_level", "side_lists"]

#: THE WINDOW the lists are built over, the carry's and the intra's.
WINDOW = list_model.WINDOW

#: THE PHASE LEDGER'S SHAPE: ``[warps][phases]``, the kernel's own `FactsPhase` order, booked over every CTA.
LEDGER_PHASES = ("load", "vote", "ranks", "lists")
LEDGER_WARPS = 8


class Facts(NamedTuple):
    """One call's facts: ``words`` the ``[BH, 2, rows, ceil(L / 32)]`` int32 class-1 table (:class:`LivenessLayout`),
    ``unions`` the ``[BH, 2, groups, ceil(L / 64)]`` int32 tile unions (``tile_unions``), and ``list_exp`` the
    ``[BH, windows, B_list, 2]`` int16 list exponents -- per window and digit of the write list level
    (:func:`list_level`) the smallest ``K`` with every token's ``b m <= 2^K`` (column 0) and ``b m s <= 2^K`` (column
    1), ``b`` its write byte, ``m`` the write gain times the write levels' scales and ``s`` the value scale; ``-126``
    for a list no token writes; ``B_list`` 0 when the write side lists nothing. ``fm`` the ``[BH, L, 2]`` fp32 factor
    products, each token's ``(F, m)`` = ``(m s, m)``; ``gains16`` ``[BH, L, 2]`` f16 each token's ``(m, m s)`` at its
    window's powers of two and ``gains_exp`` ``[BH, windows, 2]`` their exponents. A DENSE level's rows and groups
    of ``words`` and ``unions`` are left unwritten: they are its width mask over the live tokens, which a reader takes
    from the side's statics. ``write`` and ``read`` each side's lists at its list level
    (:class:`rola.engine.facts.lists.Lists`; width 0 for a side with no sparse level, ``None`` for a composite side,
    :func:`side_lists`) (docs/internals/facts/facts.md#lists-out, #gains)."""

    words: torch.Tensor
    unions: torch.Tensor
    list_exp: torch.Tensor
    fm: torch.Tensor
    write: list_model.Lists | None
    read: list_model.Lists | None
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


#: A SIDE WITH TWO OR MORE SPARSE LEVELS: its lists are composite (keyed by a digit of each), which this pass does not
#: build (:func:`side_lists` refuses them).
COMPOSITE = -2


def list_level(statics: SideStatics) -> int:
    """A SIDE'S LIST LEVEL: its one sparse level (the carve's ``lists`` for the side, by definition:
    docs/internals/engine/carve.md); ``-1`` for a side with none, whose every token is a slot; :data:`COMPOSITE` for a
    side with two or more."""
    sparse = [level for level, dense in enumerate(statics.dense_levels) if not dense]
    return sparse[0] if len(sparse) == 1 else (-1 if not sparse else COMPOSITE)


def side_lists(facts: Facts, side: str) -> list_model.Lists:
    """``side``'s (``"write"`` or ``"read"``) lists, refusing a side the pass built none for because they are composite
    (docs/internals/facts/facts.md#lists-out)."""
    lists = getattr(facts, side)
    if lists is None:
        raise ValueError(f"the facts pass builds no composite lists: the {side} side has two or more sparse levels, "
                         "and its lists would be keyed by a digit of each")
    return lists


def _dense_bits(statics: SideStatics) -> int:
    """One side's dense levels as the bitmask the kernel skips reads on."""
    return sum(1 << level for level, dense in enumerate(statics.dense_levels) if dense)


def _lists(BH: int, windows: int, width: int, device) -> list_model.Lists:
    """A side's lists out at a level of ``width`` digits (0 for none), uninitialized."""
    groups = -(-width // GROUP_DIGITS)

    def empty(*shape, dtype=torch.int16):
        return torch.empty(shape, dtype=dtype, device=device)

    return list_model.Lists(
        entries=empty(BH, windows, width, list_model.LIST_CAP, 4),
        pre=empty(BH, windows, groups, list_model.PRE_ROWS, list_model.LIST_AT),
        union=empty(BH, windows, groups, WINDOW), count=empty(BH, windows, groups, dtype=torch.int32))


def facts(routes, values, layout: LivenessLayout, read_statics: SideStatics, write_statics: SideStatics,
          read_row: int) -> Facts:
    """THE PASS: both sides' words and tile unions, the window's columns, the write list exponents and each side's
    lists from the bundle (``routes``, ``values``) in ONE launch, the read entries' row offsets in the carry arm's read
    pool rows of ``read_row`` bytes.

    Same words at every sparse level, bit for bit, as :func:`rola.engine.facts.liveness.liveness_words` over the
    bundle's rows, and their unions as :func:`rola.engine.facts.liveness.tile_unions`; the list exponents those of
    ``list_exponents`` within fp32 rounding; the lists :func:`rola.engine.facts.lists.final_lists`, bit for bit. The rows
    are read where the bundle holds them, any offset or stride of the format, each level's ``[B, T, H, w_l]`` view.
    """
    BH = routes.batch * routes.heads
    windows = -(-layout.L // WINDOW)
    device = routes.device
    words = torch.empty((BH, 2, layout.rows, layout.words), dtype=torch.int32, device=device)
    unions = torch.empty((BH, 2, layout.groups, layout.tiles), dtype=torch.int32, device=device)
    level, read_level = list_level(write_statics), list_level(read_statics)
    list_exp = torch.empty((BH, windows, layout.B[level] if level >= 0 else 0, 2), dtype=torch.int16, device=device)
    fm = torch.empty((BH, layout.L, 2), dtype=torch.float32, device=device)
    gains16 = torch.empty((BH, layout.L, 2), dtype=torch.float16, device=device)
    gains_exp = torch.empty((BH, windows, 2), dtype=torch.int16, device=device)
    write = _lists(BH, windows, layout.B[level] if level >= 0 else 0, device)
    read = _lists(BH, windows, layout.B[read_level] if read_level >= 0 else 0, device)
    count = mask_words(layout.rows)
    extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
        words, unions, list_exp, fm, gains16, gains_exp, list(write), list(read), list(layout.B),
        _dense_bits(read_statics), [read_statics.mask_word(i) for i in range(count)],
        _dense_bits(write_statics), [write_statics.mask_word(i) for i in range(count)], max(level, -1),
        max(read_level, -1), WINDOW, read_row)
    return Facts(words=words, unions=unions, list_exp=list_exp, fm=fm, write=None if level == COMPOSITE else write,
                 read=None if read_level == COMPOSITE else read, gains16=gains16, gains_exp=gains_exp)


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/facts/facts.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles, or
    ``None``."""
    extension().rola_facts_ledger_bind(ledger)
