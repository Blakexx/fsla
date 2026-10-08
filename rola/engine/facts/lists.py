# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LISTS' MODEL: the facts pass's per-(window, digit) lists and the window's gains pairs, the pure-torch spec the
kernel is graded against bit for bit (docs/internals/facts/facts.md#lists-out, #gains).

A side's LISTED level (the write side's list level, the read side's read list level) is listed when it is sparse on
that side. Per window and 32-digit group of it: the UNION -- the tokens live in any of the group's digits, in token
order -- ranks each of its tokens into a SLOT. Per digit: its tokens as slots, split into PASSES of
:data:`PASS_SLOTS` slots, each pass's entries ROUND-ROBIN by ``slot & 7`` (round ``j`` holds the ``j``-th entry of
every residue that has one, residues in order), the passes one after another with nothing between them; the list's
pass starts ``at`` (``at[q]`` the start of pass ``q``, the end from pass 4 on). A WRITE entry also carries its
coefficient pair ``c_m = f16(fp32(b m) 2^(4 - E_m))``, ``c_v = f16(fp32(b m s) 2^(4 - E_v))`` -- ``b`` the token's
level byte, ``m = ((g sw_0) sw_1)`` and ``m s`` the facts pass's fp32 columns, ``E = max(K, -123)`` the list's
exponent (``list_exp``) floored at the fold's smallest unit. The GAINS: each token's ``(m, m s)``
as f16 at its window's powers of two, ``2^s`` with ``s = -4 - floor(log2(max))`` (the largest of each in
``[2^-4, 2^-3)``; a zero largest ``s = 0``).

Mirrors ``csrc/rola/src/facts/facts.cuh``: ``kPassSlots``, ``kListCap``, ``kListAt``, ``kCoefTop``, ``kUnitMin``.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import liveness as lv

PASS_SLOTS = 128
WINDOW = 512
LIST_CAP = WINDOW
LIST_AT = 8
COEF_TOP = 4
UNIT_MIN = COEF_TOP - 127


class ListSet(NamedTuple):
    """One side's lists: ``slot``, ``cv``, ``cm`` ``[BH, windows, B, LIST_CAP]`` (``cv``/``cm`` f16, the write side's
    only), ``at`` ``[BH, windows, B, LIST_AT]``, ``union`` ``[BH, windows, groups, WINDOW]`` tokens and ``count``
    ``[BH, windows, groups]``. Past a list's end and a union's count nothing is defined."""

    slot: torch.Tensor
    at: torch.Tensor
    union: torch.Tensor
    count: torch.Tensor
    cv: torch.Tensor | None
    cm: torch.Tensor | None


def columns(routes, values) -> tuple[torch.Tensor, torch.Tensor]:
    """``(m, m s)`` ``[B, T, H]`` fp32, multiplied as the facts pass does: the gain, then each write level's scale in
    order, then the value scale."""
    m = routes.g_write.float()
    for scale in routes.scale_write:
        m = m * scale.float()
    return m, m * values.scale.float()


def _round_robin(slots: list[int]) -> list[int]:
    buckets = [[s for s in slots if s & 7 == r] for r in range(8)]
    out = []
    for j in range(max((len(b) for b in buckets), default=0)):
        out += [b[j] for b in buckets if len(b) > j]
    return out


def side_lists(bits: torch.Tensor, layout: lv.LivenessLayout, level: int, *, rows=None, cols=None,
               list_exp=None) -> ListSet:
    """One side's lists at ``level`` from its unpacked words ``bits`` ``[BH, rows, L]`` (:func:`lv.unpack`'s side
    slice); the write side's coefficient pairs from its level rows ``rows`` ``[B, T, H, width]`` u8, ``cols`` (the
    :func:`columns`) and the pass's ``list_exp`` ``[BH, windows, B, 2]``."""
    BH, L, B = bits.shape[0], layout.L, layout.B[level]
    windows, groups = -(-L // WINDOW), -(-B // lv.GROUP_DIGITS)
    base = layout.row_base(level)
    coef = rows is not None
    slot = torch.zeros((BH, windows, B, LIST_CAP), dtype=torch.int64)
    cv = torch.zeros((BH, windows, B, LIST_CAP), dtype=torch.float16) if coef else None
    cm = torch.zeros_like(cv) if coef else None
    at = torch.zeros((BH, windows, B, LIST_AT), dtype=torch.int64)
    union = torch.zeros((BH, windows, groups, WINDOW), dtype=torch.int64)
    count = torch.zeros((BH, windows, groups), dtype=torch.int64)
    if coef:
        H = rows.shape[2]
        m, ms = (c.cpu() for c in cols)
        rows = rows.cpu()
    for bh in range(BH):
        for w in range(windows):
            t0, t1 = w * WINDOW, min(L, (w + 1) * WINDOW)
            for g in range(groups):
                live = bits[bh, base + g * 32:base + min(B, g * 32 + 32), t0:t1]
                tokens = live.any(0).nonzero().flatten()
                count[bh, w, g] = len(tokens)
                union[bh, w, g, :len(tokens)] = tokens
                rank = torch.full((t1 - t0,), -1, dtype=torch.int64)
                rank[tokens] = torch.arange(len(tokens))
                for dd in range(live.shape[0]):
                    digit = g * 32 + dd
                    mine = live[dd].nonzero().flatten()
                    slots = rank[mine].tolist()
                    token_of = dict(zip(slots, mine.tolist()))
                    entries, starts = [], [0]
                    for q in range(WINDOW // PASS_SLOTS):
                        entries += _round_robin([s for s in slots if s // PASS_SLOTS == q])
                        starts.append(len(entries))
                    slot[bh, w, digit, :len(entries)] = torch.tensor(entries, dtype=torch.int64)
                    at[bh, w, digit] = torch.tensor([starts[min(q, len(starts) - 1)] for q in range(LIST_AT)])
                    if not coef:
                        continue
                    b, h = divmod(bh, H)
                    e_m = max(int(list_exp[bh, w, digit, 0]), UNIT_MIN)
                    e_v = max(int(list_exp[bh, w, digit, 1]), UNIT_MIN)
                    where = torch.arange(len(entries))
                    t = t0 + torch.tensor([token_of[s] for s in entries], dtype=torch.int64)
                    byte = rows[b, t, h, digit].float()
                    cm[bh, w, digit, where] = ((byte * m[b, t, h]) * 2.0 ** (COEF_TOP - e_m)).half()
                    cv[bh, w, digit, where] = ((byte * ms[b, t, h]) * 2.0 ** (COEF_TOP - e_v)).half()
    return ListSet(slot, at, union, count, cv, cm)


def gains_shift(top: torch.Tensor) -> torch.Tensor:
    """The window's power of two for each channel: ``-4 - floor(log2(top))`` (the exponent field's), clamped to
    [-126, 127], 0 for a zero largest."""
    e = (top.float().view(torch.int32) >> 23) - 127
    return torch.where(top > 0, (-e - 4).clamp(-126, 127), torch.zeros_like(e))


def gains(routes, values, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gains16, gains_exp)``: ``[BH, L, 2]`` f16 each token's ``(m, m s)`` at its window's powers of two, and
    ``[BH, windows, 2]`` those powers' exponents."""
    m, ms = columns(routes, values)
    B, T, H = m.shape
    both = torch.stack((m, ms), -1).permute(0, 2, 1, 3).reshape(B * H, T, 2).cpu()
    windows = -(-L // WINDOW)
    pad = torch.cat((both, both.new_zeros(B * H, windows * WINDOW - T, 2)), 1)
    top = pad.reshape(B * H, windows, WINDOW, 2).amax(2)
    shift = gains_shift(top)
    scale = torch.ldexp(torch.ones_like(top), shift.float()).repeat_interleave(WINDOW, 1)[:, :T]
    return (both * scale).half(), shift
