# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LISTS' MODEL: the facts pass's per-(window, digit) lists in the final form the carry reads, and the window's
gains pairs -- the pure-torch spec the kernel is graded against (docs/internals/facts/facts.md#lists-out, #gains).

A side's LIST LEVEL (its one sparse level) gets, per window and 32-digit group, a UNION -- the tokens live in any of
the group's digits -- RANKED: ``union[slot]`` is the token of union rank ``slot`` (``GAP`` past the union), ``count``
the union's EXTENT, its size in whole steps of 8. Per digit, its LIST: its tokens in PASSES of the carry's pass for the
side (the readout's ``pass_slots``, the fold's ``unit_slots``: ``rola.ops.carry.form``), pass ``q`` its tokens of slot
``q P .. q P + P - 1`` in token order, each pass's last step filled with PADS; ``at[q]`` the start of pass ``q``, the
end from the last pass on.

An entry is four int16 words, its two 32-bit halves the carry's ``uint2``:
- READ: the slot's pool row offset in its pass with the row's swizzle folded in, 0; the coefficient
  ``f16(byte / 256)``, the slot's panel row offset from the dummy rows. A pad at place ``c`` of its step: the pass's
  row ``c`` and dummy panel row ``c``, coefficient 0.
- WRITE: the slot, 0; ``c_v = f16(fp32(b m s) 2^(4 - E_v))``, ``c_m = f16(fp32(b m) 2^(4 - E_m))`` -- ``b`` the token's
  level byte, ``m`` and ``m s`` the facts pass's fp32 columns, ``E = max(K, -123)`` the list's exponent. A pad at place
  ``c`` of its step: the pass's slot ``c``, both coefficients 0.
A warp's HEADS a (window, group): its 4 lists ``w + 8 j``'s first 16 entries.

The GAINS: each token's ``(m, m s)`` as f16 at its window's powers of two, ``2^s`` with ``s = -4 - floor(log2(max))``
(the largest of each in ``[2^-4, 2^-3)``; a zero largest ``s = 0``).

Mirrors ``csrc/rola/src/facts/facts.cuh`` (``kListCap``, ``kListAt``, ``kCoefTop``, ``kUnitMin``, ``kStep``) and the lists'
addresses in ``csrc/rola/src/carry/carry.cuh`` (``kGap``, ``kPadBytes``, ``panel_row``, ``pool_chunk``).
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from rola.engine.facts import liveness as lv

WINDOW = 512
LIST_CAP = WINDOW
LIST_AT = 8
COEF_TOP = 4
UNIT_MIN = COEF_TOP - 127
WARPS = 8
HEAD_ENTRIES = 16
#: the carry's step: each of a list's passes is padded to whole steps, a union's extent its size in them
STEP = 8
#: a gap slot's token: token 0 flagged (``0x8000`` as int16), so a row fetch masked to the window stays inside it
GAP = -0x8000
#: the carry's 8 dummy panel rows before the panel, 64 bytes each
PAD_BYTES = 8 * 64


class Lists(NamedTuple):
    """One side's lists in the final form: ``entries`` ``[BH, windows, B, LIST_CAP, 4]`` int16, ``at``
    ``[BH, windows, B, LIST_AT]``, ``heads`` ``[BH, windows, groups, WARPS, 4 HEAD_ENTRIES, 4]``, ``union``
    ``[BH, windows, groups, WINDOW]`` (a slot's token, ``GAP`` in a gap) and ``count`` ``[BH, windows, groups]`` the
    unions' extents. Past a list's end and a union's extent nothing is defined."""

    entries: torch.Tensor
    at: torch.Tensor
    heads: torch.Tensor
    union: torch.Tensor
    count: torch.Tensor


def columns(routes, values) -> tuple[torch.Tensor, torch.Tensor]:
    """``(m, m s)`` ``[B, T, H]`` fp32, multiplied as the facts pass does: the gain, then each write level's scale in
    order, then the value scale."""
    m = routes.g_write.float()
    for scale in routes.scale_write:
        m = m * scale.float()
    return m, m * values.scale.float()


def chunk0(row_bytes: int, slot: int) -> int:
    """A pool row's chunk 0 after the swizzle (the carry's ``pool_chunk``): an 8-chunk row XORs by the slot's residue,
    a row of an odd number of 64-byte quarters by the residue's high bits within each four."""
    return slot & 7 if row_bytes == 128 else (slot >> 1) & 3


def panel_row(slot: int) -> int:
    """A slot's panel row (the carry's ``panel_row``): a slot pair a 128-byte line, each slot's halves at the segments
    the pair's residue bits XOR."""
    return 128 * (slot >> 1) + 32 * ((2 * (slot & 1)) ^ ((slot >> 1) & 3))


def _f16_bits(x: float) -> int:
    return int(torch.tensor(x, dtype=torch.float32).half().view(torch.int16))


def final_lists(bits: torch.Tensor, layout: lv.LivenessLayout, level: int, *, pass_slots: int, row_bytes: int,
                rows, cols, list_exp) -> Lists:
    """One side's lists at ``level`` from its unpacked words ``bits`` ``[BH, rows, L]`` (:func:`lv.unpack`'s side slice),
    in passes of ``pass_slots`` with pool rows of ``row_bytes`` (``rola.ops.carry.form``). Both sides' coefficients
    read the level's ``rows`` ``[B, T, H, width]`` u8; the WRITE side (``cols`` given) also takes ``cols``
    (:func:`columns`) and the pass's ``list_exp`` ``[BH, windows, B, 2]``; the READ side passes both as ``None``."""
    BH, L, B = bits.shape[0], layout.L, layout.B[level]
    windows, groups = -(-L // WINDOW), -(-B // lv.GROUP_DIGITS)
    base = layout.row_base(level)
    write = cols is not None
    entries = torch.zeros((BH, windows, B, LIST_CAP, 4), dtype=torch.int16)
    at = torch.zeros((BH, windows, B, LIST_AT), dtype=torch.int64)
    heads = torch.zeros((BH, windows, groups, WARPS, 4 * HEAD_ENTRIES, 4), dtype=torch.int16)
    union_out = torch.full((BH, windows, groups, WINDOW), GAP, dtype=torch.int64)
    count = torch.zeros((BH, windows, groups), dtype=torch.int64)
    rows = rows.cpu()
    if write:
        m, ms = (c.cpu() for c in cols)
    H = rows.shape[2]

    for bh in range(BH):
        b, h = divmod(bh, H)
        for w in range(windows):
            t0, t1 = w * WINDOW, min(L, (w + 1) * WINDOW)
            for g in range(groups):
                live = bits[bh, base + g * 32:base + min(B, g * 32 + 32), t0:t1]
                union = live.any(0).nonzero().flatten().tolist()
                slot_of = {t: s for s, t in enumerate(union)}
                count[bh, w, g] = -(-len(union) // STEP) * STEP
                union_out[bh, w, g, :len(union)] = torch.tensor(union, dtype=torch.int64)
                for dd in range(live.shape[0]):
                    digit = g * 32 + dd
                    mine = live[dd].nonzero().flatten().tolist()
                    token_of = {slot_of[t]: t for t in mine}
                    words, starts = [], []
                    for q in range(-(-len(union) // pass_slots)):
                        starts.append(len(words))
                        first = q * pass_slots
                        places = [slot_of[t] for t in mine if first <= slot_of[t] < first + pass_slots]
                        places += [None] * (-len(places) % STEP)
                        for s in places:
                            words.append(_entry(s, first, token_of, write, pass_slots, row_bytes, rows, b, h, t0,
                                                digit, m if write else None, ms if write else None,
                                                list_exp[bh, w, digit] if write else None, len(words) % STEP))
                    if len(words) > LIST_CAP or len(starts) + 1 > LIST_AT:
                        raise AssertionError(f"list {digit} of window {w}: {len(words)} entries in {len(starts)} "
                                             f"passes, over the list's {LIST_CAP} or {LIST_AT - 1}")
                    starts.append(len(words))
                    if words:
                        entries[bh, w, digit, :len(words)] = torch.tensor(words, dtype=torch.int16)
                    at[bh, w, digit] = torch.tensor([starts[min(q, len(starts) - 1)] for q in range(LIST_AT)])
                for warp in range(WARPS):
                    for j in range(4):
                        if g * 32 + warp + 8 * j < B:
                            heads[bh, w, g, warp, j * HEAD_ENTRIES:(j + 1) * HEAD_ENTRIES] = \
                                entries[bh, w, g * 32 + warp + 8 * j, :HEAD_ENTRIES]
    return Lists(entries, at, heads, union_out, count)


def _entry(s, first, token_of, write, pass_slots, row_bytes, rows, b, h, t0, digit, m, ms, list_exp, place):
    """One entry's four int16 words, ``s`` the slot or ``None`` for a pad at place ``place`` of its step."""
    def i16(x: int) -> int:
        return x - 0x10000 if x >= 0x8000 else x

    if write:
        if s is None:
            return [i16(first + place), 0, 0, 0]
        t = t0 + token_of[s]
        byte = float(rows[b, t, h, digit])
        e_m = max(int(list_exp[0]), UNIT_MIN)
        e_v = max(int(list_exp[1]), UNIT_MIN)
        cm = _f16_bits((byte * float(m[b, t, h])) * 2.0 ** (COEF_TOP - e_m))
        cv = _f16_bits((byte * float(ms[b, t, h])) * 2.0 ** (COEF_TOP - e_v))
        return [i16(s), 0, cv, cm]
    local = (s - first) if s is not None else place
    pool = local * row_bytes + 16 * chunk0(row_bytes, local)
    panel = PAD_BYTES + panel_row(local) if s is not None else panel_row(place)
    coef = _f16_bits(float(rows[b, t0 + token_of[s], h, digit]) / 256) if s is not None else 0
    return [i16(pool), 0, coef, i16(panel)]


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
