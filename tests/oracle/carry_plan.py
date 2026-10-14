# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY'S PLANS, modeled: per (batch-head, box, window) the plan `csrc/rola/src/carry/schedule.cuh` computes from
the facts pass's counts -- the readout's candidate and chunk, the fold's chunk, whether the fold's first buffer lands
beside the readout and whether the next window's first read chunk lands under the fold -- so a gate states which of
the plan's forms its cell exercises (docs/internals/carry/schedule.md#planner).

Symbols: ``W`` the window (512 tokens), ``U`` a side's union slots in a window, ``S[j]`` a side's box lists' entries
before union slot ``64 (j + 1)``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops

WINDOW, LIST_BOUND, LIST_AT = 512, 64, 8
PAD_BYTES, PANEL_ROW, BOX = 512, 64, 32
IMAGE_WHOLE, IMAGE_RING = 32 * 2048, 16 * 2048
CANDIDATES = carry_ops.READ_CANDIDATES


class WindowPlan(NamedTuple):
    """One window's plan: the readout's candidate (an index of :data:`CANDIDATES`) and chunk, the fold's chunk, fold
    buffer 0 beside the readout, the readout zone's end and the fold zone's start (bytes), the end of read buffer 0."""

    candidate: int
    runion: int
    rchunk: int
    funion: int
    fchunk: int
    beside: bool
    rzone: int
    fzone: int
    rbuf_end: int


def _a128(x: int) -> int:
    return (x + 127) // 128 * 128


def read_bytes(slots: int, entries: int, read_row: int) -> int:
    return _a128(slots * read_row) + PAD_BYTES + _a128(slots * PANEL_ROW) + _a128(entries * 8)


def fold_bytes(slots: int, entries: int, write_row: int) -> int:
    return _a128(slots * write_row) + _a128(entries * 8)


def _most(S, m: int, n: int) -> int:
    return max(int(S[min(q * m + m - 1, LIST_AT - 1)]) - (int(S[q * m - 1]) if q else 0) for q in range(n))


def _chunk_of(S, slots: int, room: int, bufs: int, size) -> tuple[int, int]:
    best, cap = 0, 0
    for m in range(1, LIST_AT):
        n = -(-slots // (LIST_BOUND * m))
        nbytes = size(LIST_BOUND * m, _most(S, m, n))
        if LIST_BOUND * m < slots and bufs * nbytes <= room:
            best, cap = m, nbytes
    return best, cap


def plan(ru: int, S_r, fu: int, S_w, read_row: int, write_row: int, area: int, prefer: int = -1) -> WindowPlan:
    """The plan of one window from its sides' unions and segment sums, as `plan_window` computes it."""
    resident = read_bytes(ru, int(S_r[-1]), read_row)
    rsize = lambda slots, e: read_bytes(slots, e, read_row)  # noqa: E731
    whole_m, whole_cap = _chunk_of(S_r, ru, area - IMAGE_WHOLE, 2, rsize)
    ring_m, ring_cap = _chunk_of(S_r, ru, area - IMAGE_RING, 2, rsize)
    lone_m, lone_cap = _chunk_of(S_r, ru, area - IMAGE_RING, 1, rsize)
    fits = (IMAGE_WHOLE + resident <= area, IMAGE_RING + resident <= area, whole_m > 0, ring_m > 0, lone_m > 0)
    pick = prefer if 0 <= prefer < len(fits) and fits[prefer] else next(k for k, ok in enumerate(fits) if ok)
    one = pick < 2
    image = IMAGE_WHOLE if pick in (0, 2) else IMAGE_RING
    rchunk = ru if one else LIST_BOUND * (whole_m, ring_m, lone_m)[pick - 2]
    rcap = resident if one else (whole_cap, ring_cap, lone_cap)[pick - 2]
    rzone = image + (1 if one else (1 if pick == 4 else 2)) * rcap
    fcap, fchunk = fold_bytes(fu, int(S_w[-1]), write_row), fu
    if fcap > area:
        m, fcap = _chunk_of(S_w, fu, area, 2, lambda slots, e: fold_bytes(slots, e, write_row))
        fchunk = LIST_BOUND * m
    fbuf0 = area - fcap
    fbuf1 = fbuf0 - fcap if fchunk < fu else fbuf0
    return WindowPlan(pick, ru, rchunk, fu, fchunk, fbuf0 >= rzone, rzone, fbuf1, image + rcap)


def _side(lists, carve_plan, routes, w: int, bh: int, group: int):
    """``(U, S)``: a side's union slots and segment sums in window ``w`` at a box whose list level's group is
    ``group``; every window token and no entries for a side with no list level."""
    if not carve_plan.lists:
        return min(WINDOW, routes.tokens - w * WINDOW), np.zeros(LIST_AT, dtype=np.int64)
    totals = (lists.pre[bh, w, group, -1].long() & 0xFFFF).cpu().numpy()
    return int(lists.count[bh, w, group]), totals


def plans(routes, facts, carve, prefer: str | None = None) -> list[list[WindowPlan]]:
    """Every distinct box's plans over the call's windows (a box per pair of the two list levels' groups, each
    batch-head), as the kernel on this card computes them (its area and pool rows from the arm's census)."""
    key = carry_ops.arm_key(carve)
    row = next(r for r in carry_ops.census() if tuple(r[1:7]) == tuple(key))
    area, read_row, write_row = row[12], row[13], row[14]
    pref = -1 if prefer is None else CANDIDATES.index(prefer)
    windows = -(-routes.tokens // WINDOW)
    write = facts_ops.side_lists(facts, "write") if carve.fold.lists else None
    read = facts_ops.side_lists(facts, "read") if carve.readout.lists else None
    groups = lambda p: max(1, carve.widths[p.lists[0]] // BOX) if p.lists else 1  # noqa: E731
    out = []
    for bh in range(routes.batch * routes.heads):
        for gr in range(groups(carve.readout)):
            for gw in range(groups(carve.fold)):
                box = []
                for w in range(windows):
                    ru, S_r = _side(read, carve.readout, routes, w, bh, gr)
                    fu, S_w = _side(write, carve.fold, routes, w, bh, gw)
                    box.append(plan(ru, S_r, fu, S_w, read_row, write_row, area, pref))
                out.append(box)
    return out


def early(this: WindowPlan, nxt: WindowPlan) -> bool:
    """The next window's first read chunk lands under this window's fold: its buffer below every fold buffer."""
    return nxt.runion > 0 and (this.funion == 0 or nxt.rbuf_end <= this.fzone)


def forms(box_plans: list[list[WindowPlan]]) -> set[str]:
    """The forms a call exercises: each readout candidate taken, ``fold-beside``, ``fold-after``, ``fold-chunks``,
    ``early`` and ``late`` boundaries."""
    out = set()
    for box in box_plans:
        for w, pl in enumerate(box):
            out.add(CANDIDATES[pl.candidate])
            if pl.funion:
                out.add("fold-chunks" if pl.fchunk < pl.funion else "fold-beside" if pl.beside else "fold-after")
            if w + 1 < len(box) and box[w + 1].runion:
                out.add("early" if early(pl, box[w + 1]) else "late")
    return out
