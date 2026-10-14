# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY'S PLANS, modeled: per (batch-head, box, window) the plan `csrc/rola/src/carry/schedule.cuh`'s plan pass
writes from the facts pass's counts -- the rule's form (or a named candidate's), its items, buffers and fills -- word
for word, so a gate asserts the kernel's plans and the forms its cell exercises (docs/internals/carry/schedule.md#planner).

Symbols: ``W`` the window (512 tokens), ``U`` a side's union slots in a window, ``S[j]`` a side's box lists' entries
before union slot ``64 (j + 1)``, ``B`` the area's bytes.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops

WINDOW, LIST_BOUND, LIST_AT = 512, 64, 8
PAD_BYTES, PANEL_ROW, BOX = 512, 64, 32
IMAGE_WHOLE, IMAGE_RING = 32 * 2048, 16 * 2048
NO_ITEM, NEXT_ITEM = 31, 30
MAXIMA, STORE, READ, DRAIN, FOLD, BOUNDARY = range(6)
FILLED, UNITS, MORE_READS, DONE = 1, 2, 4, 8
WHOLE, RING16 = 0, 2  # the image modes' codes: whole, ring of 8-plane batches (1), 16-plane halves
FOLD_OPTIONS, ORDERS = 5, 4


class Counts(NamedTuple):
    """A window's counts: each side's union and segment sums, the next window's first read buffer's bytes."""

    ru: int
    rs: tuple
    fu: int
    fs: tuple
    rnext: int


class Form(NamedTuple):
    """A window's form: its image mode, its fold chunk, its order (read chunks each drained before the next: serial;
    the last drain after the fold's first chunk: late)."""

    mode: int
    fc: int
    serial: bool
    late: bool


class WindowPlan(NamedTuple):
    """One window's plan: its head's four words and its items' two words each, as the plan pass writes them."""

    head: tuple
    items: tuple


def _a128(x: int) -> int:
    return (x + 127) // 128 * 128


def read_bytes(slots: int, entries: int, read_row: int) -> int:
    return _a128(slots * read_row) + PAD_BYTES + _a128(slots * PANEL_ROW) + _a128(entries * 8)


def fold_bytes(slots: int, entries: int, write_row: int) -> int:
    return _a128(slots * write_row) + _a128(entries * 8)


def _sum_at(S, j: int) -> int:
    return int(S[j]) if 0 <= j < LIST_AT else 0


def entries_of(S, first: int, slots: int, total: int) -> int:
    hi = LIST_AT - 1 if first + slots >= total else (first + slots) // LIST_BOUND - 1
    return _sum_at(S, hi) - _sum_at(S, first // LIST_BOUND - 1)


def chunk_max(S, m: int, total: int) -> int:
    step = LIST_BOUND * m
    return max((entries_of(S, f, min(step, total - f), total) for f in range(0, total, step)), default=0)


def largest_chunk(u: int, S, room: int, nbytes) -> int:
    """The whole union where one buffer of it fits ``room``; else the largest ``64 m`` below it two of whose largest
    buffers fit, else one (`largest_chunk`)."""
    if nbytes(u, _sum_at(S, LIST_AT - 1)) <= room:
        return u
    two, one, m = 0, 1, 1
    while LIST_BOUND * m < u:
        b = nbytes(LIST_BOUND * m, chunk_max(S, m, u))
        two = m if 2 * b <= room else two
        one = m if b <= room else one
        m += 1
    return LIST_BOUND * (two if two > 0 else one)


def read_chunk_of(ru: int, rs, read_row: int, area: int) -> int:
    return largest_chunk(ru, rs, area - IMAGE_RING, lambda s, e: read_bytes(s, e, read_row))


def first_read_bytes(ru: int, rs, read_row: int, area: int) -> int:
    first = min(read_chunk_of(ru, rs, read_row, area), ru)
    return read_bytes(first, entries_of(rs, 0, first, ru), read_row)


def rule_form(c: Counts, read_row: int, write_row: int, area: int) -> Form:
    """The image whole where it fits beside the read buffers in flight, else in 16-digit halves; the fold's largest
    chunk in the area; the reads pipelined (`rule_form`)."""
    rc = read_chunk_of(c.ru, c.rs, read_row, area) if c.ru > 0 else 0
    nr = -(-c.ru // rc) if c.ru > 0 else 0
    most = 0
    if nr > 0:
        most = read_bytes(min(rc, c.ru), chunk_max(c.rs, rc // LIST_BOUND, c.ru) if rc < c.ru
                          else _sum_at(c.rs, LIST_AT - 1), read_row)
    whole = nr > 0 and IMAGE_WHOLE + (2 if nr > 1 else 1) * most <= area
    fc = largest_chunk(c.fu, c.fs, area, lambda s, e: fold_bytes(s, e, write_row)) if c.fu > 0 else 0
    return Form(WHOLE if whole else RING16, fc, False, False)


def named_form(cand: int, c: Counts) -> Form | None:
    """A named candidate's form (an image mode by a fold chunk by an order), None where it names no form distinct
    from another (`named_form`)."""
    if not 0 <= cand < len(carry_ops.CANDIDATES):
        return None
    fopt, order = cand // ORDERS % FOLD_OPTIONS, cand % ORDERS
    f = Form(cand // (FOLD_OPTIONS * ORDERS), c.fu if fopt == 0 else LIST_BOUND * (FOLD_OPTIONS - fopt), order >= 2,
             (order & 1) != 0)
    return f if fopt == 0 or f.fc < c.fu else None


def _bits(v: int, at: int, n: int) -> int:
    return (v >> at) & ((1 << n) - 1)


def build(f: Form, c: Counts, read_row: int, write_row: int, area: int) -> WindowPlan | None:
    """A window's plan in a form (`build`): its items in order, each buffer placed, each fill hosted; None where the
    form does not fit."""
    B = area
    mode, fc, serial, late = f
    rc = read_chunk_of(c.ru, c.rs, read_row, area) if c.ru > 0 else 0
    nr = -(-c.ru // rc) if c.ru > 0 else 0
    nf = -(-c.fu // fc) if c.fu > 0 else 0
    ibytes = IMAGE_WHOLE if mode == WHOLE else IMAGE_RING

    def bytes_of(kind, first, slots):
        if kind == READ:
            return read_bytes(slots, entries_of(c.rs, first, slots, c.ru), read_row)
        return fold_bytes(slots, entries_of(c.fs, first, slots, c.fu), write_row)

    r0 = bytes_of(READ, 0, min(rc, c.ru)) if nr > 0 else 0
    if ((mode == WHOLE and nr == 0) or (serial and nr <= 1) or (late and (nr == 0 or nf == 0))
            or (nr > 0 and ibytes + r0 > B)):
        return None

    items: list[list[int]] = []
    read_at, drain_at, fold_at = [0] * LIST_AT, [0] * LIST_AT, [0] * LIST_AT

    def put(kind, flags, first, slots, dep):
        items.append([kind | flags << 3 | (first // LIST_BOUND) << 7 | slots << 10 | dep << 20 | NO_ITEM << 25,
                      NO_ITEM << 14 | NO_ITEM << 19 | NO_ITEM << 24])
        return len(items) - 1

    def drain(q):
        drain_at[q] = put(DRAIN, 0, q * rc, min(rc, c.ru - q * rc), read_at[q])

    put(MAXIMA, 0, 0, 0, NO_ITEM)
    put(STORE, DONE if mode == WHOLE else 0, 0, 0, NO_ITEM)
    for q in range(nr):
        read_at[q] = put(READ, FILLED | DONE | (MORE_READS if q + 1 < nr else 0), q * rc, min(rc, c.ru - q * rc),
                         1 if mode == WHOLE else NO_ITEM)
        if serial and not (late and q == nr - 1):
            drain(q)
        if not serial and q >= 1:
            drain(q - 1)
    if not serial and nr > 0 and not late:
        drain(nr - 1)
    for j in range(nf):
        fold_at[j] = put(FOLD, FILLED | (UNITS if j == 0 else 0), j * fc, min(fc, c.fu - j * fc), NO_ITEM)
        if j == 0 and late:
            drain(nr - 1)
    boundary = put(BOUNDARY, UNITS if nf == 0 else 0, 0, 0, NO_ITEM)

    regions: list[list[int]] = []  # [lo, hi, last user], bytes

    def set_buf(k, lo):
        items[k][1] = (items[k][1] & ~0x3FFF) | lo // 16

    if nr > 0:
        regions.append([0, ibytes, read_at[nr - 1]])
        regions.append([B - r0, B, drain_at[0]])
        set_buf(read_at[0], B - r0)
        set_buf(drain_at[0], B - r0)

    def place(nbytes, consumer):
        best, best_wait = -1, 1 << 30
        spots = [0, B - nbytes]
        for lo, hi, _ in regions:
            spots += [hi, lo - nbytes]
        for at in spots:
            if at < 0 or at + nbytes > B:
                continue
            w, ok = -1, True
            for lo, hi, last in regions:
                if lo < at + nbytes and at < hi:
                    if last >= consumer:
                        ok = False
                        break
                    w = max(w, last)
            if ok and (w < best_wait or (w == best_wait and at < best)):
                best, best_wait = at, w
        return best, best_wait

    def host(target, wait, consumer):
        fw = NO_ITEM if wait < 0 else wait
        for h in range(max(1, wait + 1), consumer + 1):
            x, y = items[h]
            if _bits(x, 25, 5) == NO_ITEM:
                items[h] = [(x & ~(31 << 25)) | target << 25, (y & ~(31 << 14)) | fw << 14]
            elif _bits(y, 19, 5) == NO_ITEM:
                items[h] = [x, (y & ~(0x3FF << 19)) | target << 19 | fw << 24]
            else:
                continue
            if wait >= 0:
                items[wait][0] |= DONE << 3
            return True
        return False

    for q in range(1, nr):
        nbytes = bytes_of(READ, q * rc, min(rc, c.ru - q * rc))
        at, wait = place(nbytes, read_at[q])
        if at < 0:
            return None
        regions.append([at, at + nbytes, drain_at[q]])
        set_buf(read_at[q], at)
        set_buf(drain_at[q], at)
        if not host(read_at[q], wait, read_at[q]):
            return None
    for j in range(nf):
        nbytes = bytes_of(FOLD, j * fc, min(fc, c.fu - j * fc))
        at, wait = place(nbytes, fold_at[j])
        if at < 0:
            return None
        regions.append([at, at + nbytes, fold_at[j]])
        set_buf(fold_at[j], at)
        if not host(fold_at[j], wait, fold_at[j]):
            return None
    if c.rnext > 0:
        wait = max([last for _, hi, last in regions if hi > B - c.rnext], default=-1)
        if not host(NEXT_ITEM, wait, boundary):
            return None

    head = (len(items) | mode << 5 | rc << 7 | fc << 17, c.ru | c.fu << 10, int(serial) | int(late) << 1, 0)
    return WindowPlan(head, tuple(tuple(it) for it in items))


def plan(c: Counts, read_row: int, write_row: int, area: int, prefer: int = -1) -> WindowPlan:
    """The plan pass's plan of one window: the named candidate's form where it builds, else the rule's, its reads
    pipelined, else serial."""
    f = named_form(prefer, c)
    got = build(f, c, read_row, write_row, area) if f is not None else None
    if got is None:
        f = rule_form(c, read_row, write_row, area)
        got = build(f, c, read_row, write_row, area) or build(f._replace(serial=True), c, read_row, write_row, area)
    return got


def _side(lists, carve_plan, routes, w: int, bh: int, group: int):
    """``(U, S)``: a side's union slots and segment sums in window ``w`` at a box whose list level's group is
    ``group``; every window token and no entries for a side with no list level."""
    if not carve_plan.lists:
        return min(WINDOW, routes.tokens - w * WINDOW), (0,) * LIST_AT
    totals = (lists.pre[bh, w, group, -1].long() & 0xFFFF).cpu().numpy()
    return int(lists.count[bh, w, group]), tuple(int(v) for v in np.asarray(totals))


def plans(routes, facts, carve, prefer: str | None = None) -> list[list[WindowPlan]]:
    """Every box class's plans over the call's windows (a class per pair of the two list levels' groups, each
    batch-head, in the plan tensor's order), as the plan pass on this card writes them (its area and pool rows from
    the arm's census)."""
    key = carry_ops.arm_key(carve)
    row = next(r for r in carry_ops.census() if tuple(r[1:7]) == tuple(key))
    area, read_row, write_row = row[12], row[13], row[14]
    pref = -1 if prefer is None else carry_ops.CANDIDATES.index(prefer)
    windows = -(-routes.tokens // WINDOW)
    write = facts_ops.side_lists(facts, "write") if carve.fold.lists else None
    read = facts_ops.side_lists(facts, "read") if carve.readout.lists else None
    groups = lambda p: max(1, carve.widths[p.lists[0]] // BOX) if p.lists else 1  # noqa: E731
    out = []
    for bh in range(routes.batch * routes.heads):
        for gr in range(groups(carve.readout)):
            for gw in range(groups(carve.fold)):
                sides = [(_side(read, carve.readout, routes, w, bh, gr), _side(write, carve.fold, routes, w, bh, gw))
                         for w in range(windows)]
                box = []
                for w, ((ru, rs), (fu, fs)) in enumerate(sides):
                    nu, ns = sides[w + 1][0] if w + 1 < windows else (0, None)
                    rnext = first_read_bytes(nu, ns, read_row, area) if nu > 0 else 0
                    c = Counts(ru, rs, fu, fs, rnext)
                    box.append(plan(c, read_row, write_row, area, pref))
                out.append(box)
    return out


def forms(box_plans: list[list[WindowPlan]]) -> set[str]:
    """The forms a call exercises: each image mode read under, and ``<mode>-chunks`` where a window reads more than one
    chunk under it; ``serial`` (a chunk drained before the next is read), ``fold-chunks``, ``late`` (the last drain
    after the fold's first chunk), ``fill-free`` (a fill waiting for nothing), ``fill-waits`` (a fill waiting for its
    region's last user), ``next`` (the next window's first read filled here)."""
    out = set()
    for box in box_plans:
        for pl in box:
            kinds = [_bits(x, 0, 3) for x, _ in pl.items]
            mode = ("whole", "ring8", "ring16")[_bits(pl.head[0], 5, 2)]
            if READ in kinds:
                out.add(mode)
            if kinds.count(READ) > 1:
                out.add(f"{mode}-chunks")
                if kinds[kinds.index(READ) + 1] == DRAIN:
                    out.add("serial")
            if kinds.count(FOLD) > 1:
                out.add("fold-chunks")
            if FOLD in kinds and DRAIN in kinds and kinds.index(FOLD) < len(kinds) - 1 - kinds[::-1].index(DRAIN):
                out.add("late")
            for x, y in pl.items:
                for target, fw in ((_bits(x, 25, 5), _bits(y, 14, 5)), (_bits(y, 19, 5), _bits(y, 24, 5))):
                    if target != NO_ITEM:
                        out.add("next" if target == NEXT_ITEM else "fill-waits" if fw != NO_ITEM else "fill-free")
    return out
