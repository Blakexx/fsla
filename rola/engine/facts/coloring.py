# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COLORING'S MODEL: each window's union of one 32-digit group, colored 0-7, and the slot each token takes
(docs/internals/facts/facts.md#coloring).

A slot's residue mod 8 is the bank group of every row the carry indexes by it (the readout's pool and panel rows, the
fold's pool rows), and the carry steps a list 8 tokens at a time. The coloring gives each union token a COLOR, and its
SLOT is ``8 x (its rank among its color's tokens, in token order) + color`` -- the RESIDUE LAYOUT, so the slot is the
color and no consumer is told one. A list placed in ROUND order (step ``i`` takes each color's ``i``-th token) then
never repeats a residue within 8 tokens. Three bounds shape a coloring:

- each digit at most ``ceil(n / 8)`` tokens a color (n its tokens): its steps are then ``ceil(n / 8)``. A token whose
  digits leave it no such color is BLOCKED and costs a step, never a conflict;
- the union at most :data:`UNION_CAP` tokens a color: always satisfiable for a union of at most a window, so a union's
  slots end by :data:`WINDOW`;
- the union's colors balanced, so its slots leave few gaps (the extent is ``8 x`` the largest color's count).

:func:`color` is the reference (the kernel's form, its concurrent claims resolved in token order); the kernel's claims
race, so its coloring is graded by :func:`check`, never by equality.
"""
from __future__ import annotations

from collections.abc import Sequence

WINDOW = 512
COLORS = 8
UNION_CAP = WINDOW // COLORS


def capacities(rows: Sequence[int]) -> list[int]:
    """Each digit's largest count a color: ``ceil(n / 8)`` of its ``n`` tokens."""
    return [-(-bin(row).count("1") // COLORS) for row in rows]


def members(rows: Sequence[int]) -> dict[int, list[int]]:
    """Each union token (a token live in any of the rows) and the digits it is live in, in token order."""
    union = 0
    for row in rows:
        union |= row
    return {t: [d for d in range(len(rows)) if rows[d] >> t & 1] for t in range(WINDOW) if union >> t & 1}


def color(rows: Sequence[int]) -> tuple[dict[int, int], int]:
    """The reference coloring of a group's ``rows`` (each a digit's 512 token bits) and its blocked tokens.

    The tokens in two or more digits go first, in token order: each takes the color least used over its digits among
    those under every digit's capacity and the union's cap (ties to the union's least-used, then ``(c - t) & 7``); a
    token no color allows is blocked and takes the union's least-used color under the cap. The kernel claims them 32 a
    batch (three-or-more first), every warp a batch a round and then warp 0's batches, the claims racing on shared
    atomics and a loser requeued -- so its coloring is graded by :func:`check`, never by equality with this one.

    The SINGLE tokens (in one digit) then fill the union to a level, the least whose quotas (``max(0, level - n_c)``,
    ``n_c`` a color's tokens so far) hold every single, pass by pass: each digit wants its singles at its room's colors
    (under its capacity) that have quota left, in the rotation ``(d + i) & 7``; each color grants in digit order while
    its quota lasts; a pass that grants nothing raises the level. Past the union's cap the rest go by quota alone, in
    digit order over the colors' room (those singles are blocked). Each digit's singles take their colors in color
    order, its tokens in token order."""
    member = members(rows)
    cap = capacities(rows)
    count = [[0] * COLORS for _ in rows]
    union = [0] * COLORS
    out: dict[int, int] = {}
    blocked = 0
    for t in (t for t in member if len(member[t]) > 1):
        pool = [c for c in range(COLORS) if union[c] < UNION_CAP and all(count[d][c] < cap[d] for d in member[t])]
        if pool:
            pick = min(pool, key=lambda c: (sum(count[d][c] for d in member[t]), union[c], (c - t) & 7))
        else:
            pick = min((c for c in range(COLORS) if union[c] < UNION_CAP), key=lambda c: (union[c], c))
            blocked += 1
        out[t] = pick
        union[pick] += 1
        for d in member[t]:
            count[d][pick] += 1

    singles = [[t for t in member if member[t] == [d]] for d in range(len(rows))]
    total = sum(len(s) for s in singles)
    level = next(lv for lv in range(UNION_CAP + 1) if sum(max(0, lv - n) for n in union) >= total)
    rem = [len(s) for s in singles]
    take = [[0] * COLORS for _ in rows]
    while any(rem) and level <= UNION_CAP:
        quota = [max(0, level - n) for n in union]
        want = [[0] * COLORS for _ in rows]
        for d in range(len(rows)):
            r = rem[d]
            for i in range(COLORS):
                c = (d + i) & 7
                w = min(r, max(0, cap[d] - count[d][c] - take[d][c])) if quota[c] > 0 else 0
                want[d][c] = w
                r -= w
        granted = 0
        for c in range(COLORS):
            for d in range(len(rows)):
                g = min(want[d][c], quota[c])
                quota[c] -= g
                take[d][c] += g
                rem[d] -= g
                union[c] += g
                granted += g
        level += 0 if granted else 1
    bound, at = 0, 0
    for c in range(COLORS):
        room = UNION_CAP - union[c]
        for d in range(len(rows)):
            g = max(0, min(at + rem[d], bound + room) - max(at, bound))
            take[d][c] += g
            union[c] += g
            blocked += g
            at += rem[d]
        at = 0
        bound += room
    for d, tokens in enumerate(singles):
        picks = [c for c in range(COLORS) for _ in range(take[d][c])]
        for t, c in zip(tokens, picks):
            out[t] = c
            count[d][c] += 1
    return out, blocked


def slots(colors: dict[int, int]) -> dict[int, int]:
    """The residue layout: each token's slot, ``8 x`` its rank among its color's tokens in token order ``+`` its color."""
    rank = [0] * COLORS
    out = {}
    for t in sorted(colors):
        c = colors[t]
        out[t] = COLORS * rank[c] + c
        rank[c] += 1
    return out


def extent(colors: dict[int, int]) -> int:
    """A union's slots: ``8 x`` its largest color's count."""
    counts = [0] * COLORS
    for c in colors.values():
        counts[c] += 1
    return COLORS * max(counts) if colors else 0


def check(rows: Sequence[int], colors: dict[int, int]) -> tuple[list[str], int]:
    """A coloring's broken bounds, and its tokens over their digits' capacities (each a step, never a conflict):
    every union token colored once, and nothing else; colors 0-7; the union at most :data:`UNION_CAP` a color."""
    member = members(rows)
    problems = []
    if set(colors) != set(member):
        problems.append(f"the colored tokens are not the union: {len(set(member) - set(colors))} uncolored, "
                        f"{len(set(colors) - set(member))} outside it")
    if any(not 0 <= c < COLORS for c in colors.values()):
        problems.append("a color outside 0-7")
    union = [0] * COLORS
    for c in colors.values():
        union[c % COLORS] += 1
    if max(union, default=0) > UNION_CAP:
        problems.append(f"a color holds {max(union)} tokens, over the union's cap {UNION_CAP}")
    cap = capacities(rows)
    over = 0
    for d, row in enumerate(rows):
        counts = [0] * COLORS
        for t in member:
            if row >> t & 1 and t in colors:
                counts[colors[t] % COLORS] += 1
        over += sum(max(0, n - cap[d]) for n in counts)
    return problems, over
