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

:func:`color` is the reference (the kernel's batched form, its claims resolved in lane order); the kernel's claims race,
so its coloring is graded by :func:`check`, never by equality.
"""
from __future__ import annotations

from collections.abc import Sequence

WINDOW = 512
COLORS = 8
BATCH = 32
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

    Tokens in two or more digits go first (three or more, then two, each in token order), 32 a batch: each takes the
    color least used over its digits among those under every digit's capacity and the union's cap (ties to the union's
    least-used, then ``(c - t) & 7``); a claim that the batch's earlier claims put over a bound is withdrawn and
    requeued, and a batch that keeps nothing runs its first token alone next. Each digit's single tokens then go, digit
    by digit in token order, to the union's least-used color under the digit's capacity and the union's cap. A token
    with no color under its digits' capacities is blocked: it takes the union's least-used color."""
    member = members(rows)
    cap = capacities(rows)
    count = [[0] * COLORS for _ in rows]
    union = [0] * COLORS
    queue = [t for t in member if len(member[t]) > 2] + [t for t in member if len(member[t]) == 2]
    out: dict[int, int] = {}
    blocked, solo = 0, False

    def free(t: int, extra: dict[tuple[int, int], int]) -> list[int]:
        return [c for c in range(COLORS) if union[c] + extra.get((-1, c), 0) < UNION_CAP
                and all(count[d][c] + extra.get((d, c), 0) < cap[d] for d in member[t])]

    def least(t: int, pool: list[int]) -> int:
        return min(pool, key=lambda c: (sum(count[d][c] for d in member[t]), union[c], (c - t) & 7, c))

    while queue:
        batch, queue = queue[:1 if solo else BATCH], queue[1 if solo else BATCH:]
        claims: dict[tuple[int, int], int] = {}
        kept, lost = [], []
        for t in batch:
            pool = free(t, {})
            stuck = not pool
            pick = least(t, pool) if pool else min((c for c in range(COLORS) if union[c] < UNION_CAP),
                                                    key=lambda c: (union[c], c))
            over = claims.get((-1, pick), 0) + union[pick] >= UNION_CAP or (
                not stuck and any(count[d][pick] + claims.get((d, pick), 0) >= cap[d] for d in member[t]))
            for d in member[t]:
                claims[(d, pick)] = claims.get((d, pick), 0) + 1
            claims[(-1, pick)] = claims.get((-1, pick), 0) + 1
            (lost if over else kept).append((t, pick, stuck))
        for t, pick, stuck in kept:
            out[t] = pick
            union[pick] += 1
            blocked += stuck
            for d in member[t]:
                count[d][pick] += 1
        solo = not kept
        queue += [t for t, _, _ in lost]

    for d in range(len(rows)):
        for t in member:
            if len(member[t]) != 1 or member[t][0] != d:
                continue
            pool = [c for c in range(COLORS) if count[d][c] < cap[d] and union[c] < UNION_CAP]
            stuck = not pool
            pick = min(pool or [c for c in range(COLORS) if union[c] < UNION_CAP], key=lambda c: (union[c], c))
            out[t] = pick
            union[pick] += 1
            count[d][pick] += 1
            blocked += stuck
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
