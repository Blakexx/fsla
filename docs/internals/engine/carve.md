# `rola/engine/carve.py` — the carve planner

Mirror of the planner the `carve` rule calls ([`rules.md`](rules.md#carve) says what the
carve is and why one descriptor). This page explains how producer v1 builds a carve and how
the check refuses one; the code is the authoritative statement of each step.

## <a id="budget"></a>The budget

The box's state is the CTA's fp32 accumulators: `WARPS` (8) warps x 32 lanes x 128
registers a lane, `BOX_VALUES` = 32,768. Its leaves are that over its channel extent: 1,024
at `CHANNELS` (32). `CAP` (32) bounds any one level's extent, and `MIN_DENSE` (16) is a
warp's least operand rows, one MMA tile.

## <a id="producer-v1"></a>Producer v1

The extents are allocated as powers of two, one bit at a time, under the cap and each
level's width:

1. `DD` levels, round robin, as long as one has room.
2. `SD` then `DS`, by turns, each to its tag's widest level with room.
3. `SS` levels, the widest first.

When the bits run out before the box is full, the channel extent doubles (the leaves
halve) and the allocation runs again, up to `d_v`. When even that cannot fill the box, the
cap lifts and the box takes the widest levels whole, smaller than the budget if the levels
are narrow (a depth-one declaration is its own whole box).

## <a id="sides"></a>Each side's plan

Each side's plan follows from the box (`_side`). A side with a sparse level lists it and
forms its operand from its dense levels, both in level order; a warp owns `box leaves / 8`
leaves, its operand rows whole while they fit (`tile_rows`), the rest in lists
(`tile_lists`). A side with no sparse level and two dense levels names one as its rows and
the other as its coefficients, in that order: its coefficients the other side's list level
where the other side lists exactly one level and it is one of this side's, else its first
dense level -- both sides of a dense-dense declaration alike, the readout of `[SD, DD]`
grouped by the fold's list level -- so at the box each side is 32 rows a warp in 4 groups,
as at the alternating carve, and the carry's one box serves both
(`docs/internals/carry/carry.md#sides`). `publish_in_registers` compares the two sides'
partitions of the box's leaves among the warps (a warp's groups -- its list levels' values,
or a side with no list level's coefficient values -- by its rows): equal partitions mean a
warp's fold state is its readout's, and the hand-off stays in its registers
(`docs/internals/carry/parts.md#same-split`).

## <a id="check"></a>The check

`check_carve` refuses, naming the reason: a declaration with a side of only sparse levels,
a width that is not a shipped power of two, an extent that does not divide its width, a
channel extent that does not divide `d_v`, a box over the state budget, a side's plan that
is not the box's, a warp with fewer operand rows than an MMA tile, and a publish route that
disagrees with the warps' ownership. The planner's own carves pass through it.

## <a id="planned-once"></a>Planned once a declaration

`plan_carve` is a pure function of the declaration (the widths, each side's per-level flags and `d_v`) and its
`Carve` is frozen, so it plans once a declaration (`functools.cache` on `_planned`) and every later call of the
shape returns the same object. Planning in Python -- the build, the check's rebuild, each side's warp ownership over
the box's 1,024 leaves -- ran twice a prefill call (the envelope's clauses and the facts node) before the first
kernel launched, the GPU idle meanwhile (`tools/op_timeline.py` found it). Planned once: the A100's sparse prefill at
`nl64k-alt-k4-h4` 10.50 -> 8.03 ms and layer g16 11.27 -> 8.72 ms, interleaved in one bench run against the tree
before (2026-11-06). A refused declaration raises every time (an exception is not cached).

## <a id="open"></a>Open

A single `DD` level's share is the cap's, not density's: at very low density a larger
share wins (by about a third on a tied layout), which a declared budget and the structural
cost model may later take. A box of fewer than 128 leaves (a narrow depth-one declaration)
is refused: it leaves a warp under an MMA tile of rows, and needs a plan that splits the
channels across warps.
