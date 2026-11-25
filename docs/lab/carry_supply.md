# `carry_supply`

Mirrors `measure/lab/experiments/carry_supply.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the LIBRARY's carry CTA ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#supply)), its
fold warpgroups' supply built and every other component a stub, list each of its sets' live tokens from a call's real
operands -- the masks, the mask-preparation pass's prepared words, the densities the carve is derived from -- and land
their value lines in its rows, exactly; and what does a window cost a CTA with that supply running?

THE FORM. This file compiles the library's device code and launches the mask-preparation pass's kernel first. A call
of one head: two levels of 256 digits, 64 value channels, a window 1,024 tokens. Its five calls: THE CELL, the write
side masked on level 0 at 4 to 5 live digits a token and whole on level 1 (the carve 32 x 16, 4 sets a fold); and the
write side masked on both, level 1 a run of 40 digits a token (the carve 64 x 8, 8 sets a fold of 32 leaves); THE WRITE SIDE WHOLE on both levels, no mask of its own (a fold is
one set and every token is live in it: the dense end); and ONE DIGIT A TOKEN on each level (sets of 16 leaves that are
mostly empty: the sparse end); and THE BOTH-LEVELS MASKS WITH THE READ SIDE'S DENSITIES CROSSED (level 0's the write
side's of level 1 and the reverse), which changes nothing the supply reads but the carve the densities give. The read side's densities are the cell's mirror and its masks are never read (its role is a
stub). The grid is the head's CTAs. A CALL HAS THREE ROWS. CHECKED through the kernel's tracer (4 windows of the
cell, the sparse call and the crossed call, 2 of the others): the tracer waits for a round's copies and copies its tokens and rows out, and on the
host every round of every set of every fold -- of each sharer's part of it, where warps share a set -- is there once, its tokens the set's list in order, its rows their value
lines in the step's swizzle order, a dead place's row zero, on its boundary; the sets' leaves tile the leaf space
once. TIMED with the library's tracer (nothing) on a call of 64 windows. STAMPED: the same call with the kernel's
phases and hand-offs stamped, whose timelines of two CTAs the record holds.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
05b6fec2; 7 repetitions; every call's check exact -- 12,288, 4,096, 16,384, 3,920 and 4,096 rounds, their tokens and
their row bytes, the sets tiling the leaf space; cycles a window a CTA, the median):

| the call | its carve, sets a fold | timed | every warp's timeline stamped |
|---|---|---:|---:|
| the cell | 32 x 16, 4 | 2,377 | 3,833 |
| the write side masked on both levels | 64 x 8, 8 | 2,340 | 3,817 |
| the write side whole on both levels | 128 x 4, 1 | 5,577 | 6,824 |
| one digit a token on each level | 128 x 4, 16 | 3,626 | 5,161 |
| both levels, the read side's densities crossed | 32 x 16, 4 | 1,339 | 2,514 |

PREDICTED, before the run (timed and stamped): the cell 2,300 to 2,450 and 3,730 to 3,970; both levels 2,270 to 2,410
and 3,700 to 3,930; whole 5,400 to 5,810 and 6,620 to 7,030; one digit 3,520 to 3,740 and 5,010 to 5,320; crossed 1,300
to 1,390 and 2,430 to 2,590 -- each 3 percent about an unrecorded held session's reading of the same form on another
box. ALL TEN HELD.

THE FORM BEFORE IT tested the pattern of listed levels a level a set inside its loops and formed a window's place in
the value lines and the call's own tokens with 64-bit multiplications a window (the record at 34b9a08d, the same card
and calls): 2,655, 2,614, 6,185, 4,007 and 1,540 timed. One switch a call to windows compiled for the pattern, and that
arithmetic kept from window to window, is 9 to 13 percent of every call.

WHAT A FOLD WARP'S WINDOW IS MADE OF, by the stamped rows (the stamps themselves add 1,100 to 1,500 cycles to a
window, so these are shares):
- THE CELL, a set a window of 3,674: the folds' hand-offs 660; the set's running sums and its tokens laid 690 (69 of
  them); the next set's words asked for 119 and had 106; then three rounds, each 234 to 272 for its 8 loads and 8
  copies, a wait for the round before of 103, 579 and 139, and 70 to 365 to its end.
- BOTH LEVELS, two sets a window of 3,746: a set's sums and lay 546 and 456; its words 370 and 179; ONE round a set, 250
  to 392, and 130 to 143 waiting.
- WHOLE, one set a fold shared by its four warps, a window of 6,666: the set's sums and a quarter of every word laid
  928; eight rounds of 179 to 274, and between them a wait for the round before of 53 to 574 -- 2,337 of the window
  waiting on copies with two rounds in flight, against 1,699 issuing them.
- ONE DIGIT, four sets a window of 5,238, most of them empty: a set 417 to 694, 2,357 a window; the words 1,510, which
  holds the wait at the window's end for the fold's slowest warp; about one round a window. An empty set runs every
  instruction a full one does -- its sums, its lay's tail, its words -- and nothing skips it.

A ROUND'S OWN COST is its 8 loads and 8 copy instructions, about 180 to 390 cycles stamped whatever number of its
places are live; what a warp waits for after it is the round before's copies. NOT MEASURED on record: more than two
rounds in flight, and a form that skips the copy instructions of a round's dead places.

WHAT IT IS NOT: the fold. No coefficient is expanded and no step runs; the other roles are stubs; the call is one
head and its values are the cache's.

Record: `measure/lab/results/carry_supply/sm_90a.json`, the five stamped rows' timelines beside it.
