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
34b9a08d; 7 repetitions; every call's check exact -- 12,288, 4,096, 16,384, 3,920 and 4,096 rounds, their tokens and
their row bytes, the sets tiling the leaf space; cycles a window a CTA, the median):

| the call | its carve, sets a fold | timed | every warp's timeline stamped |
|---|---|---:|---:|
| the cell | 32 x 16, 4 | 2,655 | 4,086 |
| the write side masked on both levels | 64 x 8, 8 | 2,614 | 4,144 |
| the write side whole on both levels | 128 x 4, 1 | 6,185 | 7,623 |
| one digit a token on each level | 128 x 4, 16 | 4,007 | 5,506 |
| both levels, the read side's densities crossed | 32 x 16, 4 | 1,540 | 2,724 |

PREDICTED, before the run (timed and stamped): the cell 2,500 to 2,800 and 3,950 to 4,300; both levels 2,200 to 2,700
and 3,600 to 4,300; whole 5,900 to 6,500 and 7,300 to 7,900; one digit 4,000 to 4,450 and 5,800 to 6,300; crossed 1,420
to 1,570 and 2,600 to 2,850 -- each about an unrecorded held session's reading of the same form, the both-levels call's
a window and two of the crossed call's sets under the carve the rule now gives it. NINE HELD. The one-digit call's
stamped row MISSED LOW: its range was that session's figure under the carve the call had then (256 x 2), and under
128 x 4 its timed row fell 5 percent and its stamped one 8 -- a stamped window's own cost was taken as the same across
a change of carve.

WHAT A FOLD WARP'S WINDOW IS MADE OF, by the stamped rows (the stamps themselves add 1,100 to 1,500 cycles to a
window, so these are shares):
- THE CELL, a set a window of 3,935: the folds' hand-offs 740; the set's running sums and its tokens laid 690 (69 of
  them); the next set's words asked for 274 and had 123; then three rounds, each 298 to 318 for its 8 loads and 8
  copies, a wait for the round before of 58, 516 and 91, and 70 to 84 to its end.
- BOTH LEVELS, two sets a window of 4,018: a set's sums and lay 561 and 457; its words 356 and 114; ONE round a set, 353
  to 357, and 52 to 103 waiting.
- WHOLE, one set a fold shared by its four warps, a window of 7,480: the set's sums and a quarter of every word laid
  1,000; eight rounds of 288 to 325, and between them a wait for the round before of 75 to 557 -- 2,119 of the window
  waiting on copies with two rounds in flight, against 2,382 issuing them.
- ONE DIGIT, four sets a window of 5,649, most of them empty: a set 422 to 677, 2,250 a window; the words 1,266, which
  holds the wait at the window's end for the fold's slowest warp; about one round a window. NOT EXPLAINED: what an
  empty set's 400 to 700 cycles are.

A ROUND'S OWN COST is its 8 loads and 8 copy instructions, about 300 to 360 cycles stamped whatever number of its
places are live; what a warp waits for after it is the round before's copies. NOT MEASURED on record: more than two
rounds in flight, and a form that skips the copy instructions of a round's dead places.

SINCE THE RECORD the fold's windows are compiled once a pattern of listed levels behind one switch a call, and a
window's own arithmetic is kept from window to window ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#supply)).
NOT RUN on record in that form.

WHAT IT IS NOT: the fold. No coefficient is expanded and no step runs; the other roles are stubs; the call is one
head and its values are the cache's.

Record: `measure/lab/results/carry_supply/sm_90a.json`, the five stamped rows' timelines beside it.
