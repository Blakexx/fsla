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
write side masked on both, level 1 a run of 40 digits a token (the carve 256 x 2; level 0's write group of 4 digits is raised to 8,
a set's least of 16 leaves, which makes 16 sets a fold); THE WRITE SIDE WHOLE on both levels, no mask of its own (a fold is
one set and every token is live in it: the dense end); and ONE DIGIT A TOKEN on each level (sets of 16 leaves that are
mostly empty: the sparse end); and THE BOTH-LEVELS MASKS WITH THE READ SIDE'S DENSITIES CROSSED (level 0's the write
side's of level 1 and the reverse), which changes nothing the supply reads but the carve the densities give. The last
three are NOT RUN on record. The read side's densities are the cell's mirror and its masks are never read (its role is a
stub). The grid is the head's CTAs. A CALL HAS THREE ROWS. CHECKED through the kernel's tracer (4 windows of the
cell, the sparse call and the crossed call, 2 of the others): the tracer waits for a round's copies and copies its tokens and rows out, and on the
host every round of every set of every fold -- of each sharer's part of it, where warps share a set -- is there once, its tokens the set's list in order, its rows their value
lines in the step's swizzle order, a dead place's row zero, on its boundary; the sets' leaves tile the leaf space
once. TIMED with the library's tracer (nothing) on a call of 64 windows. STAMPED: the same call with the kernel's
phases and hand-offs stamped, whose timelines of two CTAs the record holds.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
b64785d8; 7 repetitions; both calls' checks exact -- 12,288 and 16,384 rounds, their tokens and their row bytes, the
sets tiling the leaf space; cycles a window a CTA, the median):

| the call | timed | every warp's timeline stamped |
|---|---:|---:|
| the cell: 4 sets a fold, about 69 live tokens a set a window | 3,323 | 4,664 |
| the write side masked on both levels: 32 sets a fold, about 19 | 11,486 | 13,974 |

THE RECORD IS OF THE SUPPLY BEFORE A SET'S LEAST: at b64785d8 no group was raised, and the second call had 32 sets a
fold of 8 leaves, about 19 live tokens a set a window, 8 sets a warp a window. On this tree it has 16 sets a fold of
16 leaves, 4 a warp a window. Since then, too, the supply LAYS a set's tokens in a list where it selected them a round,
and warps that share a set split it by bit ([`internals/carry/carry_list.md`](../internals/carry/carry_list.md#lay)).
NOT RUN on record in this form.

PREDICTED, before the run: 3,100 to 3,700 and 4,300 to 5,100; 10,400 to 11,800 and 13,000 to 14,800, each range set
about an unrecorded trial reading of the same form -- ALL FOUR HELD.

WHAT A FOLD WARP'S WINDOW IS MADE OF, by the stamped rows (the stamps themselves add 1,341 and 2,488 cycles to the
window, so these are shares):
- THE CELL, a set a window: the folds' hand-offs 784; the set's running sums 289; the next set's words asked for 203
  and had 99; then three rounds, each 825 to 894 from its select to its 8 copies issued, 55 waiting for the round
  before and 70 to its end. The rounds are 2,553 of the window's 4,523, and the third holds 5 of its 32 places.
- BOTH LEVELS, as recorded, 8 sets a window: a set's sums 240 to 290; the next set's words 130 to 180 and 90 to 230; ONE round of
  840 to 950; 70 to 130 waiting and 110 after -- about 1,620 a set, the round more than half of it and 19 of its 32
  places live.

A ROUND IS PAID WHOLE: its select and its 8 copy instructions cost a warp the same whatever number of its places are
live, so a call of many small sets pays a round a set. NOT MEASURED: a form that skips the copy instructions of a
round's dead places, and what a vote or a branch round an instruction costs beside it.

WHAT IT IS NOT: the fold. No coefficient is expanded and no step runs; the other roles are stubs; the call is one
head and its values are the cache's.

Record: `measure/lab/results/carry_supply/sm_90a.json`, the two stamped rows' timelines beside it.
