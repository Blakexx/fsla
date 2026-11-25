# `carry_supply`

Mirrors `measure/lab/experiments/carry_supply.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the LIBRARY's carry CTA ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#supply)), its
fold warpgroups' supply and coefficients built and every other component a stub, list each of its sets' live tokens
from a call's real operands -- the masks, the mask-preparation pass's prepared words, the densities the carve is
derived from, the write side's rows -- land their value lines in its rows and expand their coefficients into its
tiles ([`internals/carry/carry_coefs.md`](../internals/carry/carry_coefs.md#expand)), exactly; and what does a window
cost a CTA with them running?

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

MEASURED WITH THE COEFFICIENTS (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the
experiment at ac5fc941, its first run on a device; 7 repetitions; every call's check exact -- its rounds, tokens and
row bytes as below, and every coefficient of every kept round's chunks, a column its levels' bytes multiplied in fp32
and rounded to bf16 once, a place with no token zeros, a set's chunks' columns its leaves each once; cycles a window a
CTA, the median):

| the call | its step, chunks a set | timed | every warp's timeline stamped | the supply alone, timed |
|---|---|---:|---:|---:|
| the cell | 16 x 4, 1 | 5,136 | 6,792 | 2,377 |
| the write side masked on both levels | 8 x 4, 1 | 4,381 | 5,764 | 2,340 |
| the write side whole on both levels | 16 x 4, 4 | 25,158 | 28,151 | 5,577 |
| one digit a token on each level | 4 x 4, 1 | 4,350 | 6,639 | 3,626 |
| both levels, the read side's densities crossed | 16 x 4, 1 | 2,732 | 3,794 | 1,339 |

PREDICTED, before the run (timed, and stamped): the cell 2,650 to 3,150 (4,200 to 4,800); both levels 2,420 to 2,700
(3,950 to 4,350); whole 12,500 to 17,500 (14,000 to 19,100); one digit 3,650 to 3,800 (5,200 to 5,420); crossed 1,450
to 1,700 (2,680 to 2,950) -- a stamped range its timed one and the stamps' cost in the record before -- the supply's row and a round's expansion at an instruction count (170
a lane a chunk of 64 columns at 2.1 cycles an instruction), less what was taken to hide under the supply's waits on
its copies. ALL TEN ROWS MISSED HIGH, by 1.2 to 1.9 times. THE WRONG PRICES, by the stamped rows (a fold warp, CTA 0):
- THE BYTES' LOADS WERE PRICED AS FREE AND ARE NOT. A token's bytes are register loads from global memory, scattered by
  token, and take about 700 cycles to land. In the cell a set's first round is 1,237 cycles in its coefficient phase
  and its later rounds 519 and 481; and the phase behind each round's landing, where the next chunk's bytes are first
  read, is 346 to 734 where it was 70 to 365. "Loaded a chunk ahead" is not ahead enough, and a set's first round's
  are read in the trip that loads them (the checker's `LOAD_IN_TRIP`, declared here, is that cost).
- THE EXPANSION'S OWN WORK IS ABOUT 500 CYCLES A CHUNK OF 64 COLUMNS, stamped, where the count gave 360: the cell's
  later rounds, and 655 a chunk in the whole call (2,620 a round of four chunks).
- THE WHOLE CALL: a fold warp is 21,526 of its 28,004-cycle window in the coefficient phase -- a window's 262,144
  coefficients a fold -- with 2,080 issuing its rounds and 1,608 behind their landing.
- ONE DIGIT: the coefficient phase is 532 a window, and the window grew by what the warps wait on each other at its
  end (the window phase 1,795 where it was 676, the words 1,877 where 1,036).

THE SUPPLY ALONE, before the coefficients (the same card and day, the experiment at 05b6fec2, whose rows the table
above repeats; its record is the commit's):

MEASURED (the same card, driver 580.196.07; 2026-11-25, the experiment at
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
to 1,390 and 2,430 to 2,590 -- each 3 percent about a held session's reading of the same form on another
box, a reading no record holds. ALL TEN HELD, which says that a reading of this form repeats from box to box within
3 percent and nothing of a model: the form's own prediction was the session's, made before it from the disassembly,
and it missed (the switch gained more than its instruction count gave on four of the five calls).

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

SINCE THE RECORD the bytes arrive by asynchronous copies into shared memory in the round's own group, and the
expansion reads them there ([`internals/carry/carry_coefs.md`](../internals/carry/carry_coefs.md#stage)). NOT RUN on
record in that form.

WHAT IT IS NOT: the fold. No step runs and no accumulator is held; the other roles are stubs; the rows carry no
scale; the call is one head and its values are the cache's.

Record: `measure/lab/results/carry_supply/sm_90a.json` (the run with the coefficients), the five stamped rows' timelines beside it.
