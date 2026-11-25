# `value_fetch`

Mirrors `measure/lab/experiments/value_fetch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`value_gather`](value_gather.md) read a scattered 128-byte row at 8.5 cycles fetched by four warps' own loads and
stored into shared memory, and [`state_move`](state_move.md) the bulk engine's request for it at 32.8. What does the
same row cost fetched by the THREADS' OWN ASYNCHRONOUS COPIES -- `cp.async.cg`, 16 bytes a thread an instruction,
straight into shared memory, no register holding the data -- by the warps issuing, by the table, and where a unit's
16 CTAs ask for the same rows? It is the price of a carry CTA fetching its own value lines by token index, which is
what a gather's packed stream stands in for.

THE FORM. A CTA an SM of 4 or 16 warps. A warp's instruction copies 4 rows, 8 lanes a row, into the warp's 16-row
ring in shared memory; a trip is a batch of 8 rows -- two instructions, a commit, and a wait for the batch before it,
so two batches are in flight. The table is 1,024 rows (a window's values, 128 KB), 65,536 (8 MB) or 1,048,576
(128 MB); a CTA takes its own rows, or its unit's (16 CTAs the same rows). The ring's last rows are checked exactly.

THE FORMS, at 16 warps on the window's table: the copy's size a register (the form that zero-fills a dead copy); the
ring a tensor step's tile, a row's 8 pieces in bricks 17 lines apart; both; and BY BRICK, a lane a row and an
instruction one piece of the warp's 32 rows into that piece's run of 32 consecutive lines (NOT RUN yet).

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
4a79ebda; 7 repetitions, every ring exact; cycles a 128-byte row a CTA, the median, the spread under 0.5%):

| the rows | 4 warps | 16 warps |
|---|---:|---:|
| a table of 65,536 rows (8 MB), a CTA its own | 5.79 | 3.68 |
| a table of 1,048,576 rows (128 MB), a CTA its own | 14.63 | 3.79 |
| a table of 1,024 rows (128 KB, a window's values), a CTA its own | | 2.70 |
| the same table, a unit's 16 CTAs the same rows | | 2.55 |
| the window's table, the copy's size a register | | 2.70 |
| the window's table, a row's pieces in bricks 17 lines apart | | 16.56 |
| the window's table, both | | 16.55 |

WHAT IT SHOWS.
- SIXTEEN WARPS FETCH A SCATTERED ROW IN 2.6 TO 3.8 CYCLES, straight into shared memory and through no register,
  where the row lands as 128 contiguous bytes: beside 8.5 for four warps' own loads
  ([`value_gather`](value_gather.md)) and 32.8 for the bulk engine's request ([`state_move`](state_move.md)).
- WHERE THE ROW'S PIECES LAND APART IT IS SIX TIMES THAT: 16.56 with a row's 8 pieces in bricks 17 lines apart,
  against 2.70 contiguous. The copy's form is not it: its size in a register changes nothing (2.70).
- SIXTEEN WARPS RUN FASTER THAN FOUR, and a unit's 16 CTAs asking for the same rows cost no more than each its own.

THE PREDICTIONS, WRITTEN BEFORE THE RUN. The six plain rows each within a tenth of their run at 1564c979: all HELD.
The three forms were written to part why [`carry_fetch`](carry_fetch.md)'s list-and-fetch ran 13.3 to 13.9 a line:
- the copy's size a register: 6 to 16, ON THE HYPOTHESIS that this form is the slow one -- MISSED, 2.70: the
  hypothesis is wrong.
- bricks 17 lines apart: 2.4 to 3.6, the layout putting four of an instruction's lanes on every bank as contiguous
  rows do -- MISSED, 16.56.
- both: 6 to 16 -- 16.55, inside the range for the wrong reason.

THE WRONG PRICE, NAMED: an asynchronous copy's write was priced by the banks its lanes fall on, as a thread's own
store is. That count was the same in both layouts and the cost was not: what the copies' writes follow is whether an
instruction's 32 pieces are contiguous. Why -- whether the copy engine joins contiguous pieces into wider writes -- is
NOT SHOWN.

WHAT IT IS NOT. The fetch alone: every thread issues copies and does nothing else, and the rows are never read. A CTA
that fetches its value lines by token index while it folds is NOT MEASURED.

Record: `measure/lab/results/value_fetch/sm_90a.json`.
