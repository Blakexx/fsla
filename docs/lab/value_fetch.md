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
ring a tensor step's tile, a row's 8 pieces in bricks 17 lines apart; both; BY BRICK, a lane a row and an instruction
one piece of the warp's 32 rows into that piece's run of 32 consecutive lines; and a row's 8 pieces TURNED within its
128 bytes by its place, where they land or where they are read (NOT RUN yet).

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
c1beb905; 7 repetitions, every ring exact; cycles a 128-byte row a CTA, the median, the spread under 0.5%):

| the rows | 4 warps | 16 warps |
|---|---:|---:|
| a table of 65,536 rows (8 MB), a CTA its own | 5.79 | 3.69 |
| a table of 1,048,576 rows (128 MB), a CTA its own | 14.63 | 3.80 |
| a table of 1,024 rows (128 KB, a window's values), a CTA its own | | 2.70 |
| the same table, a unit's 16 CTAs the same rows | | 2.55 |
| the window's table, the copy's size a register | | 2.70 |
| the window's table, a row's pieces in bricks 17 lines apart | | 16.57 |
| the window's table, both | | 16.57 |
| the window's table, by brick: an instruction one piece of 32 rows into a contiguous run | | 16.62 |

WHAT IT SHOWS.
- SIXTEEN WARPS FETCH A SCATTERED ROW IN 2.6 TO 3.8 CYCLES, straight into shared memory and through no register,
  WHERE EVERY INSTRUCTION MOVES WHOLE ROWS: 8 lanes a row's 128 bytes, read together and landed together. Beside 8.5 for
  four warps' own loads ([`value_gather`](value_gather.md)) and 32.8 for the bulk engine's request
  ([`state_move`](state_move.md)).
- WHERE IT DOES NOT, IT IS SIX TIMES THAT, AT EITHER END: 16.57 with a row read whole and its 8 pieces landed in
  bricks 17 lines apart; 16.62 with the writes contiguous (512 bytes an instruction) and the reads 16 bytes of 32
  rows. The two differ in which end is in pieces and cost the same.
- The copy's sized form is not a cost (2.70). Sixteen warps run faster than four, and a unit's 16 CTAs asking
  for the same rows cost no more than each its own.

THE PREDICTIONS, WRITTEN BEFORE EACH RUN.
- The six plain rows: within a tenth of their first run -- HELD, twice.
- The copy's size a register: 6 to 16, on the hypothesis that this form is the slow one -- MISSED, 2.70; the
  hypothesis was wrong.
- Bricks 17 lines apart: 2.4 to 3.6, the layout putting four of an instruction's lanes on every bank as contiguous
  rows do -- MISSED, 16.57. Both: 6 to 16 -- 16.57, inside the range for the wrong reason.
- By brick: 2.5 to 8, its writes being contiguous, which the run before had been read as what a copy's write costs
  by -- MISSED, 16.62.

TWO WRONG PRICES, NAMED. First, an asynchronous copy's write was priced by the banks its lanes fall on, as a thread's
own store is: the brick layout has a contiguous row's count on every bank and six times its cost. Second, the miss was
then read as contiguity of the WRITES alone: the form by brick writes 512 contiguous bytes an instruction and costs
the same. What the three rows agree on is that an instruction is cheap where its lanes move whole 128-byte rows at
both ends. WHY IS NOT SHOWN: the brick forms also touch more 128-byte lines an instruction, and 16 warps issue at
once, so lines touched and contention between warps are not parted from the pieces' order; the rows of a row's pieces
turned within its own 128 bytes part the last.

WHAT IT IS NOT. The fetch alone: every thread issues copies and does nothing else, and the rows are never read. A CTA
that fetches its value lines by token index while it folds is NOT MEASURED.

Record: `measure/lab/results/value_fetch/sm_90a.json`.
