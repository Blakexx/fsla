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

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
1564c979; 7 repetitions, every ring exact; cycles a 128-byte row a CTA, the median, the spread under 0.5%):

| the rows | 4 warps | 16 warps |
|---|---:|---:|
| a table of 65,536 rows (8 MB), a CTA its own | 5.78 | 3.69 |
| a table of 1,048,576 rows (128 MB), a CTA its own | 14.62 | 3.79 |
| a table of 1,024 rows (128 KB, a window's values), a CTA its own | | 2.70 |
| the same table, a unit's 16 CTAs the same rows | | 2.56 |

WHAT IT SHOWS.
- SIXTEEN WARPS FETCH A SCATTERED ROW IN 2.6 TO 3.8 CYCLES, straight into shared memory and through no register: 2.70
  from a window's 128 KB, 3.69 from 8 MB, 3.79 from 128 MB -- beside 8.5 for four warps' own loads
  ([`value_gather`](value_gather.md)) and 32.8 for the bulk engine's request ([`state_move`](state_move.md)).
- SIXTEEN WARPS RUN FASTER THAN FOUR: 1.6 times on the 8 MB table and 3.9 times on the 128 MB one, where sixteen
  warps all but hide the table the cache cannot hold (3.79 against 3.69). Sixteen warps hold four times four warps'
  copies in flight; that this is the cause is NOT SHOWN, no row varying the copies in flight alone.
- BESIDE THIS CARD'S STREAM ([`smem_rates`](smem_rates.md): a `cp.async.cg` stream from L2, every SM streaming, 3.95
  cycles a 128-byte line): sixteen warps' scattered rows of the 8 MB and 128 MB tables cost about the stream's line
  (3.69 and 3.79), and of a window's 128 KB less (2.70).
- A UNIT'S 16 CTAs ASKING FOR THE SAME ROWS cost no more than each its own (2.56 against 2.70).

THE PREDICTIONS, WRITTEN BEFORE THE RUN, AND THEIR BASIS. Every row whose table the cache holds was predicted at 6.5 to
9.5 on one price, a `cp.async.cg` stream at 7.12 cycles a 128-byte line, with more warps pushing nothing more through.
The 128 MB rows were predicted at 10 to 16 (four warps) and 9 to 16 (sixteen) from `value_gather`'s 1.6 times.
- four warps, 8 MB: 5.78 -- MISSED. Sixteen warps, 8 MB: 3.69 -- MISSED. Four warps, 128 MB: 14.62 -- HELD.
  Sixteen warps, 128 MB: 3.79 -- MISSED. Sixteen warps, 128 KB, own rows: 2.70 -- MISSED; a unit's rows: 2.56 -- MISSED.

WHAT THE MISSES SAY. THE PRICE WAS ANOTHER CARD'S: 7.12 is [`smem_rates`](smem_rates.md)'s sm_86 record, and the same
page's H100 figure for that stream is 3.95. Priced at this card's 3.95, the sixteen-warp rows of the 8 MB and 128 MB
tables are within 7% of it, the window's rows are a third under it, and the four-warp row of the 8 MB table is half
over it. The second premise, that more warps push nothing more through, the rows contradict. What binds the fetch at
sixteen warps is NOT SHOWN.

WHAT IT IS NOT. The fetch alone: every thread issues copies and does nothing else, and the rows are never read. A CTA
that fetches its value lines by token index while it folds is NOT MEASURED.

Record: `measure/lab/results/value_fetch/sm_90a.json`.
