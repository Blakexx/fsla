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
- THE COPIES IN FLIGHT SET THE RATE HERE, NOT THE PATH: sixteen warps hold four times four warps' copies in flight and
  run 1.6 times as fast on the 8 MB table and 3.9 times on the 128 MB one, where sixteen warps all but hide the table
  the cache cannot hold (3.79 against 3.69).
- A UNIT'S 16 CTAs ASKING FOR THE SAME ROWS cost no more than each its own (2.56 against 2.70).

THE PREDICTIONS, WRITTEN BEFORE THE RUN, AND THEIR BASIS. Every row whose table the cache holds was predicted at 6.5 to
9.5 on one price: the SM's path, from [`smem_rates`](smem_rates.md)'s `cp.async.cg` stream at 7.12 cycles a 128-byte
line, with more warps pushing nothing more through. The 128 MB rows were predicted at 10 to 16 (four warps) and 9 to 16
(sixteen) from `value_gather`'s 1.6 times.
- four warps, 8 MB: 5.78 -- MISSED. Sixteen warps, 8 MB: 3.69 -- MISSED. Four warps, 128 MB: 14.62 -- HELD.
  Sixteen warps, 128 MB: 3.79 -- MISSED. Sixteen warps, 128 KB, own rows: 2.70 -- MISSED; a unit's rows: 2.56 -- MISSED.

WHAT THE MISSES SAY. The price was wrong: the stream's 7.12 was taken for the path's limit, and the rows here go
through the same path at 2.6 cycles a row (about 50 bytes a cycle an SM), so 7.12 was that stream's own count of copies
in flight and no ceiling. What binds the fetch at sixteen warps -- the path, or still the copies in flight -- is NOT
SHOWN: no row holds more in flight than sixteen warps' two batches.

WHAT IT IS NOT. The fetch alone: every thread issues copies and does nothing else, and the rows are never read. A CTA
that fetches its value lines by token index while it folds is NOT MEASURED.

Record: `measure/lab/results/value_fetch/sm_90a.json`.
