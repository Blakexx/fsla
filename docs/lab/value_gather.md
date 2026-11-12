# `value_gather`

Mirrors `measure/lab/experiments/value_gather.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`state_move`](state_move.md) read the bulk engine's 128-byte load of a scattered row at 32.8 cycles, eight times
its bytes' share of a packed copy. What does the same row cost fetched by a WARP'S OWN LOAD -- a lane 4 bytes, a
row an instruction -- and stored into shared memory, by the warps issuing and by whether the cache holds the table?

THE FORM. A CTA an SM of 1 or 4 warps. Each warp fetches its own scattered rows of a table of 65,536 rows (8 MB) or
1,048,576 (128 MB) and stores each into its 16-row ring in shared memory; a row's load runs two batches of 8 ahead
of its store, at fixed registers. The ring's last rows are checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60311139, driver 595.91.07; 2026-11-12, the experiment at
1eba273a; 7 repetitions, every ring exact; cycles a 128-byte row a CTA, the median, the spread under 0.5%):

| the warps | a table of 65,536 rows | of 1,048,576 |
|---|---|---|
| 1 | 33.4 | 62.5 |
| 4 | 8.5 | 13.7 |

WHAT IT SHOWS.
- FOUR WARPS FETCH A SCATTERED ROW IN 8.5 CYCLES from a table the cache holds: a quarter of the bulk engine's
  request for the same row (32.8, [`state_move`](state_move.md)) and twice its bytes' share of a packed copy.
- ONE WARP IS FOUR TIMES THAT (33.4): the rows scale with the warps issuing, up to four.
- A TABLE THE CACHE CANNOT HOLD costs 1.6 times as much with four warps (13.7).

Record: `measure/lab/results/value_gather/sm_90a.json`.
