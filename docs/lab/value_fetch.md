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

NOT MEASURED: no run of it is recorded.
