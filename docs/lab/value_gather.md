# `value_gather`

Mirrors `measure/lab/experiments/value_gather.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`state_move`](state_move.md) read the bulk engine's 128-byte load of a scattered row at 32.8 cycles, eight times
its bytes' share of a packed copy. What does the same row cost fetched by a WARP'S OWN LOAD -- a lane 4 bytes, a
row an instruction -- and stored into shared memory, by the warps issuing and by whether the cache holds the table?

THE FORM. A CTA an SM of 1 or 4 warps. Each warp fetches its own scattered rows of a table of 65,536 rows (8 MB) or
1,048,576 (128 MB) and stores each into its 16-row ring in shared memory; a row's load runs two batches of 8 ahead
of its store, at fixed registers. The ring's last rows are checked exactly.

UNMEASURED: no card has run it.
