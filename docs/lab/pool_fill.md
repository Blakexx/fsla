# `pool_fill`

Mirrors `measure/lab/experiments/pool_fill.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A 16 × 16 box's fold takes, for each of its 16 lists, the listed tokens' values and their coefficients on the box's
16 dense leaves. [`value_gather`](value_gather.md) priced a scattered value row alone at 8.5 cycles by four warps.
What does an ENTRY cost a CTA when producer warps fill both pools from scattered rows of global memory -- the
value's 128 bytes and the token's 16 dense bytes, each by the warps' own load -- and the fold's steps consume them?

THE FORM. A CTA an SM: a fold warpgroup holding the box's state (16 accumulators of 16 leaves × 64 channels) and 1, 2
or 4 producer warps. A chunk is 4 lists of 16 entries, in one of two slots of tiles. A producer takes the entries of
its residue FOUR AN INSTRUCTION, 8 lanes an entry -- a warp's load is bound by its latency, so by the rows it has in
flight ([`load_rate`](load_rate.md)): a lane loads 16 bytes of its entry's value and stores them as one line of the
list's value tile (M|N-major; a row's 8 lanes cover the 32 banks once), and loads a word of its token's 16 dense
bytes, multiplies by the entry's byte, rounds to bf16 and stores into the list's coefficient tile, an instruction's
four entries on four bank groups. A batch is 16 entries; its loads run two batches ahead of its stores, at fixed
registers (32 rows in flight a warp). The fold, a chunk behind, runs the chunk's 4 steps (64 × 16 × 16, both
operands from shared memory) in one commit group and frees the slot.
- `the producers alone`: no step runs; the fold only hands the slots back.
- `the producers and the fold`: both, the state checked exactly after one window.
An entry's token is drawn; the list it comes from is not loaded, and the table of 65,536 tokens (8 MB of values)
is one the cache holds.

MEASURED: not yet run.
