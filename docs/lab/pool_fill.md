# `pool_fill`

Mirrors `measure/lab/experiments/pool_fill.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A 16 × 16 box's fold takes, for each of its 16 lists, the listed tokens' values and their coefficients on the box's
16 dense leaves. [`value_gather`](value_gather.md) priced a scattered value row alone at 8.5 cycles by four warps.
What does an ENTRY cost a CTA when producer warps fill both pools from scattered rows of global memory -- the
value's 128 bytes and the token's 16 dense bytes, each by the warps' own load -- and the fold's steps consume them?

THE FORM. A CTA an SM: a fold warpgroup holding the box's state (16 accumulators of 16 leaves × 64 channels) and 1, 2
or 4 producer warps. A chunk is 4 lists of 16 entries, in one of two slots of tiles. A producer takes the entries of
its residue: the value row loaded and stored into the list's value tile (M|N-major, a lane one 4-byte store), the
token's 16 dense bytes loaded, multiplied by the entry's byte, rounded to bf16 and stored into the list's
coefficient tile by 8 lanes. A row's loads run two batches of 8 ahead of its stores, at fixed registers. The fold, a
chunk behind, runs the chunk's 4 steps (64 × 16 × 16, both operands from shared memory) in one commit group and
frees the slot.
- `the producers alone`: no step runs; the fold only hands the slots back.
- `the producers and the fold`: both, the state checked exactly after one window.
An entry's token is drawn; the list it comes from is not loaded, and the table of 65,536 tokens (8 MB of values)
is one the cache holds.

MEASURED: not yet run.
