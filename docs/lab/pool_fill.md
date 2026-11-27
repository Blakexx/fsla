# `pool_fill`

Mirrors `measure/lab/experiments/pool_fill.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A 16 × 16 box's fold takes, for each of its 16 lists, the listed tokens' values and their coefficients on the box's
16 dense leaves. [`value_gather`](value_gather.md) priced a scattered value row alone at 8.5 cycles by four warps.
What does an ENTRY cost a CTA when producer warps fill both pools from scattered rows of global memory -- the
value's 128 bytes and the token's 16 dense bytes, each by the warps' own load -- and the fold's steps consume them?

THE FORM. A CTA an SM: a fold warpgroup holding the box's state (16 accumulators of 16 leaves × 64 channels) and 2, 4
or 8 producer warps. A window is the box's 16 lists of 16 entries, in one of two slots of tiles. A warp's scattered
load costs less the more rows it keeps in flight ([`load_rate`](load_rate.md)), so a producer takes the
entries of its residue in batches of 32:
- 8 VALUE instructions of FOUR entries, 8 lanes an entry: a lane loads 16 bytes of its entry's value and stores
  them as one line of the list's value tile (M|N-major; a row's 8 lanes cover the 32 banks once);
- ONE instruction of all 32 entries' dense bytes, a lane an entry's 16, which the lane multiplies by the entry's
  byte, rounds to bf16 and stores as the two lines of the list's coefficient tile, each store covering the banks
  evenly.
A batch's loads run two batches ahead of its stores, at fixed registers (64 rows in flight a warp). The fold, a
window behind, runs the window's 16 steps (64 × 16 × 16, both operands from shared memory) in one commit group and
frees the slot.
- `the producers alone`: no step runs; the fold only hands the slots back.
- `the producers and the fold`: both, the state checked exactly after two windows.
An entry's token is drawn; the list it comes from is not loaded, and the table of 65,536 tokens (8 MB of values)
is one the cache holds.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60314921, driver 595.91.07; 2026-11-12, the experiment at
a82c550e; 7 repetitions, every state value exact; cycles an entry a CTA, the median):

| the producers | alone | with the fold (lowest to highest of 7) |
|---|---|---|
| 2 | 6.1 | 6.8 (6.8 to 6.9) |
| 4 | 6.3 | 6.3 (6.3 to 6.3) |
| 8 | 5.3 | 5.4 (5.3 to 5.4) |

WHAT IT SHOWS.
- AN ENTRY OF THE FOLD'S POOLS COSTS A CTA 5.4 TO 6.9 CYCLES, its value's 128 bytes and its 16 coefficients fetched
  from scattered rows and stored as the steps' operands. The fold's 16 steps beside the producers add 0.7 cycles
  with two producers and nothing with four or eight.
- THE PRODUCERS DO NOT SCALE: two warps take an entry in 6.1 cycles, four in 6.3, eight in 5.3. Two warps load at
  the rate [`load_rate`](load_rate.md) gives a warp with these rows in flight; past two the entry's 160 bytes of
  shared stores, or its loads' path, binds, and no row here separates them.

Record: `data:records/fsla/measure/lab/pool_fill.run/20261112T055006Z-1b3afa6e/output.json`.
