# `load_rate`

Mirrors `measure/lab/experiments/load_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A warp's own load of a scattered 128-byte row costs it about 33 cycles, and four warps a quarter of that
([`value_gather`](value_gather.md)). What is that cost a property of: the instruction, its bytes, the loads in
flight, or the warps issuing?

THE FORM. A CTA an SM of 1, 4, 8 or 16 warps. Each warp fetches its own scattered rows of a table of 65,536 rows
(8 MB, one the cache holds) and folds every word into a chain a lane, nothing stored: the chain's order keeps a
word's use out of the trip that loaded it.
- NARROW: a lane loads 4 bytes, an instruction a row.
- WIDE: a lane loads 16 bytes, an instruction FOUR rows, 8 lanes a row.
A ring of 4, 16 or 64 instructions is in flight, in two halves at their own registers. In the narrow form the
toolkit copies two of a trip's loaded words at the trip's end, whatever the source's form: those two rows' loads
run half a trip ahead of their use. Every lane's chain is checked exactly.

MEASURED: not yet run.
