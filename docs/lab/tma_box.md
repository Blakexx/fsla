# `tma_box`

Mirrors `measure/lab/experiments/tma_box.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's loader stages a K-step's `x` [128 x 64] and weights [256 x 64] bf16, 48 KB a stage. What does a stage cost a CTA by the tensor copy (TMA) against `cp.async` by one warp?

THE FORM. A CTA an SM, a ring of three 48 KB stages, a loader and a warpgroup that waits on each stage, reads a word of it a thread and releases it; the operands L2-resident. `cp.async` 16 bytes a lane by one warp into wgmma's unswizzled K-major bricks; one tensor copy an operand of a 2-d box of 128-byte lines, 128-byte swizzled (what a swizzled wgmma descriptor reads); one tensor copy an operand of a 3-d box of 16-byte lines (the unswizzled bricks themselves). The last three stages landed are checked byte for byte in every CTA.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: a stage is the CTA's own x tile (16 KB of distinct lines, stream_rate's 30.8 bytes a cycle an SM: 532 cycles) and the 128 KB of weights every CTA reads (55.2 at that footprint: 594); the two paths adding give 1,126 cycles a stage, apart 594 -- the row decides which; the 16-byte-line box's 3,072 lines a stage each a request of the copy engine's):

- cp.async, a warp, 16 bytes a lane: a 48 KB stage: 560.00-1,200.00 cycles a stage.
- a tensor copy, 128-byte lines swizzled: a 48 KB stage: 560.00-1,200.00 cycles a stage.
- the 16-byte-line box over the 128-byte box: 1.50-4.00 times.
