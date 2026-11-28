# `tma_box`

Mirrors `measure/lab/experiments/tma_box.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's loader stages a K-step's `x` [128 x 64] and weights [256 x 64] bf16, 48 KB a stage. What does a stage cost a CTA by the tensor copy (TMA) against `cp.async` by one warp?

THE FORM. A CTA an SM, a ring of three 48 KB stages, a loader and a warpgroup that waits on each stage, reads a word of it a thread and releases it; the operands L2-resident. `cp.async` 16 bytes a lane by one warp into wgmma's unswizzled K-major bricks; one tensor copy an operand of a 2-d box of 128-byte lines, 128-byte swizzled (what a swizzled wgmma descriptor reads); one tensor copy an operand of a 3-d box of 16-byte lines (the unswizzled bricks themselves). The last three stages landed are checked byte for byte in every CTA.

PREDICTION (written 2026-11-28, before the first run). A stage is the CTA's own `x` tile (16 KB of distinct lines) and the weights every CTA reads (32 KB), through L2 at [`stream_rate`](stream_rate.md)'s rates past L1 (run 20261128T013030Z-737802-5ab0223a: 30.95 and 42.38 bytes a cycle an SM), taken to add: 16,384 / 30.95 + 32,768 / 42.38 = 1,302 cycles a stage. `cp.async` and the 128-byte box both 1,107-1,497 cycles a stage (1,302 within 15%); the 16-byte-line box 1.5x-4x the 128-byte box's cycles (its 3,072 lines a stage, each a request of the copy engine's).
