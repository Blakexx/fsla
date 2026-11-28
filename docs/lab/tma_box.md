# `tma_box`

Mirrors `measure/lab/experiments/tma_box.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's loader stages a K-step's `x` [128 x 64] and weights [256 x 64] bf16, 48 KB a stage. What does a stage cost a CTA by the tensor copy (TMA) against `cp.async` by one warp?

THE FORM. A CTA an SM, a ring of three 48 KB stages, a loader and a warpgroup that waits on each stage, reads a word of it a thread and releases it; the operands L2-resident. `cp.async` 16 bytes a lane by one warp into wgmma's unswizzled K-major bricks; one tensor copy an operand of a 2-d box of 128-byte lines, 128-byte swizzled (what a swizzled wgmma descriptor reads); one tensor copy an operand of a 3-d box of 16-byte lines (the unswizzled bricks themselves). The last three stages landed are checked byte for byte in every CTA.

PREDICTION (written 2026-11-27, before the first run). A stage is 49,152 bytes through L2 to the SM; at the 43.0 bytes a cycle an SM that every SM reading one shared table took ([`stream_rate`](stream_rate.md)'s L2 row, a rate past L1), 1,143 cycles. `cp.async` and the 128-byte box both 970-1,320 cycles a stage (1,143 within 15%); the 16-byte-line box 1.5x-4x the 128-byte box's cycles (its 3,072 lines a stage, each a request of the copy engine's).
