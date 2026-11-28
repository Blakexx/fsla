# `tma_box`

Mirrors `measure/lab/experiments/tma_box.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's loader stages a K-step's `x` [128 x 64] and weights [256 x 64] bf16, 48 KB a stage. What does a stage cost a CTA by the tensor copy (TMA) against `cp.async` by one warp?

THE FORM. A CTA an SM, a ring of three 48 KB stages, a loader and a warpgroup that waits on each stage, reads a word of it a thread and releases it; the operands L2-resident. `cp.async` 16 bytes a lane by one warp into wgmma's unswizzled K-major bricks; one tensor copy an operand of a 2-d box of 128-byte lines, 128-byte swizzled (what a swizzled wgmma descriptor reads); one tensor copy an operand of a 3-d box of 16-byte lines (the unswizzled bricks themselves). The last three stages landed are checked byte for byte in every CTA.

PREDICTION, run 20261128T034956Z-802475-98afad64 (written 2026-11-28, before it ran; its basis: a stage is the CTA's own x tile (16 KB of distinct lines, stream_rate's 30.8 bytes a cycle an SM: 532 cycles) and the 128 KB of weights every CTA reads (55.2 at that footprint: 594); the two paths adding give 1,126 cycles a stage, apart 594 -- the row decides which; the 16-byte-line box's 3,072 lines a stage each a request of the copy engine's):

- cp.async, a warp, 16 bytes a lane: a 48 KB stage: 560.00-1,200.00 cycles a stage.
- a tensor copy, 128-byte lines swizzled: a 48 KB stage: 560.00-1,200.00 cycles a stage.
- the 16-byte-line box over the 128-byte box: 1.50-4.00 times.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00fyq0107h186k284, driver 580.196.07; 2026-11-28, the experiment at
b7215a57, run 20261128T034956Z-802475-98afad64; 7 repetitions, the 3 checked variants exact; the median, the widest
spread of a row 1.7%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| cp.async, a warp, 16 bytes a lane: a 48 KB stage | 5,663.72 cycles a stage | 560.00-1,200.00 cycles a stage | MISSED |
| a tensor copy, 128-byte lines swizzled: a 48 KB stage | 966.36 cycles a stage | 560.00-1,200.00 cycles a stage | held |
| the 16-byte-line box over the 128-byte box | 5.55 times | 1.50-4.00 times | MISSED |

The record's rows:
- cp.async, a warp, 16 bytes a lane: a 48 KB stage: 5,663.72 cycles a stage a CTA.
- a tensor copy, 128-byte lines swizzled: a 48 KB stage: 966.36 cycles a stage a CTA.
- a tensor copy, 16-byte lines (the unswizzled bricks): a 48 KB stage: 5,359.88 cycles a stage a CTA.

Record: `data:records/fsla/measure/lab/tma_box.run/20261128T034956Z-802475-98afad64/output.json`.
