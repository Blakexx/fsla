# `projection_tile`

Mirrors `measure/lab/experiments/projection_tile.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's projection tile alone: a job is 128 tokens x 256 columns over the hidden axis, two math warpgroups of 64 tokens each issuing `wgmma` m64n256k16 from shared memory, one loader warp staging a K-step (64 hidden) of `x` [128 x 64] and the job's weights [256 x 64] by `cp.async` into a ring of three stages. What does a job cost at hidden 256, 512 and 2,048, against its tensor floor (16 x hidden cycles: two warpgroups' steps on one pipe, 128 cycles a step)?

THE FORM. A CTA an SM, each its own 128-token tile of `x`, every CTA the same four jobs' weights (a write head's, from L2); a job's K-loop then an epilogue folding the accumulators; the next job re-stages the tile (a write head's jobs, each re-reading its tile). The operands are small integers in bf16, so every sum is exact: the last job's accumulators of two CTAs are checked against the host's products.

PREDICTION, run 20261128T034916Z-771677-038e8f02 (written 2026-11-28, before it ran; its basis: a K-step's tensor floor 1,024 cycles (8 steps x 128, two warpgroups on one pipe); its x, 16 KB of the CTA's own lines, through L2 at stream_rate's distinct-line rate (30.8 bytes a cycle an SM, an aggregate limit at 132 SMs), its weights, the job's 128 KB (hidden 256), 256 KB (512) or 1 MB (2,048) every CTA reads, at the shared rate of that footprint (55.2 at 128 KB, 42.6 at 4 MB: ~53 and ~48 between); whether the two paths add on the SM or run apart is unmeasured, so a K-step is 1,024-1,126 cycles at hidden 256 (532 + 594 adding, or the tensor floor) and 1,024-1,154 at 512; at 2,048 x comes from HBM (69 MB, re-read each job, overruns L2) at 12.95, 1,265 cycles; a job its K-steps plus ~900 (the ring's first fill's share, its end's wait, fold and clear)):

- hidden 256: a job over its tensor floor: 1.18-1.36 times.
- hidden 512: a job over its tensor floor: 1.08-1.28 times.
- hidden 2,048: a job over its tensor floor: 1.18-1.36 times.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: the first run's 5.3x at every hidden was the one loader warp's cp.async (~9 bytes a cycle an SM, lg_throttle); tma_box's 128-byte swizzled tensor copy staged the same 48 KB in 966 cycles (50.9 bytes a cycle an SM), under the tensor floor's 1,024; so a K-step is the floor's 1,024 plus what the ring fails to hide (~0-100) at hidden 256 and 512, and at 2,048 x's 16 KB from HBM (12.95 bytes a cycle an SM: 1,265) binds; a job its K-steps plus ~900 (the ring's first fill's share, its end's wait, fold and clear); the cp.async form repeats within 5%):

- hidden 256: the cp.async job over its floor: 5.00-5.60 times.
- hidden 256: the tensor-copy job over its floor: 1.12-1.40 times.
- hidden 512: the tensor-copy job over its floor: 1.05-1.30 times.
- hidden 2,048: the tensor-copy job over its floor: 1.15-1.40 times.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00fyq0107h186k284, driver 580.196.07; 2026-11-28, the experiment at
b7215a57, run 20261128T034916Z-771677-038e8f02; 7 repetitions, the 3 checked variants exact; the median, the widest
spread of a row 1.4%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| hidden 256: a job over its tensor floor | 5.34 times | 1.18-1.36 times | MISSED |
| hidden 512: a job over its tensor floor | 5.27 times | 1.08-1.28 times | MISSED |
| hidden 2,048: a job over its tensor floor | 5.33 times | 1.18-1.36 times | MISSED |

The record's rows:
- hidden 256, a job 128 tokens x 256 columns, three stages: a job: 21,859.78 cycles a job a CTA.
- hidden 256, a job 128 tokens x 256 columns, three stages: against the tensor floor: 5.34 the job's cycles over 16 x hidden.
- hidden 512, a job 128 tokens x 256 columns, three stages: a job: 43,210.79 cycles a job a CTA.
- hidden 512, a job 128 tokens x 256 columns, three stages: against the tensor floor: 5.27 the job's cycles over 16 x hidden.
- hidden 2048, a job 128 tokens x 256 columns, three stages: a job: 174,622.74 cycles a job a CTA.
- hidden 2048, a job 128 tokens x 256 columns, three stages: against the tensor floor: 5.33 the job's cycles over 16 x hidden.

Record: `data:records/fsla/measure/lab/projection_tile.run/20261128T034916Z-771677-038e8f02/output.json`.
