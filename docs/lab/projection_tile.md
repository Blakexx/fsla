# `projection_tile`

Mirrors `measure/lab/experiments/projection_tile.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's projection tile alone: a job is 128 tokens x 256 columns over the hidden axis, two math warpgroups of 64 tokens each issuing `wgmma` m64n256k16 from shared memory, one loader warp staging a K-step (64 hidden) of `x` [128 x 64] and the job's weights [256 x 64] by `cp.async` into a ring of three stages. What does a job cost at hidden 256, 512 and 2,048, against its tensor floor (16 x hidden cycles: two warpgroups' steps on one pipe, 128 cycles a step)?

THE FORM. A CTA an SM, each its own 128-token tile of `x`, every CTA the same four jobs' weights (a write head's, from L2); a job's K-loop then an epilogue folding the accumulators; the next job re-stages the tile (a write head's jobs, each re-reading its tile). The operands are small integers in bf16, so every sum is exact: the last job's accumulators of two CTAs are checked against the host's products.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: a K-step's tensor floor 1,024 cycles (8 steps x 128, two warpgroups on one pipe); its x, 16 KB of the CTA's own lines, through L2 at stream_rate's distinct-line rate (30.8 bytes a cycle an SM, an aggregate limit at 132 SMs), its weights, the job's 128 KB (hidden 256), 256 KB (512) or 1 MB (2,048) every CTA reads, at the shared rate of that footprint (55.2 at 128 KB, 42.6 at 4 MB: ~53 and ~48 between); whether the two paths add on the SM or run apart is unmeasured, so a K-step is 1,024-1,126 cycles at hidden 256 (532 + 594 adding, or the tensor floor) and 1,024-1,154 at 512; at 2,048 x comes from HBM (69 MB, re-read each job, overruns L2) at 12.95, 1,265 cycles; a job its K-steps plus ~900 (the ring's first fill's share, its end's wait, fold and clear)):

- hidden 256: a job over its tensor floor: 1.18-1.36 times.
- hidden 512: a job over its tensor floor: 1.08-1.28 times.
- hidden 2,048: a job over its tensor floor: 1.18-1.36 times.
