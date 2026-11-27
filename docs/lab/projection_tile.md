# `projection_tile`

Mirrors `measure/lab/experiments/projection_tile.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's projection tile alone, as its design states it: a job is 128 tokens x 256 columns over the hidden axis, two math warpgroups of 64 tokens each issuing `wgmma` m64n256k16 from shared memory, one loader warp staging a K-step (64 hidden) of `x` [128 x 64] and the job's weights [256 x 64] by `cp.async` into a ring of three stages. What does a job cost at hidden 256, 512 and 2,048, against its tensor floor (16 x hidden cycles: two warpgroups' steps on one pipe, 128 cycles a step)?

THE FORM. A CTA an SM, each its own 128-token tile of `x`, every CTA the same four jobs' weights (a write head's, from L2); a job's K-loop then an epilogue folding the accumulators; the next job re-stages the tile (the design's form). The operands are small integers in bf16, so every sum is exact: the last job's accumulators of two CTAs are checked against the host's products.

PREDICTION (written 2026-11-27, before the first run). A K-step a CTA: tensor 1,024 cycles (8 steps x 128); 48 KB through L2 to the SM; at hidden 2,048 `x`'s 16 KB a K-step comes from HBM (`x` is 69 MB, re-read each job, over the 50 MB L2). Cycles a K-step = max(1,024, 49,152 / R_L2 [, hidden 2,048: 16,384 / R_HBM an SM]), a job that times the K-steps plus 300-800 for its end, with R from [`stream_rate`](stream_rate.md) plugged in before this row runs. The design's 1.1-1.3x holds only if R_L2 is at least ~45 bytes a cycle an SM.

NOT YET MEASURED: it runs on the producer's held H100 session.
