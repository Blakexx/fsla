# `projection_tile`

Mirrors `measure/lab/experiments/projection_tile.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's projection tile alone: a job is 128 tokens x 256 columns over the hidden axis, two math warpgroups of 64 tokens each issuing `wgmma` m64n256k16 from shared memory, one loader warp staging a K-step (64 hidden) of `x` [128 x 64] and the job's weights [256 x 64] by `cp.async` into a ring of three stages. What does a job cost at hidden 256, 512 and 2,048, against its tensor floor (16 x hidden cycles: two warpgroups' steps on one pipe, 128 cycles a step)?

THE FORM. A CTA an SM, each its own 128-token tile of `x`, every CTA the same four jobs' weights (a write head's, from L2); a job's K-loop then an epilogue folding the accumulators; the next job re-stages the tile (a write head's jobs, each re-reading its tile). The operands are small integers in bf16, so every sum is exact: the last job's accumulators of two CTAs are checked against the host's products.

PREDICTION (written 2026-11-27, before the first run). A K-step (64 hidden) a CTA: the tensor floor 1,024 cycles (8 steps x 128, two warpgroups on one pipe); 48 KB through L2 to the SM at the 43.0 bytes a cycle an SM [`stream_rate`](stream_rate.md) measured for a shared table, 1,143 cycles; at hidden 2,048 `x`'s 16 KB a K-step also comes from HBM (`x`, 69 MB, re-read each job, overruns the 50 MB L2) at 13.0 bytes a cycle an SM, 1,260 cycles. A job is its K-steps at the binding one, plus its share of the ring's first fill (~1,850 cycles over 4 jobs, ~460) and its end (the wait for all, the fold and the clear, ~400): hidden 256, 4 x 1,143 + 860 = 5,430 cycles, 1.33x its floor (band 1.20-1.46); hidden 512, 8 x 1,143 + 860 = 10,000, 1.22x (1.10-1.34); hidden 2,048, 32 x 1,260 + 860 = 41,180, 1.26x (1.13-1.39).
