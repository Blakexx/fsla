# `newton_row`

Mirrors `measure/lab/experiments/newton_row.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's entmax threshold is Newton from the left on the projection's accumulators (Blake's ruling of 2026-11-27: Newton, gated by this row). On a `wgmma` m64n256 accumulator's rows: how many steps does a row take, and a warp (its 16 rows' vote); what does a 64-token block cost; is the support EXACT against the fp64 sorted closed form?

THE FORM. A warpgroup holds a 64-token block of 256 logits a token in the m64n256 accumulator's layout ([`wgmma_regs`](wgmma_regs.md)), loaded before the clock; one or two warpgroups a CTA, a CTA an SM. The solve is the `ampere` branch's `entmax_rows` (`produce_kernel.md#threshold`) on that layout: each row shifted by its max, then from -2 (alpha 1.5) or -1 (alpha 2) each step the count, sum and sum of squares past the bound (one compare and three predicated adds a value), the closed form on that support, the settled test's count (one compare, one predicated add), a Newton step; the warp's vote over its 16 rows ends the loop, cap 16. The inputs are `tests/integration/test_production_levels.py`'s stress kinds at 256 columns (random, near the boundary, a collapsed support, five times the scale); the oracle the entmax package's sorted closed form (`entmax15`, `sparsemax`) in fp64. Every row's support, threshold and steps are written and checked.

PREDICTION (written 2026-11-27, before the first run). Steps (a desk float32 emulation of the same algorithm, 4,096 rows a kind): a row at most 7; a warp's mean, alpha 1.5: random 5.6, near the boundary 5.5, collapsed 1.0, five times 2.8; alpha 2: 4.3, 4.0, 1.0, 2.1 (each within 0.6). Support EXACT on every row, none capped; the threshold within 2e-6 of fp64. Cycles a 64-token block = 260 (the max and shift) + S c, S the warp's steps: one warpgroup a CTA c in [700, 1,000] (a step's pass is 128 values x 6 instructions = 768 a warp, plus the shuffles', divide's and vote's latency), two warpgroups c in [650, 900]. The design's 'under 1K cycles a 64-token block' cannot hold: one step's pass alone is 768 instructions a warp.

NOT YET MEASURED: it runs on the producer's held H100 session.
