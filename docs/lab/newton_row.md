# `newton_row`

Mirrors `measure/lab/experiments/newton_row.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's entmax threshold is solved on the projection's accumulators by Newton from the left. On a `wgmma` m64n256 accumulator's rows: how many steps does a row take, and a warp (its 16 rows' vote); what does a 64-token block cost; is the support EXACT against the fp64 sorted closed form?

THE FORM. A warpgroup holds a 64-token block of 256 logits a token in the m64n256 accumulator's layout ([`wgmma_regs`](wgmma_regs.md)), loaded before the clock; one or two warpgroups a CTA, a CTA an SM. THE MATHEMATICS. With a row shifted by its max (`x = z - max`), alpha 1.5's level is `((x - tau) / 2)_+^2` and alpha 2's `(x - tau)_+`; `f(tau) = sum level - 1` is convex and decreasing, and `f(-2) >= 0` at alpha 1.5 (`f(-1) >= 0` at 2: the max alone), so Newton from there never passes the root. At a bound `lo`, with `n`, `P` and `Q` the count, sum and sum of squares of the support `{x > lo}`: alpha 1.5's `f(lo) = (Q - 2 P lo + n lo^2) / 4 - 1` and `f'(lo) = -(P - n lo) / 2`, so its step is `lo + (Q - 2 P lo + n lo^2 - 4) / (2 (P - n lo))`; the closed form on that support is alpha 1.5's smaller root of `n tau^2 - 2 P tau + Q - 4`, alpha 2's `(P - 1) / n` (the step itself); it is the root exactly when no member of the support lies below it, the count of `x >= tau` equal to `n`. Each step is a pass over a thread's 128 values -- one compare and three predicated adds a value (no select), then the settled count's compare and predicated add -- and two shuffles a sum; the warp's vote over its 16 rows ends the loop, cap 16. Two forms of the step: SERIAL, each row's pass, sums, closed form and count in turn; INTERLEAVED, both rows' passes in one loop with every sum split into two chains (twelve chains of 32), the six sums' shuffles issued together, both counts in one loop. A row's steps are its 4 lanes' most (it settles when every lane holding it does). The check is the support against fp64, a row that differs passing only where every disagreeing logit lies within 2 units of fp32's last place at the threshold (the sorted closed form in fp32 differs from fp64 on such rows too). The inputs are `tests/integration/test_production_levels.py`'s stress kinds at 256 columns (random, near the boundary, a collapsed support, five times the scale); the oracle the entmax package's sorted closed form (`entmax15`, `sparsemax`) in fp64. Every row's support, threshold and steps are written and checked.

PREDICTION, the run to come (written 2026-11-28, before it ran; its basis: the first run's blocks fit cycles = 420 + S c within 4% at one warpgroup (c 1,157 alpha 1.5, 1,100 alpha 2, the steps S a block's from the host's float32 iteration: 6.0 and 4.8 random); a step issues 861 (alpha 1.5) or 685 (alpha 2) instructions, so ~300-400 cycles of latency are exposed; the interleaved form issues 872 and 690 with twelve chains of 32 (the chains' latency under the issue) and the phases' own latency (the shuffles, the root and divide, the vote: ~150-180) left, so c ~1,050 and ~840 at one warpgroup and issue-bound at two (872, 690 a block); the steps a row are now the row's (its 4 lanes' most), as the host's iteration counts them):

- alpha 1.5, random, 1 WG, serial: a block: 6,990.00-7,720.00 cycles.
- alpha 1.5, random, 1 WG: the interleaved block over the serial: 0.82-0.98 times.
- alpha 1.5, near the boundary, 1 WG: the interleaved block over the serial: 0.82-0.98 times.
- alpha 2, random, 1 WG: the interleaved block over the serial: 0.68-0.85 times.
- alpha 2, near the boundary, 1 WG: the interleaved block over the serial: 0.68-0.85 times.
- alpha 1.5, random, 2 WG: the interleaved block over the serial: 0.80-0.98 times.
- alpha 2, random, 2 WG: the interleaved block over the serial: 0.80-0.98 times.
- alpha 1.5, random: steps a row, the mean: 3.89-4.75 steps.
- alpha 1.5, random: steps a block, the mean: 5.50-6.50 steps.
- alpha 2, random: steps a row, the mean: 2.53-3.09 steps.
- alpha 1.5, near the boundary, 1 WG: supports differing within fp32's resolution: 0.00-12.00 rows.
