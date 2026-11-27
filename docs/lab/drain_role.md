# `drain_role`

Mirrors `measure/lab/experiments/drain_role.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`drain_rate`](drain_rate.md) put an SM's rate for fp32 adds into global memory at one 128-byte line every 9.6
cycles, whatever issues them. A carry's CTA holds a token's partial twice, once a warpgroup, where its two boxes
share their read-sparse digits. What does a token cost when a DRAIN ROLE of warps pulls both partials from shared
memory, adds them, and issues ONE line of f16 pairs -- and what does an f16 line cost by itself? Do value loads on
the same path add to it?

THE PLAN STATES THE SUM'S WIDTH ACROSS CTAS, f16 or fp32
([precision.md](../architecture/precision.md)); what the f16 sum gives up in accuracy is
[`arith_model`](../internals/tools/arith_model.md#f16)'s, not this experiment's.

THE FORM. A CTA an SM.
- `direct`: 4 warps add a token's 64 channels as 32 f16 pairs, one line, from registers.
- `role`: two source warpgroups each store a chunk of 64 tokens x 64 channels fp32 into their panels. 1, 2 or 4
  drain warps, a chunk behind, take the tokens of their residue: a token's two rows loaded (two batches of 8 at
  fixed registers, a batch's rows loaded two batches before its adds), added in fp32, rounded once to f16 pairs,
  and added as one line into the token's row -- the CTA's own rows, or rows drawn from 1,024 that every CTA shares.
  The sources hand a chunk over by an arrival a warp at a barrier; the drain warps free its panels by another.
- `loads`: one thread bulk-loads 32 KB a chunk from global memory, two in flight, nothing else running.
- `role and loads`: both in one CTA.
Every drained sum is checked exactly. A role's `__syncwarp` before its one-lane arrival compiles to a convergence
call, once a chunk; the experiment declares those.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60297524, driver 615.71.09; 2026-11-12, the experiment at
8c03d7d9; 7 repetitions, every drained sum exact; SM-cycles a token a CTA, the median, the spread under
1.4%):

| the form | cycles a token |
|---|---|
| an f16 line from registers, 4 warps | 9.8 |
| the drain role, 1 drain warp, the CTA's own rows | 36.2 |
| the drain role, 2 drain warps, the CTA's own rows | 21.1 |
| the drain role, 4 drain warps, the CTA's own rows | 11.7 |
| the drain role, 2 drain warps, rows spread over 1,024 | 21.1 |
| the loads alone, 32 KB a chunk | 10.5 |
| the drain role, 2 drain warps, the CTA's own rows, beside the loads | 24.0 |
| the drain role, 4 drain warps, the CTA's own rows, beside the loads | 16.1 |

WHAT IT SHOWS.
- AN F16 LINE COSTS WHAT AN FP32 LINE DOES: 9.8 cycles, and a token's 64 channels as f16 pairs are one line where
  fp32 is two ([`drain_rate`](drain_rate.md): 19.2 a token).
- THE DRAIN ROLE IS ITS WARPS' UNTIL THERE ARE FOUR: 36.2 cycles a token with one drain warp, 21.1 with two, 11.7 with
  four -- within 19% of the line's own rate, the two sources' stores and the hand-over included, and with four no
  load running ahead (the header).
- ROWS SPREAD OVER A WINDOW: 21.1 against 21.1 with two drain warps.
- LOADS AND DRAINS SHARE THE PATH IN PART. 32 KB a chunk of loads costs 10.5 cycles a token alone (21 a kilobyte, two
  in flight). Beside the two-warp role they add 2.9; beside the four-warp role, where the path is near full, 4.4
  (16.1 against 11.7) -- 42% of their own time, neither hidden nor added whole.

Record: `data:lab/drain_role/sm_90a.json`.
