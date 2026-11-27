# `drain_rate`

Mirrors `measure/lab/experiments/drain_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`bulk_drain`](bulk_drain.md) read a token's fp32 partial at about 20 cycles into global memory, by one bulk
reduction a panel and by 4 warps' own adds alike. Is that the SM's CAPACITY for such adds, or only what one issuer or
4 warps can keep in flight? And what do the adds cost when the CTAs' rows are spread over a window, as a carry's are,
and not all the same? [`carry_floor`](../internals/tools/carry_floor.md) prices a partial's drain from this record.

THE FORM. A CTA an SM, of 4, 8 or 16 warps. A round is a partial of 64 tokens x 64 channels from registers, one
`red.global.add.f32` a value, a warp's 32 lanes one 128-byte line; the same 128 line adds a round at every warp
count.
- `own`: each CTA adds into its own panel of 64 rows.
- `one`: every CTA adds into the same 64 rows.
- `spread`: a token's row is drawn, a (CTA, round, token), from a panel of 1,024 rows every CTA shares.
Every sum is checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60292535, driver 615.71.09; 2026-11-12, the experiment at
41730f9b; 7 repetitions, every sum exact; SM-cycles a token a CTA, the median, the spread under 0.9%):

| the rows | 4 warps | 8 warps | 16 warps |
|---|---|---|---|
| each CTA its own rows | 19.2 | 19.1 | 19.2 |
| every CTA the same rows | 46.7 | 46.7 | 46.9 |
| rows spread over 1,024 | 24.8 | 24.9 | 24.8 |

WHAT IT SHOWS.
- THE RATE IS THE PATH'S, NOT THE WARPS': 4, 8 and 16 warps issue the same partial in the same 19.2 cycles. More
  warps in flight push nothing more through; an SM takes a 128-byte line of adds about every 9.6 cycles.
- ROWS SPREAD OVER A WINDOW COST 1.30 TIMES ROWS OF A CTA'S OWN (24.8), where every CTA on the same 64 rows costs
  2.44 times (46.7). A carry's drains sit near the first.
- Every SM drains at once here: whether the rate is an SM's own or the card's shared between them is NOT separated.

Record: `data:records/fsla/measure/lab/drain_rate.run/20261112T015827Z-bc35ef7c/output.json`.
