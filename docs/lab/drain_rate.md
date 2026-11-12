# `drain_rate`

Mirrors `measure/lab/experiments/drain_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`bulk_drain`](bulk_drain.md) read a token's fp32 partial at about 20 cycles into global memory, with 4 warps a CTA
issuing the adds, and [`carry_floor`](../internals/tools/carry_floor.md) prices every partial at that. Is that the
SM's CAPACITY for such adds, or only what 4 warps can keep in flight? And what do the adds cost when the CTAs' rows
are spread over a window, as a carry's are, and not all the same?

THE FORM. A CTA an SM, of 4, 8 or 16 warps. A round is a partial of 64 tokens x 64 channels from registers, one
`red.global.add.f32` a value, a warp's 32 lanes one 128-byte line; the same 128 line adds a round at every warp
count.
- `own`: each CTA adds into its own panel of 64 rows.
- `one`: every CTA adds into the same 64 rows.
- `spread`: a token's row is drawn, a (CTA, round, token), from a panel of 1,024 rows every CTA shares.
Every sum is checked exactly.

UNMEASURED: no card has run it.
