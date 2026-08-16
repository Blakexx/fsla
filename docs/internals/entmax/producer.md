# `entmax/producer.cu` — the fused producer kernel

The producer plane used to be five kernels and three DRAM round trips: a cuBLAS fp32
router GEMM, a materialized fp32 logit plane, one solve launch per `(operator, operand
set)`, a fold reduction, and — before P74's python plane landed — a separate gain
projection and a pack. This kernel is the endgame the P74 card named: **x in, the
finished packed-bf16 bundle out**, in one launch, with the fp32 logits never reaching
DRAM at all.

It is assembled from two components rather than written as one body:

* `tile_gemm.cuh` — the bf16-operand / fp32-accumulate tile GEMM;
* `solve_row.cuh` — the exact per-row solves (entmax `alpha in {1.5, 2}`, softmax, the
  union split).

<a id="components"></a>
## Why components and not one kernel body

Because the gate has to run the shipped binary. The component-sharing probe pattern (the
growth-probe / `walk_probe` precedent) says: build the shipping kernel out of parts,
build a second kernel out of the *same* parts that emits an intermediate, ship and ratify
both, and then "the gated intermediate is the shipped intermediate" is a call-graph fact
instead of a claim about two implementations that happen to agree.

<a id="probe"></a>
## The probe

`producer_probe` is the tile-GEMM component with the fp32 logit tile **emitted** to a
gate buffer instead of solved in place. It is shipped, ratified and SASS-gated like any
other kernel, and reached from tests only — it has no place on the shipped path, and a
production caller that wanted logits would be asking for the round trip this kernel
exists to delete.

It carries the projection half of the entmax seam split (P74's oracle-precision
doctrine): the projection is gated **numerically** against the fp64 projection with a
derived envelope, and the solve is gated **exactly** on production logits.

### The projection floor model, which is ours for the first time

Because the tile order is ours, the floor is stateable rather than opaque: operands are
rounded to bf16 (unit roundoff `u_b = 2^-8`), every accumulation is fp32
(`u = 2^-24`), and the K axis is accumulated as `hidden/16` sequential `mma.m16n8k16`
steps. Per element,

    floor = (2*u_b + (hidden/16)*u) * sum_k |x[t,k]| * |W[h,k,c]|

and the gate reports **ratio-to-floor**. Measured across five cells (ragged tokens,
ragged columns, `hidden` 64–512, 1–8 heads): 0.072–0.163. A kernel at ratio <= 1 is doing
exactly the arithmetic it declares; a kernel at ratio ~0.1 is doing it with the operand
term dominating and no accumulation pathology.

This is the upgrade P74's fused-kernel ruling anticipated: ratio-to-floor now extends
from the consumer to the whole producer.

<a id="smem-alias"></a>
## Shared memory

One dynamic allocation serves the K-block staging buffers and then the fp32 logit tile;
they alias, because the staging is dead once the last `mma` retires. See
`tile_gemm.md#smem-alias` for the barrier that separates them and the size arithmetic.
