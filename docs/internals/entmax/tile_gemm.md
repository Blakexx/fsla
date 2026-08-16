# `entmax/tile_gemm.cuh` — the producer's tile-GEMM component

One CTA computes the fp32 logit tile `[WORD_TOKENS, C]` for one `(batch, head)` stream:

    logits[t, c] = sum_k x[t, k] * W[head, k, c]  (+ bias[c])

with **bf16 operands and fp32 accumulators** on the tensor cores
(`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`). It is a `__device__` function
rather than a kernel because two kernels compose it — the shipping fused producer and
the probe — and the whole point of the probe is that the agreement it certifies is a
call-graph fact.

## Why bf16 operands, and what it is worth

The declared arithmetic is the industry-standard class (P74's precision ruling, which
reversed `effd23c` for the projection while leaving the solve's support-exactness bar
untouched). On GA102 the class is worth measuring rather than assuming, because the same
board's TF32 tensor rate equals its FP32 rate and that is what killed the tf32 lever.
It does not repeat here: measured 69.9 TFLOP/s bf16-with-fp32-accumulate against 23.9
TFLOP/s fp32 SIMT, a 2.9x class ratio, and 2.49–2.76x at the producer's own GEMM shape.

## The tile shape, and why the token axis is 32

`WORD_TOKENS = 32` — the support word's token quantum, frozen for nine downstream
consumers. A CTA that owns exactly the tokens of one support word can emit that word
with no atomic and no second pass, which is what makes the solve fusible at all. The
cost is a short M axis, so the GEMM's arithmetic intensity is low and the phase is
L2-bound on `W` rather than tensor-core bound; `W` is small enough per head to stay
L2-resident, which is what makes that affordable.

The N axis is the CTA's whole packed router width: `NT` n-tiles of `MMA_N = 8` per warp
over `GEMM_WARPS = 8` warps, so `NT = ceil(C / 64)` and `MAX_PACKED_COLUMNS = 512`.

<a id="accumulator"></a>
## The accumulator is a register array, so `NT` is a template parameter

`acc[NT][M_TILES][4]` — the whole tile lives in registers for the length of the K loop
and never reaches DRAM. That is the fusion. A runtime n-tile count would force a
worst-case array, so the count is a template parameter selected by `int_switch` from a
tile count (structural), not from a width threshold.

<a id="smem-alias"></a>
## The shared-memory alias

Two regions want the same shared memory and never want it at the same time: the K-block
staging buffers (`xs` = `[32][TILE_K+8]` bf16, `ws` = `[TILE_K][ws_stride]` bf16) and the
fp32 logit tile the solve then reads. The staging is dead once the last `mma` retires, so
the caller passes overlapping pointers and the K loop's trailing `__syncthreads` is the
barrier that separates them. The allocation is therefore `max(staging, tile)` and not
their sum — at the pinned cell 33.8 KB rather than 52.9 KB, which is what keeps the
block inside the default 48 KB and off the dynamic-shared-memory opt-in.

`XS_STRIDE = TILE_K + 8` pads the `A`-fragment reads so the eight rows one lane group
touches land in eight distinct banks.

<a id="ragged-tile"></a>
## The ragged tile carries zeros, it does not branch

`mma` is a warp-wide collective, so a per-lane column bound cannot gate it: the n-tile
guard is `n0 >= columns`, which is warp-uniform, and the columns between `columns` and
the tile boundary are staged as zeros. The store back to the tile is guarded per element,
where divergence is free.

This is why the `N = 257` ragged-tile penalty that the fp32 `ampere_sgemm_128x64_nn`
charged for the fused gain column (P74's lever 2, +0.0306 ms at
`packed_router_width = 256`) does not exist here: the tile order is ours, and a partial
n-tile costs one tile's arithmetic whether it is full or not — the same quantum the
assembler charged, but now on a 64-column quantum inside a kernel that was going to run
anyway.
