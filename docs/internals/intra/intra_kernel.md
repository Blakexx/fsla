# `csrc/rola/src/intra/` — the K31 window-intra kernel

The within-window causal term, as its own deterministic side kernel. It computes
nothing the carry kernel computes: the carry kernel's readout sees the state as
of a window's start, and every pair of tokens INSIDE one window belongs here.
The design it implements is `development/plans/K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md`
§2.1 (the global window grid) and §2.5 (the factorized, windowed, tiled intra).

## What it computes

For a window `w` of `W` tokens (`W` is an instantiation axis; the built arms are
`W = 384` and `W = 512` — the two ends of the architecture doc's band — and `W = 64`,
the conformance cell. They are the same three the carry family builds, because the two
kernels share ONE global window grid),

    A[t, t'] = gain(t') * prod_l G_l[t, t'],   G_l = R_l · W_l^T over level l's B = 256 digits
    o[t]   += sum_{t' <= t in w} A[t, t'] · v[t']
    den[t] += sum_{t' <= t in w} A[t, t']

The gain is a per-COLUMN scalar applied AFTER the level product, which is the
shipped kernel's gain law: support bits are ungained. `den` is the same `A`
summed — the mass the ones-column of the oracle's recurrence produces — so the
kernel carries no second matrix for it.

`[W x W]` is never materialized. The window's `(W/64) x (W/64)` tile grid of `[64 x 64]`
output blocks is walked lower-triangular (21 tile-pairs at `W = 384`, 36 at `W = 512`);
the diagonal blocks carry the within-tile staircase. Per tile-pair the per-level Gram
accumulates in registers over the full 256-digit axis, the levels are
Hadamard-multiplied there, and the result becomes the A operand of the value MMA
**in place** — the accumulator-to-A fragment identity, no shared-memory round trip
on sm_80.

## The output block is C = 128 reader tokens

The unit of the walk is not the tile pair but the BLOCK: two row tiles, 128 reader
tokens, sweeping the column tiles together. One staged column tile — its `W_l`
operand slabs, its value rows, its gains — is consumed by both row tiles, so a
window of 8 tiles stages 20 column tiles instead of 36. What the block does NOT
change is the row side: each row tile's operands are still re-staged per column
tile. Cutting that needs two column tiles per block as well, which multiplies the
Gram tiles a warp must hold; the census already refuses two of them (below).

A window whose grid is a single tile has no pair to share with, so the block
collapses onto it and the CTA halves with it — `row_tiles_of(W)`, `rows_of(W)` and
`threads_of(W)` are one derivation, and the arm's tile count is what picks it.

## The carve, and why it has no reductions

One CTA per `(bh, window)`, one warp per `kWarpRows = 16` rows of the block —
eight warps at the paired arms, four at a single-tile one. A warp holds the Gram
of ONE tile pair and owns 16 output rows and every column; the row tiles are
shared ACROSS warps rather than stacked in one warp's registers, which is what
keeps the accumulator budget independent of `C`. Output rows are disjoint per
warp and the causal triangle's imbalance lands nowhere — neither between warps
nor between CTAs; the block's last column tile leaves the low half of the warps
idle for that one sweep, which is the strictly-upper pair not being computed.

The block's whole reader side is resident, so the footprint clears the 48 KB a
kernel gets unasked (65 664 B at `W = 512`) and the arms launch on opted-in
dynamic shared memory, re-declared per launch because the opt-in is per device.

Write-back is `fan_in_add` (a non-returning RED) into caller-owned `o`/`den`
planes, so the kernel is composable with a concurrent carry kernel and with
itself; the caller owns zeroing.

## The pipeline stage is wider than the contraction step

`kStageDigits = 64` digits are staged per `cp.async` fill, and four `kSlabDigits
= 16` MMA steps ride each fill. Barriers are per stage, not per MMA step: 16 per
tile-pair rather than 64. Measured on sm_86, narrowing the stage to the MMA step
cost 1.66x on the dense cell.

Operand rows are padded (`kOperandPitch`, `kValuePitch`) so that the eight row
addresses `ldmatrix` reads per 8x8 tile land in eight distinct banks, while every
row base stays 16-byte aligned for `cp.async`.

## The slab gate

The one sparsity tier the study kept. Before the tile walk, the CTA summarizes
each `(tile, level, side)` as a 256-bit digit union over that tile's 64 tokens,
built from the operand values themselves. A slab's MMA group issues only when the
row side's 16-bit window AND the column side's is nonempty; the staging for a
whole stage is skipped when all four of its slabs are dead.

A side the launch's mode word does not declare sparse is never summarized and
answers all-ones, so the gate self-deactivates at dense routing rather than
becoming a second code path. The modes are a LAUNCH ARGUMENT, not a compile-time
pattern axis: `DENSE_BOTH`, `READ_SPARSE`, `WRITE_SPARSE`, `SHARED_SPARSE` (union
routing, which runs on exactly this machinery).

The gate is an exact zero certificate, so it cannot move a number — the battery
asserts bitwise equality between a gated and an ungated run of the same operands.

## Where the time goes (sm_86, measured)

The Π→Σ collapse that makes this term cheap in FLOPs also collapses its
arithmetic intensity. Per window the Gram issues `W^2 · sum_l B_l` flops against
`4 · W · sum_l B_l` operand bytes, so the ideal intensity is `W/4` — 128 flop/byte
at `W = 512`, against a 3080 Ti's 94 flop/byte compute/bandwidth balance. Ideal
intensity is therefore NOT what binds; the tile walk's operand RE-READ is, and it
grows with `W`, not away from it.

Both levers are pulled. `W` is an instantiation axis and the flagship is 384–512
(the architecture doc's sizing law); the output block is `C = 128`. The re-read
each addresses is a tile-staging count against the window's distinct tiles — at
`W = 512`, 16 distinct tiles against 72 stagings before the block and 56 after
(the row side's 36 are untouched, the column side's 36 became 20).

Measured at `W = 512`, `BH = 1`, `L = 65536`, dense (`ncu`, four launches, and the
free-running wall beside it):

| | dur ms | wall ms | DRAM read MB | vs ideal | tensor | occ | long-scoreboard |
|---|---|---|---|---|---|---|---|
| one row tile per sweep | 1.322 | 1.33 | 646.7 | 4.5x | 17.2% | 13.2% | 18.8% |
| `C = 128` | 0.938 | 0.74 | 507.2 | 3.6x | 26.0% | 16.6% | 10.1% |

The ideal is 142.7 MB — the two operand planes, the value plane and the gains,
read once. Traffic fell 1.28x and the wall fell 1.8x, because the block also
halves the column stagings and the barriers per unit of issued MMA: the same
Gram work now rides half as many `cp.async` fills on the column side. The term is
no longer bandwidth-dominated at the dense cell.

What remains is the ROW side's 4.5x, and what would cut it is more tile pairs
per warp — which is the budget the census refuses. Stacking just the block's two
row tiles in one warp's registers (the four-warp form of the same block)
assembles 255 registers with 72 spill stores and 17 in-loop local accesses at
`W = 512`; two column tiles as well would want twice that again. The shipped
block, sharing the row tiles across warps instead, is register-clean at 168
registers (`W = 512`), 170 (`W = 384`), 86 (`W = 64`), zero spill, zero local.

The CTA is register-bound to one per SM (256 threads x 168 registers), so a grid
of 128 CTAs — `BH = 1` at `L = 65536` — runs 1.6 waves and pays a tail. Cells short
enough to be tail-dominated show it: the cohort-64 cells run 0.42/0.46 ms against
0.39/0.43 before at `BH = 1`, while the same cells at `BH = 8` run 2.85/3.10 ms
against 3.34/3.90. Dense and scattered are faster at both.

The slab gate, measured, is a CLUSTERING mechanism and not a sparsity one: at
`k_tok = 4` SCATTERED the kernel runs 1.09 ms against dense's 0.94 (near
indistinguishable), and at cohort 64 it runs 0.53 ms with DRAM read down 2.5x.
That is the same fact the ballot-intersection tier's `k_tok ~ 7` crossover
records, seen from the built side, and it is why that tier is not built.
