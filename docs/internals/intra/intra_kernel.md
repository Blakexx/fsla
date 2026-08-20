# `csrc/rola/src/intra/` — the K31 window-intra kernel

The within-window causal term, as its own deterministic side kernel. It computes
nothing the carry kernel computes: the carry kernel's readout sees the state as
of a window's start, and every pair of tokens INSIDE one window belongs here.
The design it implements is `development/plans/K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md`
§2.1 (the global window grid) and §2.5 (the factorized, windowed, tiled intra).

## What it computes

For a window `w` of `W` tokens (`W` is an instantiation axis; the built arms are
`W = 384`, the flagship, and `W = 64`, the conformance cell — the same two the carry
family builds, because the two kernels share ONE global window grid),

    A[t, t'] = gain(t') * prod_l G_l[t, t'],   G_l = R_l · W_l^T over level l's B = 256 digits
    o[t]   += sum_{t' <= t in w} A[t, t'] · v[t']
    den[t] += sum_{t' <= t in w} A[t, t']

The gain is a per-COLUMN scalar applied AFTER the level product, which is the
shipped kernel's gain law: support bits are ungained. `den` is the same `A`
summed — the mass the ones-column of the oracle's recurrence produces — so the
kernel carries no second matrix for it.

`[W x W]` is never materialized. The window's `(W/64) x (W/64)` tile grid of `[64 x 64]`
output blocks is walked lower-triangular (21 tile-pairs at `W = 384`); the diagonal blocks
carry the within-tile staircase. Per tile-pair the per-level Gram accumulates in
registers over the full 256-digit axis, the levels are Hadamard-multiplied there,
and the result becomes the A operand of the value MMA **in place** — the
accumulator-to-A fragment identity, no shared-memory round trip on sm_80.

## The carve, and why it has no reductions

One CTA per `(bh, window)`, four warps. A warp owns 16 rows of the row-tile the
CTA is currently on and every column; the CTA walks the 10 tile-pairs in
sequence. Output rows are therefore disjoint per warp, every warp does exactly
the same amount of work, and the causal triangle's imbalance lands nowhere —
neither between warps nor between CTAs. The price is four warps per CTA.

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
`4 · W · sum_l B_l` operand bytes, so the ideal intensity is `W/4` — 64 flop/byte
at `W = 256`, against a 3080 Ti's 94 flop/byte compute/bandwidth balance. The
term is bandwidth-balanced BEFORE any re-read, and the tile walk re-reads each
plane tile 2.5x (10 tile-pairs over 4+4 distinct tiles). `ncu` confirms it: DRAM
at 51% of peak and the tensor pipe at 12.5%.

The two levers that raise the intensity are structural, not tuning: a larger `W`
(intensity is linear in it) and a larger output tile (`C = 128` takes the re-read
factor from 2.5x to 1.5x for +20% issued MMA).

The FIRST lever is now pulled. The integration rung made `W` an instantiation axis and
moved the flagship to 384: ideal intensity 96 flop/byte, just over the 94 balance, which
is the architecture doc's sizing law. The cost is register-neutral — the per-window tile
LOOP grows 4→6 row-tiles and 10→21 tile-pairs, but the `gram[D][8][4]` accumulators are
per-tile-PAIR and do not — measured at 154 registers (`W = 384`) against 152 at
`W = 256`, zero spill, zero local. Shared memory grows only by the masks, one word per
(tile, level, side, 32 digits): 46 976 B at `W = 384`. The tile-re-read factor rises with
the grid (21 pairs over 6+6 distinct tiles = 3.5x), so the second lever — the `C = 128`
output tile — is the named follow-up and is gated on its own register census.
