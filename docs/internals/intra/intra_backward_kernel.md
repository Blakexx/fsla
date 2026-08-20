# `intra_backward_kernel.cuh` — the within-window term's reverse pass

The mirror of `csrc/rola/src/intra/intra_backward_kernel.cuh`. One kernel over the same
tile-pair grid the forward intra walks, emitting the WITHIN-WINDOW half of all four
cotangents.

Symbols at first use: `t` a read token, `j` a write token, `e` a digit, `l` a level,
`G_l` the level's Gram, `P = prod_l G_l`, `A` the pairwise coefficient, `W` the window.

## Recompute, never save, at both grains

`A` is `[W x W]` per window — 134 MB per head at the flagship, the snapshot class the
rung's constraint forbids, applied to the intra. So the Grams are rebuilt from the same
staged slabs the forward reads, under the same slab-occupancy mask, and `A` is formed in
registers. The slabs are staged twice per tile pair: once for the Grams and once for the
two digit MMAs, which is why the kernel's MMA count is roughly three times the forward
intra's and why that ratio is the number to measure it against.

## Why a separate kernel and not a fused one

The forward intra must remain a pure forward for inference. `dA` needs the seed, which
exists only in training, and fusing would put a training-only operand in an inference
kernel's signature.

## Two gates, not one

The Gram pass keeps the forward's INTERSECTION certificate: a digit slab contributes to
`G_l` only where both sides have support in it.

The digit passes gate on ONE side each, and the difference is not an optimization —
`d_pr_l[t,e] = sum_j dG_l[t,j]·pw_l[j,e]` drops level `l`'s own read factor, so it is
live wherever the WRITE side has support at `e` whatever the read side does there.
Gating it on the intersection would drop real terms. The masks are unions over a whole
tile's tokens, so a skipped slab contains only exact zeros and the intra's digit
gradients are COMPLETE — unlike the carry's, which the owner schedule restricts to each
level's own support.

## The leave-one-out survives here

The K33 factorization principle applies where a zero pattern is one factor's support
along ONE axis and the bucketed operation keeps a dense free axis. The intra's sparsity
is a pairwise INTERSECTION, which has no axis to enumerate, so the per-level
leave-one-out is the genuine article. It is computed as a prefix/suffix product over the
`D` Grams already in registers — `2D` multiplies per accumulator element against
`D(D-1)` for the naive form — and never as `A / G_l`: `G_l` is exactly zero whenever the
two tokens' level-`l` supports are disjoint, which under an ALT structure is the common
case, so the division is `0/0`.

## Where the transposes are

`A` and `dG_l` are accumulated with `m = t` and `n = j`, which is the orientation
`d_pr_l` wants: its A operand is the accumulator by an in-lane pack, exactly as the
forward's `A·V` is. The two outputs indexed by the COLUMN token — `d_v` and `d_pw_l` —
need `m = j`, and that is the accumulator's transpose. One `movmatrix` per 8x8 tile
supplies it, which is cheaper than the shared-memory round trip of an object the warp
already holds.

## Why the accumulators are 16 digits wide

`d_pw_l` is indexed by the column token, so its `m` axis is the whole 64-token tile: at a
64-digit stage its accumulator would be 128 registers per thread. The digit axis is
therefore taken one 16-digit slab at a time — the forward's own slab granularity — which
puts both digit accumulators at 32 registers and leaves the RED volume unchanged, since
the number of `(token, digit)` cells written does not depend on how they are staged.

## The fan-in

`d_pr` rows are indexed by the tile pair's ROW axis, which is warp-disjoint under the
forward's row carve, so no cross-warp reduction appears there. `d_pw` and `d_v` rows are
indexed by the COLUMN axis, so all warps contribute to every one; they are REDed
directly rather than through a CTA tree, which is the simple form the rung ships and
measures.
