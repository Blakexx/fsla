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

Both stagings ride the forward's cp.async ring, in the same pair of operand buffers: the
digit pass's fill for step `s+1` is issued while step `s` computes, and its FIRST fill is
issued before the seed contraction and the value cotangent so it lands under them rather
than in front of itself. The two passes' rings are one continuous chain — the digit
pass's prologue is the Gram pass's own trailing commit slot — so the wait depth is one
group throughout and no phase drains the pipe.

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

## The second launch bound

The kernel declares the CTA target `design::cta_target` computes from its own block
footprint, as the carry family's three kernels do. It is not decoration: given only a
thread count, ptxas budgets registers for the residency that thread count suggests rather
than the one this block's shared memory leaves reachable, and the narrow-window arm —
whose 128 threads suggest several resident blocks where its footprint permits one — spills
without the declaration. `kBwdRegFloor` is the measured live set of the built arms and
enters only the target's register ceiling.

## The fan-in, and why it relays through shared memory first

`d_pr` rows are indexed by the tile pair's ROW axis, which is warp-disjoint under the
forward's row carve, so no cross-warp reduction appears there. `d_pw` and `d_v` rows are
indexed by the COLUMN axis, so all warps contribute to every one, and they meet in the
destination plane's own atomics rather than in a CTA tree.

That leaves ADDRESSING as the whole cost of the fan-in, and it is the kernel's largest
single one. The MMA accumulator's lane map puts a `[16 x 16]` tile's row on `lane>>2` and
its column on `(lane&3)*2`, so one reducing instruction's 32 values land on EIGHT
destination rows — a plane pitch apart — at four scattered columns of each. That is eight
memory sectors to carry 128 bytes, and at the flagship the reverse pass issues about 1400
such instructions per thread per tile pair. Measured on sm_86, the resulting store-queue
backup (LG throttle, MIO throttle and short scoreboard together) was 59 % of the kernel's
warp cycles per executed instruction.

So every reduced tile relays through `xpose`, a warp-private `[16 x 18]` plane, before it
reduces: the accumulator is written in the MMA's lane map (two column-adjacent values per
eight-byte store) and read back in the destination's, so a reducing instruction covers
two rows of CONTIGUOUS digits. That is the sector minimum for the value count, and it is
warp-private, so the relay costs two shared accesses per value and NO barrier.

The relay is the ONE reducing path — `d_pr`, `d_pw` and `d_v` are all `[16 x 16]` tiles
and all take it. `d_gw` does not: it is one value per column token, already reduced
across the warp by three lane shuffles, and its four surviving lanes are 0.04 % of the
kernel's reductions.

Choosing the relay over the alternatives is a measurement, not a preference. Meeting the
column-indexed partials in a CTA-wide shared plane — which would cut the reduction COUNT
eightfold — measures WORSE: `atomicAdd` on shared `float` compiles to `ATOMS.CAST.SPIN`, a
lock-and-retry emulation, and eight warps contending one cell turns the saving into a
3.2x instruction blow-up. The count was never the problem; the sectors were.
