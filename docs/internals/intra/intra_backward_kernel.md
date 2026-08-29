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

The reverse's stage is HALF the forward's digit width, and that is what buys the image
below. Its block carries the forward's operands plus the seed, the relay plane and the
image, and the operand slabs are the only term large enough to pay for the last of those;
halving them frees more than the image costs and leaves the block inside the opt-in
limit. The price is twice as many staging steps and barriers per pass, which is why the
cap is the reverse's own constant and not the forward's — the forward has nothing to buy
with it. The digit pass unrolls its stage loop by the ring's own period of two, so a
body's buffer parity is a compile-time property of the body rather than a value it
carries; the ring is what sets the factor, not a tuning sweep.

## Why a separate kernel and not a fused one

The forward intra must remain a pure forward for inference. `dA` needs the seed, which
exists only in training, and fusing would put a training-only operand in an inference
kernel's signature.

## Two gates, and they answer to different authorities

The Gram pass keeps the forward's INTERSECTION certificate: a digit slab contributes to
`G_l` only where both sides have support in it.

A digit pass's OPPOSITE-side gate is algebra. `d_pr_l[t,e] = sum_j dG_l[t,j]·pw_l[j,e]`
drops level `l`'s own read factor, so it is live wherever the WRITE side has support at
`e` whatever the read side does there; gating it on the intersection would drop real
terms. The masks are unions over a whole tile's tokens, so a skipped slab contains only
exact zeros.

A digit pass's OWN-side gate is the CONSUMER's, and it is the one thing in this kernel
justified from outside the kernel's own equations. The two planes leave the operator at
`rola/ops/prefill.py` as the returned gradients of `read_levels` and `write_levels`, and
they have no other consumer in the tree: whatever produced the routing factors is the
only thing that reads them. Where that producer is entmax, its VJP restricts to the
support in BOTH of its terms — `dz_e = g_e (dp_e - q)` with `g_e` zero off support and
`q = <g,dp>/<g,1>` — and the shipped kernel realizes that by never LOADING an
off-support cotangent, which is stronger than multiplying it by zero. Where the producer
is softmax there is no support: `dz_e = p_e (dp_e - <p,dp>)` needs the whole digit axis,
and truncating it moves the logit gradient by tens of percent.

So the own-side gate is applied EXACTLY where the mode word declares that side sparse,
and `side_bits` answering `0xFFFF` for an undeclared side is what makes that a single
expression rather than a branch: joined to the opposite-side gate it is a no-op on a
softmax duty and the intersection certificate on an entmax one. It is applied at two
grains, the digit slab (through the certificate) and the individual `(token, digit)`
cell (through the own-side operand, resident at the reducing address). The test is the
MAGNITUDE bits, so a signed zero is off support like an unsigned one.

What this changes about the planes the operator returns: a cell the gate drops is no
longer written BY THIS KERNEL. It is not thereby zero — the carry's two scans own the
cross-window half of the same planes and do not yet restrict — so the intra's digit
gradients are no longer COMPLETE in the sense the old dense sweep made them. They are
complete on the union of the two declared supports, which is the set the producer reads,
and that is the property the gradcheck gate holds them to.

## The leave-one-out survives here

The K33 factorization principle applies where a zero pattern is one factor's support
along ONE axis and the bucketed operation keeps a dense free axis. The intra's sparsity
is a pairwise INTERSECTION, which has no axis to enumerate, so the per-level
leave-one-out is the genuine article. It is computed as a prefix/suffix product over the
`D` Grams already in registers — `2D` multiplies per accumulator element against
`D(D-1)` for the naive form — and never as `A / G_l`: `G_l` is exactly zero whenever the
two tokens' level-`l` supports are disjoint, which under an ALT structure is the common
case, so the division is `0/0`.

## The image, and why the reducing phase is carved differently from the accumulating one

The tile pair's coefficient object is `A` and the `D` cotangents `dG_l`, each a
`[C reader tokens x 64 write tokens]` plane spread over the block's warps. It is
accumulated ROW-carved because the Gram's `m` axis IS the reader axis — a warp owns 16
readers and there is no choice about it. Every phase that CONSUMES it, though, contracts
over the reader axis: `d_pw_l` and `d_v` sum over `t`, which under a row carve is the
carved axis, so every warp holds a partial of every destination cell and the whole block
reduces each cell eight times.

So the object is PUBLISHED once per level, as a bf16 plane in shared memory, and the
consuming phase is re-carved by DIGIT: a warp owns one `kSlabDigits` slab of the stage
and one slice of the destination token axis, its contraction closes inside the warp, and
every destination cell is reduced ONCE per block. The publish is arithmetically free —
the consuming phase's A operand was already rounded to bf16 in registers before every
MMA, so the image is that same rounding at that same point — and it costs one plane at
the pitch the operand slabs already prove conflict-free in BOTH `ldmatrix` flavours,
which is what lets one image serve `d_pr_l`'s `(m = t, k = j)` map through `ldmatrix` and
`d_pw_l`'s and `d_v`'s `(m = j, k = t)` map through `ldmatrix.trans`. No in-register tile
transpose and no per-`(slab, m-tile)` re-pack survive it.

The carve is a statement about the block, not about a shape: the digit part is one slab
of the stage and the group is the rest of the warps, so the row group is `C / groups`
readers wide and the column group is `64 / groups` writers wide, at every built window.
`d_pr_l` is the one gradient the row carve already left warp-disjoint, and the digit
carve keeps it so — a digit belongs to exactly one warp.

## Why the accumulators are 16 digits wide

`d_pw_l` is indexed by the column token, so its `m` axis is the whole 64-token tile: at a
wider digit granule its accumulator would be many times 32 registers per thread. The
digit axis is therefore taken one 16-digit slab at a time — the forward's own slab
granularity, and the unit the digit carve hands out — which puts both digit accumulators
at 32 registers and leaves the RED volume unchanged, since the number of `(token, digit)`
cells written does not depend on how they are staged.

`d_pw_l`'s contraction runs over the whole block of readers rather than one warp's 16, so
it is gated per row tile: a tile the certificate left unstaged is one whose operand rows
are all exact zeros, and a tile the causal walk excludes is one whose image rows are
exact zeros, so both are skipped rather than read.

## The second launch bound

The kernel declares the CTA target `design::cta_target` computes from its own block
footprint, as the carry family's three kernels do. It is not decoration: given only a
thread count, ptxas budgets registers for the residency that thread count suggests rather
than the one this block's shared memory leaves reachable, and the narrow-window arm —
whose 128 threads suggest several resident blocks where its footprint permits one — spills
without the declaration. `kBwdRegFloor` is the measured live set of the built arms and
enters only the target's register ceiling.

## The fan-in, and why it relays through shared memory first

Under the digit carve every reduced cell has exactly one owning warp, so the fan-in is
not a cross-warp reduction at all: what remains in the destination plane's atomics is the
meeting of the tile pairs of one column, and of the blocks and windows the carry's scans
also write.

That leaves ADDRESSING as the whole cost of the fan-in, and it is the kernel's largest
single one. The MMA accumulator's lane map puts a `[16 x 16]` tile's row on `lane>>2` and
its column on `(lane&3)*2`, so one reducing instruction's 32 values land on EIGHT
destination rows — a plane pitch apart — at four scattered columns of each. That is eight
memory sectors to carry 128 bytes. Measured on sm_86, the resulting store-queue backup
(LG throttle, MIO throttle and short scoreboard together) was 59 % of the kernel's warp
cycles per executed instruction.

So every reduced tile relays through `xpose`, a warp-private `[16 x 18]` plane, before it
reduces: the accumulator is written in the MMA's lane map (two column-adjacent values per
eight-byte store) and read back in the destination's, so a reducing instruction covers
two rows of CONTIGUOUS digits. That is the sector minimum for the value count, and it is
warp-private, so the relay costs two shared accesses per value and NO barrier.

The relay is the ONE reducing path — `d_pr`, `d_pw` and `d_v` are all `[16 x 16]` tiles
and all take it. `d_gw` does not: it is one value per column token, already reduced
across the warp by three lane shuffles, and its four surviving lanes are a rounding error
in the kernel's reductions.

Choosing the relay over the alternatives is a measurement, not a preference. Meeting the
column-indexed partials in a CTA-wide shared plane — which would cut the reduction COUNT
eightfold — measures WORSE: `atomicAdd` on shared `float` compiles to `ATOMS.CAST.SPIN`, a
lock-and-retry emulation, and eight warps contending one cell turns the saving into a
3.2x instruction blow-up. The count was never the problem; the sectors were.

## Why the restricted reduce is predicated and not branched

The own-side cell gate is a property of the DESTINATION CELL, and the relay's lane map
gives one lane eight destination cells: eight rows, two apart, of one digit column. So a
`if (live) atomicAdd(...)` inside the reducing walk is eight control-flow regions, each
of which every value crossing the walk must stay live across — measured at 96 bytes of
spill on the two wide arms, against zero for the same arithmetic issued as a predicated
`red.global.add.f32`.

The walk then unrolls by the relay's own period, TWO, which is the number of row
parities the lane map splits the tile into. Unrolling past that holds all eight
certificate loads live at once and spills 80 bytes; unrolling to one buys nothing over
two. This is the same rule pass 2's stage loop follows, and for the same reason: the
factor a body may be replicated by is the period of the structure it walks.

A DENSE duty takes the unrestricted walk unchanged. The declaration is uniform over the
block, so that is one branch; leaving it as a per-cell test that a dense side can never
fail costs the dense arm a shared load per reduced cell for nothing, and it measured as
0.80x of baseline's wall below one wave against 0.98x with the branch.

The certificate is one shared load at an address the reducing lane already owns. The
own-side operand slab is resident — the pass just contracted against it — and the eight
cells are eight rows of one of its digit columns, so the whole mechanism is a base
formed once in the prologue plus a two-row stride. Nothing is staged for it and no new
plane exists.

## The reducing addresses are one pointer and constant displacements

`d_gw`'s drain reduces sixteen values per column tile at eight-float spacing. Written as
one sum per destination — base plus a level's worth of index arithmetic per value —
`ptxas` materializes sixteen live 64-bit addresses, thirty-two registers, in the region
where the Gram accumulators are also live. Written as one pointer formed before the
sweep plus fifteen compile-time displacements it is two registers and the displacements
fold into the reducing instruction. That difference is what the own-side certificate is
paid for out of: it is the whole of the headroom the restricted reduce needed.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/intra/intra_backward_kernel.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kbwdstagecap"></a>
### `kBwdStageCap`

THE REVERSE STAGES HALF AS WIDE AS THE FORWARD.  Its block carries the forward's
operands plus the seed, the relay plane and the `dG_l` image, and the image is what
the digit carve is built on, so the staging ring is the term that pays for it.  The
cap is the reverse's own because the forward has nothing to buy with it.

<a id="kxposerows"></a>
### `kXposeRows`

THE RELAY PLANE, one `[kMmaM x kSlabDigits]` tile per warp: every cotangent this
kernel reduces has that shape, and the accumulator's lane map scatters a reducing
instruction's 32 values over eight destination rows a plane pitch apart.  The pitch
is EVEN because the accumulator's two column-adjacent values relay as one eight-byte
store.  See docs/internals/intra/intra_backward_kernel.md.

<a id="kimagepitch"></a>
### `kImagePitch`

THE `dG_l` IMAGE: the tile pair's coefficient object, `[C reader tokens x kTile write
tokens]`, published in bf16 so the phase that READS it can be carved differently from
the phase that accumulated it.  The pitch is the one the operand slabs already prove
conflict-free in both `ldmatrix` flavours.

<a id="live-rows-side"></a>
### `live_rows_side`

THE GRAM PASS'S GATE IS THE INTERSECTION and the digit passes' is ONE SIDE.  A
level's digit gradient drops that level's own factor, so `d_pr_l[t,e]` is live
wherever the WRITE side has support at `e` whatever the read side does there; gating
it on the intersection would drop real terms.

<a id="publish-image"></a>
### `publish_image`

THE PUBLISH.  One row-carved `[kWarpRows x kTile]` accumulator becomes bf16 rows of
the image, at the rounding pass 2's A operand carried anyway.  `row` addresses the
warp's first accumulator row at the lane's column; the accumulator's second row group
is eight rows below it.

<a id="reduce-tile"></a>
### `reduce_tile`

THE ONE REDUCING PATH.  `get` is the lane's cursor into the relayed plane and
`lane_off` its destination offset -- row half in the high byte, digit column in the
low; both are invariant over the whole kernel.  `base` addresses the tile's first
destination row and `stride` its row pitch, as an `int` because the widest offset a
reduced tile reaches is fifteen row pitches.
docs/internals/intra/intra_backward_kernel.md.

<a id="fan-in-add-if"></a>
### `fan_in_add_if`

A PREDICATED reducing instruction rather than a branched one: the gate is a property
of the destination CELL, so eight cells of one lane carry eight independent gates and
a branch per cell gives the walk eight control-flow regions to keep values live
across -- measured, 96 bytes of spill against zero.

<a id="reduce-tile-on-support"></a>
### `reduce_tile_on_support`

THE SAME PATH, RESTRICTED TO THE DESTINATION PLANE'S OWN SUPPORT.  `sup` walks the
own-side operand at the reducing address and `sup_step` is its two-row stride.  A side
whose duty is DENSE has no support to restrict to and takes the unrestricted path
unchanged: the declaration is uniform over the block, so the choice is one branch and
not a per-cell test the dense arm would pay for nothing.  The restricted walk unrolls
by the relay's own period -- the lane's eight cells are the two row parities of a
`[16 x 16]` tile, and unrolling past that holds eight certificate loads live and
spills.  docs/internals/intra/intra_backward_kernel.md.

<a id="kbwdregfloor"></a>
### `kBwdRegFloor`

THE SECOND LAUNCH BOUND, from the arm's own footprint.  Without it ptxas budgets
registers for the residency the THREAD COUNT suggests rather than the one this
block's shared memory leaves reachable, and the narrow-window arm pays the
difference in spill.  `kBwdRegFloor` is the measured live set of the built arms.

<a id="kparts"></a>
### `kParts`

THE DIGIT CARVE.  A warp is a (digit part, group) pair: the part names one
`kSlabDigits` slab of the stage and the group names the slice of the destination
token axis the warp owns, so both digit gradients close their contraction inside
the warp and every destination cell is reduced ONCE per block.

<a id="kelem"></a>
### `kElem`

THE OPERAND BASES.  In the unrolled bodies the buffer, the k-tile and the m-tile
are compile-time constants, and everything else an `ldmatrix` address carries --
the lane map, the warp's digit part, its group -- is invariant over the whole
kernel, so each address is ONE base computed here plus a constant.

<a id="note-l612"></a>
### near line 612

`A`, `dA` and every `dG_l` are formed IN PLACE over the Gram registers, and
(I3)'s row sum is drained inside the same sweep so it never becomes an array.
ITS SIXTEEN DESTINATIONS ARE ONE POINTER AND FIFTEEN CONSTANT DISPLACEMENTS: a
reducing address written as one sum is sixteen live 64-bit addresses instead.

<a id="syncthreads"></a>
### `__syncthreads`

THE IMAGE IS THE RE-CARVE.  `A` and each `dG_l` are accumulated ROW-carved --
the Gram's `m` axis IS the row axis -- and published once as bf16, which is the
rounding pass 2's A operand carried in registers anyway.  Every phase that
reduces over the reader axis then reads the image back under a carve of its
own, and closes its contraction inside one warp.

<a id="note-l741"></a>
### near line 741

(I6)'s contraction runs over the WHOLE block of readers, so it is gated
per row tile: a tile the certificate left unstaged is one whose operand
rows are all zero, and the rows of a tile the causal walk excludes are
zero in the image.

<a id="near-line-77"></a>
### near line 77

The forward's block footprint plus the ROW side's seed rows, the per-warp relay
plane and the image.  The seed is staged once per output block, not once per column
tile: its rows are the block's readers.

<a id="near-line-111"></a>
### near line 111

the mask helpers read one CTA-local summary, so the reverse pass's two one-sided
predicates cost nothing beyond the forward's intersection one.

<a id="kslabsperstage"></a>
### `kSlabsPerStage`

§6: bwd_slabs_per_stage(B) is compile-time (B is this function's template
parameter) but the heuristic can't see through the function call, so name it.

<a id="krowtiles"></a>
### `kRowTiles`

§6: row_tiles_of(W) is compile-time (W is this function's template parameter)
but the heuristic can't see through the function call, so name it.

<a id="near-line-149"></a>
### near line 149

THE GRAM PASS'S OWN GATE: both sides' supports must meet in the digit slab, which is
the forward's intersection certificate read off this kernel's own mask summary.

<a id="krowtiles-2"></a>
### `kRowTiles`

§6: row_tiles_of(W)/bwd_slabs_per_stage(B) are compile-time (D, B, W are this
function's template parameters) but the heuristic can't see through the
function calls, so name them.

<a id="near-line-183"></a>
### near line 183

THE HALF-SPLIT CERTIFICATE PASS needs a level at least two words wide.  A
narrower level is a REFUSAL here rather than a second pass shape, because the
reverse family is built at depth two only (`intra_backward.cu`).

<a id="maskwords"></a>
### `mask_words`

§6: both bounds are compile-time (kTiles/D are locals/template params, mask_words(B)
a constexpr function of the template param B) but the heuristic can't see through
the arithmetic/function-call form, so name them.

<a id="kmaskwordshalf"></a>
### `kMaskWordsHalf`

§6: mask_words(B)/2 is compile-time (B is this function's template parameter)
but the heuristic can't see through the function call/arithmetic, so name it.

<a id="kcoldigitoctets"></a>
### `kColDigitOctets`

§6: these are all compile-time (B, W are this function's template parameters)
but the heuristic can't see through the function call/arithmetic, so name them.

<a id="kcolchanneloctets"></a>
### `kColChannelOctets`

§6: kColChannels/8 is compile-time but the heuristic only reads a bare
identifier, so name it.

<a id="near-line-284"></a>
### near line 284

THE SEED, staged ONCE PER OUTPUT BLOCK: its rows are the block's readers, which do
not change with the column tile.

<a id="krowchanneloctets"></a>
### `kRowChannelOctets`

§6: kRowChannels/8 is compile-time but the heuristic only reads a bare
identifier, so name it.

<a id="ktileoctets"></a>
### `kTileOctets`

§6: kTile/8 is compile-time (kTile is a file-level constexpr) but the
heuristic only reads a bare identifier, so name it.

<a id="relaytile"></a>
### `relay_tile`

THE RELAY.  `acc` is one `[kMmaM x kSlabDigits]` accumulator in the MMA's lane map
and `put` the lane's cursor into its own warp-private plane; the round trip is
warp-private, so it needs no barrier.

<a id="kxposerowshalf"></a>
### `kXposeRowsHalf`

§6: kXposeRows/2 is compile-time (kXposeRows is a file-level constexpr) but the
heuristic only reads a bare identifier, so name it.

<a id="kxposerowshalf-2"></a>
### `kXposeRowsHalf`

§6: kXposeRows/2 is compile-time (kXposeRows is a file-level constexpr) but the
heuristic only reads a bare identifier, so name it.

<a id="grptile"></a>
### `grp_tile`

the row group's own output tile: a group is `kPrRows <= kTile` rows wide, so it
lies inside one row tile and inherits that tile's causal predicate.

<a id="basesupr"></a>
### `base_sup_r`

THE OWN-SIDE CERTIFICATE'S per-cell address: a lane's eight destination cells are
eight rows, two apart, of ONE digit column of its own side's staged operand.

<a id="kstagesperlevel"></a>
### `kStagesPerLevel`

§6: bwd_stages_per_level(B)/bwd_slabs_per_stage(B) are compile-time (B is
this function's template parameter) but the heuristic can't see through the
function calls, so name them once for this loop nest.

<a id="pre"></a>
### `pre`

the leave-one-out over LEVELS, as a prefix/suffix product and never as
`A / G_l`: `G_l` is exactly zero on disjoint supports.

<a id="dpwbase"></a>
### `dpw_base`

THE FIRST FILL IS ISSUED BEFORE THE SEED CONTRACTION ABOVE, so it lands under
it; from there the ring is the forward's, in the buffers pass 1 already owns.

<a id="prown"></a>
### `pr_own`

A SOFTMAX DUTY IS DENSE BY THEOREM, so its cotangent plane is live on the
whole digit axis and no cell of it may be dropped; only a side the mode word
declares sparse carries the producer's off-support annihilation.

<a id="near-line-679"></a>
### near line 679

§6: kStagesPerLevel is the same compile-time quantity aliased above
(bwd_stages_per_level(B)), reused here -- same enclosing ct-loop scope.
