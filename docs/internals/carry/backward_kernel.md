# `backward_kernel.cuh` — the carry's two reverse scans

The mirror of `csrc/rola/src/carry/backward_kernel.cuh`. Two `__global__` kernels and
the owner-level bodies they share: Scan A, the ascending replay that emits `d_pr`, and
Scan B, the descending adjoint that emits `d_v`, `d_gw`, `d_pw` and `d_state_in`.

Symbols, each at first use: `D` levels, `B` digits per level, `BC` the leaves one owner
holds, `W` the global window length in tokens, `C` the private batch's token tile,
`d_v = 64` value channels, `cols = d_v + 1`, `s_l` an owner's span at level `l`, `r_l` a
leaf's run offset inside its owner at level `l`.

## Why the state is replayed and not stored

The state is `O(N·cols)` per head and the window count is `T/W`, so a snapshot at any
granularity is `O(N·T/W·cols)` — 33 GB per head at the flagship. With no decay the
recurrence is affine in the state, so the state at a window boundary is the SUM of the
deposits below it: replaying it forward costs one pass over deposits the forward already
read and nothing else, and the window grid's global edges are already the forward's own
fold boundaries. The same argument is applied to the SCHEDULE: neither scan inherits a
list, a threshold or a bucket partition from the forward — both rebuild the two packed
lists from the facts table, in the forward's order, at the forward's cost.

## Why there are two kernels and not one

A fused form holds two fp32 slabs in one thread's registers. At the flagship the slab is
80 registers per thread on the measured forward, so two slabs plus a working set that
cannot be smaller than the forward's 96 exceeds the hard 255-register cap before any of
the backward's own operands are counted. No term couples the replayed state to the
adjoint — the read-side gradients read the state only, the write-side ones read the
adjoint only — so the two scans are data-independent given the seeds. They are launched
concurrently on two streams: two slabs are resident in the MACHINE, and neither thread
holds two.

## Why the slab is leaf-major, and the carve with it

The forward holds the state channel-major because its readout contracts over LEAVES, so
the accumulator's `m` axis is the readout's `m` axis and the accumulator becomes an A
operand by an in-lane pack. Both backward readouts contract over CHANNELS:

    dR[leaf, tok] = sum_c  Sreplay[leaf, c] · dn[tok, c]
    dW[leaf, tok] = sum_c  Lambda[leaf, c] · vt[tok, c]

so `m = leaf` and `k = channel`. Holding the slab leaf-major keeps the same
accumulator-as-operand identity the forward has; holding it channel-major would make
`k` the carve's own axis and turn every readout into a cross-warp partial sum. The carve
follows the `m` axis as it always has, so warps own LEAF slices — the transpose of the
forward's channel slices, and a consequence of the same law rather than an exception to
it.

The price is paid once, on `d_v`: `d_v[tok, :] = sum_leaf U[tok, leaf]·Lambda[leaf, :]`
contracts over the carved axis, so it is the one MMA whose B operand is the slab in the
wrong lane map. It is repaired by an in-register `movmatrix` per 8x8 tile and reduced
across the four warps in the output staging buffer. The deleted P65 design paid that
price on EVERY readout; here it is confined to one MMA on one path.

## The digit reduction's pass geometry

Under the mixed-radix local order a level's digit field is a contiguous bit run of the
owner's local index, and the leaf-split carve slices that index at a bit boundary. This
pass geometry needs every level to fall cleanly on one side of that boundary: either
warp-private (its field covers the whole warp id, so its digits PARTITION across warps)
or warp-replicated (its field lies entirely below the warp id). At the built arms,
`local = (r_0 << 4) | r_1` and warp `u` owns `local in [32u, 32u+32)`:

| level | segment | reduction | cross-warp |
|---|---|---|---|
| `l = 0` (outer, `s_0 = 8`) | a whole MMA m-tile | three lane shuffles | no — the warp owns its two digits |
| `l = 1` (inner, `s_1 = 16`) | one accumulator row | one add | yes — every warp holds every row |

No runtime index into a register array appears at either level, which is the constraint
that governed the deleted design's whole pass law.

### Where the inner level's four warps meet

The inner level is warp-REPLICATED: all four warps hold a partial of the same thirty-two
`(token, digit)` cells, so they have to meet. Three places were built and measured on
this kernel, and the choice is not the obvious one.

| meeting place | what it costs |
|---|---|
| a SHARED-memory atomic | **refused.** `atomicAdd` on a shared `float` has no hardware reduction on sm_80/sm_86; nvcc lowers it to `ATOMS.CAST.SPIN`, a lock-and-retry loop whose body is a five-instruction serial chain. Four warps on the same cells retried **2.08 times per site**, and the block carried **51.6 %** of Scan A's stall samples and **32.1 %** of Scan B's |
| barrier-separated ROUNDS | correct and deterministic, but three rendezvous per batch: measured `short_scoreboard` 3.57 → 2.24 against `barrier` 0.39 → 2.66, and the wall did not move. At four warps per CTA a CTA-wide barrier idles three of them |
| **the GLOBAL plane** | **taken, on BOTH sides.** The plane is already the OWNERS' meeting place — two owners sharing a level-1 group reduce into the same cell — so a fourth grain of partial adds no mechanism that was not there. No shared cell, no atomic, no rendezvous |

The global meeting is at the reduction-addressing floor: for one `(nt, e)` a warp's
thirty-two lanes cover FOUR token rows (`lane & 3`) at EIGHT CONTIGUOUS digits each
(`lane >> 2`), so one reducing instruction touches four 32-byte sectors for 128 bytes.
Measured `lg_throttle` stays at 0.00 with the two scans' global reductions at 9.4 M and
22.0 M.

### The gain, and why it may ride the warp-grain partial

`(R7)` is `d_pw_l[j, e] = gw[j] · sum_s (…)`, with the gain OUTSIDE the sum. A
warp-grain fan-in computes `sum_over_warps (gw[j] · partial_w)` instead: the same value
in exact arithmetic, a different rounding in this one. **This is ratified, and the reason
it is not a new liberty is that the plane already does exactly this at the OWNER grain** —
every owner CTA that touches token `j` reduces its own `gw[j] · partial_owner` into the
cell, so the gain has always been distributed across the owner decomposition. The write
side distributes it one grain finer, inside an accumulator whose summation order is
nondeterministic by construction and whose outputs are not bit-reproducible run to run on
any build. The read side's `(R2)` has no such factor and is unaffected either way.

The OUTER level is the exception and does not need the argument: it is warp-private, its
sum completes inside one warp under three lane shuffles, and it stages through `sdg` so
the drain can coalesce its scattered token rows. The gain is applied there, on a
completed sum, so `(R7)` is honoured exactly on those rows. The staged plane therefore
carries the outer level ALONE, and the clear and the drain walk exactly those cells —
which is why `sdg` is currently allocated for the whole digit run while only `s_0` of its
rows are live. Reclaiming those 2 048 bytes does not change `cta_target` at any built
arm and is left as a separate resource change.

### The two conditions, and why only depth two meets them

Whether a level falls cleanly is a property of the SPAN VECTOR, not of the order, so the
refusal is stated as the two conditions rather than as a depth:

* the ROW map, `s_{D-1} == kMmaM` — one accumulator row is one innermost digit;
* the WARP map, `span_bits(0) == kLocalBits - log2 kMmaM` — the bits above the innermost
  level's field are ONE level's, so that level's digits partition across warps and no
  level straddles the warp-id boundary.

Only depth two can meet both: the innermost level owns the four row bits and one other
level must own everything above them. The two deep arms each fail exactly the conditions
that name their obstruction — `s = (2,4,16)` splits the upper bits between levels 0 and
1, so level 1's field straddles the boundary (two warps hold each of its digits, and the
staging would need a per-warp digit base plus a second cross-warp meeting); `s =
(2,2,4,8)` fails the row map as well, since `s_3 = 8` is half an m-tile and one
accumulator row then carries two levels' digits.

Extending either is a per-arm derivation carrying a wrong-gradient risk, so the gap is
NAMED rather than served: `rola.ops.prefill.prefill_autograd` refuses depth by the same
sentence, and the intra's reverse arms stop at the same place.

## What the digit gradients are, and where they are exact

A level's digit gradient is a weighted marginal of the one per-batch gradient tile along
that level's index axis, weighted by the product of the OTHER levels' amplitudes:

    d_p_l[t, e] = sum_{s : r_l(s) = e}  tile[t, s] · prod_{l' != l} p_l'[t, r_l'(s)]

The weight is a prefix/suffix product over the `D` per-level amplitudes the produce
already loads, so it costs `2D` multiplies and NEVER a division — `p_l` is exactly zero
off its support under exact entmax, and the leave-one-out written as `prod_l p_l / p_l`
would be `0/0` with no epsilon that does not invent a gradient.

**The result is exact on each level's own support and partial off it.** The schedule
visits an owner only where EVERY level's support meets that owner's run, while
`d_p_l` drops level `l`'s own factor — so at a digit outside level `l`'s support the
kernel holds the sum over the live owners rather than over all of them. Nothing reads
it: the exact-entmax Jacobian is zero off the support in both indices, and a softmax
level has no off-support entry at all. `tests/oracle/test_carry_backward_vs_reference.py` states
that as an algebraic gate, not as a comment.

## The mass channel

`cols = 65` is not a whole number of MMA k-tiles, so the mass channel does not ride the
readout's k axis. It is the slab's last n-tile, one eighth occupied, and the readout adds
its contribution as a broadcast product against the panel's own last column — which is
`d_den` on the seed side and 1 on the value side, and getting those two confused is the
one arithmetic difference between the two scans' readouts.

## The ordering law, mirrored

The forward's law is window-grain read-before-write on the owner's own slots. Scan B's is
its exact mirror: **writes of window `w` execute before `w`'s reads fold.** A batch that
crossed a window line would owe the within-window pairs inside it to neither this scan
(whose readout is against the adjoint at window EXIT) nor the intra backward (which owns
within-window pairs). Inside a window a scan batches freely between its own-side events,
which is the forward's private-quantum freedom iterated downward.

## Register floors are declared per kernel

`cudaFuncGetAttributes` is a property of a KERNEL, so one floor for two kernels is how a
spill goes green: the build honors a bound the heavier scan cannot meet and only the
census notices. `BwdGeom` declares `kRegFloorA` and `kRegFloorB` separately, and each
scan's ring depth and CTA target are derived from its own.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/backward_kernel.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kdv"></a>
### `kDv`

THE REVERSE FAMILY'S OWN VALUE WIDTH, AND WHY IT IS A CONSTANT HERE (K35 S3b).
`d_v` became a per-arm parameter of the FORWARD carry family; this body did NOT
follow it.  The two scans are a separate design -- leaf-major slabs, a fixed
`kColsPad = 72` pre-packed operand plane, its own arm list -- and they carry a
standing seam gap against the forward's arms already.  Widening them is that gap's
work, not this one's, so the constant stays here, named where this family uses it,
and nothing in `carry_kernel.cuh` defines it any more.

<a id="kcolspad"></a>
### `kColsPad`

THE SEED AND VALUE PLANES ARE PRE-PACKED `cols`-WIDE AND PADDED TO WHOLE MMA
n-tiles, so ONE staging body serves both and the mass channel is an ordinary
column rather than a carve.  Columns past `kCols` are zero at the seam, which is
what lets the last n-tile be read with no partial-tile path.

<a id="kregfloora"></a>
### `kRegFloorA`

TWO FLOORS, TWO CENSUSES.  One floor for two kernels is how a spill goes green:
the build honors a bound the heavier scan cannot meet and only the census
notices.  Both are DECLARED -- the measured live set of the heaviest built arm.

SCAN A SITS AT THE ARCHITECTURE'S PER-THREAD CAP, and that is recorded rather
than smoothed over: 255 of 255 on every built arm and both arches, zero spill,
`cta_target` 2.  It has NO headroom, so the next change to that kernel starts by
measuring what it costs in registers, and a hoist of the forward's item-1 shape
(which cost that kernel twelve to sixteen) is not available here without giving
something back first.

<a id="bwdparams-note-l135"></a>
### in `BwdParams`, near line 135

THE SLOT TABLES OF THE TWO SLABS (`ops.cuh`'s `page_slot`,
docs/internals/paging/paging.md).  Null is the dense backing -- the slot IS the
atom -- so a paged ENTRY plane and a dispatch-allocated dense EXIT plane are the
same body with two tables, not two paths.

<a id="static-assert"></a>
### `static_assert`

THE DIGIT REDUCTION'S ROW MAP, stated as the two conditions it rests on rather
than as a depth.  A warp owns `kWarpLeaves` consecutive owner-local indices, so
a level's digit field reduces WITHOUT a cross-warp meeting exactly when the
field lies inside the warp's own bit range, and an accumulator ROW is one digit
exactly when the innermost span is one m-tile.  Depth two at `s = (8, 16)`
satisfies both, and only depth two can: the innermost level owns the four row
bits and ONE other level must own everything above them, including the whole
warp id, so its digits PARTITION across warps.  `s = (2, 4, 16)` splits those
upper bits between two levels and level 1's field then straddles the warp-id
boundary -- two warps hold each of its digits, so the staging acquires a
per-warp digit base and a second cross-warp meeting.  `s = (2, 2, 4, 8)` breaks
the row map as well (`s_3 = 8`, half an m-tile, so one accumulator row carries
two levels' digits).  Each is a per-arm derivation with a wrong-gradient risk,
and a named gap beats one.

<a id="produce"></a>
### `produce`

ONE BATCH'S PANEL, expanded from factors.  `SAMP` also publishes the per-level
amplitude RUNS: they are the digit reduction's leave-one-out weights, and they
are the same values this body already holds -- never a second gather.

ONLY THE PANEL-GAINED SIDE MULTIPLIES.  Three of the family's four produce sites
pass a gain that is exactly `1.0` -- Scan A's read phase and BOTH of Scan B's --
and `rn(a * 1) = a`, so for them the trailing packed multiply over all `SPT`
slots is deleted outright and the accumulator is stored as it stands.  What that
multiply also carried, the DEAD-ROW ZEROING, moves onto LEVEL 0's run instead:
level 0 is the outermost and therefore the narrowest at every built arm, and
every panel entry has exactly one level-0 factor, so zeroing that run zeroes the
whole row for one or two selects rather than `SPT/2` packed multiplies.  `samp`
is published BEFORE the zeroing and so still carries the raw amplitudes, exactly
as it did when the zeroing rode the panel gain.

<a id="bwdowner-note-l546"></a>
### in `BwdOwner`, near line 546

THE MASS COLUMN IS READ ONCE, NOT ONCE PER m-TILE.  Its panel entry is indexed
by the TOKEN alone, and a thread's token depends on `nt` and `e & 1` only, so
the two m-tiles want the same eight values.  Read as `[nt][e & 1]` ahead of the
`pm` loop: eight shared loads and eight converts per readout, against the
thirty-two the inner loops used to repeat.

<a id="for-outer-cells"></a>
### `for_outer_cells`

THE OUTER LEVEL'S STAGED CELLS, walked by a CARRYING cursor.  Both the clear and
the drain visit exactly these, so the walk is written once: `sdg` keeps its full
`kDigitRows` stride (`samp` and `sdcol` share the run's indexing) while only the
outer level's rows are live in it, the inner level having fanned itself in.

The cursor advances instead of dividing.  The stride between a thread's cells is
the constant `kThreads`, so the quotient and remainder carry; one division per
walk, at the seed.  At the built arms `s_0` divides `kThreads` and the row is in
fact invariant, but the carry is written out because that is a property of the
SPAN VECTOR and not of the pass.

<a id="digit-reduce-outer"></a>
### `digit_reduce_outer`

THE DIGIT REDUCTION.  A level's digit gradient is the tile's weighted marginal
along that level's index axis, the weight being the OTHER levels' amplitude
product -- one staged amplitude at `D = 2`, and never a division.  The staging is
TOKEN-MAJOR because the drain below writes one contiguous digit run per token, and
a digit-major buffer would make every one of those stores its own sector.

<a id="bwdowner-note-l636"></a>
### in `BwdOwner`, near line 636

outer level: a whole m-tile is one digit and a warp owns its two exclusively, so
the segment is three lane shuffles and no cross-warp term at all.  Its sum is
therefore COMPLETE inside one warp, and it stages into `sdg` so that the drain
can coalesce its scattered token rows into one reducing instruction per run.

<a id="digit-reduce-inner-direct"></a>
### `digit_reduce_inner_direct`

THE INNER LEVEL'S FOUR WARP PARTIALS MEET IN THE GLOBAL PLANE.  One accumulator
ROW is one digit, so a warp's segment is an add -- but every warp holds every row,
over the SAME thirty-two cells, so the four have to meet somewhere.  They meet
where the OWNERS already meet: the plane is a fan-in point by construction (two
owners sharing a level-1 group write the same cell), so adding a fourth grain of
partial to it introduces no mechanism that was not already there.

WHY NOT IN SHARED MEMORY, both ways measured on this kernel:
  * `atomicAdd` on a shared `float` has NO hardware reduction on this
    architecture.  nvcc lowers it to `ATOMS.CAST.SPIN`, a lock-and-retry loop
    whose body is a five-instruction serial chain; with four warps contending the
    same cells it retried 2.08 times per site and carried a THIRD of each scan's
    stall.  (The intra reverse pass reached the identical verdict from the other
    side when it measured a CTA-wide shared atomic plane at 1.65x its wall.)
  * barrier-separated rounds, the mechanism the `ung` meeting below uses, cost
    three rendezvous per batch and traded the whole saving back: measured
    `short_scoreboard` 3.57 -> 2.24 against `barrier` 0.39 -> 2.66, wall unmoved.
    At four warps per CTA a CTA-wide barrier idles three of them.

ADDRESSING: for one `(nt, e)` the warp's thirty-two lanes cover FOUR token rows
(`lane & 3`) at EIGHT CONTIGUOUS digits each (`lane >> 2`), so one reducing
instruction touches four 32-byte sectors for 128 bytes -- the sector floor.

BOTH SIDES MEET HERE, AND THE WRITE SIDE'S GAIN DISTRIBUTES OVER THE FOUR WARPS.
`(R7)` reads `d_pw_l[j, e] = gw[j] * sum_s (...)`, with the gain outside the sum;
a warp-grain fan-in computes `sum_over_warps (gw[j] * partial_w)` instead, which
is the same value in exact arithmetic and a different rounding in this one.  It is
taken deliberately, and the reason it is not a new liberty is that THE PLANE
ALREADY DOES EXACTLY THIS AT THE OWNER GRAIN: every owner CTA that touches token
`j` reduces its own `gw[j] * partial_owner` into the cell, so the gain has always
been distributed across the owner decomposition.  This distributes it one grain
finer, inside an accumulator whose summation order is nondeterministic by
construction and whose outputs are not bit-reproducible run to run on ANY build.
The read side `(R2)` has no such factor and is unaffected either way.

Measured: Scan B 15.554 -> 13.546 ms at the flagship dense cell, `barrier`
1.39 -> 1.53, `lg_throttle` 0.00 at 22.0 M reducing instructions.

<a id="digit-drain"></a>
### `digit_drain`

THE DRAIN CARRIES THE OUTER LEVEL ALONE, on both sides: the inner level fans
itself in from the reduction.  The gain is applied HERE and not at the reduction,
because the outer level's sum is complete inside one warp -- there is no meeting
to distribute it over, so `(R7)` is honoured exactly on these rows.

<a id="note-l822"></a>
### near line 822

(G5) and (G6) off ONE accumulator: the panel is UNGAINED, so the value
columns times `gw` are `d_v` and the same accumulator dotted with the ones-
column-bearing value row is `d_gw`.  `sum_s dW` is not `d_gw` and is never
formed.  `k = leaf` is the carved axis, so this is the backward's one
cross-warp term -- four rounds through the output staging buffer.

<a id="note-l850"></a>
### near line 850

`k = leaf` is the carved axis, so this is the backward's one cross-warp term:
four warp partials of `[C x cols]` meeting in the output staging buffer.  The
meeting is BARRIER-SEPARATED ROUNDS rather than a shared-memory RED: every
warp writes the same cells, so a RED here is a four-way conflict on all
`cols` of every token and measures worse than the three extra barriers.

<a id="kbwdbuildmagic"></a>
### `kBwdBuildMagic`

THE BACKWARD FAMILY'S OWN DEVICE STAMP.  `cudaFuncGetAttributes` is a property of
a KERNEL, and so is a stale binary; the forward's magic says nothing about these.

<a id="gainsite"></a>
### `GainSite`

the gain's place in a produce: nowhere, folded into the panel (the forward's own
form, which Scan A replays), or staged for the accumulator's TOKEN axis -- which
is what (G6) needs, since `sum_s dW` is not `d_gw` unless the panel is ungained.

<a id="bwdparams"></a>
### `BwdParams`

THE OPERAND PACK IS SHARED so the two scans cannot drift on what a plane means;
each ignores the outputs it does not own.  `vt` and `dn` are `[BH, L, LdD]` bf16,
`(v, 1, 0...)` and `(d_num, d_den, 0...)`.

<a id="near-line-134"></a>
### near line 134

ONE OWNER'S ADDRESSING AND THE BODIES BOTH SCANS SHARE.  What differs -- the walk
direction, which list drives which phase, which slab is held -- lives in the
kernels below and nowhere else.

<a id="bdf"></a>
### `bdF`

THE PANEL RING'S BASES, AS NAMED SCALARS.  A two-element array indexed by the
batch parity is a runtime index into a register array, which is a stack frame;
a select is one instruction and no memory.

<a id="liveness"></a>
### `liveness`

THE LIVENESS PASS, RECOMPUTED rather than saved -- the R4 constraint applied to
the SCHEDULE as well as to the state.  Both scans rebuild the SAME two packed
lists from the same table in the same order, so no list crosses the boundary.

<a id="knl"></a>
### `kNL`

§6: PM::n_l(l) is compile-time (l is the static_for's constexpr index) but the
heuristic can't see through the qualified call, so name it.

<a id="near-line-371"></a>
### near line 371

A padding row is zeroed -- by a zero gain where the gain is on the panel, and by
level 0's zeroed run everywhere else -- so a short batch needs no separate path.

<a id="stageslot"></a>
### `stage_slot`

ONE `cols`-WIDE SLOT OF THE RING, gathered by the driving side's list.  The two
planes have one layout, so one body stages either.

<a id="slabload"></a>
### `slab_load`

THE PAGE SWEEP, in the leaf-major orientation, at CANONICAL atom ids (P80):
atom `a` is warp `a>>1`'s
m-tile `a&1`, because the carve slices the local index at a bit boundary.

<a id="slot"></a>
### `slot`

AN ABSENT ATOM IS SKIPPED, AND WITH IT THE TWO BARRIERS: the slab keeps the
zero `slab_zero` set, which is what a never-written state contributes.

<a id="slot-2"></a>
### `slot`

THE EXIT STORES ONLY PLANNED ATOMS, which is what leaves a skipped slot
carrying exactly what it held.

<a id="channelreadout"></a>
### `channel_readout`

THE GRADIENT TILE: the slab against the batch's `cols`-wide panel, contracted
over CHANNELS -- `dR` in Scan A, `dW` in Scan B, one body.  `[leaf x token]`.

<a id="m0"></a>
### `m0`

THE MASS CHANNEL RIDES THE SAME READOUT, and its PANEL entry is not 1: it is
`d_den` on the seed side and 1 on the value side.  `cols` is not a whole number
of k-tiles, so the term is a broadcast product rather than a fifth k-step.

<a id="digitzero"></a>
### `digit_zero`

THE DIGIT PLANE IS CLEARED AT THE HEAD OF THE BATCH, so the leading rendezvous
publishes it and the reduction below needs no barrier of its own.

<a id="near-line-666"></a>
### near line 666

THE DESCENDING ADJOINT.  Per window: the WRITE side against the adjoint at window
EXIT, then the fold of that window's own reads.  The reverse ordering law is the
mirror of the forward's, and it is what keeps the within-window pairs out.

<a id="near-line-737"></a>
### near line 737

the inner level meets in the GLOBAL plane here too, and `(G7)`'s gain rides
the warp-grain partial: RATIFIED, the plane already distributing that gain
across the owner decomposition.  See `digit_reduce_inner_direct`.

<a id="kfanpasses"></a>
### `kFanPasses`

§6: OW::k* are compile-time constexpr static members but the heuristic can't
see through the qualified name, so alias them locally.

<a id="kqbhalf"></a>
### `kQBHalf`

§6: (kSlabN+1)/2 is compile-time (kSlabN is a file-level constexpr) but the
heuristic only reads a bare identifier, so name it.

<a id="near-line-886"></a>
### near line 886

THE ASCENDING REPLAY.  Per window: the read side against the window's ENTRY state,
then the forward's own fold of that window's writes -- so the slab at the end of
the walk is the forward's `state_out`, which is the rung's cheapest strong gate.

<a id="near-line-948"></a>
### near line 948

the inner level meets the other three warps in the GLOBAL plane and never
touches `sdg`; ungained here, `(R2)` having no factor on the sum.

<a id="kqbhalf-2"></a>
### `kQBHalf`

§6: (kSlabN+1)/2 is compile-time (kSlabN is a file-level constexpr) but the
heuristic only reads a bare identifier, so name it.
