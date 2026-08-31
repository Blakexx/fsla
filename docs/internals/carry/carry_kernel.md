# `carry/carry_kernel.cuh` — the carry body's frame

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`, `carry/box.cuh` and `carry/params.cuh`.
The shared-memory tenants have their own page, `smem_ledger.md`; the host side has
`carry.md`.

Symbols, each at first use. `D` = routing depth (levels). `B_l` = level `l`'s padded digit
count. `N = prod_l B_l` = leaf capacity. `DV` = the padded value width. `BC` = the CTA
box's leaf count. `W` = the window, 512 tokens. `M`/`N_mma`/`K` = the MMA's 16/8/16.
`kSeg` = a segment, 16 tokens. The three shared strides — the transit region's, the state
page image's and the V tile's — are named and sized in
[`smem_ledger.md`](smem_ledger.md).

## The frame, in one picture

```
prologue: the parameter block -> uniform registers; the box's two call-level bits ->
          die fast; the page slots resolved once; the per-side warp sub-box selected
          once from the generated member set; the state slice loaded
for each window:
  [out leg]    publish hi(state) -> COVER SLOT A -> edge -> pull as read-side A fragments
  PHASE 1:     per read batch [ stage(b+1) | produce(b+1) | mma(b) | fan-in(b-1) ]
  [return leg] store hi back -> COVER SLOT B -> edge -> rejoin over the kept lo
  PHASE 2:     per write batch [ stage(b+1) | expand(b+1) | fold-mma(b) ]
  [window line]
exit: the state slice stored, iff this call writes it
```

Every bracketed slot is a NAMED function in the source, whether or not it has a body yet.
The frame is what a schedule is built into; a stage that arrived later and had to invent
its own edges would be a schedule retrofitted, which is the shape this arrangement exists
to prevent.

<a id="edges"></a>
## The three edges, and why there are exactly three

`Edge::kOutLeg`, `Edge::kReturnLeg`, `Edge::kWindowLine`. Each is a NAMED BARRIER id, and
each is crossed AFTER its cover slot: a unit publishes its own contribution, runs work that
depends on nobody, and only then arrives at the barrier it needs the others to have reached.

The split the frame wants cannot be had from a named barrier with EQUAL PARTICIPATION — a
thread that arrives and later syncs on one barrier contributes two arrivals, and the arrive
is a no-op in the pair. It is had from ORDER instead, and the effect is the same: the cover
work is issued by every warp before the barrier, which is exactly what a split arrive bought.
What is not paid is `mbarrier.test_wait`'s SPIN, which was 21% of issue slots where the
streams used it and turned every spin into a shared-memory read; a named barrier sleeps.

* the OUT LEG separates publishing the accumulator's high halves into the transit region
  from reading them back in the other side's carve;
* the RETURN LEG separates the reader writing its high halves back into the region from the
  writer reading them there;
* the WINDOW LINE separates this window's rejoin, which READS the region, from the next
  window's publish, which writes it — and the fold that lies between them touches neither.

Nothing else in the loop is CTA-wide. A segment's producer/consumer handshake belongs to
its STREAM and is a warp-local wait (`cp.async` groups today) or, where the producer and
the consumer are different warps, that stream's own barrier — not the frame's. There are
three such stream edges: the fold's `kFilled` and `kEmpty`, and the readout's `kReadBatch`.

The frame's edges are the ONLY CTA-wide points the frame owns, and each of the three is a
READ-AFTER-WRITE on the region. That is why the region is the transit's ALONE: a phase-local
tenant aliased into it would introduce a WRITE-AFTER-READ the frame does not name, and the
one tenant that does alias — PHASE 1's output blocks — is separated from the pull by the
readout stream's own first crossing and restored by the store-back before the return leg.

The cover slots are load-bearing rather than opportunistic: at one CTA per SM there is no
second CTA to run while this one waits, so every leg cycle is either covered by
construction or exposed. LEG EXPOSURE — uncovered cycles per leg — is a ledger number from
this frame on.

<a id="accumulator"></a>
## The accumulator's orientation is forced, not chosen

`C[m = channel][n = leaf]`, `kMCh` m-tiles by `kNL` n-tiles per warp, `kMCh * kNL * 4`
fp32 registers per lane.

The readout's A fragments ARE the fold accumulator's top halves (KERNEL_STANDARDS §2), and
only this orientation admits that. Written out:

```
fold      C[c][leaf] += sum_t  V^T[c][t]      . Wcoef[t][leaf]     -> C = (m = c, n = leaf)
readout   out^T[c][t] = sum_l  S^T[c][leaf]   . R^T[leaf][t]       -> A = (m = c, k = leaf)
```

The `m16n8k16` accumulator's thread-value map is `lane -> m = lane>>2 (+8), n = 2*(lane&3)
(+1)`; the A operand's is `lane -> m = lane>>2 (+8), k = 2*(lane&3) (+1) (+8)`. They agree
on `m`, so two adjacent n-tiles compose one k-tile and the conversion is a RELAYOUT with no
data movement. The transposed orientation puts the leaf on `lane&3` in `C` and on
`lane>>2` in the B operand: no relayout exists, and the accumulator would have to make a
round trip through memory to become an operand — which `common/design.cuh` already
static_asserts is a different algorithm rather than a slower arm of this one.

The cost of the orientation is that the channel count is padded to a multiple of `M = 16`:
at `DV = 64` the 65 channels (64 values and the mass) occupy 5 m-tiles of 16, so 80 of the
lane's registers hold 65 elements of state. The mass is the deposit value's trailing one,
so it is not a special case anywhere except that its A fragment is a CONSTANT rather than a
loaded tile.

<a id="transit"></a>
## The hybrid transit: the same halves, re-owned

The transit region is `[row = sb_swizzle(local)][col = channel]`, `BC` rows of `kLdX`
halfwords. `local` is the OWNER-LOCAL CANONICAL INDEX (`geom.md#local-row`) — one law that
both sides read, never two maps — and `sb_swizzle` is the region's GF(2)-linear row
permutation. Two facts keep every displacement an immediate:

* `local(strm, P) = local(strm, 0) | scatter(P)`, because a stream's offset at a level and a
  slot's digits at that level occupy DISJOINT bit fields — so the `|` is a `^` and `scatter`
  is linear in `P`'s bits;
* `sb_swizzle` is linear, so `row(strm, P) = row(strm, 0) ^ swizzle(scatter(P))` and the
  `P`-part is a constant wherever `P` is.

The four legs, in the order the frame runs them:

| leg | what moves | how |
|---|---|---|
| publish | the accumulator's high halves, write carve | `st.shared.u16`, 32 lanes covering 4 rows × 8 consecutive channels per issue |
| pull | the same halves as the readout's A operand, read carve | `ldmatrix.trans.x4`, one per (k-step, channel m-tile) |
| store back | the halves the reader still holds, read carve | the pull run backwards, `st.shared.u16` |
| rejoin | the halves as the writer's accumulator, write carve | `ldmatrix.trans.x4`, one per (n-tile PAIR, channel m-tile) |

`ldmatrix.trans` serves both directions because the A operand's thread map and the
accumulator's thread map AGREE on the channel and compose on the leaf: `lane -> m = lane>>2
(+8)` in both, and the accumulator's `n = 2*(lane&3) (+1)` is the A operand's `k` of two
adjacent n-tiles. So one `x4` returns four registers that land exactly on
`acc[m][j][0..3]` and `acc[m][j+1][0..3]`, and the rejoin is `join_hi_lo` over the LOW
halves, which never left the register file and are never truncated.

THE MASS CHANNEL is scalar on all four legs. It opens the last channel m-tile
(`kMassRow == 0`), so an `ldmatrix.x4` at that tile would address the eight channels above the row's width; the four lanes that hold it read and write it as halfwords instead. Every OTHER
element of that m-tile is structurally zero — the fold's mass A fragment has a single live
row and the state sweep never loads a channel at or above `DV + 1` — so the rejoin states
that rather than carrying sixteen zero registers through PHASE 1.

<a id="readout"></a>
## PHASE 1: the readout

`P[m = channel][n = token] += S^T[c][leaf] . Rcoef[leaf][t]` — the fold with two indices
swapped, against the state the window ENTERED with. Per batch of `kSeg` tokens: `kDw` k-steps
(16 leaves each) by `kTokTiles` token n-tiles by `kMCh` channel m-tiles of `mma.m16n8k16`,
into a panel that is dead at the end of the batch.

The read ring is TWO raw slots and it takes its own bytes rather than the region's: cover
slot A fills it while the region still holds the published halves. At iteration `b` the loop
WRITES slot `b % 2` and READS slot `(b + 1) % 2`, and the next iteration swaps them — so ONE
edge a batch carries both the release and the publish, where the fold's single value slot
needs two. Issue order inside a batch:

```
stage(b + 2) -> slot b%2   | claimed, cp.async, never waited on until the batch ends
fan-in(b - 1)              | the CTA drains the other panel into num/den and leaves it zero
mma(b)                     | against the pulled A fragments, in the tensor pipe
reduce(b) -> block b%2     | red.shared.add.f32, 4 rows x 8 consecutive fp32 = 32 banks once
produce(b + 1)             | the next batch's coefficients, in the SIMT pipe under the MMAs
cp.async.wait_group; edge  |
```

<a id="read-expansion"></a>
## The read coefficient needs no transpose

The MMA's B fragment is `k = 2*(lane&3) + 8h (+1)`, `n = lane>>2`. On the READ side `k` is
the LEAF and `n` is the TOKEN — the mirror of the fold, where `k` is the token. So a
register's two elements are two ADJACENT LEAVES at ONE token, and two adjacent leaves are
two adjacent amplitude columns of that token's row, which the packed plane already stores
adjacent. The readout therefore expands from the staged rows AS THEY LIE: one 32-bit shared
access per (token n-tile, k-step, `h`), and no column block, no transpose pass and no gain.

The memo is the fold's j-slice law on the `k` axis. The innermost level's sub-box span is at
least the page rectangle, so `P`'s low four bits ARE `local`'s low four bits and every level
above the innermost has a sub-box shift of at least four — its digit is a function of the
k-step ALONE, uniform across the warp and across `h`. Both column sets are walked once in
the prologue (`rcin`, `rcout`) and never inside a window.

<a id="fan-in"></a>
## The fan-in: one contribution per owner box per token

`num` and `den` are planes the readout REDUCES into, and every owner box of every batch-head
contributes to every token. The warps of a CTA therefore meet FIRST, in a shared panel of
`kSeg` tokens by the page image's own stride, in fp32 (`red.shared.add.f32`), and the CTA then drains that panel into the planes
with one `red.global.add.f32` per cell. The alternative — each warp reaching the plane
directly — issues the same number of instructions with the same sector efficiency, and
`warps_per_cta` times the L2 atomic traffic.

There are TWO panels, ping-ponged a batch apart, so the drain of one and the adds into the
other never meet; the drain leaves the panel it emptied at zero for the batch two ahead, and
only the window's first pair is cleared explicitly. The panel's stride is the page image's,
and its `4 mod 32` congruence makes the adds conflict free for exactly the reason it makes
the page image's reads conflict free (`smem_ledger.md#out-block`).

A token past the window's end carries a zero coefficient by construction, but it has no
cell in the planes — so the drain is the one place in the readout that tests a token against
the window's length.

THE ORDER OF THE CONTRIBUTIONS IS THE HARDWARE'S. `num` and `den` are therefore
associativity-dependent: the same call twice agrees to fp32 reduction-rounding, not bit for
bit. The state plane is not — it is accumulated in registers in a fixed order, which is why
the dense/paged byte gate is a byte gate.

<a id="slot-map"></a>
## From a warp's slot to a leaf, a page and a factor column

A side's warp sub-box is a member of the generated set `sub_boxes(D, BC, streams)`. The
member fixes, per level, how many of the box's span bits this warp keeps (`sub_bits`) and
where they sit in the warp's own slot index (`sub_shift`). `slot_local` composes the
warp's stream offset with the slot's digits into the OWNER-LOCAL CANONICAL INDEX — the same
quantity `geom_xch_row` swizzles for the transit region, so the two are one law and not two.

From that index:

* the PAGE and the ROW split at the page rectangle: `page = local >> 4`, `row = local & 15`.
  This holds because the owner span law gives the innermost level at least the page's four
  bits and no shift below it — asserted in the kernel, not assumed.
* the FACTOR COLUMN of level `l` is `col_base[l] + geom_digit(l)`, the owner's grid digit
  above the box's span and the slot's run below it. It depends on the lane's leaf and not
  on the token, so it is computed once in the prologue and never inside a window.

<a id="state-io"></a>
## The state slice

The box's pages are resolved to slots ONCE, at the prologue, into a shared array of
`BC/16` int32 — so the page table is read `atoms per box` times per launch and never per
fragment. A null table is the dense backing, where the slot IS the page (`ops::page_slot`).

The sweep runs in rounds sized by how many page images the transit region's bytes hold. A
round stages its pages with `state_page::load` — a coalesced 32-bit gather with a `PRMT`
interleave and no other arithmetic — issues one edge, and then every warp lifts its own
fragments out of the images by the slot map above. The exit sweep is the mirror. A page
this call only READS moves its high plane alone; one it will WRITE moves both. A page the
box does not touch at all moves nothing, and an absent slot moves nothing.

The die-fast is the same fact at the box's grain: the OR of the box's page bytes, read
before any shared-memory touch. Both bits clear means this CTA has no work in this call and
returns.

<a id="fold"></a>
## PHASE 2: the fold

The fold is WINDOW-BLIND. A deposit never reads the accumulator, so a whole window's
segment MMAs chain back to back with no accumulator dependency, and the segment buffer's
refill is covered by the previous segment's chain rather than by a barrier. This is the
structural difference from an attention mainloop, which must read-modify-write its
accumulator on every tile.

Per segment, in issue order: the value tile is drained out of the shared slot into A
fragments by a transposing tile load — the segment is stored token major and the operand is
channel major, so the transpose is the load's, not a pass; the NEXT segment's factor rows
are transposed into their column block; the EMPTY edge releases the value slot and
publishes the column block; the segment `kSegmentSlots` ahead is claimed and staged; the
NEXT segment's coefficients are expanded in registers; this segment's deposits issue; and
the FILLED edge publishes the staging to the warps that will drain it.

That order is the schedule's cover, chain by chain. The expansion's shared loads are issued
BEFORE the deposits that do not depend on them, so their latency sits under the MMA chain
rather than in front of it; the staging is not waited on until the next iteration, so it is
covered by an expansion and a whole segment of deposits; and the deposits of segment `b` are
still draining in the tensor pipe while segment `b + 1`'s expansion issues in the SIMT pipe.
The fold's window-blindness is what allows the last of those.

A lane holds one leaf per n-tile and two token PAIRS of the segment, and its coefficient
pair is the gain times the product of that leaf's `D` factor amplitudes, one packed bf16
multiply per level, which is the declared operand arithmetic. A token past the window's end
has zero amplitudes staged and its value quantum zero-fills, so a partial window is the same
code and not a tail case — nothing in the loop tests a token against the end.

The segment slot count is arch-derived: where the MMA sources both operands from the
issuing unit's registers, ONE slot suffices and the registers are the second buffer
(§4's two-stages-in-flight, in its register form); where it does not, the ring is two slots.

<a id="factor-rows"></a>
## What the fill stages, and why it is that little

A CTA box reaches ONE CONTIGUOUS RUN of digits per level — `owner_span(l)` of them, at
`col_base(l) + owner_digit(l) * owner_span(l)` in the packed plane — because the owner grid
fixes the leading digits and the box holds every combination of the rest. So the whole
routing footprint of a segment is `kSeg` tokens times `sum_l owner_span(l)` amplitudes plus
`kSeg` gains: 66 bytes a token at the flagship arm, against the 512-column row a token
occupies. The fill stages exactly those runs, as they lie, with `cp.async` at the widest
quantum each run's length, base and row stride admit.

Nothing is staged per warp. Every warp of the box reads from the same runs (its leaves are a
sub-box of the same box) and every warp's A operand is the same `V^T`, so the segment
carries the values and the factor rows ONCE and the warps differ only in the coefficient
they expand from them.

<a id="fill"></a>
## The fill is claimed, not assigned

A fill UNIT is `kUnitTokens` tokens: their value rows are one sixteen-byte quantum per lane
and their factor runs one eight-byte quantum per lane at the flagship, so a claim is one
lane-parallel burst of each. A warp takes a unit with a single shared `atomicAdd` on the
stream's cursor and keeps taking until the segment is gone, so the fill shares out by
arrival rather than by assignment — which is the property the sparse form needs, where a
warp's units are not all the same size.

The cursor is MONOTONE and never reset. A warp leaves a fill with exactly one failing claim,
so a production point advances the cursor by its unit count plus one per warp, every thread
can carry the next point's base in a register, and no reset — hence no barrier of its own —
is needed.

There is one cursor PER PRODUCTION POINT (`Fill`), and the reason is the split edges: a
cover slot runs between an edge's arrive and its wait, so a warp can be inside the cover
while another has passed the wait and reached the next point. Two points may share a cursor
only when a completed edge separates them; the cover's does not qualify, and the three
points the window's schedule has each carry their own.

<a id="column-block"></a>
## The one transpose, and the gain that rides in it

The MMA's B fragment pairs two ADJACENT TOKENS in one 32-bit register, so a coefficient pair
is one leaf column at two tokens — and the packed plane stores those two amplitudes a whole
row apart. `cp.async` writes what it reads, contiguously, so the staged row cannot be in the
pairing order; the segment's raw rows are therefore transposed ONCE, cooperatively, into a
column block that stores each column's `kSeg` tokens adjacent. It is a 2x2 halfword block per
thread: two 32-bit loads, the halfword shuffles between them, two 32-bit stores, one word per
thread per segment.

The GAIN rides in the same pass. It is per token, and the coefficient needs it exactly once,
so the transpose multiplies the OUTERMOST level's columns by the token pair's gains as it
writes them — one packed multiply per thread instead of one per leaf per lane. The gains
themselves are the one operand a `cp.async` cannot stage: a token's gain is a single bf16 in
a `[BH, L]` plane, so the run's alignment is a property of `BH * L` rather than of the
descriptor. They are read instead as one coalesced row by one warp, a segment ahead, and
stored to shared after that segment's deposits have issued.

<a id="expansion"></a>
## The expansion, memoized

A lane's `kNL` leaf tiles do not use `kNL` different columns of a level. The tile index
contributes the slot's bits ABOVE the lane's three, so a level's column depends only on the
tile bits its span reaches — its J-SLICE — and every tile with the same slice shares one
column. The expansion loads each level's DISTINCT columns once into registers and multiplies
them into the tiles that use them: at the flagship member that is two columns a level rather
than four, eight packed loads a segment rather than sixteen.

The slice is compile time (`slice_shift`/`slice_bits` on the member's own span layout), so
the memo is a register array of a compile-time size and the walk that fills `scol` runs once
in the prologue, per (level, slice) rather than per (level, tile).

<a id="stream-edges"></a>
## The stream's two edges, which are not the frame's

The frame owns three edges and the loop adds no fourth: the two edges below belong to the
STREAM, and a stream edge is what the schedule always said would appear here. FILLED
publishes a segment's staging to every warp that consumes it — a warp waits its OWN
asynchronous copies and the edge makes them visible CTA-wide. EMPTY releases the value slot
and the raw factor slot, and, because the transpose runs between the two, it publishes the
column block in the same crossing.

They are NAMED BARRIERS, not mbarriers, and that is a measured choice rather than a
stylistic one: a stream edge here has EQUAL PARTICIPATION (every thread arrives once) and
runs once a segment rather than once a window, and `mbarrier.test_wait` SPINS where
`bar.sync` sleeps. Measured on the frame's own cells, the spin cost 21% of issue slots to branch resolution and
turned every spin into a shared-memory read; the named barrier costs neither, and it takes no
shared bytes to allocate.

They are what the SHARED value slot costs. Staging the segment per warp needs no edge at all
and is what the frame started from; it also fetches every value row `kWarps` times and holds
`kWarps` copies of it. Two CTA-wide crossings a segment is the price of fetching it once,
and at `kSegmentSlots = 1` both are structural: one releases the slot before its refill, the
other publishes the refill.

<a id="forms"></a>
## The two forms per side, and where each switch sits

Each phase has two bodies, `Form::kUnion` and `Form::kSparse`, and the launch picks one per
side. The switches are ADDITIVE — "two plus two bodies", four in all, not four combinations
— so each sits at its own phase: inside the window loop, outside the batch loop that the
in-kernel-switch ruling calls the hot loop. One uniform branch per phase per window.

The two per-side MEMBER switches are different: they fix the addressing every stage on that
side uses, so they sit in the prologue, outside every loop, nested. That nesting is
MULTIPLICATIVE in the two sides' member counts. At the flagship arm the generated set has
two members per side and the product is four; a deep topology can generate up to twelve per
side, and the product is then the compile budget's problem rather than the schedule's.

<a id="ledger"></a>
## The primitive ledger: where each quantity lives

| class | quantity | where it lives |
|---|---|---|
| 1 arm axes (compile-time free) | `D`, `DV`, `warps_per_cta` | the arm key; template parameters of `carry_kernel`, declared in `tools/manifests/shipped_set.json` |
| 2 the problem (runtime, untuned) | `B_l`, the level modes, `N` | `CarryParams::g`, derived on the host by `derive_carry_geom` |
| 2 the problem (runtime, untuned) | `BH`, `L` | the grid's `y`, `CarryParams::L` |
| 3 runtime dials (the only tuned class) | the form per phase | `CarryParams::readout_form` / `fold_form` |
| 3 runtime dials | the stream grain, the active leaf count, the carve order, the producer:consumer split | not yet consumed; they arrive with the stages that read them |
| 4 arch-deterministic | `M`, `N_mma`, `K` | `common/ops.cuh` |
| 4 arch-deterministic | `warps_per_sm`, the launch table, the cluster size, the state's share of the register file | `common/geom.cuh`, `common/constants.cuh` |
| 4 arch-deterministic | `W`, `kSeg`, the segment slot count | `common/constants.cuh` |
| 5 derived quotients | `leaves_per_sm`, `leaves_per_warp`, `BC`, the atoms per box, the accumulator's tile counts, the shared ledger's rows | `BoxPlan` in `carry/box.cuh` |

A class-4 or class-5 quantity appearing as a knob anywhere is a bug, and that is what makes
the table worth writing down rather than inferring.

<a id="box-plan"></a>
## `BoxPlan`

One `struct` per arm holding every class-5 quantity above as a `constexpr`, with the
static_asserts that make the derivation refuse rather than round: the box is a whole number
of pages, a warp's leaves are a whole number of n-tiles, the window is a whole number of
segments, the page images fit the region, the ledger fits the architecture's per-CTA
maximum. Nothing in it is a literal that a reader has to trust.

<a id="measured"></a>
## What the frame costs, measured

The one probe over four cells, with the counters the stage gate reads. The frame is what
every later stage subtracts, so these are the numbers to subtract.

| cell | duration | IPC | barrier | long scoreboard | short scoreboard | not selected |
|---|---|---|---|---|---|---|
| one deposit, 16 tokens | 15.2 µs | 0.99 | 7.1% | 11.0% | 6.7% | 3.0% |
| dense, 256 tokens | 35.8 µs | 1.40 | 3.6% | 17.8% | 2.8% | 3.8% |
| sparse k=4, 256 tokens | 37.4 µs | 1.38 | 3.6% | 18.3% | 2.9% | 3.7% |
| dense, 1024 tokens, N = 65,536 | 455.9 µs | 1.28 | 1.1% | 33.9% | 1.0% | 2.8% |

Each line, against what the design says it should be:

* **barrier** falls as the window count and the fold's size grow. The three edges and the
  two state sweeps' rendezvous are the only CTA-wide synchronisation in the kernel, so this
  is the whole of it. At this stage the cover slots are empty, so every leg cycle is exposed
  by construction and this number IS the leg exposure — which makes it the thing to watch
  when the legs start moving bytes: covered, it should not grow.
* **long scoreboard** is the dominant line and it grows with the FOLD, not with the state
  I/O. Its seat is the coefficient expansion's per-lane scalar reads of the gain and the
  write amplitudes: the expansion itself is in registers, which is the design's form on
  every architecture, but its FEED is unstaged here. Cooperative staging and a memoized walk
  are what answer it.
* **short scoreboard** and **mio throttle** (≈1%) are the tile loads' own latency.
* **not selected** does not go to zero at one CTA per SM and should not: the SM's eight
  warps still sit two to a scheduler partition, so a warp that loses a cycle to its partner
  is counted here. Three per cent is that share. A second CTA would read very differently,
  which is what makes the counter worth reading at all.
