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
  PHASE 1:     per read batch [ stage(b+1) | produce(b+1) | mma(b) | fan-in(b) ]
  [return leg] store hi back -> COVER SLOT B -> edge -> rejoin over the kept lo
  PHASE 2:     per write batch [ stage(b+1) | expand(b+1) | fold-mma(b) ]
  [window line]
exit: the state slice stored, iff this call writes it
```

Every bracketed slot is a NAMED function in the source, whether or not it has a body yet.
The frame is what a schedule is built into; a stage that arrived later and had to invent
its own edges would be a schedule retrofitted, which is the shape this arrangement exists
to prevent.

THE FAN-IN IS THE ONE SLOT THAT DOES NOT RIDE A BATCH BEHIND, and the picture says `fan-in(b)`
because that is what the source does. Two structural facts forbid the skew, and both are
arithmetic rather than preference: the panel is ONE set of slices, because `warps_per_cta` of
them already fill 34,816 of the region's 36,864 bytes ([`smem_ledger.md#panel`](smem_ledger.md#panel)),
so a second set to deposit into while the first is being summed does not exist at any declared
value width; and skewing would need `po`/`pm` double-buffered, which is 40 more fp32 registers a
lane on a census that has none to give. The cover the readout actually gets is the next batch's
staging in flight and the other warps' drift across the panel line.

<a id="edges"></a>
## The three edges, and why there are exactly three

`Edge::kOutLeg`, `Edge::kReturnLeg`, `Edge::kWindowLine`. Each is a NAMED BARRIER id, and
each is crossed AFTER its cover slot: a unit publishes its own contribution, runs work that
depends on nobody, and only then arrives at the barrier it needs the others to have reached.

The split the frame wants cannot be had from a named barrier with EQUAL PARTICIPATION — a
thread that arrives and later syncs on one barrier contributes two arrivals, and the arrive
is a no-op in the pair. It is had from ORDER instead, and the effect is the same: the cover
work is issued by every warp before the barrier, which is exactly what a split arrive bought.
What is not paid is the spin of the split-barrier object this frame used to carry (deleted,
DELETIONS.md): its wait was a branch loop and a shared-memory read, measured at 21% of issue
slots where the streams used it. A named barrier sleeps.

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
WRITES slot `b % 2` and READS slot `(b + 1) % 2`, and the next iteration swaps them, so the
ring itself needs only the batch line. Issue order inside a batch:

```
stage(b + 2) -> slot b%2   | claimed, cp.async, never waited on until the batch ends
mma(b)                     | against the pulled A fragments, in the tensor pipe
produce(b + 1)             | the next batch's coefficients, in the SIMT pipe under the MMAs
deposit(b) -> slice[warp]  | this warp's panel, written whole, no atom in the path
edge                       | the panel line: the slices are published
fan-in(b)                  | the CTA sums the slices and reduces into num/den
cp.async.wait_group        |
edge                       | the batch line: the ring turns and the panel is free again
```

TWO crossings a batch, which is the fold's count: the panel is ONE set of slices, because a
second set does not fit the region at every declared value width, so the fan-in cannot run a
batch behind the deposits that fill it.

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
contributes to every token. The warps of a CTA therefore meet FIRST, in shared memory, and
the CTA then reduces the sum into the planes with one native `red.global.add.f32` per cell.

THE ALTERNATIVE WAS BUILT AND MEASURED, and it lost. Each warp reaching the plane directly — the
form the SPARSE readout is obliged to use — issues 288 warp-level reductions per CTA per batch
against this form's 33, an 8.73x ratio that is structural and that the counters read back exactly.
Both forms spend four filled 32-byte sectors an issue, so the whole difference is the issue count:
L2 traffic 168 -> 654 MB at the flagship, of which the fan-in's own share is 69 -> 555 MB. It does
not even buy the instructions it appears to: the direct form pays a per-token global address and a
dead-token test PER WARP (IMAD 32, IADD3 11, LOP3 8, LEA 8, and 37 branch instructions a warp a
batch) where the panel computes one address per cell FOR THE WHOLE CTA and reads the eight slices
at a constant stride, so the two sections cost 145.0 and 154.6 instructions a warp a batch — a 6%
saving, against 1.39x the wall time and 3.2x the stall samples. THE PANEL AMORTISES ADDRESSING
ACROSS THE CTA; THE DIRECT FORM REPLICATES IT PER WARP. That is the measured reason this form
stands, and the direct form's price falls with the density rather than being paid flat, which is
why the sparse body is right to take it: below one contribution per token it issues FEWER global
reductions than this floor.

THE WARPS MEET WITHOUT AN ATOM. Each warp writes its own PANEL SLICE whole — an accumulator
element belongs to exactly one lane of exactly one warp, so the slices are disjoint by
construction and nothing contends. The CTA then reads all `warps_per_cta` slices of a cell
and adds them in a fixed order. The shared-memory atomic this began as is not available as a
native operation here: `red.shared.add.f32` assembles to a compare-and-swap SPIN, which the
constitution's no-fallback clause refuses and the stage's ledger measured at roughly seven
and a half instructions per issue.

The panel's TOKEN GRAIN is derived, not chosen: the widest power of two, up to `kSeg`, whose
`warps_per_cta` slices fit the region they alias into (`smem_ledger.md#panel`). At the
flagship arm it is the whole batch and the fan-in runs once per batch; at a value width whose
slices are twice as wide it halves and the fan-in runs twice.

A token past the window's end carries a zero coefficient by construction, but it has no
cell in the planes — so the fan-in is the one place in the readout that tests a token against
the window's length.

THE ORDER OF THE CONTRIBUTIONS ACROSS OWNER BOXES IS THE HARDWARE'S. `num` and `den` are
therefore associativity-dependent: the same call twice agrees to fp32 reduction-rounding, not
bit for bit. The state plane is not — it is accumulated in registers in a fixed order, which
is why the dense/paged byte gate is a byte gate.

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

There is ONE cursor for every production point of both union bodies. Two points may share a
cursor exactly when a completed edge separates them, and in this frame every point is
preceded by one since the last: the covers by their leg's edge, the readout's batches by
`kReadBatch`, the fold's segments by `kFilled`/`kEmpty`. K1 gave each point its own word
while the covers still ran between an edge's arrive and its wait; the covers now run BEFORE
a whole edge (`#edges`), so the five words are one, and the four registers each warp used to
carry across both phases are four the phases' working sets get instead — which is the
difference between a spilling body and a body at 242 registers.

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

They are NAMED BARRIERS, and that is a measured choice rather than a stylistic one: a stream
edge here has EQUAL PARTICIPATION (every thread arrives once) and runs once a segment rather
than once a window, and the split-barrier object this family used to carry (deleted,
DELETIONS.md) WAITED BY SPINNING where `bar.sync` sleeps. Measured on the frame's own cells,
that spin cost 21% of issue slots to branch resolution and turned every spin into a
shared-memory read; the named barrier costs neither, and it takes no shared bytes to
allocate. No spinning wait primitive survives anywhere in this body.

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

THE SPARSE BODIES ARE THE OTHER TWO. `#fold-sparse` and `#readout-sparse` describe them;
what the switch has to be told is that they are bodies, not combinations — the four-way
nesting of the two forms would compile four copies of the window loop and spill (K0's own
finding on the member switches), while two independent binary switches compile four bodies
at additive cost.

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
| 3 runtime dials | the stream grain `G` per side | `CarryParams::readout_grain_bits` / `fold_grain_bits`, as log2 leaves |
| 3 runtime dials | the carve order | `CarryParams::g`, the side's `level_at` |
| 3 runtime dials | the producer:consumer split | not yet consumed; it arrives with the rotation |
| 5 derived quotients | `active_leaves` = `warps_per_cta * G`, live streams = `warps_per_cta`, the stream catalog = `leaves_per_warp / G` | the sparse bodies' loop bounds |
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
## What the body costs, measured

One probe over the fixed cells, and the section ledger the stage gate reads. Sections are
anchored on the frame's own barrier instructions: the three frame edges and the three stream
edges are named barriers with DISTINCT IDS, so a section boundary is a fact of the
disassembly rather than a judgement about it.

| section | instructions | share | stall samples | share |
|---|---|---|---|---|
| prologue + entry sweep | 3,670,016 | 4.8% | 3,855 | 10.0% |
| out leg: publish + cover A | 1,994,752 | 2.6% | 440 | 1.1% |
| out leg: pull | 495,616 | 0.6% | 85 | 0.2% |
| PHASE 1, the readout | 43,319,296 | 56.6% | 17,056 | 44.4% |
| return leg: store-back + cover B | 1,445,376 | 1.9% | 370 | 1.0% |
| return leg: rejoin | 1,218,560 | 1.6% | 295 | 0.8% |
| PHASE 2, the fold | 21,658,624 | 28.3% | 15,647 | 40.7% |
| exit sweep | 2,723,840 | 3.6% | 679 | 1.8% |

Per warp per 16-token batch the two phases are 165 instructions (fold) against 330
(readout), and each issues exactly 20 MMAs — the two phases' arithmetic is the same size,
which is what the orientation says it should be. Everything above 20 is that phase's FEED.

* THE READOUT'S FEED IS ITS CROSS-WARP FAN-IN, and it is structural rather than fixable. The
  readout contracts over the LEAF, the leaf is what the carve divides among warps, so every
  warp produces a full `[tokens × channels]` partial and the CTA sums `warps_per_cta` of
  them. The floor is 32 stores, 32 loads and 28 adds per lane per batch and the measurement
  sits on it; per token it is 2 + 2 + 1.75 and it does not fall with a bigger batch. The
  addressing above that floor is the one line with room in it.
* THE LEGS COST 6.7% OF THE INSTRUCTIONS AND 3.1% OF THE STALLS, for four passes that move
  the box's whole high plane twice through shared memory. The publish and the store-back are
  at their floor of `BC × (DV + 1)` halfwords each way per window; the pull and the rejoin
  move the same bytes in a quarter of the issues, because `ldmatrix` reads sixteen rows an
  instruction.
* LEG EXPOSURE IS ZERO. The three frame edges are crossed once per window per warp and carry
  no measured stall at all; the covers absorb them, which is what the cover slots are for.
  Every remaining barrier cycle belongs to a STREAM edge — the fold's two per segment and
  the readout's two per batch — and the aggregate barrier line FELL from 26.7% of issue
  slots to 20.5% while the crossings per window more than doubled, which is the named
  barrier not spinning.
* SHORT SCOREBOARD and MIO THROTTLE rise (7.1% → 10.9%, 0.8% → 2.8%): the readout's feed is
  shared memory, and the panel puts 32 stores and 32 loads a batch through the LSU pipe on
  top of the fold's traffic.
* BANK CONFLICTS rise with the legs' halfword stores: eight lanes write eight CONSECUTIVE
  halfwords, so two lanes share a 32-bit word and the store is two-way by construction. A
  region column layout that put a lane's two channels adjacent would pair them, and it would
  move the pull's `ldmatrix` — named here, not taken.
* NOT SELECTED stays at 1.5% and should: the SM's eight warps still sit two to a scheduler
  partition. A second CTA would read very differently, which is what makes it worth reading.

<a id="streams"></a>
## The sparse form: streams keyed on leaf ranges

A STREAM keys on a LEAF RANGE of `G` leaves (the nesting law: stream boxes and accumulator
boxes are power-of-two leaf ranges, one dividing the other). `G` is a runtime dial in
`[16, leaves_per_warp]`, so a stream lies inside exactly ONE warp's sub-box: its accumulator
tiles are that warp's registers and its segment is that warp's to fill and to consume. That
is the whole structural difference from the union form, where the accumulator is sharded over
every warp and every warp therefore has to consume every segment.

THE REAP is per (token, stream) PAIR. A token is live for a stream when the stream's leaf
range contains a leaf whose coefficient is nonzero, and because the range is a bit-rectangle
of the owner-local index — a run of digits at each level, independently — that question
factorises exactly:

    live(token, stream) = AND over levels l of ( OR over the run of digits d at level l
                                                of liveness[side][row_base(l) + d][token] )

`stream_masks` folds it once per stream per window, one lane per 32-token word, `sum_l cnt_l`
coalesced word loads (17 at the flagship: sixteen innermost digits and one outer digit). A
dead pair is then never staged, never expanded, never sent and never MMA'd — and the term it
would have contributed was exactly `+0.0f`, because a dead digit's amplitude is exactly zero
by the liveness pass's own predicate. THE REAP CHANGES NO SUM.

THE ACTIVATION is stream-major and symmetric: `warps_per_cta` streams are live at once, one
per warp, and the catalog (`leaves_per_warp / G` streams per warp) is walked in TURNS over
the window. `active_leaves = warps_per_cta * G` is therefore derived here rather than free;
the rotation that makes it free — half the warps producing, half consuming — is a role
assignment on top of this schedule, not a restructure of it.

THE COMPACTION is a ballot. For each 32-token word, `rank = popc(live & lanemask_lt)` is a
lane's position among the live tokens; lanes with `rank < capacity - held` take, claim
`numProduced` with one `atomicAdd`, and stage their token's rows at `slot & (kSeg - 1)`.
A LANE THAT TAKES NO TOKEN MUST NOT ISSUE THE COPY AT ALL: a `cp.async` whose source run is
predicated off still writes its destination — it zero-fills it, which is exactly what the
union body wants for a token past the window's end — and under compaction that destination
belongs to a neighbour. The staging is branched, not predicated; racecheck names the
alternative as a write-write hazard on the value tile.

THE TAIL of a turn is a partial segment. Its unfilled rows are ZEROED — the value rows, the
factor rows and the gains — rather than predicated, because the deposit reads both operands
whole: a zero coefficient over a stale value is a zero product only if the stale value is
finite, which shared memory is not obliged to be, and reading it at all is an initcheck
violation.

<a id="counters"></a>
## The stream counters

Each live stream carries `numProduced` and `numConsumed`, 32-bit shared words counted in
TOKENS. `numProduced` is fetch-added to CLAIM a placement (K1's monotone cursor, generalised
per stream); the consumer starts a segment at `produced - consumed >= kSegmentTokens`, drains
it in one `ldmatrix` burst and fetch-adds `numConsumed` to RELEASE the slot; a producer whose
claim would reach `consumed + capacity` waits, and the capacity IS one segment, because the
register fragments are the second buffer.

THE WAIT PRIMITIVE IS NEITHER A POLLED COUNTER NOR AN MBARRIER. Under warp-private streams
the consumer is the producer, so the only completion the consumer can wait on is its own
`cp.async` group: `cp.async.wait_group 0` and `__syncwarp()`, a warp-local edge with no spin
loop and no CTA crossing. K1's measured spinning wait — 21% of issue slots — is
not merely avoided here but unreachable. The counters are the PLACEMENT and RELEASE contract,
which is what a rotated producer needs from them; the WAIT half becomes the cross-warp edge
when a warp fills a stream it does not consume.

<a id="fold-sparse"></a>
## PHASE 2, the sparse form

Per turn: fold the stream's liveness into 16 word masks; then, per segment, claim and stage,
`cp.async.wait_group` + `__syncwarp`, pad the tail if any, drain the value tile, expand the
coefficient, release, deposit.

THE EXPANSION READS THE RAW ROWS, with no transpose and no column block. The union body's
transpose exists because the B fragment pairs two ADJACENT TOKENS, which the packed plane
stores a row apart; it is worth its cost when eight consumer warps share one block, and it is
not when one warp does. The pair is assembled where it is used instead — two `ld.shared.u16`
and a pack, against the union's one `ld.shared.u32` plus a share of a cooperative pass — and
the gain rides level 0's factor exactly as the transpose applied it.

THE PHASE CROSSES THE CTA ONCE, and that crossing is not a data edge: the rings alias the
transit region, so `StreamEdge::kEmpty` closes the write-after-read against the rejoin at the
phase's opening. Against the union body's two crossings per segment — 64 a window — that is
the stall-budget gate discharged by construction.

THE COVER IS THE OTHER WARPS'. Carrying the drained fragments across the next segment's claim
would hide its copies behind the current segment's deposits, at 24 registers of live range
this box has not got. The sparse form crosses no CTA edge inside the phase, so the eight warps
drift freely and one warp's staging latency is another's issue slots — an overlap the union
body's per-segment barriers make impossible.

<a id="readout-sparse"></a>
## PHASE 1, the sparse form

The same streams against the accumulator instead of into it. The ring carries NO value tile
(the A operand is the transit's registers) and NO column block (the read side's B fragment
pairs two adjacent LEAVES, which the plane stores adjacent), so it is the raw factor rows plus
a 16-entry row of window-relative TOKEN IDS.

THE FAN-IN CANNOT SURVIVE COMPACTION. The panel (`#fan-in`) sums `warps_per_cta` per-warp
slices of one [token x channel] block and issues one contribution per owner box per token —
which works only because every warp's slice is indexed by the SAME token. A compacted stream's
token list is its own, so there is no common row to sum, and the sparse readout deposits
directly: `red.global.add.f32` at `num[tokid[row]][channel]`, four tokens by eight consecutive
channels per issue, four filled 32-byte sectors. The partial is carried a channel m-tile at a
time rather than whole, because a direct deposit can be issued before the next group's MMAs.

THE PRICE IS A DENSITY CROSSOVER, NOT A REGRESSION: one contribution per WARP per live token
instead of one per BOX per token is `warps_per_cta * density` times the union's atomic traffic,
against an instruction count with no panel round (32 shared stores, 32 shared loads and 28
adds a batch) and no panel edges. Above density `1 / warps_per_cta` the union wins that term
and below it the sparse form does; the launch picks the form, and the measurement is reported
per density rather than averaged.
