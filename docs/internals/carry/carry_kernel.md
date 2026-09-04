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
  [out leg]    publish hi(state) -> COVER SLOT A -> edge -> PHASE 1 reads the region as B
  PHASE 1:     per read batch [ stage(b+1) | produce(b+1) | mma(b) | fan-in(b) ]
  [return leg] COVER SLOT B -> edge -> rejoin over the kept lo
  PHASE 2:     per write batch [ stage(b+1) | expand(b+1) | fold-mma(b) ]
  [window line]
exit: the state slice stored, iff this call writes it
```

Every bracketed slot is a NAMED function in the source, whether or not it has a body yet.
The frame is what a schedule is built into; a stage that arrived later and had to invent
its own edges would be a schedule retrofitted, which is the shape this arrangement exists
to prevent.

THE READOUT'S WRITE-OUT RIDES A STREAM BEHIND, and a batch behind: a slice's rows are
staged as their segments complete in the batch's last live stream and drained under the
MMAs that follow -- the next segment's, then the next batch's first stream's -- so the
reductions to the planes never appear as a burst (`#readout`).

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
two such edges: the readout's stream edge `kReadBatch`, and the fold's batch edge `kFoldRound` -- the pool's rows visible before a batch, dead after it.

The frame's edges are the ONLY CTA-wide points the frame owns, and each of the three is a
READ-AFTER-WRITE on the region. That is why the region is the transit's ALONE through PHASE 1: a phase-local tenant aliased
into it would introduce a WRITE-AFTER-READ the frame does not name, so the read pool and
the staging tiles take their own bytes behind it and the readout reads the region in place.

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

The two legs, in the order the frame runs them:

| leg | what moves | how |
|---|---|---|
| publish | the accumulator's high halves, write carve | `st.shared.u16`, 32 lanes covering 4 rows × 8 consecutive channels per issue |
| rejoin | the halves as the writer's accumulator, write carve | `ldmatrix.trans.x4`, one per (n-tile PAIR, channel m-tile) |

Between them the region is READ ONLY, and PHASE 1 reads it as its B operand where it lies
(`#readout`): row = leaf, column = channel, `ldmatrix.trans` at a stream's k-step rows returns
`B[k = leaf][n = channel]`. Nothing pulls the halves into registers and nothing stores them
back; the read pool takes its own bytes behind the region for exactly this reason
(`smem_ledger.md#pool`).

`ldmatrix.trans` serves the rejoin because the B operand's thread map and the accumulator's
thread map AGREE on the channel and compose on the leaf: `lane -> n = lane>>2` in both, and
the accumulator's `n = 2*(lane&3) (+1)` is the B operand's `k` of two adjacent n-tiles. So one `x4` returns four registers that land exactly on
`acc[m][j][0..3]` and `acc[m][j+1][0..3]`, and the rejoin is `join_hi_lo` over the LOW
halves, which never left the register file and are never truncated.

THE MASS CHANNEL is scalar on both legs and in the readout. It opens the last channel m-tile
(`kMassRow == 0`), so an `ldmatrix.x4` at that tile would address the eight channels above the row's width; the four lanes that hold it read and write it as halfwords instead. Every OTHER
element of that m-tile is structurally zero — the fold's mass A fragment has a single live
row and the state sweep never loads a channel at or above `DV + 1` — so the rejoin states
that rather than carrying sixteen zero registers through PHASE 1.

<a id="readout"></a>
<a id="readout-sparse"></a>
## PHASE 1: the readout, at the position carve

`num[t][c] += sum over leaves of Rcoef[t][leaf] . S[leaf][c]` against the state the window
ENTERED with, and `den[t] += sum over leaves of Rcoef[t][leaf] . mass[leaf]`. The read pool
(`#pool`, the read tenant) holds the window's live positions' amplitude rows by rank, in
batches of `kReadPos = warps_per_cta * kSeg` positions; a warp OWNS the sixteen consecutive
positions `[16w, 16w + 16)` of every batch -- its SLICE -- and every channel of their rows,
held in its MMA accumulator (`kMassTile * 2` channel n-tiles and the mass, 36 registers)
for the whole batch. The stream is the loop variable:

```
for each batch:
  fill (cp.async amplitude rows, the token table)  ->  stage_wait  ->  edge
  per warp: the streams live in its slice (stream_count_before at the slice's two ranks)
  for each live stream s:                                   one uniform skip per dead one
    A  = the slice's coefficients for s, gathered ONCE      ldmatrix per inner block, digits per k-step
    for each channel m-tile m, for each k-step i of s:
      B  = the region's rows at s's leaves, m-tile m         ldmatrix.trans, one m-tile ahead
      acc[2m], acc[2m+1] += A . B ; mass += A . B_mass       (B_mass: the region's mass column)
  edge                                                       before the next fill
```

EVERY OPERAND IS AT FULL REUSE. A stream's coefficients are gathered once and drive every
channel n-tile and the mass; a k-step's B rows are read once and drive sixteen positions;
nothing is replicated across warps, because the warps partition the positions and a row is
complete inside one warp. The sum across streams happens in the accumulator, so there is no
pool of partial rows, no read-modify-write and no atom, and a stream's tile is
`(kMassTile * 2 + 1) * kKs` MMAs against roughly the same number of shared-memory
instructions -- the tensor pipe is the readout's binding pipe.

THE REAP is per (stream, slice): a stream with no live position in the slice is one uniform
test and no work. A dead position INSIDE a live stream contributes exact zeros -- its
amplitude at the stream's outer digit is zero by the liveness pass's own predicate -- so
correctness never depends on the lists, and the read side mints none. The grain is coarser
than the list form's (stream, sixteen live tokens): at a random quarter live the slices are
all live and the MMAs run dense; at a sixteenth, most still are. The MMAs are the cheap
resource here and the per-tile cost is a fraction of the list form's, so the form wins or
ties at every density measured (`#cost`); where the tensor pipe were the bound instead, a
list-sourced tile is the same body with a different tile source.

THE ROWS LEAVE ONCE A BATCH, AT THE LINE LAW. On this architecture a `red.global.add.f32`
costs the write path one 128-byte line per line it touches and only a position's 32
consecutive channels is one line, so the fragment (each lane two positions by channel PAIRS)
goes through the warp's STAGING TILE (`Smem::stage`, `[position][channel | mass]` fp32 at a
`DV + 8` stride: the fragment's 64-bit pair stores bank free, a row contiguous) and each
position's row leaves as `DV / 32` reductions plus one for the mass: 33 an instruction-minimum
per sixteen positions, once per batch per warp. THE DRAIN IS PROGRESSIVE, so it never appears
as a burst: a SEGMENT (32 channels, two m-tiles) is final when its second m-tile's last MMA
lands, so in the batch's last live stream each segment but the last is staged as its m-tile
pair ends and its four groups of four positions drain under the two m-tiles after it; the
last segment and the mass are staged after the loop and drain under the NEXT batch's first
live stream's m-tiles; the window's last batch is flushed at the phase's end with nothing to
hide under. The seventeenth staging row holds the slice's token ids twice, by batch parity,
because a batch's own drain and the batch before's are in flight together. A slice past the
window's count carries the dead token and its rows are skipped by a uniform test per
position; every other batch drains untested.

WHY REDUCTIONS, AND WHY THIS FEW. `num` and `den` are planes every owner box of every
batch-head reduces into, and the reduction count is the readout's price on this arch: an
ablation of the list-sourced form at the state carve (the slice form's predecessor) removed
its reductions alone and recovered a quarter of the whole kernel, while removing its MMAs,
its staging stores or its staging loads recovered under 4% together -- the reductions cost
by how many leave the SM, not by issue, and no port on the way to L2 was above 55% while
they did. The state carve issued one per (stream, token) line; this form issues one per
(token) line per CTA per batch, the stream multiplicity fewer: 4.33M -> 0.54M at the dense
flagship cell, and the memory-queue-full stall in the readout from 25% of its samples to 1%.

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

The fold deposits every live token of the window into the accumulator: `C[c][leaf] += sum_t
V^T[c][t] . Wcoef[t][leaf]`, one `mma.m16n8k16` per (channel m-tile, leaf n-tile) per tile of
sixteen tokens. It is WINDOW-BLIND -- a deposit never reads the accumulator, because each
carries its decay-to-window-end factor -- so a window's tiles chain with no dependency on `C`.

Its operands come from THE POOL (`#pool`): the CTA's live positions, in rank order, filled by
their owner warps and gathered by index. Nothing in the fold is per stream in shared memory; a
"stream" is a warp's LIST of live slots, and the reap is the list. The phase runs the window's
live positions in BATCHES of the pool's capacity, `kPoolPos` at a time -- ONE batch whenever
the window's live count fits:

    the union's words and their prefix, once            -- every warp, one lane per round
    for each batch of kPoolPos live ranks:
      pool_fill_round, each round by its owner warp      -- the round's live positions in the batch
      cp.async.wait_group 0; bar.sync kFoldRound         -- every row visible to every warp
      pool_pass per box the warp owns                    -- its list, its tiles, its tail padded
      bar.sync kFoldRound                                -- the rows are dead before the next batch's fill

Two CTA crossings per batch and nothing else CTA-wide inside the phase; no position is
reclaimed inside a batch, so nothing is tracked. THE FOLD'S GRAIN is a launch parameter that SELECTS A BODY, `CarryParams::fold_grain_bits`
through `uniform_switch` at the phase's entry: a warp's box holds `leaves_per_warp / G` stream
boxes and walks them as passes, and inside the chosen body the pass count, the n-tile range and
the k-range are compile-time constants -- no MMA is predicated on the grain. The readout's grain
(`readout_grain_bits`) selects its body the same way.

<a id="pool"></a>
## The pool: filled by rank, gathered by index

Per slot the pool holds the VALUE ROW (`DV` bf16, `kVRowBytes`) and the AMPLITUDE ROW (the
box's `D` runs of the write plane, `kAmpStride` halfwords, laid out as the pool's amplitude rows are,
at `span_base` per level). Row `kPoolPos` is the DEAD ROW: zero, written once per window by
`pool_zero_dead_row` because the pool shares the transit region's bytes, and read by every
entry a padded tile carries.

THE POOL HOLDS THE UNION OF THE WARPS' LIVE POSITIONS, BY RANK. At the cover slot in front
of a phase every warp reads its streams' liveness words once -- one stream per pass of the
side's grain -- and writes each stream's word and its exclusive prefix per round into
`stream_words` and `stream_prefix` (lane `r` holds round `r`, `cover_live_words`); at the
phase's start every warp reads the union off those words (`union_word`) and scans it -- lane
`r` holds round `r`'s word and its exclusive prefix -- so a live position's SLOT is its rank
among the CTA's live positions less the batch's base. A window whose live count fits
`kPoolPos` is one batch; the pool fills as far as there are rows to fill.

THE FILL HAS NO CURSOR: round `r` is copied by warp `r mod kWarps` (`pool_fill_round`), four
live positions per step -- lane group `g` takes the g-th remaining live bit -- and each position
by eight lanes (`pool_copy_position`): the value row's eight quanta and every amplitude run
by `cp.async` at the widest quantum their stride admits -- nothing on the fill's path waits on
a global load. THE GAIN IS THE GATHER'S: cover slot B copies the window's gains into a table
(`stage_gain_table`, `kGainTableOffset`), the fill writes each position's gain beside its rows
(`pool_gain`, from the table by round and lane), and the gather folds the two tokens a register
holds into the outermost run's digit it already multiplies -- one packed multiply per slice
per half, once per tile rather than once per position at a global load's latency. A position
no box lists is never copied.

THE LISTS ARE MINTED BY THE FILL, one per stream, of pool SLOTS: the same walk that copies a
round's positions has every stream's word for the round in hand, so a lane whose position is
in the batch writes its slot into each stream that holds it, at the stream's prefix plus its
rank within the word less the stream's count before the batch (`stream_count_before`: the
union rank's round and lane, read back off the stream's own words). A pass consumes its
stream's list -- `send - sbase` entries, the tail padded to the dead row -- and builds nothing.
A tile is sixteen consecutive entries.

<a id="fill"></a>
<a id="factor-rows"></a>
## The factor rows, as they lie

The pool's amplitude rows are a position's `D` runs of the box's own digits AS THEY LIE in
the packed plane -- `abase[l]`, the level's row base plus the owner's digit, is also the
class-1 table's row base -- copied at the widest `cp.async` quantum the stride admits
(`pool_copy_position`). Both sides' pools share this shape; the write side's pool carries
the gain beside the rows, one halfword a position.

<a id="memo"></a>
## The read memo

Everything the readout's B side needs of the geometry depends on the k-step of the box and
the lane alone: each lane's region row for a k-step's B fragment, the four rows the mass
lanes read, and the amplitude columns (the inner run's block per lane half, the outer runs'
digit). `derive_read_memo` computes them once per launch in the prologue -- warp `K mod
warps` takes k-step `K` -- into three tables behind the live words (`kMemoRowsOffset`,
`kMemoMassOffset`, `kMemoColOffset`; `smem_ledger.md#memo`), and a stream reads its k-steps'
entries at its start: a handful of shared loads in place of the owner-local index law's
integer arithmetic per stream, which was about a hundred instructions of the stream body's
hundred and fifty.

<a id="expansion"></a>
## The j-slice of a level

A warp's n-tiles index a level's digits through `slice_shift` / `slice_bits`: which bits of
the tile index a level's digit depends on, from the sub-box's shift and span at that level. The
column memo `scol[l][v]` is one entry per slice `v`, and tile `j` reads slice `(j >> shift) & mask`.

<a id="stream-edges"></a>
## The readout's stream edge

`StreamEdge::kReadBatch` is the readout's batch edge -- after a batch's fill and before the
next -- a named barrier id disjoint from the frame's (`Edge`).

<a id="pool-tile"></a>
## One tile: gathered fragments

`pool_gather` gathers a tile with `ldmatrix.trans` at per-lane row addresses: lane `l` supplies
row `(l & 7) + ((l >> 4) & 1) * 8` -- its list entry -- for every operand it loads, so ONE list
read serves all of them. The value rows, channel major, give the A fragments (one `ldmatrix.x4`
per channel m-tile, exactly the transposing load the old segment drain issued, now at a
gathered row); the mass m-tile is the constant fragment `mass_frag`. The INNERMOST run gives
the B fragments directly: sixteen tokens by sixteen digits, addressed as the four 8x8 blocks
(tokens 0-7 | 8-15) x (digits 0-7 | 8-15), so n-tile `j`'s inner half is `j & 1` and its two
registers are blocks `j & 1` and `(j & 1) + 2`. That is the token pairing the MMA's B fragment
wants and the hardware does it; there is no transpose.

The OUTER runs are per-token scalars for a leaf tile: for each outer level and each of its
slices (`scol`, indexed as `#expansion` indexes it), the lane reads its token pair's two
halfwords and packs them; one `mul_bf16x2` per fragment register applies the product. At
`D = 2` that is four shared halfword loads and eight packed multiplies per tile.

Then `pool_deposit`: `kMCh x kNL` MMAs, the tile as the k-step -- issued after the NEXT tile's `pool_gather`, so each gather's shared-memory latency sits under twenty MMAs (`Frags`, two sets in flight).

A PARTIAL TILE at a batch's end is padded to the dead row (`pool_pad_tile`) and deposited; the
padding rows contribute exactly zero. That padding is the fold's reap floor: one tile per box
per batch, `16 / kPoolPos` of the batch's positions.


<a id="forms"></a>
## One readout body, and where its switch sits

The readout has ONE body (`#readout`); what the launch picks is its STREAM GRAIN, and the
grain is a body switch (`uniform_switch<kGrainCases>`) that sits at its phase: inside the
window loop, outside the batch loop that the in-kernel-switch ruling calls the hot loop --
one uniform branch per window, each case with its k-step count, stream count and pass
structure as compile-time facts. The fold's pass grain is the same kind of switch. The one-stream point of the family (the grain at the warp sub-box) is what the bench's
`box` schedule name selects.

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
| 3 runtime dials (each selects a compiled body at the phase's entry) | the two sides' stream grains `G` | `CarryParams::readout_grain_bits`, `fold_grain_bits`, as log2 leaves |
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
<a id="cost"></a>
## What the body costs, measured

Flagship arm, sm_86, `tools/probe_cells.py` against the shipped body at the same schedule
(`sparse-g32`), milliseconds; per-PC stall sampling and pipe counters by ncu, dense cell:

| cell | shipped | this body |
|---|---|---|
| flagship-dense | 1.420 | 0.697 |
| flagship-alt-k16 | 0.802 | 0.433 |
| flagship-alt-k4 | 0.395 | 0.206 |
| flagship-cohort-k16 | 0.264 | 0.157 |
| flagship-cohort-k4 | 0.201 | 0.129 |

* THE READOUT IS TENSOR-BOUND, which is the regime the design wants: its top line is the
  MMA (math-pipe throttle 25% of its samples, fixed-latency wait 27%), memory-queue-full
  stalls are 1%, and the tensor pipe runs 26% of the kernel's cycles at two warps a
  scheduler. Instructions per MMA in the readout are about 1.5.
* THE FOLD IS NOW THE LARGER PHASE on the dense cell: 43% of the samples against the
  readout's 51%, its MMA line at 88% math-pipe throttle. The fold's turn is next.
* THE REDUCTIONS are 0.54M on the dense cell, one per (position, line) per CTA per batch,
  against the 4.33M the state carve issued; the L2 atomic units run at 30%.
* THE BATCH EDGE shows on the sparse cells (12% barrier in the readout at alt-k4): a slice's
  live streams differ across warps, and the batch line waits for the longest. That is the
  position carve's imbalance, and it is the one cost the list form did not have.
* THE COVER-WORD WALK reads a level's run of words eight at a time; what remains of it is
  the prologue's share of the sparse cells (11% at alt-k4).

<a id="streams"></a>
## The sparse form: streams keyed on leaf ranges

A STREAM keys on a LEAF RANGE of `G` leaves (the nesting law: stream boxes and accumulator
boxes are power-of-two leaf ranges, one dividing the other). `G` is a launch dial in
`[16, leaves_per_warp]`, so a stream lies inside exactly ONE warp's sub-box. On the WRITE
side that warp's accumulator tiles are the stream's and its list is that warp's to consume;
on the READ side every warp consumes every stream against its own slice of positions
(`#readout`), and the stream's leaf range is what the B rows are read at.

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

THE COMPACTION IS THE POOL'S. Each side has its pool of rows by rank (`#pool`), filled by
the same walk from the side's live words that its cover slot writes (`cover_live_words`:
each stream's word per round and its exclusive prefix). On the write side the walk mints
each stream's list of pool slots as it copies, and a stream's pass walks its list in whole
tiles, the tail padded to the dead row. On the read side the slice IS the tile and the
words answer the one question the readout asks, how many of a stream's positions fall in a
slice (`stream_count_before` at the slice's two ranks). No stream stages, claims or counts
anything else: the batch edges are the phase's only CTA crossings, and there are no counters.

