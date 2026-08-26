# `carry_kernel.cuh` — the K31 box-native carry kernel (rung R1)

The mirror of `csrc/rola/src/carry/carry_kernel.cuh`. A NEW KERNEL FAMILY, built
beside the shipped chunk consumer: nothing under `csrc/rola/src/chunk/` is touched and
no shipped dispatch reaches this file. The shipped arms stay the baseline until this
family beats them cell for cell (`development/plans/K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md`
§6). The ratified study governs; this doc records the construction.

**Symbols.** `D` levels, `B` digits per level, `K` the uniform sub-box span, `M` the
capacity multiplier, `BC` the leaves one owner holds, `W` the global window length in
tokens, `C` the private batch's token tile, `d_v` (`DV`) the value channel count — AN
ARM PARAMETER since K35 S3b, not a constant of the file. `k_tok` (a token's nonzero
digits per level) appears nowhere in the kernel, which is the point: the schedule is
computed, never branched on.

## What it computes, and what it does not

Per window, per owner:

    O^T[:, t] += S^T[:, box] . R~^T[box, t]      the readout, vs state-at-window-start
    S^T[:, box] += V~^T[:, t] . W~[t, box]       the fold, of the window's own writes

`R~` / `W~` are the read/write leaf coefficients restricted to the owner's box, and
`V~` is the value row with the ones column appended (the mass channel rides the state,
so there is one recurrence and one readout). The within-window (INTRA) pairs are NOT
computed here — rung R2 builds that kernel, and R1's conformance adds the contribution
on the host from the fp64 oracle. `inter(W) + intra(W)` is `W`-invariant and equals the
canonical oracle; `tests/k31/test_carry_vs_oracle.py::test_m2_decomposition_is_exact`
is the gate that licenses grading a half-kernel at all.

<a id="the-batch-line"></a>
## The ordering law: all of a window's readouts precede any of its folds

The only ordering law is window-grain read-before-write on the owner's own slots, and
it is a statement about the WINDOW, not about a batch. A structure that read and folded
per batch would let batch 1's readout see batch 0's folded writes — and batch 0 and
batch 1 are inside one window, so those pairs belong to the intra term. They would be
counted twice. The kernel therefore runs two phases per window: PHASE 1 reads every
read batch against the window's entry state, PHASE 2 folds every write batch.

The same argument closes the batch at the window line. Study §2.1 licenses batching
"ACROSS windows until a tile fills — legal precisely between the owner's own-side
events"; a batch spanning windows `w0 < w1` is correct only if it contains no
(read in `w1`, write in `w0`) pair, and a window that is live on both sides generally
has both. So the batch line IS the window line, except across windows where the owner
is side-dead — which is exactly the whole-window-skip harvest and nothing more.

## The two-sided schedule

Per level at most ONE side is sparse, so there is one stored row set per level and a
dense side is absent-means-full. The two sides' CLAUSE SETS are different level
subsets, so the schedule is two-sided in general: two packed lists with independent
counts `nr` and `nw`, and V gathered by the WRITE list. The merged one-list form is not
built; a cell whose two sides happen to share supports simply produces two lists with
identical content.

The liveness vote is a VALUE test — the OR of the owner's `s_l`-wide amplitude run per
level, ANDed over levels, per side. A dense level's OR is nonzero for every token and
drops out of that side's conjunction on its own, so "AND over exactly the levels where
this side is sparse" is obtained with no structure fact carried into the kernel and no
pre-launch branch on a computed statistic. The test is EXACT rather than conservative:
if every level contributes some nonzero digit inside the owner's run, the leaf combining
those digits has a nonzero product, and conversely an all-zero run zeroes every leaf.

Per-side lists make activity airtight: a side's simplex weights are strictly positive on
its own support, so a listed entry is live for that side by construction. There is no
union-superset padding logic anywhere in this file, by design.

The ROWS ARE VOTED AT THE CARVE GRAIN. Each level's owner run is split at the FINER of
the two sides' spans there (`CarveRows`), so a level carved by either side emits
`s_l / min(S^W_l, S^R_l)` times the rows and any coarser range is the OR of the rows it
covers. Nothing in the facts kernel changes: the carve rows are the same
`chunk_facts_tables` pass instantiated at the finer spans. Each side reads its OWN
stream's range at every level it carves, and the owner-grain word — the shared walk's
input and the whole-window skip's — is the OR over the run. The table grows from 48 to
192 rows per side at the flagship's two-sided carve, and the facts pass by 6% of itself
(0.496 → 0.527 ms at `L = 65536`), against a `Σ_l B/s_l` shape that still does not grow
with `N`.

<a id="the-schedule-has-no-barrier"></a>
### The schedule is ONE BALLOT WORD PER LANE, and it has no barrier

A window is `LW = W/32` ballot words, and a lockstep unit has 32 lanes, so GIVE EACH
LANE ONE WORD. A row then costs ONE `LDG.32` per warp, which is cheap enough that every
warp can afford to hold BOTH SIDES' owner-grain ballots — and once it does, the whole
schedule is a warp-private computation of CTA-IDENTICAL inputs and the warps agree by
construction rather than by publication:

* the WHOLE-WINDOW SKIP is `lane_ballot(read-live | write-live) == 0`, CTA-uniform with
  no barrier and no shared word;
* a warp's READ list comes from its own group row and never leaves the warp; its count
  stays in a register;
* the WRITE list is one list with `kWarps` WRITERS. Every warp folds every write entry,
  so the list is shared — and because every warp holds the same ballot, each takes a
  fixed slice of every word's BITS (`32/kWarps` of them) and derives its own offsets
  from the same lane prefix. It is therefore built BALANCED AT ANY DISTRIBUTION, with no
  CTA prefix, no serial section and no warp carrying it alone; the window line is what
  publishes it, and the write count is a lane reduction of a word each warp already has.

What that replaces is a per-WORD walk — `LW` popcounts, `LW` prefix adds and `LW`
predicated stores per list — plus a CTA-wide packed-count exchange across two
rendezvous. The trip count of the scatter is now the window's LIVENESS rather than its
width, which is the property the design needs and the property a dead-word skip could
not deliver.

THE ROWS ARE NOT STAGED COOPERATIVELY, and the reason is the barrier and not the load.
Staging the window's row-words through shared memory once per CTA is the obvious
economy — it takes the per-warp load count from `2*(kOuter + kWarps)` to about one — but
it forces a barrier between the stage and the walk, and every warp waits at it for the
slowest global load with nothing to overlap: measured 0.53 → 1.50 barrier stall per
issue at the scattered cell and 0.46 → 5.00 at the clustered one, against a load term
worth 7% (`rola-scratch/prototypes/k31_listprobe/FINDINGS.md`, A1 vs A2). One `LDG.32`
per row per warp is what makes the walk barrier-free, and that is worth more than the
loads it duplicates.

A token appears in the group lists its innermost live digits fall in — one of them at
93.2% of read entries at the graded cell, two at the rest — and the several partials of a
multi-group entry meet in the global RED, which is associative. The union of the
`kWarps` lists is the single list the owner-grain vote would have produced, EXACTLY,
because the owner row is the OR of the group rows.

The tail of a partial window is a WORD mask rather than a per-lane ballot, and the word
index is clamped to the row buffer's last word: a past-the-end word is read in range and
masked to zero, which keeps the lockstep unit converged without a branch. A lane past
the window's last word reads word zero and is emptied by the same mask.

The list ORDER is part of the contract, not an accident of the implementation: entries
are ascending in token position, which is what makes a batch a contiguous token range
and the fold's accumulation order reproducible. Conformance cannot see a reordering (the
readout is per-row and the fan-in is a sum); the state plane can, and is the instrument.

## The carve: a per-level SPAN VECTOR, and the EXECUTOR MAP over it

A STREAM owns a box and walks its data into tiles; an EXECUTOR runs streams — a warp on
sm_80 (the terminology law, P76). The two counts are INDEPENDENT DIALS: streams are the
in-CTA sparsity dial, warps the execution budget. An arm declares `WPS`, the WARPS PER
WRITE STREAM, and the rest is derived (`CarryGeom`, all `static_assert`ed):

    kStreams = SVW::kStreams     kWarps = kStreams * WPS     SVR::kStreams == kWarps
    kLeafW   = SVW::kLeaf        kLeafE = kLeafW / WPS       kLeafE == SVR::kLeaf
    strm     = the rotated executor index = this warp's READ stream
    wstrm    = strm >> log2 WPS  = its WRITE stream
    col      = strm & (WPS - 1)  = its COLUMN HALF of that stream's leaves

The READ carve is declared at ONE STREAM PER WARP, which is what leaves the readout, the
exchange's pull and the fan-in untouched by the split. At `WPS = 1` the mask is
zero, `col` folds to a constant and every column term below vanishes from the generated
code.

### The COLUMN SPLIT: `WPS` warps folding one stream's rows

With `WPS = 2` both warps of a write stream read the SAME segment rows and fold DISJOINT
halves of that stream's n-tiles. The half is a STATIC choice of n-tiles, not a runtime
test, and it enters the code in exactly three places:

* the accumulator is sized to `kLeafE` (`st[MCH][kLeafE/8]`), so a warp holds half the
  leaves and issues half the MMAs — which is the register lever the card's container
  argument is about;
* `leafW_of` adds `colBase = col * kLeafE` to its private slot, so the publish, the
  reclaim, `page_regs` and the fold all address the right leaves with NO new nest;
* the consumer's coefficient base `myC` carries `colBase`, so the two warps'
  `ldmatrix.trans` pulls read disjoint slot runs of one row.

The per-(token, stream) instruction total is therefore unchanged — warp-count invariance
(P76 F2) — and the extra executors buy issue slots over a fixed instruction total. The
fetch share is `i % kWarps` over ALL executors, so an executor resolves `TPW * kStreams`
candidate `(token, stream)` pairs per batch rather than the whole batch's `kSegRows`.

A stream's box is a per-level SPAN VECTOR `S_l` — the digits of level `l` one stream
holds — so the box is an AFFINE RECTANGLE of digits and contiguity in leaf order is a
property of carving a prefix, never a requirement. The single-level carve is simply the
vector with one non-full entry; there is no second path for it. The two sides carve the
SAME box along DIFFERENT axes, which is why the state evacuates registers at each phase
change (the exchange), and each side's vector is declared per arm in
`CARRY_ARMS_X` — the write side's `SW`, the read side's `SR`.

Constraints, all `static_assert`ed: `s_l % S_l == 0`, every span a power of two, `WPS` a
power of two, and the two declarations agreeing with the executor map —
`SVW::kStreams * WPS == SVR::kStreams == kWarps`, and a column half of the write stream's
box being exactly one read stream's box.

### `d_v` is the THIRD DIAL, and it is a parameter rather than a path (K35 S3b)

`d_v` is a template parameter of both forward bodies. `CarryGeom` names it `kDv` and
everything the width touches falls out of it — `kCols = d_v + 1`, `LdV = d_v + 8`,
`LdO = d_v + 4`, `LdX = d_v + 8`, `MCH = d_v / 16` (the channel m-tiles a warp
holds), `kChunkV = d_v / 8` (a value row's asynchronous quanta) and every byte count
derived from those. Nothing tests the width: there is no `if (d_v == 128)` anywhere, and
the `d_v = 64` arms are register-for-register and byte-for-byte what they were.

The clauses on the width are STRUCTURAL, not a list: the channels tile into whole MMA
m-tiles, a value row is a power-of-two number of sixteen-byte quanta, and a lockstep
unit covers at least one whole row. The CHANNEL-SPLIT body carries one more, and it is
the reason no wide arm takes that body: it gives each of a FIXED four warps exactly one
m-tile of channels, so its width IS its executor count (`d_v == kWarps * 16`) and 128 is
not a declaration it can express. At the wide plane the design's answer is the carved
body at sixteen leaves per warp — a different arm, never a branch.

**What the width costs, per the card's CONTAINERS entry.** Eight resident warps per SM
is the target; at `d_v = 128` sixteen leaves per warp is REQUIRED (the state, its packed
`lo`, the working set and one tile of output fragments are ~250 registers at that box).
Three containers reach it and all three are built at the flagship cell (`W = 384`, ALT):
1x8 at `BC = 128` by a column split (`WPS = 2`, four write streams), 1x8 at `BC = 128`
by eight write streams, and 2x4 at `BC = 64` — the CTA's box halved rather than its warps
doubled, four streams of sixteen leaves under the ruled one-warp-per-stream map, two CTAs
of four warps per SM. The 2x4 row is `(k, m) = (4, 4)` on the flagship topology: spans
`[4, 16]`, 1024 owners.

**Two things the width reaches that the narrow arms never exercise.**

* THE PAGE SWEEP AT A ROW WIDER THAN THE CTA. The state page is `kCols = 129` columns and
  the 2x4 container runs 128 threads, so `kRowStep = kThreads / kCols` is ZERO. The sweep
  is unchanged: its step is a carry, `kPageCarry = kThreads mod kCols` is below `kCols`
  by construction, so the cursor crosses at most one row boundary per step whether the
  step spans several rows or less than one. What was asserted as `kRowStep >= 1` is now
  asserted as the carry's own bound, which is the statement that was always true.
* THE VALUE COPY. `kValPass = kCand * kChunkV / 32` doubles, and it must: the copy moves
  `d_v` channels. It is the one nest the width reaches; every other nest is O(fragments)
  in `MCH` and `NL`, which is what keeps compile time and instruction count linear in the
  data rather than in the leaves.

**One toolchain fact, recorded because it looks like a style choice.** Making `MCH`
dependent (which it becomes the moment `d_v` is a template parameter) SEGFAULTS the
pinned front end — cudafe++, CUDA 12.4.99 — when a lambda parameter's type is a
reference to an array of dependent bound, at PARSE and before any instantiation. So
`pull_read`, `publish_write` and `reload_hi` take their fragment stacks by DEDUCED reference
(`auto&`) rather than as `uint32_t (&)[MCH][4]`. The caller still passes the one declared
stack; nothing about the types is loosened. Do not spell the bound back in.

### The exchange, and its swizzle (K35 S4a)

The two sides carve the same box along different axes, so the state is wanted in two slot
orders per window and an accumulator holds one. EVERY PHASE CHANGE RE-LAYS IT (P76, THE
ORDER LAW), and the form is WRITER-SIDE PLACEMENT: the publisher stores EACH LEAF'S ROW of
channels at that leaf's slot in the CONSUMER's private order, into the consumer's own
contiguous segment; a CTA barrier; the consumer pulls its whole segment into fragments
with `ldmatrix.trans` and the segment is DEAD. Twice per window — `publish_read` /
`pull_read` before the readout, `publish_write` / `reload_hi` after it — over ONE region,
because the two changes are consecutive moments and never co-live.

`XchgSegments` is `[consumer][leaf][channel]` over `BC` rows (the consumers partition the
box on either side, `kWarps * kLeafE == BC`) at row stride `LdX = d_v + 8` — the `kCols`
channels padded to a sixteen-byte quantum, 72 and 136 at the two widths.

* THE PUBLISH is `st.shared.b16` per fragment ELEMENT. The accumulator's `(row, col)` is
  `(channel, private slot)`, so a lane's four elements of one (m-tile, n-tile) are two
  channels eight apart at two ADJACENT slots — and adjacent slots of one order belong to
  different consumers in the other, so nothing can be packed. The `lo` halves the warp
  keeps are packed INSIDE the publish, n-tile by n-tile, so the fp32 accumulator dies tile
  by tile rather than living whole beside them.
* THE ROW MAP IS AFFINE, `XchRow`. A slot's level-`l` digit is `offset_l | field_l` (the
  stream's offset is a multiple of the span, so it sits above the field); the consumer's
  group and private digit are a BIT SPLIT of that digit; the group terms sum into the
  consumer index in mixed radix and the private terms into disjoint fields of the
  consumer's slot. Every sum is an OR, so `row(slot) = row(0) + sum over the slot's SET
  BITS of a per-bit STEP` and every step is COMPILE-TIME — it depends on the two orders
  and the two stream grids, never on which stream is publishing. So the publisher pays
  the base (its digit offsets, its column half, its lane's slot pair) ONCE PER WARP before
  the window loop, and a fragment's row is that base plus a constant while its address is
  a per-warp pointer plus an IMMEDIATE: no per-row multiply, no per-row map.
* THE PULL is `ldmatrix.trans` and nothing else: the consumer's rows ARE its own private
  slots, so its address is an offset and not a map, and one instruction covers a whole
  (k-step, channel m-tile) tile. `[leaf][channel]` is `[k][m]`, which is exactly what the
  transposed atom wants for an A operand, in BOTH directions — the readout's A fragments
  and the fold accumulator's hi halves have the same thread-value map.
* THE MASS is one COLUMN (`chan = d_v`) and cannot be an `ldmatrix` tile: two
  `ld.shared.u16` at the pair's two rows build the one fragment word whose row 0 is the
  mass. Its accumulator's rows 1..15 are identically zero by construction of `ones_a`, so
  moving them would move zeros.

STORE INSTRUCTIONS PER LANE PER CHANGE: `4 * NL * MCH` channel halves plus `2 * NL` mass
halves — 64 + 8 at the 32-leaf `d_v = 64` arms, 32 + 4 at the 16-leaf ones and 64 + 4 at
`d_v = 128`; the pull is `KDw * MCH` `ldmatrix.trans` (8, 4, 8) plus `KDw * 2` mass pairs.
Twice a window, against the retained plane's `2 * NL * MCH` `st.shared.u32` once — the
extra instructions buy the readout's whole tenant back and remove the pair-level pin.

THE SWIZZLE, DERIVED FROM BOTH PATTERNS. `LdX / 2` is `4 (mod 32)` at both widths, so
consecutive rows sit ONE four-bank block apart and a bank is
`(4*row + 4*granule + word) mod 32`.

* THE PULL is serviced eight lanes at a time (eight lanes x 16 B is the 32 banks x 4 B the
  memory delivers per cycle), and a phase is EIGHT CONSECUTIVE rows at ONE granule — so it
  already walks eight distinct blocks and asks only that the swizzle be CONSTANT over an
  aligned group of eight rows.
* THE PUBLISH covers the lane quad's FOUR consumer rows x EIGHT CONSECUTIVE channels, and
  two lanes sharing a 32-bit word are a BROADCAST and not a conflict — so it is four rows
  x four words inside ONE granule, and it needs the four rows in four DIFFERENT blocks.
  Those rows are the consumer-side images of four leaves and the carves scatter them, so
  the ROW INDEX MUST REACH THE GRANULE: `xch_swz(row) = ((row>>3) ^ (row>>4) ^ (row>>6)) &
  7` gives every row bit a distinct nonzero image and is constant over an aligned eight.
* THE MASS is a single element and must spread over its pad granule's four words on its
  own: `xch_mass` puts it at `d_v + 2 * (((row>>3) ^ (row>>5)) & 3)`.

`test_segment_layout.py` models BOTH legs of BOTH changes at every carved arm and pins
MAX-WAY ONE, and it proves the ROUND TRIP where it actually lives: the composite of the
two row maps is the IDENTITY ON LATTICE LEAVES, so a stride, an order or a stream-grid
term that disagreed between publisher and consumer shows up as a leaf in the wrong row.
The device leg is the oracle at every arm — the state plane is exactly the readout->fold
change's round trip and `num`/`den` are the fold->readout one.

### The straddle: three moments, one region (K35 S4a)

Allocation is the MAXIMUM OVER MOMENTS and never a sum. Below the window's INDEX — the
lists, the published ballots, the prefix counts and the liveness mask, live in every
moment and nobody's straddle — there are three sets of tenants and NO TWO OF THEM ARE
EVER CO-LIVE:

    FOLD     the per-stream segments, the rolling staging, the gain, the page buffer
    CHANGE   the exchange segments          (dead the moment every consumer has pulled)
    READOUT  the reader segments, the read staging, the batch's row index, the ring

so all three begin at the region's base and the allocation carries `straddle_bytes =
max(...)`. What this costs is ONE extra CTA barrier per change — after the pull, before
any warp writes the readout's tenants into the same bytes — and what it buys is the
plane's 18.7 / 37 KB leaving the READOUT moment entirely.

The SEGMENT's value rows are the same story at `LdV = 136`: a 128-bit shared access is
serviced EIGHT LANES AT A TIME (eight lanes × 16 B is the 32 banks × 4 B the memory
delivers per cycle), and a group's eight quanta cover all thirty-two banks exactly once
whatever word the row starts at. The wide row is two such groups rather than one, which
is access WIDTH and not a conflict; the consumer's `ldmatrix` leg is conflict-free
because `136 * 2 / 4 = 68` is `4 (mod 32)` exactly as `72 * 2 / 4 = 36` was.

### The private order, and why it is a RULE and not a choice

A stream numbers its leaves in a MIXED RADIX over its own spans, `p = sum_l r_l <<
shift(l)`, so `p` is a bit permutation of the digit offsets exactly as `local` is and the
expansion is a handful of amplitude loads times products at CONSTANT indices. A span-1 level
contributes a constant and drops out of the order on its own. The order itself is fixed
by the span vector:

**the levels this side CARVES go OUTERMOST**, keeping their lattice order among
themselves, and **the levels it spans whole go inside them in lattice order**. NO LEVEL IS
PINNED (K35 S4a): the retained plane published `bf16x2` words of two LATTICE-ADJACENT
leaves, which forced the lattice's innermost level to lie lowest in both private orders
and its span to be at least two; writer-side placement stores each leaf on its own, so the
carve vectors are fully general on both sides.

The PAIR LEVEL — which seeds the packed pairs and carries the dead-row zeroing — is
therefore simply the innermost level OF THE ORDER: a lane mints two adjacent private slots
at once and they differ in that level's digit, so its two amplitudes are adjacent in the
staged run and one four-byte access covers the pair. Levels of span one occupy no bits, so
the level wanted is the lowest with a span of at least two, and a box holds at least two
leaves, so exactly one qualifies. It is a statement about the ORDER only; the lattice is
not consulted.

A side carves where its declaration marks it sparse, so this is the PRIVATE-ORDER RULE
in the vector's vocabulary — *dense levels innermost, sparse levels outermost*: an
eight-slot n-tile spans dense digits, and each sparse level indexes WHOLE tiles, where a
digit skip is exact. The order is not cosmetic: it is the readout's MMA **k-order**, so
it fixes the fp32 summation order and therefore the bits.

Dropping the pin MOVES the read order at every carved arm, and no write order at all
(their carved level is never the lattice's innermost): at the `D = 2` ALT arms the read
order becomes `0` innermost with the carved `1` outermost where the pin had `1` innermost,
and at `deep4` / `carve4` it becomes `2,1,0,3` and `2,0,3,1`. That order is the readout's
MMA **k-order**, so it changes the fp32 summation order and therefore `num`/`den` in the
last bits; the FOLD's arithmetic is per-leaf and per-level and does not move, so the state
plane stays bit-identical. The oracle is the gate.

### Per-level coordinates

The stream index decomposes in MIXED RADIX over the per-level stream grids, MSB-first
exactly as the owner index does, giving `grp_l` at every level. Its digit offset
`grp_l * S_l` is added to that level's amplitude column (`lvCol`/`lvColW`) and to that
level's digit in the leaf map (`leaf_of`/`leafW_of`) — at EVERY carved level, each side
under its own vector. Both grids are compile-time powers of two, so every use site is a
shift and a mask, and the offsets are computed ONCE per warp before the window loop.

The stream a warp takes is ROTATED per owner, `strm = (warp + ((o + (o>>5)) & (W-1))) &
(W-1)`. The obvious `o mod kWarps` is defeated: the two CTAs co-resident on an SM are one
grid wave apart and the wave is a multiple of the warp count, so both would rotate
identically and still collide on one processing block. The wave-stride form costs three
setup instructions once per kernel and nothing in any loop.

### Liveness is ONE conjunction, and it is exact at any carve

The ballot table is refined at every level EITHER side carves — the per-level MINIMUM of
the two vectors (`CarveRows`) — and stays sigma-shaped, `sum_l B/s_l` rows, so it does
not grow with `N`. A digit RANGE's word is the OR of the table rows it covers, and the
levels are ANDed. That is EXACT, because a rectangle of digits holds a live leaf iff
every level's range holds a live digit — which is the same statement that makes an
n-tile skip exact once tile-grain rows exist.

Two ranges are asked for per side: the STREAM's own (its ownership test, which compacts
the fold onto the tokens that reach its leaves) and the OWNER's whole run (the shared
walk's derivation and the whole-window skip). The conjunction is PRUNED BY THE
DECLARATION: a side conjoins a level only where that side is declared sparse, because a
dense side's row is all-ones and ANDing it is a load for nothing.

`CV` survives on the SEAM — the dispatch's `carve` scalar — so a caller asking for a
carve this kernel does not build is refused rather than silently served another one.

## Two bodies, and the DECLARATION chooses between them

The carry family has TWO kernels and the arm's declared carve selects one of them at
compile time (K33 law 3; `csrc/rola/src/carry/carry.cu`'s `kCarves`). This file holds the
CARVED body; [`carry_split_kernel.md`](carry_split_kernel.md) holds the CHANNEL-SPLIT
body, which holds the whole box in every warp and splits the channels instead. Neither is
a fallback for the other: the axis is resolved at the seam and never appears at run time,
so an arm declared with full spans on both sides runs a kernel with no carve in it at all
rather than a carved kernel with the carve turned off.

The seam's rule for an unbuilt declaration is DENSE_BOTH, which is always correct and
only ever slower. The TOPOLOGY is never degraded: an unbuilt `(D, B, k, m, W, C)` is
refused outright.

## The state source is one seam

The readout's A fragments for a k-step come from `state_operand` and from nowhere else:
`MCH` value m-tiles and the mass row, packed out of the warp's own accumulator by
`pack_bf16x2`. It is the only place the drain touches the state's RESIDENCY, so a state
held somewhere else substitutes its own load there and no other line of the drain moves.

## The atom walk's shape is a register-indexing fact

A state fragment is a register array and a runtime index into one is a stack frame —
measured, on this kernel's first build: walking `kAtoms` at runtime put the whole state
(320 B/thread) in local memory. So the walk is FULLY UNROLLED over the box's atoms, and
so are the two loops inside it, because the accumulator element that carries a given
leaf of a given page is a compile-time function of `(atom, group digit, outer)`. A page
holds `kAtomLeaves/s_{D-1}` outer runs times EVERY digit of the carve level, so each warp
owns `kGS` × that many of its own leaves in every page and the CTA still moves exactly
one page per atom. The same measured law governs every expansion's slot loops: they are
`static_for` and not `#pragma unroll`, because with a runtime slot index the level maps
are evaluated at RUNTIME and the amplitude run lands in local memory — observed at 64-67
local accesses per deep-arm body before the loops were made compile-time.

## The cooperative fetch (K35 F2/F4): one union walk, writer-side placement

BOTH SIDES DO THIS. What follows states the FOLD's, which came first (S2); the readout's
is its mirror and is stated under "The read side's cooperative fetch" below, along with
the one fact of shape that differs.

THE FOLD'S OPERAND PATH IS NOT PER-STREAM ANY MORE. The CTA walks the UNION of its
streams' write lists ONCE — that list is `wlistS`, which the shared derivation already
builds. Warp `warp` takes the fixed share `i % kWarps` of the union, IN ASCENDING UNION
ORDER (no dynamic assignment, no cursor), and for every token it handles and every STREAM
that wants that token it places, into that stream's segment, BOTH

* the value row, by `cp.async` (the copy never touches registers), and
* the FULLY EXPANDED coefficient row for that stream's box — `kLeafW` bf16 in the
  stream's private order, gain applied.

The consumer does NO produce: tile skip → `ldmatrix` → HMMA. Each stream's row is
minted ONCE, so the total multiply count is unchanged; what is removed is every warp
re-loading and re-expanding the factor runs the other warps also want.

THERE IS NO BATCH AND NO BARRIER INSIDE THE WALK (K35 S2c, P76 CORRECTION-2). A stream's
rows ACCUMULATE across the walk and the stream fires when ITS OWN segment is full, so it
issues exactly `ceil(n_s / kSegRows)` k-steps per window instead of one per union chunk it
appears in. The walk still advances in GROUPS of `kSegRows` union tokens, and that is a
FETCH QUANTUM and nothing else: it is one rolling staging block and one lane map over
`kCand` candidate pairs, it is PER WARP, and no stream's firing is tied to it.

### The rank IS the address, and the bound is the type's

The union list is ascending in token and a stream's compacted list is a subsequence of
it, so `rank(t, s)` — the token's position in stream `s`'s list — is the PREFIX POPCOUNT
of that stream's write ballot below `t`. Each executor therefore publishes, once per
window, its own ballot word per lane (`kStreams × kLW` words) together with the per-word
prefix popcount; two shared loads and a `popc` turn any (token, stream) pair into its
rank, and the fetcher needs no knowledge of a stream it does not run beyond those.

THE ADDRESS IS THE RUNNING RANK: the segment row is `rank mod kSegRows` and the TILE is
`rank >> log2 kSegRows`. The segment bound is the type's — `kSegRows == CP`, a
`static_assert` — and there is no truncation path and no runtime test, because the rank is
reduced modulo the segment's own height and the flow counters below are what keep a row
from being overwritten before it is read.

The shared loads of a rank are paid ONCE PER PAIR: lane `q < kCand` of each warp owns
candidate pair `q = (i, s)`, resolves it, and every lane that fills that row reads the
result by shuffle.

### Coverage is by construction

`0 × NaN = NaN`: a slot row the MMA reads must never be arbitrary shared memory, and an
accounting argument about which rows a partial batch left dead is exactly the kind of
argument that failed before. THE DEAD-ROW LAW: a dead row is a FINITE (stale) value row
with a ZERO coefficient row — finite operand, zero weight, contributing exactly nothing —
and coverage is a ONCE-PER-WINDOW clear of the whole segment plus ONE zeroing of the dead
COEFFICIENT rows, at the window's LAST, PARTIAL tile. The window's clear is what makes
every row finite (the published exchange segments straddle these very bytes); thereafter a
value row is either this tile's placement or an earlier tile's, both finite. EVERY TILE
BUT THE LAST IS FULL BY CONSTRUCTION — that is what "fires when full" means — so there is
nothing dead in it and the per-batch clear of the batch form is gone with the batch.

The two clears split over a stream's `WPS` warps DIFFERENTLY, and that is a correctness
fact rather than a preference:

* the WINDOW clear splits BY ROWS. It is followed by a rendezvous before any placement,
  so any partition of the rows is legal and a row split keeps both legs contiguous.
* the LAST TILE's dead-row clear splits BY COLUMNS. A warp may only zero bytes IT ITSELF
  reads — the stream's other executor holds the other column half and reads it. A column
  half of a row is `kLeafE` slots, contiguous and sixteen-byte aligned, so the run is a
  predicated `STS.128` nest with a row stride carried per quantum.

No row's liveness is computed twice and no row is ever unwritten.

### The expansion, and its declared order

Under the private-order rule the levels a side spans WHOLE lie lowest in `p` and the
levels it CARVES lie outermost, so the product splits cleanly:

    memo  = gain, then × f_l for every level the side SPANS WHOLE, ascending l
    w     = memo, then × f_l for every level the side CARVES,      ascending l

`memo` does not depend on the stream, so it is minted ONCE per (token, lane) and reused
by every stream that wants the token — that is the sub-product memo, and its shape is
the span vector's, at compile time. The level lying LOWEST in the order
(`PMW::kPairLevel`) supplies BOTH slots of a packed pair with one four-byte access, because its two
digits are adjacent; every other level's digit is constant across the pair and is
splatted. At the `D = 2` ALT arms that is ONE shared access and ONE `mul_bf16x2` per
(token, stream) row per lane.

THE ORDER IS DECLARED. It is not the order of the private per-warp expansion the
cooperative fetch replaced (which multiplied the pair level first and the gain last), so
the coefficients are not bit-identical to it; the fp64 oracle is the gate.

### The rolling factor staging

The fetcher expands from SHARED MEMORY: the compressed factor rows of a batch's union
tokens are `cp.async`ed ONE BATCH AHEAD into `StagedFactorRow`, so no per-token global
factor load ever appears in the fetch loop. A staged row is the OWNER's digit run at
each level — `Σ_l s_l` amplitudes, not the topology-wide `kWtot` packed row — laid out
level by level with each run aligned to its own asynchronous quantum (4, 8 or 16 bytes);
the alignment holds by construction because a level's column base is a multiple of `B`
and an owner's run starts at a multiple of `s_l`.

IT IS PER WARP (K35 S2c). A warp expands exactly the rows of its own share `i % kWarps`
of the union, so it stages exactly those, and the shares are disjoint — which is what lets
the fold's walk run with no barrier inside it at all. The block's LAYOUT is unchanged
(`[slot][row]`, the row being the union index inside the group), so the two sides still
share one body and one round trip.

IT IS A ROLLING BLOCK AND NOT THE WHOLE WINDOW. The card's ledger line stages `W × kWtot`
at window start; at `W = 512` the owner-run form of that is 24.5 KB, which costs the
second CTA on the RTX. Two blocks of `kSegRows` rows is 1.5 KB, is independent of `W`,
loads strictly fewer rows (only the union's tokens, never a dead one), and keeps the
property the staging exists for — the expansion carries no global latency. The gain
rides the same block: its load is issued at the TOP of a fetch phase and its store at
the BOTTOM, so the whole expansion covers it.

### The per-row tile-live mask

The EXPANSION knows which slots it wrote nonzero, so it emits the liveness itself: one
ballot names every live packed pair of a pass's rows, a byte of that ballot is an n-tile
pair, and the bits are OR-ed across the batch by `red.shared.or`. The consumer skips an
n-tile pair whose batch OR-bit is zero — both its `ldmatrix` and its MMAs — and the skip
is EXACT: a tile with no live leaf contributes nothing. No facts-table row is involved,
and the grain is the MMA's own.

The mask word is per EXECUTOR, not per stream, so that A WARP RESETS ONLY WHAT IT READS:
the reset sits between a warp's own read of the mask and the `drained` bump that lets the
fetchers OR the NEXT tile's bits into it, and a word shared with the stream's other warp
would be reset out from under that warp's read. The `WPS` words of one stream carry the SAME bits — one
lane per column half publishes them, which is the single predicated `or.shared` the
one-word form issued, with more lanes live rather than more instructions — and each
consumer tests only its own field, `[col*kNJ, (col+1)*kNJ)`, through two constants
computed once per warp.

## The flow counters: streams fire when full (K35 S2c)

THE TWO FACTS ARE TWO COUNTERS AND NOT TWO BARRIERS. `landed[s]` counts the rows of write
stream `s` that have ARRIVED in this window; `drained[e]` counts the tiles executor `e` has
PULLED. Both are monotone within a window, both are read by warps that do not write them,
and on sm_90+ the pair IS an mbarrier's arrive/wait — this is the portable form of the
same two facts.

    FETCHER   place rank `r` of stream `s` only while `r >> log2 kSegRows <= min_e drained[e]`
              then, after `cp.async.wait_group` and `__threadfence_block`:
              `red.shared.add.u32 landed[s], (rows placed)`   — one lane per stream touched
    CONSUMER  pull tile `k = drained[e]` when `landed[s] >= (k + 1) * kSegRows`
              `ldmatrix` → reset the mask word → `__syncwarp` → fence → publish `drained[e]`

WHY A COUNT IS AS GOOD AS A PREFIX. Placements go only into tiles `<= min drained <=
drained[e]`, so every row `landed` counts lies in tiles `[0, drained[e]]`, whose capacity
is `(drained[e] + 1) * kSegRows` — the count reaches that bound ONLY when every one of
those rows has arrived. That is what lets the arrival be one `red.shared.add` instead of a
prefix structure.

DEADLOCK FREEDOM, and the code is written in this order:

* a warp PUBLISHES `landed` for everything it has issued BEFORE it waits on any `drained`.
  The group's placements are therefore made in ROUNDS of what there is room for: a group
  spans `kSegRows` union tokens and can straddle a stream's tile boundary, so an
  all-or-nothing group would have a warp wait for a tile to be drained on rows it is
  itself holding back. A stream's ranks inside one group span at most two tiles, so there
  are at most two rounds and one in the ordinary case;
* every wait SERVICES the waiting warp's own stream — it keeps draining and folding — so
  the executor another warp waits on is never itself stuck behind that other warp;
* a warp that has finished its share keeps servicing until the `done` count reaches
  `kWarps`, because a warp parked at a barrier drains nothing;
* the walk is in ascending union order, which is what makes the minimal-position argument
  go through: every row of the tile the least-advanced blocked warp waits on lies at a
  lower union position, so it has been issued and published by every warp, the tile is
  complete, and its executor drains it.

THE WALK IS ONE LOOP WITH THE FOLD AT ITS HEAD, and every wait in its body is a `continue`
back to that head. That is what makes the servicing rule structural rather than a habit,
and it is also why exactly ONE `pull_and_fold` and ONE `expand` reach the generated code:
a second inlining of the fold costs 16 registers and spills at the two `D = 4` carved arms,
measured.

THE WINDOW'S EDGE is ONE barrier. After it `landed[s] == n_s` is final and AT MOST ONE
TILE is left undrained — a fetcher cannot write tile `k+1` before every executor has
drained tile `k`, so `n_s <= (drained + 1) * kSegRows` holds there. That tile is the
window's last, its dead coefficient rows are zeroed then, and the stream has fired exactly
`ceil(n_s / kSegRows)` times.

## The segments, and the two facts of the fold's sync

`SegValueRows` is `[stream][row][channel]` at row stride `LdV = d_v + 8`;
`SegCoeffRows` is `[stream][row][slot]` at row stride `LdS = kLeafW + 8` — the WRITE
STREAM's whole row, which is what the `WPS` warps of that stream share. The readout's own
panels are a separate tile at `LdR = kLeafE + 8`, the READ stream's width. They are
adjacent halves of one per-stream segment and separate only because `ldmatrix` reads
each at its own row stride. ONE SLOT PER STREAM: the second stage is REGISTERS (F4) —
`ldmatrix` drains the slot, the `__syncwarp` after it is the drained fact, the slot is
refilled for the next batch, and the MMAs run from fragments underneath the copies just
issued. A second SMEM slot is the declared fallback arm and is not built.

BANK BEHAVIOUR, derived from both access patterns and gated in
`tests/k31/test_segment_layout.py`:

| leg | pattern | max-way |
|---|---|---|
| fetcher, value (`cp.async` 16 B, 8 lanes to a row) | a row's eight quanta cover all 32 banks once | 1 |
| consumer, value (`ldmatrix.trans` at `rT/cT`) | `LdV = 72` words puts consecutive rows 4 banks apart | 1 |
| consumer, coefficient (`ldmatrix.trans` at `rF/cF`) | `20·lr mod 32` is a permutation of the 8 bank quads | 1 |
| fetcher, coefficient (`st.shared.u32`, two rows at once) | two 16-word rows share their banks or do not | 2 |

The two-way store is the FLOOR and not a defect: a `kLeafW/2`-word row occupies
`kLeafW/2` banks by definition, so two rows written in one instruction cannot be made
disjoint at every placement, and the alternative — one row per instruction with half the
lanes idle — costs the same two cycles UNCONDITIONALLY. The pair mapping is never worse.

THE FOLD CARRIES THE TWO FACTS IN COUNTERS, NOT BARRIERS (K35 S2c, above): "landed" is
all-fetchers → one consumer and "drained" is one consumer → all fetchers, and with a
single slot neither folds into the other — but neither is CTA-wide, so neither is a
`__syncthreads`. The readout still carries them as the batch form's two barriers.

## The barriers: the window line, the phase change, and the fold's two per window

THE SCHEDULE HAS NO BARRIER AT ALL (above), so outside the fold the CTA rendezvouses
only at the phase change and at the window line, and a window the owner is dead in costs
zero barriers. THE FOLD'S WALK HAS NONE EITHER (K35 S2c): it opens on one — the segment
clear and the counters' zero — and closes on one, and between them the flow counters carry
everything.

THE READ PHASE IS FORM (b) TOO (K35 S4b). Its fetch is cooperative and its fan-in meets
in shared memory, so it carries the fold's two barriers per batch and the same two facts —
LANDED and DRAINED — plus one at the end of the window's readouts, before the ring is
drained, that says every contribution has landed.

| edge | carried by |
|---|---|
| every fetcher's expansion of read batch `b` → the reader's drain of it | the LANDED rendezvous |
| a reader's drain and dead-row clear of `b` → any fetcher's writes of `b+1` | the DRAINED rendezvous |
| every warp's ring contributions → the drain of the ring's last `C` rows | the readout's closing rendezvous |
| a contributor's `red.shared` → the reclaimer that flushes that row | `publish_fence` before `arrived`, and the CAS |
| a reclaimer's clear → every contributor to the row it claimed | `publish_fence` before the tag, and the tag load |
| every warp's reads → any warp's folds of the same window | the WINDOW-LINE rendezvous |
| the segment clear and the counters' zero → any fetcher's placement | the fold's OPENING rendezvous |
| every fetcher's placement of a tile → the stream's pull of it | `cp.async.wait_group` + `publish_fence` + `landed[s]` |
| a stream's pull of tile `k` → any fetcher's writes of tile `k+1` | `__syncwarp` + `publish_fence` + `drained[e]` |
| every warp's last placement → the last, partial tile's pull | the `done` count, then the fold's CLOSING rendezvous |
| the staged factor block for group `g+1` → the expansion that reads it at `g+1` | `cp.async.wait_group` in the same warp (the block is that warp's own) |
| the published ballots and prefixes → the fetch that ranks against them | the phase change's rendezvous |
| the window's list writes → the previous window's list reads | the WINDOW-LINE rendezvous of the previous window |

The mass channel is a channel like any other under this carve: the warp holds its own
leaves' mass partial and REDs it to `den` itself, so no cross-warp sum of the denominator
exists.

### The READOUT still carries the batch form, and the reason is the ring

The same mechanism is NOT on the read side, and that is a design question rather than an
omission. The fan-in ring below is `C = 32` rows keyed on the UNION READ RANK and never
evicts an incomplete row — so it is correct only while the union-rank SKEW BETWEEN READERS
stays under `C`, and the union BATCH is exactly what bounded that. Per-stream firing does
not: a reader's sixteen own tokens span an arbitrary stretch of the union list, hundreds of
ranks at scattered routing. Worse, the resulting spin is a DEADLOCK and not slowness —
every warp is both a reader and a fetcher, so a warp blocked in `ring_acquire` still owes
the placements another reader's tile needs to fill. The fold has no such cycle because its
consumer writes only its own accumulator. The fix is a NON-BLOCKING acquire (an incomplete
row may be evicted, the flush being an `atomicAdd`, at the price of an in-flight count and
a direct-to-global arm for the contributor that loses its slot), which changes P76 §2.6-C's
protocol and is Blake's call.

## The read side's cooperative fetch (K35 S4b)

THE READ SIDE IS THE WRITE SIDE'S MIRROR, and the private per-warp expansion, the
two-slot read panel and the per-warp compacted read list are DELETED with it. What differs is one fact
of shape and everything follows from it: on the readout the B operand is
`R~^T [leaves × tokens]`, so the LEAVES are the contraction axis and the TOKENS are `n`,
in eights. A reader's segment is therefore coefficient rows and nothing else — the A
operand is the state, already in registers — and the n-tile the fold skips by LEAF the
readout skips by TOKEN.

A BATCH is `kSegRows` consecutive tokens of the CTA's union READ list; executor `warp`
takes the fixed share `i % kWarps` of them, and for each READER that reads a token it
handles it expands that reader's `R` row — `kLeafE` bf16 in the reader's private order —
into that reader's segment at `rank(t, reader)`. Each warp IS a reader stream
(`SVR::kStreams == kWarps`, the executor map's declaration), so a reader owns its whole
segment and the ranks come out of the reader's own published ballot exactly as the write
side's do. `kLists` is 2: neither side keeps a per-stream list any more.

THE EXPANSION'S DECLARED ORDER on this side is the write side's without the gain: the
levels the read carve spans whole ascending (the MEMO, shared by every reader of the
token), then the levels it carves ascending. The read side's scalar is identically one,
so THE FIRST FACTOR OF THE ORDER IS THE PRODUCT'S SEED (`first_whole_level`) and no
multiply by one is issued — the same saving the deleted private expansion's dead trailing
multiply had, taken structurally.

THE STAGING IS ONE BODY FOR BOTH SIDES. `StagedFactorRow` holds the OWNER's digit run per
level, which is what either side's expansion indexes with its own stream offsets, so the
read side stages from `pread` into its own rolling two-block window and nothing else
differs. `factor_at` and `stream_off` are likewise one body each, taking the side's
`SpanMap`/`StreamSpans` as a type argument.

### The readout's tile-live mask, and what it may gate

The mask bit is per (reader, EIGHT-TOKEN n-tile) and the expansion sets it iff it wrote a
nonzero coefficient in any ROW of that tile — a ballot over the row's own lanes, at bit
`rank >> 3`. It gates the `ldmatrix` and the MMAs AND NOTHING ELSE.

THE SCATTER IS GATED BY `rank < n_reader` INSTEAD, and that separation is a correctness
requirement rather than a preference. `expected[r]` is the ballot popcount — the number of
readers that HOLD the token — so a reader holding it must arrive at the ring even when its
tile is dead by VALUE, and it does: its accumulator is zero and it adds zero. Gating the
scatter on the mask would leave `arrived < expected` standing for that rank and the next
warp to want the slot would wait forever.

Dead rows are zeroed on this side too (`clear_dead_read`), under the fold's own law: a row
the MMA reads is never arbitrary shared memory. Because a reader owns its whole segment
the clear is a ROW split and needs none of the fold's column reasoning.

## Fan-in: the CTA ring (P76 §2.6-C, K35 S4b)

RED-DIRECT-FROM-THE-ACCUMULATOR IS DELETED. It issued one global reduction per (owner
warp, token, element) — uncoalesced by construction, four 32-byte segments per issue — and
it was the source of the run-to-run `num`/`den` nondeterminism every stage has recorded.

THE RING IS KEYED ON THE UNION RANK. Rank `r` — a token's position in the CTA's union read
walk — lives at slot `r % C`, `C = kRingC` rows deep (an arm field, 32 declared). A slot
carries `{tag, arrived, expected}` beside `d_v` output channels and the mass in fp32.

* CONTRIBUTE: a warp finishing an eight-token n-tile scatters each token's partial to
  `slot(rank)` — one `red.shared.add.f32` per element and ONE arrival per warp-row.
* RECLAIM: the warp that finds `tag == r - C` with `arrived == expected` CASes the tag to
  RECLAIMING, flushes the old row to global and clears it in the same pass, sets
  `arrived = 0` and `expected`, and publishes `tag = r`. CAS losers spin until `tag == r`.
* SPIN: `tag == r - C` but INCOMPLETE means a laggard's contribution is still coming, and
  the race-free form of waiting for it is to wait. No incomplete row is ever evicted.
* DRAIN: the last `C` rows have no next writer, so each executor flushes the complete
  slots of its share at the window's phase barrier. A claimed row short of its expected
  arrivals is a lost contribution and traps.

THE MEMORY ORDER. `tag == r` is the ONLY statement that the row belongs to rank `r`, and
it is published LAST — after the clear, after `arrived`/`expected`, behind a block-scope
fence — so a contributor that reads it has seen a cleared row. In the other direction a
contributor reduces first, fences, and only then increments `arrived`, so a reclaimer that
reads `arrived == expected` has seen every element. Block scope is the right scope and not
a shortcut: the participants are warps of ONE CTA talking through shared memory.

TAGS ADVANCE MONOTONICALLY, so the first contributor to a rank is by construction its
reclaimer and NO ZERO PASS EXISTS — the clear rides the flush, which is reading the row
anyway. The window's first generation is the same statement: slot `i` starts at tag
`i - C`, which is NEGATIVE, and a reclaim of a negative generation clears without
reducing.

WHY THE SPIN IS PRESENT AND WHY IT IS NOT TAKEN HERE. The protocol is correct for any
`C ≥ kSegRows` with no help from any barrier. Under form (b) the readout's per-batch CTA
barrier bounds the skew at one batch, so at `C = 32 = 2·kSegRows` a slot's stale rank is
two batches old and is always complete: the wait is the protocol's, not this schedule's,
and `C` is what buys tolerance when the schedule stops providing it (F7's dial).

THE COUNT, which is the payoff stated as a number: global reductions per (token, element)
fall from one per OWNER WARP that reads the token to ONE PER CTA, and the surviving one is
issued by 32 lanes over 32 consecutive channels — a 128-byte transaction. Determinism is
unchanged in kind (the output was already order-nondeterministic fp32 RED across CTAs);
what changes is that the in-CTA order is now the ring's arrival order.

THE RING'S LAYOUT. `LdRing = d_v + 8` fp32 words, and the `8` is derived: a scatter
instruction covers the accumulator's four lane quads = four TOKENS = four ring rows at one
eight-channel granule, so what the row must give is eight whole banks, and
`LdRing % 32 == 8` puts consecutive rows exactly one bank block apart. The rows of an
issue are DATA (four union ranks), so four-way is the floor for any permutation; the
granule swizzle `(row >> 1) & 3` makes the DENSE case — ranks `r, r+2, r+4, r+6` — a
permutation of the four blocks at every alignment and both widths, where the unswizzled
map is two-way always. The flush's own leg is conflict-free unconditionally. Both legs and
the round trip are gated in `tests/k31/test_segment_layout.py`.

### The measurement the ring supersedes

The warp-private RED-direct fan-in the ring replaces was measured before it was deleted,
and the numbers are kept because they say WHY the ring's payoff is structural rather than
cell-specific. At the graded cell 93.2% of read entries had ONE writer and replacing every
RED with a plain store moved the `k4` wall by −0.03% against a 1.78% noise floor — so the
in-CTA MEETING was never the cost there. At the DENSE cell the same fan-in multiplied by
exactly `kWarps` (RED instructions 4.19×, RED L2 sectors 4.00×, `l1tex` 42.5% → 55.7%) and
the cell cost 1.38× of the owner-grain kernel's wall with IDENTICAL HMMA. The ring pays at
BOTH ends of that spread and pays even at multiplicity one, because what it fixes there is
the COALESCING (four 32-byte segments per issue → one 128-byte transaction), which is a
property of the fragment's order and not of the data's density.

## Addressing

One pinned 64-bit base per (plane, `bh`) and a 32-bit byte displacement after it, for
every plane: amplitudes, gain, V, `o`, `den`, the state plane. That is the shipped
kernel's discipline (`docs/internals/chunk/chunk_kernel.md#state-addressing`) and it is
what keeps `IMAD.WIDE` out of the loop bodies. The window contributes one uniform
displacement per plane per window, formed once.

## Numerics

No bit-identity target against the shipped kernel: the summation orders differ by
construction (a global window grid instead of per-CTA super-chunks, a two-MMA loop
instead of the fused drain). Conformance is the fp64 oracle within the repository's one
bf16 family bound (`tests/oracle/tolerances.py`). Operands are bf16, accumulators fp32,
the state plane fp32 — the shipped kernel's conventions.

<a id="state-addressing"></a>
## Plane addressing: one 64-bit base, 32-bit displacements

Every gathered row is a base formed ONCE per plane plus a 32-bit BYTE displacement,
never a base plus an element index. The arithmetic is the contract: at element type the
compiler is entitled to re-derive `base + index` as a widening multiply per access, and
it does — measured at 30 `IMAD.WIDE` in a loop the displacement form emits none for. The
caller owns the 32-bit bound, and every plane the family reads is inside it at every
admitted length.

The base is additionally declared a GLOBAL address. An address whose storage class the
compiler cannot infer turns the fan-in's reduction into a generic read-modify-write and
every gathered load into a generic one, so the assumption restores what the displacement
arithmetic hid. Both halves live in `common/ops.cuh`'s `gather_ptr` and `pin_address`.

## The schedule scatter

The window's live tokens are turned into a list by `walk`. Two properties of it are
deliberate and were established by measurement rather than by argument; both are
recorded here because the code that results looks unremarkable.

### The store is addressed, not indexed

`dst[k++]` through a `uint16_t*` lets the compiler re-derive `dst + k` as a widening
multiply on every iteration — the same hazard the gather comment in `common/ops.cuh`
names on the load side, here on the store side. The loop's instruction ledger showed
it plainly: at ALT `k4 cohort=64` the scatter cost 447 warp-instructions per
`(owner, window)` with top opcodes `IMAD:114, IADD3:69, BRA:49, BSYNC:34` and **no
`STS` in the top five at all**, for a loop whose entire work is about two stores per
lane.

Carrying the shared address in a 32-bit register (`store_shared_u16`) makes the
per-iteration cost one add:

| | ALT k4 cohort=64 | ALT k4 scattered |
|---|---|---|
| indexed (`dst[k++]`) | 447.5 | 461.4 |
| addressed | **397.4** | **429.1** |

The list, its order and its contents are bit-identical either way — this is an
addressing change and nothing else, which is why no consumer and no contract moves
with it. The state-plane gate (`torch.equal` on all five cells against the mainline
baseline) is what holds that claim.

### The loop stays divergent, and the word-major repair is refused

After the addressing fix the residual is reconvergence: `BRA` and `BSYNC` dominate.
The obvious repair is to invert the loop so it is warp-uniform — round `r` takes
ballot word `r` whole (broadcast, so every lane sees the same word), hands its bit
`lane` to lane `lane`, and does one ballot rank plus one predicated store. That form
has no per-lane trip count, no reconvergence, a constant trip count of `LW`, and it
preserves rank order exactly (word ascending, then lane ascending, is token
ascending), so it needs no re-baseline of the replay contract.

It was built. It is **2.0x to 3.1x worse**:

| | ALT k4 cohort=64 | ALT k4 scattered | kernel total |
|---|---|---|---|
| per-lane (divergent) | 397.4 | 429.1 | 3,342 / 8,257 |
| word-major | 807.3 | 1,324.0 | 3,679 (+10%) / 9,028 (+9%) |

The reason is that the `LW` rounds are **unconditional**. The running prefix needs
`bit_count` of every word whether that word is live or not, so an empty word cannot be
skipped without losing the prefix — a sparse window pays all sixteen rounds of
`SHFL + POPC + address` in order to store two bits. The divergent form's trip count is
the window's *liveness*, and a lane holding no bits leaves immediately.

**Divergence is the cheaper failure mode when the data is sparse.** This kernel's
operating point is sparse by construction, so the divergent scatter stays.
