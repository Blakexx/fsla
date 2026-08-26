# `carry_kernel.cuh` — the K31 box-native carry kernel (rung R1)

The mirror of `csrc/rola/src/carry/carry_kernel.cuh`. A NEW KERNEL FAMILY, built
beside the shipped chunk consumer: nothing under `csrc/rola/src/chunk/` is touched and
no shipped dispatch reaches this file. The shipped arms stay the baseline until this
family beats them cell for cell (`development/plans/K31_BOX_NATIVE_KERNEL_STUDY_2026-08-18.md`
§6). The ratified study governs; this doc records the construction.

**Symbols.** `D` levels, `B` digits per level, `K` the uniform sub-box span, `M` the
capacity multiplier, `BC` the leaves one owner holds, `W` the global window length in
tokens, `C` the private batch's token tile, `d_v = 64` value channels. `k_tok` (a
token's nonzero digits per level) appears nowhere in the kernel, which is the point:
the schedule is computed, never branched on.

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
transit plane's pull and the fan-in untouched by the split. At `WPS = 1` the mask is
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
change (the transit plane), and each side's vector is declared per arm in
`CARRY_ARMS_X` — the write side's `SW`, the read side's `SR`.

Constraints, all `static_assert`ed: `s_l % S_l == 0`, every span a power of two, `WPS` a
power of two, and the two declarations agreeing with the executor map —
`SVW::kStreams * WPS == SVR::kStreams == kWarps`, and a column half of the write stream's
box being exactly one read stream's box.

### The private order, and why it is a RULE and not a choice

A stream numbers its leaves in a MIXED RADIX over its own spans, `p = sum_l r_l <<
shift(l)`, so `p` is a bit permutation of the digit offsets exactly as `local` is and the
produce is `sum_l n_l` amplitude loads times products at CONSTANT indices. A span-1 level
contributes a constant and drops out of the order on its own. The order itself is fixed
by the span vector:

1. **level `D-1` is INNERMOST.** The transit plane publishes `bf16x2` words of two
   LATTICE-ADJACENT leaves, so the level lying lowest in `p` must be the level lying
   lowest in `local`, and its span must be at least two. This is the PAIR LEVEL: it seeds
   the pairs and carries the dead-row zeroing. **The span-≥2 clause is a limitation of
   the CURRENT exchange, not of the design** — it comes from the plane publishing pairs,
   and the rebuilt exchange (writer-side placement into per-reader `[leaf][channel]`
   segments) makes the carve vectors fully general.
2. **the remaining levels keep the LATTICE order, except the levels this side CARVES,
   which move OUTERMOST** keeping their lattice order among themselves.

A side carves where its declaration marks it sparse, so rule 2 is the PRIVATE-ORDER RULE
in the vector's vocabulary — *dense levels innermost, sparse levels outermost*: an
eight-slot n-tile spans dense digits, and each sparse level indexes WHOLE tiles, where a
digit skip is exact. The order is not cosmetic: it is the readout's MMA **k-order**, so
it fixes the fp32 summation order and therefore the bits.

The rule reproduces the two single-level maps it replaces EXACTLY — the read side's
q-major order is the lattice order (`3,2,1,0` at `D = 4`), the write side's `CL`-major
order is `CW` moved outermost (`3,1,0,2` at `D = 4`, `1,0` at `D = 2`) — which is what
makes the span-vector build byte-identical to the single-level one at every arm that
existed before it.

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
one page per atom. The same measured law governs the produce's slot loops: they are
`static_for` and not `#pragma unroll`, because with a runtime slot index the level maps
`n_l`/`idx_of` are evaluated at RUNTIME and the amplitude run lands in local memory —
observed at 64-67 local accesses per deep-arm body before the loops were made
compile-time.

## The cooperative fetch (K35 F2/F4): one union walk, writer-side placement

THE FOLD'S OPERAND PATH IS NOT PER-STREAM ANY MORE. The CTA walks the UNION of its
streams' write lists ONCE — that list is `wlistS`, which the shared derivation already
builds — and a BATCH is `kSegRows = CP = kMmaK` consecutive tokens of it. Executor
`warp` takes the fixed share `i % kWarps` of a batch's union tokens (no dynamic
assignment, no cursor), and for every token it handles and every STREAM that wants that
token it places, into that stream's segment, BOTH

* the value row, by `cp.async` (the copy never touches registers), and
* the FULLY EXPANDED coefficient row for that stream's box — `kLeafW` bf16 in the
  stream's private order, gain applied.

The consumer does NO produce: tile skip → `ldmatrix` → HMMA. Each stream's row is
minted ONCE, so the total multiply count is unchanged; what is removed is every warp
re-loading and re-expanding the factor runs the other warps also want.

### The rank IS the address, and the bound is the type's

The union list is ascending in token and a stream's compacted list is a subsequence of
it, so `rank(t, s)` — the token's position in stream `s`'s list — is the PREFIX POPCOUNT
of that stream's write ballot below `t`. Each executor therefore publishes, once per
window, its own ballot word per lane (`kStreams × kLW` words) together with the per-word
prefix popcount; two shared loads and a `popc` turn any (token, stream) pair into its
rank, and the fetcher needs no knowledge of a stream it does not run beyond those.

Inside a batch of `kSegRows` union tokens a stream can hold at most `kSegRows` of them,
so the batch-local rank is BELOW `kSegRows` by construction. The segment bound is
therefore the type's — `kSegRows == CP`, a `static_assert` — and there is no truncation
path and no runtime test.

The four shared loads of a rank are paid ONCE PER PAIR: lane `q < kSegRows` of each warp
owns candidate pair `q = (i, s)`, resolves it, and every lane that fills that row reads
the result by shuffle.

### Coverage is by construction

`0 × NaN = NaN`: a slot row the MMA reads must never be arbitrary shared memory, and an
accounting argument about which rows a partial batch left dead is exactly the kind of
argument that failed before. THE DEAD-ROW LAW: a dead row is a FINITE (stale) value row
with a ZERO coefficient row — finite operand, zero weight, contributing exactly nothing —
and coverage is a ONCE-PER-WINDOW clear of the whole segment plus a PER-BATCH zeroing of
the dead COEFFICIENT rows only. The window's clear is what makes every row finite (the
published transit plane straddles these very bytes); thereafter a value row is either
this batch's placement or an earlier one's, both finite, and a full batch writes nothing
at all between the barriers.

The two clears split over a stream's `WPS` warps DIFFERENTLY, and that is a correctness
fact rather than a preference:

* the WINDOW clear splits BY ROWS. It is followed by a rendezvous before any placement,
  so any partition of the rows is legal and a row split keeps both legs contiguous.
* the PER-BATCH dead-row clear splits BY COLUMNS. It runs between a batch's `ldmatrix`
  drain and the barrier that releases the writers, so a warp may only zero bytes IT
  ITSELF reads — under a row split it would overwrite rows the stream's other warp is
  still draining. A column half of a row is `kLeafE` slots, contiguous and sixteen-byte
  aligned, so the run is the same predicated `STS.128` nest with a row stride carried
  per quantum.

No row's liveness is computed twice and no row is ever unwritten.

### The expansion, and its declared order

Under the private-order rule the levels a side spans WHOLE lie lowest in `p` and the
levels it CARVES lie outermost, so the product splits cleanly:

    memo  = gain, then × f_l for every level the side SPANS WHOLE, ascending l
    w     = memo, then × f_l for every level the side CARVES,      ascending l

`memo` does not depend on the stream, so it is minted ONCE per (token, lane) and reused
by every stream that wants the token — that is the sub-product memo, and its shape is
the span vector's, at compile time. The level lying LOWEST in the order (level `D-1`, by
rule 1) supplies BOTH slots of a packed pair with one four-byte access, because its two
digits are adjacent; every other level's digit is constant across the pair and is
splatted. At the `D = 2` ALT arms that is ONE shared access and ONE `mul_bf16x2` per
(token, stream) row per lane.

THE ORDER IS DECLARED AND IS NOT THE DELETED PRODUCE'S, which multiplied the pair level
first and the gain last. The coefficients are therefore NOT bit-identical to the produce
they replace; the fp64 oracle is the gate.

### The rolling factor staging

The fetcher expands from SHARED MEMORY: the compressed factor rows of a batch's union
tokens are `cp.async`ed ONE BATCH AHEAD into `StagedFactorRow`, so no per-token global
factor load ever appears in the fetch loop. A staged row is the OWNER's digit run at
each level — `Σ_l s_l` amplitudes, not the topology-wide `kWtot` packed row — laid out
level by level with each run aligned to its own asynchronous quantum (4, 8 or 16 bytes);
the alignment holds by construction because a level's column base is a multiple of `B`
and an owner's run starts at a multiple of `s_l`.

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
the reset sits between a warp's own read of the mask and the barrier that lets the
fetchers OR into it, and a word shared with the stream's other warp would be reset out
from under that warp's read. The `WPS` words of one stream carry the SAME bits — one
lane per column half publishes them, which is the single predicated `or.shared` the
one-word form issued, with more lanes live rather than more instructions — and each
consumer tests only its own field, `[col*kNJ, (col+1)*kNJ)`, through two constants
computed once per warp.

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

FORM (b) COSTS TWO CTA BARRIERS PER BATCH, and both are structural. "Landed" is
all-fetchers → one consumer; "drained" is one consumer → all fetchers; with a single
slot neither can be folded into the other, and on sm_80 there is no cheaper carrier than
`__syncthreads`. On sm_90 both become mbarrier arrivals on the same two facts.

## The barriers: the window line, the phase change, and the fold's two per batch

THE SCHEDULE HAS NO BARRIER AT ALL (above), so outside the fold the CTA rendezvouses
only at the phase change and at the window line, and a window the owner is dead in costs
zero barriers.

The READ PHASE SHARES NOTHING. The list, the panel region and the fan-in are the warp's
own, so the executors run their own batch counts to the window line without one CTA
barrier; the only ordering the phase needs is a lane's panel `STS` against another lane's
`ldmatrix` of the same region, which is `__syncwarp`.

| edge | carried by |
|---|---|
| pre-loop read produce → MMA(0), produce(i+1) → MMA(i+1) | `__syncwarp` (the panel region is warp-private) |
| every warp's reads → any warp's folds of the same window | the WINDOW-LINE rendezvous |
| every fetcher's placement of batch `b` → the stream's drain of it | the LANDED rendezvous |
| a stream's drain and clear of batch `b` → any fetcher's writes of `b+1` | the DRAINED rendezvous |
| the staged factor block for `b+2` → the expansion that reads it at `b+1` | the LANDED rendezvous of `b+1` |
| the published ballots and prefixes → the fetch that ranks against them | the phase change's rendezvous |
| the window's list writes → the previous window's list reads | the WINDOW-LINE rendezvous of the previous window |

The mass channel is a channel like any other under this carve: the warp holds its own
leaves' mass partial and REDs it to `den` itself, so no cross-warp sum of the denominator
exists.

## The read side's produce

THIS IS THE READ SIDE'S PRODUCE AND ONLY THE READ SIDE'S. The write side's expansion
moved to the fetcher (above) and `produce`'s `WRITE` arm, its gain, its `lvColW` columns
and the `panW0`/`panW1` panels are DELETED; what stays here is the read panel's, until
the readout takes the same treatment.

Per token per warp, `Σ_l n_l` amplitudes are loaded and everything else is computed; the
gate plane is never materialized. A LANE owns a whole panel row — the batch tile is the
warp width — and expands its stream's `kLeafW` coefficients in the PRIVATE order `p`.
Because `p` is a bit permutation of the levels' digit offsets exactly as `local` is, the
expansion is `Σ_l n_l` amplitude loads followed by products of register values at
constant indices: no dynamic indexing, no spill. `load_run<n>` covers a level's run at
the widest access that fits it, `n ∈ {1, 2, 4, 8, 16}`.

The level lying LOWEST in `p` supplies the adjacent bf16 PAIR and every other level a
per-pair scalar. THE READ SIDE HAS NO GAIN: its scalar is identically 1, and on sm_86 —
where there is no `mul.rn.bf16x2` and a packed multiply is two FMUL, an F2FP pack and two
PRMT — paying for `rn(a*1) = a` was 44% of the read produce, so the body stores its
accumulators directly.

A dead row (a padding row of a partial batch, which has no token) is zeroed at the PAIR
LEVEL's run rather than at a trailing multiply. That run is the factor every slot of the
row carries, so the whole row is covered by one or two selects and no arithmetic; a zero
coefficient contributes exactly nothing to the MMA.

## Addressing

One pinned 64-bit base per (plane, `bh`) and a 32-bit byte displacement after it, for
every plane: amplitudes, gain, V, `o`, `den`, the state plane. That is the shipped
kernel's discipline (`docs/internals/chunk/chunk_kernel.md#state-addressing`) and it is
what keeps `IMAD.WIDE` out of the loop bodies. The window contributes one uniform
displacement per plane per window, formed once.

## Fan-in

All output accumulation is RED at the top, and under this carve it is WARP-PRIVATE and
DIRECT — no meeting of any kind. The warp stages one `kMmaN`-token n-tile of its
accumulators into its own region of the output block, then REDs it with **lane =
CHANNEL**, which is what makes every issue one coalesced 128-byte segment; the mass
column rides the same region and reaches `den` the same way. Entries past `n` are
padding and contribute nothing rather than contributing zero to somewhere.

The several partials of an entry that reaches two groups are two REDs of an associative
reduction, which is correct at any interleaving. The accumulate-and-evict RING is NOT
built: at the graded cell 93.2% of read entries have ONE writer, and the fan-in probe
measured replacing every RED with a plain store at −0.03% of the k4 wall against a 1.78%
noise floor. Where it WOULD pay is dense data, and the price is measured: at the dense
cell the fan-in multiplies by exactly `kWarps` (RED instructions 4.19x, RED L2 sectors
4.00x, `l1tex` 42.5% → 55.7%) and the cell costs 1.38x of the owner-grain kernel's wall
with IDENTICAL HMMA. That is K33 §1.4's dense-declaration escape — a carve chosen at
compile time from the per-level mode declaration — becoming a live question rather than
a hypothetical.

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
