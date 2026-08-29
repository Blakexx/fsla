# `carry_split_kernel.cuh` — the channel-split carry body

The mirror of `csrc/rola/src/carry/carry_split_kernel.cuh`. One of the carry family's
TWO bodies; [`carry_kernel.md`](carry_kernel.md) holds the other and states how the
per-level MODE DECLARATION chooses between them.

**Symbols** are `carry_kernel.md`'s.

`d_v` is a template parameter here as it is in the carved body (K35 S3b), but it is not a
free dial: this body gives each of a FIXED four warps exactly ONE MMA m-tile of channels,
so its width IS its executor count and `static_assert(kDv == kWarps * kMmaM)` says so.
`d_v = 128` is therefore not a declaration this body can express, and no wide arm takes
it — at the wide plane the design's answer is the carved body at sixteen leaves per warp,
which is a different arm and not a branch inside this one.

## What it is for

A declaration with NO sparse level on the carve axis leaves the leaf carve nothing to
exploit: at DENSE routing every read entry is live at every digit of every level, so a
warp owning one digit group would drain every entry anyway and the only thing the carve
could still do is multiply the fan-in by `kWarps`. This body is the answer — every warp
owns `kMmaM` of the `d_v` CHANNELS across ALL `BC` leaves, so the warps' readout rows
are DISJOINT and the kernel's only fan-in is the RED at the top.

It is not a fallback. The declaration resolves at the seam and the axis never reaches
run time, so a dense declaration runs THIS kernel and a sparse one runs the other; a
declaration the family has no carved body for is served this one, which is always
correct and only ever slower.

## What differs from the carve body

| | channel-split (here) | leaf carve |
|---|---|---|
| a warp owns | `kMmaM` channels x all `BC` leaves | all `kCols` channels x `BC/kWarps` leaves |
| the state's n-tiles | `NDV = BC/kMmaN`, the box's own | `NL = BC/(kWarps*kMmaN)`, the warp's own |
| the mass channel | split along the LEAVES (`kMassUnits`), one channel being unsplittable along channels | a channel like any other — the warp holds its own leaves' partial |
| the panel | ONE CTA-shared panel per ring slot, produced cooperatively (`kRowThr = kThreads/C` threads per row) | one PRIVATE region per warp per slot, one lane per row |
| the ballot rows | the OWNER's, one per (level, owner group) | the CARVE grain, `kWarps` group rows at the carve level |
| the readout's fan-in | the warps hold different CHANNELS of one token, so they MEET: the `so` transpose, the `sden` cross-warp mass sum and two rendezvous per read batch | warp-private and DIRECT: no meeting, no `sden` |
| CTA barriers per window | ~13 | 1 outside the fold |

## Why it is a separate file rather than a branch inside the other

The two bodies differ in the state's register map, the panel's ownership, the schedule's
list count, the mass channel's split and the fan-in's shape — five things, not one — and
a single template carrying both would express that as five `if constexpr` blocks whose
joint reading is neither body. Two files each state one kernel, and the seam states
which. The cost is the text they share (the page sweep, the run loader, the window
grid); the benefit is that either body can be read end to end.

## The identity this file carries

It is `perf/pipeline` 7825d31's carry kernel, and the gate is that it still IS: the SASS
of every arm is byte-identical to that commit's, which is what makes "a dense
declaration costs nothing" a measurement rather than a claim. See
`rola-scratch/prototypes/k31_formb_build/FINDINGS.md`.

## Comments moved from source (S5b)

Verbatim `//:` prose blocks from `csrc/rola/src/carry/carry_split_kernel.cuh`, in source order, one heading per declaration. Source keeps only a short decl block pointing here.

<a id="kdv"></a>
### `kDv`

THE VALUE WIDTH IS THIS ARM'S (K35 S3b), exactly as the carved body's is.  The
channel split holds `kDv / kWarps` channels in each of a FIXED four warps, so a
width the four warps cannot tile -- or one whose state row outgrows the CTA's page
sweep -- is REFUSED by the clauses below rather than served a narrower path.  No
`d_v = 128` declaration takes this body: at that width the design's answer is the
carved 16-leaf executor, which is a different arm and not a branch inside this one.

<a id="panel-bytes"></a>
### `panel_bytes`

THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA.  A window's
readouts all precede its folds, so the write panel is dead through phase 1 and
the read panel is dead through phase 2: each phase double-buffers in the
other's buffer and produces one batch AHEAD, which is what puts the produce's
global-load latency underneath the previous batch's MMA block.

<a id="kregfloor"></a>
### `kRegFloor`

THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR,
exactly as the shipped `Geom::stages` is: the shallowest ring that reaches the
best residency this arm can reach at any depth.  `kRegFloor` is the DECLARED
floor for this family -- the measured census of the heaviest built arm,
recorded in the findings doc and re-measured at every milestone.

<a id="kmassallwarps"></a>
### `kMassAllWarps`

when the split reaches every warp the mass guard is a TAUTOLOGY, and it must be
spelled as one: `__launch_bounds__` does not let the compiler prove
`threadIdx.x < kThreads`, so the guard would otherwise cut the readout and the
fold into basic blocks and pin the lookahead produce out of their HMMA shadows.

<a id="rowbytes"></a>
### `rowBytes`

AND THE OWNER'S BALLOT ROW, per level: the SAME `o_l`, addressing the row the
facts pass voted for this owner group at this level (K31-i-d).  The liveness
pass reads these words instead of re-reading the amplitude runs itself -- once
per (owner, window, level, side) 32 tokens, against `2*D*s_l` bf16 per TOKEN.

<a id="note-l250"></a>
### near line 250

THE ATOM WALK'S SHAPE IS A REGISTER-INDEXING FACT, NOT A CONVENIENCE.  A
state fragment is a register array, and a runtime index into one is a stack
frame -- measured, on this kernel's first build: `kAtoms` walked at runtime put
the WHOLE state (320 B/thread) in local memory.  So the walk is FULLY
UNROLLED over the atoms, and every warp owns every atom (its tile index is a
compile-time `2*a`) because the carve is along the channels.

<a id="note-l257"></a>
### near line 257

---- THE ENTRY STATE.  The fragments are SET, never accumulated: they are
provably zero here, and with no decay the recurrence is affine in the state
with an inclusive-after-write readout, so this IS the oracle's seeding.
NULL = NO STATE OBJECT AT THIS END: nothing is read, allocated, or touched
(docs/internals/state.md section 1). An instantiated-but-empty paged state is
NOT null -- it is engaged, and its absent atoms skip below at the per-atom
`slot < 0` grain, orthogonal to this check. This predicate is CTA-uniform:
`s_in` is a launch argument, identical for every thread, so the branch cannot
diverge within or across warps -- one compiled body, no STATE template axis.

<a id="write"></a>
### `WRITE`

A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS.  A padding row of a partial
batch has no token, and a zero coefficient contributes exactly nothing to either
MMA (the produce's dead-row zeroing law).  Level 0's run is the outermost one and
therefore the narrowest at every built arm, so zeroing it covers the whole slot
run for one or two selects and costs no arithmetic.

THE READ SIDE HAS NO GAIN.  Its scalar is identically 1 and the trailing
`mul_bf16x2(acc, 1)` was 44% of the read produce -- 32 FMUL, 32 PRMT and 16 F2FP
for `rn(a*1) = a` -- so the side is a compile-time argument and the read body
stores its accumulators directly.

<a id="note-l317"></a>
### near line 317

THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run
is loaded, folded into every slot and dropped inside one `static_for` body, so
no `[D][n]` array is ever live: `SPT/2` packed pairs plus one run, against the
`sum_l n_l` shorts a per-level array would hold at once.
the innermost level supplies the ADJACENT PAIR (its run offset owns bit 0),
every level above it a per-pair scalar -- the shipped producer's chain.

<a id="nb"></a>
### `nb`

ONE BATCH'S PANEL, expanded from factors.  The list, the row's token and the
gain are all resolved here so the two phases call ONE body; `have` false
zeroes the leading factor, and a zero coefficient contributes exactly nothing
to either MMA (the produce's dead-row zeroing law).

<a id="kscaniters"></a>
### `kScanIters`

THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19, study section 2.0 as
rewritten).  Per level at most ONE side is sparse, so there is one stored row
set per level -- but the two sides' CLAUSE SETS are different level subsets,
and a dense side's clause is absent-means-full.  ANDing each side's OWN rows
IS that conjunction over exactly the levels where that side is sparse: a dense
level's ballot word is all ones for every token and drops out on its own, with
no structure fact carried into the kernel.  Union routing (one support serving
both sides) is then the special case where the two row sets coincide, and it
needs no separate code.

Per-side lists make activity AIRTIGHT: a side's simplex weights are strictly
positive on its own support, so a listed entry is live for that side by
construction.  There is no union-superset padding anywhere, by design.
THE SCAN IS A TABLE READ (K31-i-d).  A warp's thirty-two tokens are one
BALLOT WORD of the owner's row, so the conjunction over levels is `D` word
ANDs per side for the whole warp, and the lane reads its own bit out of the
result -- where the amplitude scan it replaces re-read `2 * sum_l s_l`
bf16 PER TOKEN, per owner, at owner-specific columns that coalesced with
nothing.  A level the owner spans WHOLE emits no row: its clause is true for
every token and drops out of the conjunction with no structure fact carried
into the kernel, which is exactly what the amplitude form's `or_run` did.

THE COMPACTION IS TWO RENDEZVOUS, both load-bearing.  Every warp holds its
whole window's ballots in registers and publishes its `kScanIters x 2` live
counts ONCE; the first barrier publishes those counts, and after it EVERY
thread re-derives the CTA-wide exclusive prefix from them in registers -- the
running totals are CTA-uniform and never need a shared round trip, so neither
the per-iteration barrier pair nor the `tid == 0` serial walk over the warps
exists.  The second barrier publishes the LISTS, which the phases read from the
first batch onward, and it doubles as the separation between this window's
count reads and the next window's count writes -- which is why it precedes the
whole-window skip rather than following it.

<a id="rendezvous"></a>
### `rendezvous`

THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring,
UNCONDITIONALLY and AFTER the batch's MMA block.  Unconditionally because a
guard is a basic block and ptxas schedules within one; after, because the
produce ends in panel `STS` and the MMA begins in panel `LDSM`, and the two
shared ranges are not provably disjoint -- in the other order that aliasing
edge pins every produce instruction ahead of the HMMAs.  The trailing batch is
entirely dead rows by the produce's zeroing law and lands in a panel no MMA
reads.
THE PRE-LOOP PRODUCE IS THE ONLY THING THE HEAD RENDEZVOUS EVER PUBLISHED, so
it is taken OUT of the loop: for every later batch the publish is already
carried by the readout's own two rendezvous (docs/internals/carry/carry_kernel.md,
"The rendezvous the ring needs").

<a id="nbw"></a>
### `nbw`

V IS A WRITE-SIDE QUANTITY and is gathered by the WRITE list; the read list
never touches it.  The value ring is `STAGES` deep and asynchronous: issue
order is FIFO, one implicit group per batch, and the wait leaves `STAGES-2`
groups in flight so the gather for batch `i + STAGES - 1` is airborne while
batch `i` folds.

<a id="rendezvous-2"></a>
### `rendezvous`

the mirror of the entry load, after the last fan-in has consumed the staging
buffer -- which is what the one extra rendezvous below is for.
NULL = NO STATE OBJECT AT THIS END: nothing is written (state.md section 1),
symmetric with the entry check above.

<a id="kliststride"></a>
### `kListStride`

TWO packed lists per window, one per SIDE (study section 2.0, two-sided
schedule): `nr` read entries then `nw` write entries, `kListStride` apart.

<a id="near-line-61"></a>
### near line 61

THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
the same geometry the ring depth is.  A ring sized for a residency the launch
bound does not ask for is a ring sized for nothing.

<a id="ndv"></a>
### `NDV`

every warp holds the WHOLE box and a slice of the channels, so the state's
leaf tiling is the box's and is not carved.

<a id="kmassunits"></a>
### `kMassUnits`

THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel
axis, so it is split along the axis the mma leaves free -- the leaves.

<a id="vrowelems"></a>
### `vRowElems`

ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the
shipped addressing discipline (docs/internals/carry/carry_kernel.md#state-addressing).

<a id="near-line-144"></a>
### near line 144

`o_l` is a bit-field extract of the owner index: the lattice's owner
coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).

<a id="rowsr"></a>
### `rowsR`

ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the side's
whole row block is `kRows * tok_words * 4` bytes, which is 32-bit at every
admitted length.

<a id="brt0"></a>
### `brT0`

THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by
the batch parity would be a runtime index into a register array, which is a
stack frame; a select is one instruction and no memory.

<a id="konespair"></a>
### `kOnesPair`

THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so
the fold's A fragment for it is the constant whose row 0 is ones.

<a id="atombh"></a>
### `atomBH`

THE PLANE IS ADDRESSED BY CANONICAL ATOM ID (P80) -- the same re-key the carved
body makes, so the two bodies stay one ABI and one base translation.

<a id="activity"></a>
### the activity masks

THE ACTIVITY OF THIS OWNER'S ATOMS, READ ONCE AT THE PROLOGUE (P80b): the carved
body's two `uint32` masks, folded here over the same `kAtoms` and consulted by the
same two sweeps (`carry_kernel.md#activity`).  One byte per atom per CTA is the
whole cost, and the window loop never sees it.

<a id="kpagebytes"></a>
### `kPageBytes`

THE SLOT, NOT THE ATOM, ADDRESSES THE PLANE (`ops.cuh`'s `page_slot`,
docs/internals/paging/paging.md) -- the same one translation the carved body
makes, so the two bodies remain one ABI and one accumulation order.

<a id="pagevalue"></a>
### `page_value`

element `k` of an accumulator tile sits at row `(lane>>2) + 8*(k>>1)` and
column `(lane&3)*2 + (k&1)`; here rows are channels and columns are leaves.

<a id="slot"></a>
### `slot`

AN ABSENT OR IDLE ATOM IS SKIPPED, AND WITH IT THE TWO BARRIERS: the load
predicate is RESIDENT AND (read or written), the carved body's exactly
(`carry_kernel.md#decltype`, `common/ops.md#atom-activity`).  The two bodies keep
one predicate for the same reason they keep one base translation: a declaration
must not be able to change which atoms a call touches.

<a id="knl0"></a>
### `kNL0`

§6: PM::n_l(0) is compile-time (a constexpr static member function of a
template class called with a literal) but the heuristic can't see through
the qualified call, so name it.

<a id="near-line-381"></a>
### near line 381

the two sides ride ONE count word, low half read and high half write.  A side's
window total is at most `W`, so no low-half sum ever carries into the high half
and one packed add carries both prefixes.

<a id="wordb"></a>
### `wordB`

the word is the WARP's, not the lane's: `tok0` and `t * kThreads` are both
whole numbers of words, so warp `u` owns word `(tok0 + t*kThreads)/32 + u`.

<a id="nr"></a>
### `nr`

THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
mechanism's harvest, kept (study section 2.1).

<a id="f"></a>
### `f`

the ring slot is advanced on EVERY batch, live or not, so the wait/commit
accounting never depends on the data.

<a id="near-line-606"></a>
### near line 606

the ring's unconsumed groups are retired before the next window reuses the
value slots.

<a id="slot-2"></a>
### `slot`

THE EXIT STORES ONLY WRITTEN, PLANNED ATOMS (P80b), the carved body's predicate
exactly (`carry_kernel.md#decltype-2`).  An atom this call does not write is not
stored and an unplanned one is not addressed at all, which is what leaves a skipped
slot carrying exactly what it held.
