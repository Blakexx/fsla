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

The ROWS ARE VOTED AT THE CARVE GRAIN. The carve level's owner run is split into
`kWarps` digit groups (`BoxPlan::carve_s`, `CarveRows`), so that level emits `kWarps`
times the rows and an owner-grain word is the OR of a group's `kWarps`. Nothing in the
facts kernel changes: the carve rows are the same `chunk_facts_tables` pass instantiated
at the finer spans. The READ side takes the rows as `kWarps` ballots — one per digit
group, and therefore one per warp — and the WRITE side ORs them back into the owner's
own clause. The table grows from 48 to 96 rows per side at the flagship, and the facts
pass by 6% of itself (0.496 → 0.527 ms at `L = 65536`), against a `Σ_l B/s_l` shape that
still does not grow with `N`.

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

## The carve: one digit group of the innermost level, per warp

A warp owns one GROUP of the innermost level's digits — `kGS = s_{D-1}/kWarps` of them —
across every outer digit, which is `BC/kWarps` leaves and ALL `kCols` channels
(K33 clean-sheet §1.2A, tier A; the drain microbench that decided it is
`rola-scratch/prototypes/k31_tieraprobe/FINDINGS.md`). The warp then TILES ITS OWN
LEAVES: its private index is `p = (r_{D-1} - grp*kGS) * kOut + outer`, the carve level
SLOWEST, so one of its digits is a contiguous run of `kOut = BC/s_{D-1}` leaves and a
whole number of accumulator n-tiles.

That private order is observed by NOTHING outside the warp. The accumulator's n-indexing
is a register-array indexing the kernel chooses; the lattice's `local` order, the state
plane's page layout, the atom rectangle, decode and the facts geometry are all untouched.
The consequence is the whole design: the readout's A operand is the warp's own
accumulator packed in place, so there is **no state image, no `ldmatrix` for A, no leaf
order flip and no CTA barrier in the read phase**.

Two things follow that are not free, and both are measured
(`rola-scratch/prototypes/k31_tiera_build/FINDINGS.md`):
* a READ entry is drained by the warps whose groups its digits reach, so at DENSE routing
  every entry is drained by all four and the fan-in multiplies by `kWarps` exactly;
* a WRITE entry is live at ~1.04 of the outer level's digits and at ALL of the innermost
  level's, so it belongs to every warp whatever the group split is. The write side is
  therefore UNCHANGED in schedule — one union list, the shared V ring, a rendezvous per
  batch — and only its operands re-carve with the state.

The group a warp takes is ROTATED per owner, `grp = (warp + ((o + (o>>5)) & 3)) & 3`.
The obvious `o mod kWarps` is defeated: the two CTAs co-resident on an SM are one grid
wave apart and the wave is a multiple of `kWarps`, so both would rotate identically and
still collide on one processing block. The wave-stride form costs three setup
instructions once per kernel and nothing in any loop.

`CV` survives on the SEAM — the dispatch's `carve` scalar — so a caller asking for a
carve this kernel does not build is refused rather than silently served another one.

## Two bodies, and the DECLARATION chooses between them

The carry family has TWO kernels and the per-level MODE DECLARATION selects one of them
at compile time (K33 law 3; `csrc/rola/src/carry/carry.cu`'s `carve_level`). This file
holds the LEAF CARVE, for a declaration that marks a level read-sparse;
[`carry_split_kernel.md`](carry_split_kernel.md) holds the CHANNEL-SPLIT body, for one
that does not. Neither is a fallback for the other: the axis is resolved at the seam and
never appears at run time, so a dense declaration runs a kernel with no carve in it at
all rather than a carved kernel with the carve turned off.

The CARVE LEVEL is a template parameter `CL`, threaded through `BoxPlan::carve_s`,
`CarveRows`, `CarveMap`, `CarryGeom` and the kernel, and named by the declaration rather
than by this file. `carve_level<D, MODES>` is the innermost level the declaration marks
read-sparse, or `-1` for the channel-split body; `CarveMap`'s private leaf order is exact
only while that level is the innermost one, so a declaration naming an outer level is
served the channel-split body — correct, and stated rather than silently carved at a
level nothing declared.

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

## The rings

The V ring is `STAGES` slots deep, asynchronous, FIFO, one implicit group per batch,
`stage_wait<STAGES-2>` before the consuming rendezvous, and a group committed on every
batch — live or not — so the wait/commit accounting never depends on the data.
`STAGES` is DERIVED by `design::stages_for(slot, fixed, threads, kRegFloor)`, the same
function `chunk`'s `Geom::stages` uses: the shallowest ring that reaches the best
residency this arm can reach at any depth.

The PANEL ring is two slots, and each slot is carved into `kWarps` PRIVATE REGIONS of
`C × (BC/kWarps + kPadE)` — the warp expands only its own leaves, so the CTA's total
produce work is unchanged and only the region a lane stores into moves. A lane owns a
whole panel ROW (the batch tile is the warp width), which publishes as four
conflict-free `STS.128`: the wide store path is what compaction lost to
(`rola-scratch/prototypes/k31_compactprod/FINDINGS.md`) and it is kept here by
construction. Because all of a window's readouts precede all of its folds, the write
panel is dead through phase 1 and the read panel is dead through phase 2; each phase
double-buffers in the other's buffer and produces one batch AHEAD. Its two operand bases
are NAMED SCALARS selected by the batch parity — a two-element array indexed at runtime
would be a stack frame.

The lookahead produce is UNCONDITIONAL and sits AFTER its batch's MMA block, and both
halves of that are scheduling laws rather than style. A `i + 1 < nb` guard is a basic
block, and ptxas schedules within one, so a guarded produce cannot place a single
instruction in an HMMA's 16-cycle tensor-pipe shadow; the trailing batch it costs
instead is entirely dead rows under the produce's zeroing law and lands in a panel no
MMA reads. Placement after the MMA is forced by ALIASING: the produce ends in panel
`STS` and the MMA begins in panel `LDSM`, and ptxas cannot prove the two runtime panel
ranges disjoint, so in the other order that memory edge pins every produce instruction
ahead of the HMMAs even with the branch gone. In this order the produce's amplitude
`LDG` hoist above the MMA and its expansion arithmetic fills the tensor pipe's shadows.

## The barriers: the window line, and nothing else outside the fold

THE SCHEDULE HAS NO BARRIER AT ALL (above), so the window line is the only CTA
rendezvous outside phase 2's per-batch one, and a window the owner is dead in costs zero
barriers.

The READ PHASE SHARES NOTHING. The list, the panel region, the staging block and the
fan-in are the warp's own, so the four warps run their own batch counts to the window
line without one CTA barrier; the only ordering the phase needs is a lane's panel `STS`
against another lane's `ldmatrix` of the same region, which is `__syncwarp` and costs no
instruction on provably converged lanes. Phase 1's body is `[MMA(i) reads panel P(i)]`,
`[stage an n-tile, RED it]`, `[produce → P(i+1)]`, `__syncwarp`.

The WINDOW LINE keeps its barrier, and it is the one place run-ahead stops. It is
required, not decorative: without it a warp that finished its reads early would begin
staging the next window's V into ring slots another warp is still folding from, and the
per-window barrier sequence would stop being CTA-uniform. Coherence costs latency, not
work — the barrier is where that is paid, and it is where a COHERENT window's cost
appears: the carve gives a warp one digit group for the whole kernel, so a window whose
live support falls in one group is drained by one warp while the others reach the line
and wait. That is a property of REGISTER-resident state (the slice is a register set, so
only its owner can drain it) and not of the schedule; the measurement, and what an
addressable state costs to fix it, are in
`rola-scratch/prototypes/k31_formb_build/FINDINGS.md`.

Phase 2 is unchanged: `BAR_D`, `[stage V slot f; fold(i) reads V slot i and P(i)]`,
`[produce → P(i+1)]`, one rendezvous per write batch.

| edge | carried by |
|---|---|
| pre-loop produce → MMA(0), and produce(i+1) → MMA(i+1) | `__syncwarp` (the panel region is warp-private) |
| MMA(i) writes the staging region → its own fan-in | `__syncwarp` |
| fan-in(i) reads the region → MMA(i+1) rewrites it | `__syncwarp` |
| every warp's reads → any warp's folds of the same window | the WINDOW-LINE rendezvous |
| pre-loop produce → fold(0), produce(i+1) → fold(i+1) | `BAR_D(i+1)` |
| the V ring's `cp.async` group → fold(i) reads the slot | `BAR_D(i)`, after the per-thread wait |
| fold(i-1) reads V slot → `stage_batch` rewrites it | `BAR_D(i)` |
| the window's list writes → the previous window's list reads | the WINDOW-LINE rendezvous of the previous window |

The mass channel is a channel like any other under this carve: the warp holds its own
leaves' mass partial, stages it in its own region's mass column and REDs it to `den`
itself, so no cross-warp sum of the denominator exists.

## The produce

Per token per warp, `Σ_{l<D-1} s_l + kGS` amplitudes are loaded and everything else is
computed; the gate plane is never materialized. A LANE owns a whole panel row — the
batch tile is the warp width — and expands its warp's `BC/kWarps` coefficients in the
PRIVATE order `p`. Because `p` is a bit permutation of the levels' digit offsets exactly
as `local` is, the expansion is `Σ_l n_l` amplitude loads followed by products of
register values at constant indices: no dynamic indexing, no spill. `load_run<n>` covers
a level's run at the widest access that fits it, `n ∈ {1, 2, 4, 8, 16}`; at the flagship
(`s = (8, 16)`, `kWarps = 4`) that is one 16-byte access at level 0 (`n_0 = 8`) and one
8-byte access at the carve level (`n_1 = kGS = 4`, at the group's runtime base) — 12
amplitudes per token per warp, against 18 for the owner-grain map.

The level lying LOWEST in `p` — the carve level's neighbour outward — supplies the
adjacent bf16 PAIR, and every other level a per-pair scalar, which is the shipped
producer's multiply chain read in the private order. At `D = 2` that chain is
`rn(a_0 * a_{D-1})` where the owner-grain chain was `rn(a_{D-1} * a_0)`: multiplication
is commutative and both round once, so **the coefficient is bit-for-bit the same number
and the state plane is bit-identical to the owner-grain kernel's**. At `D ≥ 3` the
association genuinely moves (the pair comes from a different level), so the deep arms'
coefficients differ in the last bf16 bit and only the oracle grades them.

The SIDE is a compile-time argument to the produce, because the two sides' trailing
scalars are not the same kind of thing. The write side ends in `mul_bf16x2(acc, gain)`
over all `SPT/2` packed pairs; the read side's scalar is identically 1, and on sm_86 —
where there is no `mul.rn.bf16x2` and a packed multiply is two FMUL, an F2FP pack and
two PRMT — paying for `rn(a*1) = a` was 44% of the read produce. So the read body stores
its accumulators directly.

A dead row (a padding row of a partial batch, which has no token) is then zeroed at the
CARVE LEVEL's group run rather than at the trailing multiply. That run is the narrowest
factor the private order has (`kGS ∈ {2, 4}`), so the whole slot run is covered by one or
two selects and no arithmetic; a zero coefficient contributes exactly nothing to either
MMA. On the write side the same zeroing rides the gain, which is already a select.

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
