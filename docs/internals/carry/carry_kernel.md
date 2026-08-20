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

## The carve is settled, so it is not an axis

Each warp owns `kMmaM` of the `d_v` channels across all `BC` leaves. Its readout rows are
therefore DISJOINT from its siblings': nothing in the owner loop reduces across warps,
and the kernel's only fan-in is the RED at the top. The fold-residency coupling law
holds by construction — the warp carve of the state IS the output carve of the fold.

Study §2.8 recommended the alternative (warps owning `BC/kWarps` leaves across all
channels, meeting in an intra-CTA shared-memory tree) and named this one the measured
control; R1 measured, and the control won on sm_80. Blake ruled the axis closed on
2026-08-20 (commit `46a3d3e`), and the integration rung removed the template machinery
the ruling left standing — the `Carve` enum, the `MDV` channel-tile loop, the
`kStageUnits` atom-staging loop and the shared-memory readout tree with its
`atomicAdd` partials. The math of the surviving arm did not change; the register census
did not move (176 registers, zero spill, zero local, on both arms, sm_80 and sm_86).
The losing arm's revival is `git`, not a dead branch. R1's measurement is in
`rola-scratch/workflows/k31-r1-carry/FINDINGS.md`.

`CV` survives on the SEAM — the dispatch's `carve` scalar — so a caller asking for the
deleted carve is refused rather than silently served the other one.

## The atom walk's shape is a register-indexing fact

A state fragment is a register array and a runtime index into one is a stack frame —
measured, on this kernel's first build: walking `kAtoms` at runtime put the whole state
(320 B/thread) in local memory. So the walk is inverted. The outer, fully unrolled loop
is fully unrolled over the box's atoms, which makes each one's tile index a compile-time
`2*a`. Every warp owns every atom, because the carve is along the channels; the CTA
stages one atom at a time through the output buffer.

## K33: the factorized bucket contraction

<a id="buckets"></a>**A side's KEY LEVEL factors its contraction.** An entry's in-box
deadness is axis-aligned — the dense spans are fully live, one sparse level is not — so
the sparse level's digit becomes a BUCKET INDEX and leaves the contraction. Bucket `d`
of a side keyed at level `l*` owns the `BC / s_{l*}` leaves whose level-`l*` run offset
is `d`, and its panel is the Kronecker product of the DENSE levels alone. A side with
no sparse level has no key: ONE bucket of `BC` leaves whose panel is the whole box,
which is the pre-K33 kernel written in the new vocabulary — no branch, no second path.
The key levels come from the per-level support-mode DECLARATION (law 3), resolved on
the host at the plan seam; nothing here reads a density.

<a id="fold-key"></a>**The fold keys the outermost level, and only it.** A bucket's
leaves are a contiguous run of whole state tiles exactly when no level above the key
has a span, and the fold's accumulator IS the state — it cannot be gathered. A write
side sparse only at an inner level therefore runs the trivial one-bucket fold with a
partially zero panel: exact, uniform, and the saving it forgoes is a consequence of the
leaf order the fold was given ([`box.md`](box.md)).

<a id="state-image"></a>**The readout's operand transits SHARED MEMORY, and it has to.**
For `mma.m16n8k16` the A operand's `k` axis and the accumulator's `n` axis occupy the
lane's low bits IDENTICALLY, so an A fragment can be formed from accumulator registers
with no cross-lane traffic only when its sixteen leaves are two WHOLE in-order n-tiles.
A read bucket's slice is strided, and its values sit in the wrong LANES — no register
pack can move them, and the shuffle form costs 16–32 shuffles and a four-way select per
drain to set up two to four MMAs. So the state is packed `fp32 -> bf16` into a per-warp
`[chan][bucket-major leaf]` image ONCE PER READOUT PHASE — not once per batch, which is
what the pre-K33 kernel paid — and each bucket's A operand is one `ldmatrix`. On sm_90
that image is exactly the shared-memory-resident A operand `wgmma` requires.

<a id="late-scalar"></a>**The panel is ungained and unscaled on both sides.** The
bucket scalar and the write gain are applied LATE, on the entry axis: the readout scales
its ACCUMULATOR COLUMN — so the accumulator literally is the backward's unscaled
`q^{(d)}` at the instruction before scaling (R4 §4.2) — and the fold scales its A
operand's TOKEN axis, four packed multiplies rather than `d_v` of them, which is §4.3's
ungained-panel rule. The mass column's A fragment carries the same scalar in row zero:
moving the gain off the panel moves it there too, or the mass is the one channel that
loses it.

<a id="schedule"></a>**The flush schedule, and the invariant it buys.** Reads of a
window must see the state as of its start, so every pending FOLD lands first; pending
READS were taken against a state those folds would advance, so they land before them.
Each phase empties the other side's buckets before filling its own, which makes "AT MOST
ONE SIDE IS EVER PENDING" an invariant — and cross-window batching is then exactly what
happens across a run of windows that touch one side only. The invariant is also a
MEMORY-SAFETY one: the read buckets and the write buckets are the phase union's two
tenants and share bytes.

## The rings and the phase union

<a id="rings"></a>Two panel slots per side: a group's panel is expanded while the
PREVIOUS group's mma block runs, so the amplitude gather sits underneath arithmetic
instead of in front of it. The value ring is two slots on the same principle. The state
image, the mass row, the read panel ring and the READ buckets are phase-1 tenants; the
value ring, the write panel ring and the WRITE buckets are phase-2 tenants; they are
disjoint in time and share bytes at a fixed offset — the architecture doc's V/staging
alias, widened.

<a id="fill"></a>**The work follows the fill, not the slot.** A drain group is `kGrp`
entries wide and a bucket rarely fills it, so a dead panel row is WRITTEN zero rather
than gathered and multiplied by zero, a dead V row likewise, and the mma trip counts are
predicated on the group's live count (the accumulator index stays a constant — the loops
are unrolled — and the predicate is on the tile). A padding entry's stored scalar is read
as ZERO through the same bound: an mma sums `A[m,k]*B[k,n]` over `k`, so a NaN in `A`
meets `B = 0` and poisons the accumulator rather than vanishing from it.

## Addressing

One pinned 64-bit base per (plane, `bh`) and a 32-bit byte displacement after it, for
every plane: amplitudes, gain, V, `o`, `den`, the state plane. That is the shipped
kernel's discipline (`docs/internals/chunk/chunk_kernel.md#state-addressing`) and it is
what keeps `IMAD.WIDE` out of the loop bodies. The window contributes one uniform
displacement per plane per window, formed once.

## Fan-in

All output accumulation is RED at the top, exactly as the shipped drain does it: the
partial staged for read entry `i` belongs to token `r[i]`, entries past `n` are padding
and contribute nothing rather than contributing zero to somewhere. The denominator's
per-unit partials meet in shared memory first and reach global as one contribution per
token.

## Numerics

No bit-identity target against the shipped kernel: the summation orders differ by
construction (a global window grid instead of per-CTA super-chunks, a two-MMA loop
instead of the fused drain). Conformance is the fp64 oracle within the repository's one
bf16 family bound (`tests/oracle/tolerances.py`). Operands are bf16, accumulators fp32,
the state plane fp32 — the shipped kernel's conventions.
