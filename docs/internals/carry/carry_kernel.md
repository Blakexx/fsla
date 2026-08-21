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

The compaction that turns the votes into those lists costs TWO rendezvous per window and
no serial section. A warp's 32 tokens are one ballot word, so each warp holds its whole
window's ballots in registers and publishes one packed live-count word per (scan
iteration, warp) — the read side in the low half, the write side in the high half, which
is exact because a side's window total is at most `W`. The FIRST barrier publishes those
counts; every thread then re-derives the CTA-wide exclusive prefix from them IN
REGISTERS, because the running totals are CTA-uniform and a shared round trip buys
nothing. The SECOND barrier publishes the lists themselves, which the phases read from
their first batch onward, and it doubles as the separation between this window's count
reads and the next window's count writes — which is why it precedes the whole-window
skip rather than following it.

The list ORDER is part of the contract, not an accident of the implementation: entries
are ascending in token position, which is what makes a batch a contiguous token range
and the fold's accumulation order reproducible. Conformance cannot see a reordering (the
readout is per-row and the fan-in is a sum); the state plane can, and is the instrument.

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

## The rings

The V ring is `STAGES` slots deep, asynchronous, FIFO, one implicit group per batch,
`stage_wait<STAGES-2>` before the consuming rendezvous, and a group committed on every
batch — live or not — so the wait/commit accounting never depends on the data.
`STAGES` is DERIVED by `design::stages_for(slot, fixed, threads, kRegFloor)`, the same
function `chunk`'s `Geom::stages` uses: the shallowest ring that reaches the best
residency this arm can reach at any depth.

The PANEL ring is two slots and costs no shared memory at all. Because all of a
window's readouts precede all of its folds, the write panel is dead through phase 1 and
the read panel is dead through phase 2; each phase double-buffers in the other's buffer
and produces one batch AHEAD. Its two operand bases are NAMED SCALARS selected by the
batch parity — a two-element array indexed at runtime would be a stack frame.

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
For the same reason the mass row's `warp < kMassUnits` guard is written so that it
disappears when the mass split reaches every warp: `__launch_bounds__` does not let the
compiler prove `threadIdx.x < kThreads`, so the tautology would otherwise cut the
readout and the fold into basic blocks.

## The rendezvous the ring needs

Three per read batch and two per write batch is what the naive placement gives; the
kernel runs on TWO and ONE, and the difference is an argument about the ring, not a
tuning choice. Phase 1's body is `[MMA(i) reads panel P(i); writes so and sden]`,
`[produce -> P(i+1)]`, `BAR_B`, `[fan-in reads so and sden]`, `BAR_C`. Phase 2's is
`BAR_D`, `[stage V slot f; fold(i) reads V slot i and P(i)]`, `[produce -> P(i+1)]`.

Every cross-thread edge and the barrier that carries it:

| edge | carried by |
|---|---|
| pre-loop produce -> MMA(0) | the ONE rendezvous before the phase-1 loop |
| produce(i+1) -> MMA(i+1) | `BAR_B(i)` and `BAR_C(i)` |
| MMA(i) writes `so`, mass(i) writes `sden` -> fan-in(i) | `BAR_B(i)` |
| fan-in(i) reads `so`/`sden` -> MMA(i+1)/mass(i+1) rewrite them | `BAR_C(i)` |
| MMA(i-1) reads `P(i-1)` -> produce(i+1) rewrites it (two-slot ring) | `BAR_B(i-1)`, `BAR_C(i-1)` |
| pre-loop produce -> fold(0) | `BAR_D(0)` |
| produce(i+1) -> fold(i+1) | `BAR_D(i+1)` |
| the V ring's `cp.async` group -> fold(i) reads the slot | `BAR_D(i)`, after the per-thread wait |
| fold(i-1) reads V slot -> `stage_batch` rewrites it | `BAR_D(i)` |
| fold(i-1) reads `P(i-1)` -> produce(i+1) rewrites it | `BAR_D(i)` |

So phase 1's HEAD rendezvous is load-bearing only at `i = 0`, where the pre-loop
produce is published by nothing else, and it moves OUT of the loop; phase 2's TAIL
rendezvous carries no edge that `BAR_D(i+1)` does not already carry, and it is gone.
`BAR_C` survives only because `so` and `sden` are single-buffered — double-buffering
them costs `out_bytes + den_bytes` and would put phase 1 on one rendezvous; the cost is
recorded, the change is not made.

## The produce

Per token, `Σ_l s_l` amplitudes are loaded and everything else is computed; the gate
plane is never materialized. A row's `kThreads/C` threads split the box's `BC`
coefficients into `SPT` contiguous LATTICE slots each, and because the lattice index is
a bit permutation a thread's slot run is a sub-lattice: per level the digits it touches
are a CONTIGUOUS run of the owner's own run at a runtime base, and WHICH element of that
run each of its slots wants is a compile-time function of the unrolled slot index. So
the expansion is `Σ_l n_l` amplitude loads followed by `SPT` products of register values
at constant indices — no dynamic indexing, no spill. `load_run<n>` covers a level's run
at the widest access that fits it, `n ∈ {2, 4, 8, 16}`; at the built arms (`SPT = 32`,
`s = (8, 16)`) that is one 4-byte access at level 0 (`n_0 = 2`) and two 16-byte accesses
at level 1 (`n_1 = 16`).

The innermost level supplies the adjacent bf16 PAIR (its `c` field owns bit 0) and every
level above it a per-pair scalar, which is the shipped producer's multiply chain.

The SIDE is a compile-time argument to the produce, because the two sides' trailing
scalars are not the same kind of thing. The write side ends in `mul_bf16x2(acc, gain)`
over all `SPT/2` packed pairs; the read side's scalar is identically 1, and on sm_86 —
where there is no `mul.rn.bf16x2` and a packed multiply is two FMUL, an F2FP pack and
two PRMT — paying for `rn(a*1) = a` was 44% of the read produce. So the read body stores
its accumulators directly.

A dead row (a padding row of a partial batch, which has no token) is then zeroed at
LEVEL 0's run rather than at the trailing multiply. Level 0 is the outermost level and
therefore the narrowest run at every built arm (`n_0 ∈ {1, 2}`), so the whole slot run
is covered by one or two selects and no arithmetic; a zero coefficient contributes
exactly nothing to either MMA. On the write side the same zeroing rides the gain, which
is already a select.

## The produce's thread map, and its fine axis

`ProduceMap<D, B, K, M, C, FINE>` carries the map, and `FINE` — the level the thread
takes WHOLE when the span allows it — is the whole of its freedom. The construction:
order the levels `[FINE, then D-1, D-2, …, 0 with FINE removed]`, lay their `span_bits`
end to end, and cut at `kJBits = log2 SPT`. Bit positions below the cut are supplied by
the thread's own unrolled slot index; positions above are supplied by `rIn`, the
thread's index within its token row. A level may straddle the cut, in which case its
digit is the concatenation of the two halves.

At `FINE = D-1` the order IS innermost-outward, the cut positions coincide with
`run_shift(l)`, and every derived quantity reduces algebraically to the contiguous map:
`n_l = SPT >> run_shift(l)` clamped into `[1, s_l]`, `slot_of(j) = j`, and
`slot0 = rIn·SPT`. That is why `FINE = D-1` is the default and why the reverse family,
which shares this type, is untouched by the parameter's existence.

A side whose declaration names exactly one sparse level takes THAT level as its fine
axis, so the compaction below walks the thread's own digits and the trip count is
identical across the four threads of a token row. At the flagship's spans `(8, 16)` the
read side's declared sparse level is already the fine one; the write side's is not, and
its map becomes all eight level-0 digits by four level-1 digits — the four threads of a
row splitting level 1 instead of level 0.

## The compacted produce

A level declared SPARSE for the side being produced contributes only at the digits the
support plane marks live, so its factor loop is an `__ffs` walk over the thread's
support slice — one aligned 32-bit load, because a level's thread base is a multiple of
`n_l` and the owner's run base a multiple of `s_l ≥ n_l`. The dense levels' product,
which every live digit reuses, is formed once into `ND/2` packed pairs beforehand; every
slot the walk does not reach is zero-filled at the FOOTPRINT's granularity (the
innermost level owns bit 0 of the slot index, so the footprint is `SPT / n_{D-1}`
contiguous runs) rather than at the walk's.

The arithmetic is the dense body's, factor for factor. The write side is
`dp = level (D-1); dp *= splat(sparse); store rn(dp·gain)` — the dense chain verbatim.
The read side is `dp = level 0; dp *= splat(level 1)` against the dense body's
`acc = level 1; acc *= splat(level 0)`: two factors, and bf16 round-to-nearest multiply
commutes. Skipped digits test EXACTLY zero and `rn(0·x) = 0`, so a skipped slot's dense
value is the zero already written. The state plane is therefore bit-identical to the
dense binary's on every cell, which is what the gate asserts.

The live amplitude is picked out of a per-thread SHARED scratch. A runtime index into a
register array is a stack frame; a select ladder over one costs more issue than the
multiply chain it replaces; and a second global gather puts two serialized round trips
on the produce's critical path. Shared memory is the only store on this machine that
indexes at run time without any of the three.

The form is built for ONE sparse and ONE dense level per side. A longer factor chain
would have to reconcile its multiply order with the dense body's before it could claim
the same rounding, and the refusal is a `static_assert` rather than a runtime branch.
The declaration itself is resolved at the seam: a declaration this family has no
compacted form for is served `DENSE_BOTH`, which is always correct.

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
