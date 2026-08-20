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

## The carve, as one compile-time axis of one body

`Carve::STATE` gives each warp `BC/kWarps` leaves across all `d_v` channels; its readout
is a partial over the whole output, so the four warps meet in an intra-CTA shared-memory
tree. `Carve::CHANNEL` gives each warp 16 of the 64 channels across all `BC` leaves; its
readout rows are disjoint and it reduces nothing. Study §2.8 recommended the former and
named the latter the measured control. R1's measurement is in
`rola-scratch/workflows/k31-r1-carry/FINDINGS.md`.

The fold-residency coupling law is binding either way: the warp carve of the state IS
the output carve of the fold, and both branches of this body obey it.

## The atom walk's shape is a register-indexing fact

A state fragment is a register array and a runtime index into one is a stack frame —
measured, on this kernel's first build: walking `kAtoms` at runtime put the whole state
(320 B/thread) in local memory. So the walk is inverted. The outer, fully unrolled loop
counts the atoms THIS WARP owns, which makes its tile index a compile-time `2*a`; the
inner loop names which warp's atom the CTA is cooperatively staging through the output
buffer. Under the CHANNEL carve `kStageUnits == 1` and the walk collapses to the shipped
kernel's shape exactly.

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
and produces one batch AHEAD, which puts the amplitude gather's latency underneath the
previous batch's MMA block. Its two operand bases are NAMED SCALARS selected by the
batch parity — a two-element array indexed at runtime would be a stack frame.

## The produce

Per token, `Σ_l s_l` amplitudes are loaded and everything else is computed; the gate
plane is never materialized. A row's `kThreads/C` threads split the box's `BC`
coefficients into `SPT` contiguous LATTICE slots each, and because the lattice index is
a bit permutation a thread's slot run is a sub-lattice: per level the digits it touches
are a CONTIGUOUS run of the owner's own run at a runtime base, and WHICH element of that
run each of its slots wants is a compile-time function of the unrolled slot index. So
the expansion is `Σ_l n_l` amplitude loads followed by `SPT` products of register values
at constant indices — no dynamic indexing, no spill.

The innermost level supplies the adjacent bf16 PAIR (its `c` field owns bit 0) and every
level above it a per-pair scalar, which is the shipped producer's multiply chain. `have`
false makes the gain zero, and a zero coefficient contributes exactly nothing to either
MMA — one select on the leading factor covers the whole run.

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
