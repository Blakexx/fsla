# `csrc/rola/src/consumer_kernel.cuh` — the RoLA V3 consumer kernel body

THE ENDGAME EXECUTOR, and the ONLY consumer. This doc owns the kernel BODY: the
schedule ring's producer (the streamed membership scan and the cooked group
descriptor), the compact-into-registers atom walk, the four GEMMs in their phase
order, and the split-K epilogue.

**The body composes two layers and each has its own document:**

| document | source | owns |
|---|---|---|
| [`smem_layout.md`](smem_layout.md) | `smem_layout.cuh` | the carve — address functions, conflict proofs, the register/residency budget, the byte ledger |
| [`mma_tile.md`](mma_tile.md) | `mma_tile.cuh` | the operand maps — fragment order, the `ldmatrix` atoms, the GEMMs' inner sequences |

The kernel body's own `__syncthreads()` carries every intra-CTA phase ordering — there
is no separate item-stream layer: [below](#visit-barrier).

Its sibling is [`consumer.md`](consumer.md), which owns `ConsumerParams` and the host
side and is NOT restated here: the leaf-digit decomposition, the paged state layout
and `kAbsentPage`, the §6.4 work list and its workspace, the three-pass driver, and
the instantiation matrix all live there. Two descriptions of one kernel is one that
can go stale.

## Reference conventions

The same in all three consumer-kernel documents — [`consumer_kernel.md`](consumer_kernel.md),
[`smem_layout.md`](smem_layout.md), [`mma_tile.md`](mma_tile.md):

- **`§N` is ALWAYS a SPEC reference.** `ROLA_V3_SPEC.md` unless another spec is named
  at the site; `GLA_MASS_DECAY_SPEC.md` §12 is the decay numerics contract. It never
  means a section of these documents.
- **A section of these documents is referenced by LINK**, never by its number, so a
  reference resolves without the reader knowing which of the four holds it.
- Section numbers run continuously across the four documents, so a number identifies
  one section — but it is the link, not the number, that is the reference.

The spec sections this document leans on: §3 topology-agnostic kernel structure, §4
schedule, §5 consumer, §6.4 sequence parallelism, §7.4.1 the decay lowering.

---

## <a id="orientation"></a>1. Orientation — what the kernel does, in the order it does it

One CTA per `(owner, segment)` work item, `kConsumerThreads == 256` threads (8 warps),
run to completion. The owner's state block lives in SHARED memory for the CTA's whole
life, is read from HBM ONCE at entry and written ONCE at exit.

**Per CTA, once:**

1. resolve the work item (`bh`, `owner`, §6.4's `seg` / `nseg` / `owner_slot0`) and
   this segment's word range `[w_lo, w_hi)` over the scan stream;
2. decompose the owner into per-level digit bases `base[l]` (§3's rectangle);
3. hoist the owner's leaf digits `dig[l][i]` — `D * BC` integer divisions, once,
   instead of once per (group, token, leaf, level);
4. under `DECAY`, form the per-leaf `log_keep[i] = log1pf(-rate[i])`;
5. load `S[BC][SP]` — zeros in pass 1 (the summary pass), from the §6.4 workspace on
   a split owner's non-entry segment, else atom-wise from `p.state` through the page
   table;
6. warp 0 [cooks](#producer) the schedule ring's first `kSchedDepth` group
   descriptors — the cold start — and a `__syncthreads()` publishes them.

**The walk**, once per GROUP of `kGroupTokens == 16` gathered tokens (every group is
full except each segment's final ragged flush; visit count is exactly
`ceil(gathered / 16)`):

7. the visit [reads its cooked descriptor](#group-assembler) straight off the
   [schedule ring](#sched-ring) — `t_idx`, the read/write atom masks, the side flags,
   the pre-multiplied row scales `sr`/`sw` — and assembles nothing; concurrently, warp
   0 cooks the ring's NEXT descriptor (the [run-ahead](#sched-ring)), published by this
   visit's own end-of-visit barrier;
8. **STAGE** — under `DECAY`, the clock prefix and entry scale; the §3 step-3 row
   scales `sr`/`sw`; the fragment-ordered read panel `Rf`; the write panel `WT`; the
   scale-folded value tile `Vw` — every leaf-axis loop walking LIVE ATOMS ONLY;
9. **GEMM 1, the readout** — `y_group = R . S`, live read atoms only, accumulating
   into each warp's own registers; the mass column's [TERM 1](#mass-term1), one more
   `n8` tile on WARP 0, beside it;
10. **GEMM 2, the Gram** — `G += R_q . W_q^T` over atoms live on BOTH sides, then §5's
    inclusive causal mask and the publication of `Gm`;
11. the visit's [ONE internal barrier](#visit-barrier), guarded `mixed`, carrying both
    the readout-before-fold edge on `S` and the Gram-before-phase-C edge on `Gm` at
    once;
12. **PHASE C**: **GEMM 3, the intra term** — `y_group += G . Vw` into the same
    registers the readout used, plus the mass column's [TERM 2](#phase-c);
13. **EMIT** — the split-K `atomicAdd` of `sr[t] * y` into `num`, and, out of the
    [mass column's D fragment](#mass-column), of `sr[t] * dm` into `den`;
14. **GEMM 4, the fold** — `S += W^T . Vw`, live write atoms only, walked as a STATIC
    STRIDED PARTITION over `(atom, n-tile)` items, with `DECAY`'s exit scale riding its
    read-modify-write and the mass column's [TERM 3](#gemm-fold) — one more `mma.sync`
    per live atom, on the same tensor core as the fold itself — beside it.

**Per CTA, at exit:** under pass 1 (the summary pass), publish the sparse `(A, B)`
written-leaf summary; otherwise, if this is the owner's last segment, store `S` back
atom-wise.

Three facts a reader must not miss:

- **The state is SMEM-resident, not register-resident.** That is what decouples the
  thread count from the state layout, frees the CTA to 8 warps, and puts every
  per-visit term on the tensor cores as one of four GEMMs.
- **The one-copy invariant is a property of the LAUNCH SHAPE**, not of a policy the
  kernel implements: one CTA per owner, run to completion, one HBM read and one HBM
  write of the owner's state block. There is no wave machinery anywhere.
- **Every structural axis is a template parameter and no axis has a runtime
  fallback.** `BC in {16, 64}`, `DV in {32, 64}`, `D in {1..4}`, `DECAY in {0, 1}` —
  32 arms total, see [`consumer.md`](consumer.md#instantiation-matrix). The endgame
  walk IS the path; dense (full-support) routing self-heals to identity grouping
  rather than taking a different code path.
- **The visit pays exactly TWO unconditional `__syncthreads()`** — the post-staging
  barrier (staging into GEMM 1/GEMM 2) and the end-of-visit barrier that closes the
  loop and publishes the producer's run-ahead cook — plus the [visit barrier](#visit-barrier)
  (readout before fold on `S`, Gram before phase C on `Gm`), which is CTA-UNIFORM but
  not unconditional: it is guarded `mixed`, because both of its edges are (a
  read-only visit runs no fold, a write-only visit reads no state, and neither has a
  Gram), so a single-sided visit pays nothing for it. `DECAY` arms pay two more, both
  inside staging: after the clock-prefix pass (before it scales `S`) and at
  [`clk-dead`](#clk-dead) (before `Vw` may overwrite `clk`'s aliased bytes). There is
  no separate group-entry barrier: the ring publishes
  a visit's descriptor a whole visit ahead of the visit that reads it.

## <a id="glossary"></a>2. Symbol glossary

| symbol | what it is |
|---|---|
| `BC` | leaves per owner's compute block; `{16, 64}` |
| `DV` | value width `d_v`; `{32, 64}` |
| `D` | level count of the topology; `1 <= D <= 4` (spec §1) |
| `j` | `first_spanned_level`. Levels `l < j` are FIXED for this owner |
| `kMmaKQuantum` | `16` — the hardware `mma.sync` k-extent on the sm80 bf16 path, and the sector invariant's other half ([`smem_layout.md#cute-pin`](smem_layout.md#cute-pin)). Defined once, in `arch_table.cuh` |
| `kGroupTokens` / `kGT` | `16` — the work unit: one group of gathered tokens, == `kMmaKQuantum`. `kGT` is the kernel body's local alias |
| `QPO` / `kQPO` | atoms per owner, `BC / kMmaKQuantum`. An ATOM is one `kMmaKQuantum`-leaf block in canonical leaf order, and also §4.5's page granule |
| `NT_DV` | `DV / 8` — the `n8` column tiles of a `DV`-wide output |
| `kWarps` | `8` — `kConsumerThreads / 32` |
| `kFullMask` | the atom masks' FULL value, `(1u << kQPO) - 1u` (or all-ones past 32 atoms). `mask == kFullMask` selects the statically-unrolled dense arm |
| `S` | the STATE, `[BC][SP]` fp32 SMEM, resident for the owner's whole run |
| `SP` | `S`'s row pitch, `DV + 4` ([below](#state-pitch)) |
| `cols` | `DV + 1`, UNCONDITIONAL — the mass column always exists |
| `WT` | the write coefficient panel, LEAF-MAJOR, `[BC][BTP]` bf16 |
| `BTP` | `WT`'s row pitch, `kWtPitch = kGT + 8 = 24` ([`smem_layout.md#stage-format`](smem_layout.md#stage-format)) |
| `Vw` | the scale-folded value tile, `Vw[t,ch] = sw[t] * v[t0+t, ch]`, bf16, swizzled, no pad |
| `Gm` | the causal effective-score block `[kGT][kGT]`, pitch `GMP = kGT + 8 = 24` |
| `Rf` | the read coefficients, MMA-FRAGMENT-ORDERED, `frag_a_words(BC)` words, atom-blocked |
| `clk` | the visit's per-leaf INCLUSIVE clock prefix `[kGT][CKP]` fp32, `CKP = BC + 4`; ALIASES `Vw`'s bytes ([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)) |
| `log_keep`, `p_ref`, `s0` | `DECAY`'s three `[BC]` vectors — per-owner log retention, §12 rule 1's reference exponent, and the entry/exit scale |
| `sr`, `sw` | the §3 step-3 per-token read/write ROW SCALES, `[kGT]` each — retargeted every visit into `g_up`'s matching ring slot, [cooked by the producer](#upper-product) |
| `dm0..dm3` | the [mass column's D fragment](#mass-column), register-resident on WARP 0 — one `[16 x 8]` accumulator, to the denominator exactly what `y0..y3` is to the numerator |
| `dig` | the hoisted leaf digits, `[D][BC]` |
| `t_idx` | the visit's `kGT` gathered GLOBAL token indices, ascending — retargeted every visit onto the [schedule ring](#sched-ring)'s matching slot, never copied |
| `kSchedDepth` / `kSchedSlots` | the producer's run-ahead depth (`ROLA_SCHED_DEPTH`, a `constexpr` dial) and the ring's slot count, `kSchedDepth + 1` ([`smem_layout.md#sched-ring`](smem_layout.md#sched-ring)) |
| `g_ring` | the [cooked schedule ring](#sched-ring), `kSchedSlots` group descriptors, `kSchedSlotInts` ints each |
| `g_up` | the ring's per-owner [upper-product](#upper-product) pairs, `kSchedSlots * 2 * kGT` floats |
| `p_st`, `p_chunk` | the [producer's cursor state](#producer-state): seven uniform scalars (`p_st`) and five 32-lane planes (`p_chunk`, the cooked chunk's `fr`/`fw`/`member` and the selected word's two atom-bitlist rows) |
| `A`, `B` | §6.4's per-segment summary: the per-leaf survival product and the accumulated deposit |
| `w_lo`, `w_hi` | this segment's word range over the scan stream, `[w_lo, w_hi)` |
| atom | one `kMmaKQuantum`-leaf block of the owner's leaf axis; also one 32-byte fetch sector ([`smem_layout.md#cute-pin`](smem_layout.md#cute-pin)) |
| group | the work unit: `<= kGT` gathered, causally-ordered member tokens |

## <a id="one-precision-story"></a>The one precision story

bf16 storage everywhere (staged operands, amplitude streams, the value stream); fp32
accumulation, state, clock, row scales, and — after U07-d — the decay panel
application; fp64 torch reference as the oracle. There is no `MmaPrec` axis, no
tf32x3, no `PREC` template parameter. U07-d built an fp16 arm for the decay panel
application and CARVED it out on a measured overflow breach: an admitted hot routing
pushed the survival factor past fp16's 65504 max (measured NaN, not a rounding
error) — see the kernel header comment (`consumer_kernel.cuh` lines ~36-42, ~395-401)
and `workflows/perf/p26_evidence/m2_fp16_breach.txt` for the account; revival is git
history, not this doc's job to narrate further.

---

# Part I — the compile-time budget

## <a id="force-ctas"></a>3. `ROLA_FORCE_CTAS` — the forcing-census instrument

When defined, it replaces the DERIVED launch bound
(`ROLA_CONSUMER_CTAS_EXPR = ctas_for_residency(consumer_smem_bytes<...>(),
consumer_reg_budget(...))`) with a fixed CTA count on every instantiation, so a
candidate register budget can be censused over the whole dispatch matrix in one probe
compile. `tools/ratify.py --census` is its only legitimate definer, and a manifest is
only ever ratified off the UNFORCED compile — the extension build defines no such
flag. Forcing changes the register side only; the SMEM derivation (`consumer_smem_bytes`)
stays shipped, which is what the census wants: the register file is the axis under
measurement, the footprint is not.

---

# Part II — the carve, as the body spells it

## <a id="state-pitch"></a>4. `SP = DV + 4` — the state's row pitch

`SP` is DERIVED from the bf16 MMA B map: `S` as GEMM 1's B operand is read with
`k = k0 + 2r`, so at `SP == DV` (a multiple of 32) the bank index `(2r + g) % 32`
COLLIDES up to four ways (`2r + g = 6` is hit by `(r,g) = (0,6), (1,4), (2,2), (3,0)`).
`DV + 4` makes the bank `(8r + g) % 32` with `r in [0,4)`, `g in [0,8)` — every one of
the 32 combinations distinct, covering all 32 banks exactly once. This is the kernel's
ONE state pitch; there is no separate tf32 derivation to reconcile it with, because
there is no tf32 arm.

## <a id="pitches"></a>5. Every operand pitch proves its own conflict-freedom

Each buffer's row pitch is pinned by a `static_assert` at the point the kernel body
carves it, so a future `BC`/`DV` that broke the proof fails the BUILD:

| assert | what it pins |
|---|---|
| `banks_distinct(1, 2 * SP)` | `S` as GEMM 1's B operand |
| `banks_distinct_u16(1, 2 * BTP)` | `WT` as GEMM 2's B operand |
| `banks_distinct_u16(BTP, 2)` | `WT` as GEMM 4's A operand |
| `pitched_ldsm_banks_ok<kMmaKQuantum, kGT, kWtPitch>()` | `WT`'s `ldmatrix` fetch conflict- and alignment-freedom at `kWtPitch` |
| `pitched_ldsm_banks_ok<kGT, kGT, GMP>()` | `Gm`'s `ldmatrix` fetch conflict- and alignment-freedom at `GMP`, as GEMM 3's A operand |
| `stage_offset_matches_cute<kGT, DV>()` | `Vw`'s closed-form address IS CuTe's layout |
| `stage_layout_is_bijection<kGT, DV>()` | the swizzled `Vw` tiles `[0, kGT*DV)` exactly |
| `stage_ldsm_banks_ok<kGT, DV>()` | `Vw`'s `ldmatrix.x2.trans` fetch |
| `stage_store_banks_ok<kGT, DV>()` | `Vw`'s staging store |

The predicates themselves, and the swizzle CuTe owns for `WT`/`Vw`, are
[`smem_layout.md#bank-predicates`](smem_layout.md#bank-predicates) and
[`smem_layout.md#stage-layout`](smem_layout.md#stage-layout).

## <a id="ldsm-alignment"></a>6. `ldmatrix` takes a 16-byte-aligned ROW ADDRESS

So every `ldmatrix` source base (`WT`, `Vw`, and `Gm` — GEMM 3's A operand and the
intra mass term's, both `ldmatrix`-fed) must be 16-byte aligned — a requirement
`align4` does NOT imply. Stated rather than trusted: `kWtOffsetBytes`,
`kVwOffsetBytes` and `kGmOffsetBytes` are computed as explicit expressions from the
carve and `static_assert`ed `% 16 == 0`, because CUDA aligns the dynamic shared window
itself to 16 B but says nothing about what a mid-carve `align4` offset lands on.

## <a id="atom-row-base"></a>7. `atom_row_base` — §4.5's paged address, re-keyed to the ATOM

A page is ONE `kMmaKQuantum`-leaf ATOM, not one owner, so an owner spans `QPO =
BC / kMmaKQuantum` INDEPENDENTLY PAGED atoms whose slots are not contiguous in
general — a CTA resolves `QPO` bases at startup instead of one. `QPO <= 4` and both
`BC` and the quantum are compile-time, so the resolving loop is `#pragma unroll`ed
over a compile-time trip count, never a runtime-indexed register array.

`atom_row_base<BC>(p, bh, owner, q)` returns the first STATE ROW of atom `q` of
`owner`, or `-1` when the atom has no committed slot — which, under total admission,
means no write candidate at all, hence a dense state that is identically zero. ONE
EXPRESSION SERVES BOTH LAYOUTS: unpaged is the identity slot map (atom `q` of owner
`o` is row `bh * N + o * BC + q * kMmaKQuantum`), so paged and unpaged are the same
addressing strategy at different slot maps. `atoms_per_bh = N / kMmaKQuantum` is
DERIVED rather than carried — the host sets `p.N = owners_total * BC` under paging, so
there is no second place for the table's stride to be wrong.

---

# Part III — CTA entry and the group scheduler

## <a id="entry-state"></a>8. Where this segment's entry state comes from

```
pass 1 (summary)      : ZERO.
pass 3, split owner,
  not first segment   : the §6.4 workspace slot owner_slot0 + seg.
pass 3, unsplit owner,
  or split owner's
  first segment        : the owner's own state, paged or dense atom-wise.
```

The dense read is ATOM-WISE: the outer loop is `#pragma unroll`ed over a compile-time
`kQPO <= 4`, each base resolved once via `atom_row_base`, and the inner walk covers
`kMmaKQuantum * cols` contiguous elements per atom. An atom with no committed slot
reads as ZEROS and is never stored back — the same arithmetic the dense path applies
to a never-written atom, which is what makes paging BITWISE INVISIBLE rather than
approximately equivalent.

`cols = DV + 1` is unconditional; there is no `raw`/`GLOBAL` axis left to branch it on.

## <a id="pass-mode"></a>9. `pass_mode` — §6.4's three passes, LAUNCH-UNIFORM

`p.pass_mode` is `kPassSummary` (pass 1: entry state zero, no read emitted, publishes
the sparse `(A, B)` summary) or `kPassRead` (pass 3: reads emitted, entry state as
[above](#entry-state), the last segment publishes). Pass 2 is a separate launch
(`scan_kernel`) and never appears here. `summary = (p.pass_mode == kPassSummary)` is a
LAUNCH-UNIFORM boolean, the same structural status as `is_read`/`is_write` below — it
drives branches that are structure, never a distributional property.

## <a id="digit-hoist"></a>10. The leaf-digit hoist, and `log_keep`

§3's leaf-digit decomposition, HOISTED TO THE OWNER: `leaf_digit(level, base, i) =
base + (i / stride) % modulus` is a runtime integer division and modulo — hardware has
no integer divide, so each is a multi-instruction sequence — and it depends on none of
the group, the token or the side, only on `(level, owner, leaf)`. Computed once per
CTA it costs `D * BC` evaluations for the owner's whole run (`D * BC <= 4 * 64 = 256 ==
kConsumerThreads`, at most one iteration per thread) rather than once per (group,
token, leaf, level).

Under `DECAY`, §12 rule 2's per-leaf log-retention runs once per owner, in log space:
`rate[i] = prod_l dial_l[level_row_offset[l] + leaf_digit(l, base[l], i)]`,
`log_keep[i] = log1pf(-rate[i])`. `log1pf` IS the composition rule 2 requires — never a
rounded linear-domain `1 - rate` then logged — and it re-evaluates `leaf_digit` rather
than reading the `dig` table computed just before, because `dig` is indexed
`l * BC + i` (written by a different thread than would read `i` here) and using it
would need a barrier the digit hoist itself is cheap enough to not need.

---

# Part IV — the walk

## <a id="producer"></a>11. The producer — warp 0 cooks the schedule ring

`w_lo`/`w_hi` bound this segment's range over the SCAN STREAM's word axis — the
32-token machine word is the metadata granule under §6.4's P > 1 partition, not a tile
range. WARP 0 ALONE runs the membership scan and the ballot: it ANDs the owner's
fixed-level (`l < j`) `read_tok_bits` / `write_tok_bits` rows and the token-granular
slice-union word, ballots members into a group ascending, ORs the per-token atom-union
bytes into the group's read/write atom masks, and publishes the result as a COOKED
GROUP DESCRIPTOR into [the schedule ring](#sched-ring) — once, for every consumer.

The cook happens ONCE PER OWNER's group, not once per warp. The scan's result — which
tokens the group gathers, which atoms are live on each side — is a fact of the owner
and the group alone; every one of the CTA's 8 warps would derive the identical
`rmask`/`wmask`/`t_idx` from the identical inputs, so computing it 8 times is redundant
ISSUE, not redundant latency. That distinction is the whole rationale: this kernel's
binding constraint at scale is instruction issue (the P11 verdict — a prefetch that
buys stall movement with instructions loses here), not memory or execution latency, so
the win available is cutting issued instructions, and a single producer warp cuts the
scan's ballot ALU and its `2 * QPO` atom-bitlist loads by 8x.

**Stability is the correctness requirement.** Words ascend and each word is consumed
LSB-first, so a descriptor's `kGroupTokens` slots are GLOBALLY ascending — the intra
Gram's inclusive causal mask is order-relative, and this ordering IS what makes it
correct.

A side whose visit mask is empty (`rmask == 0` or `wmask == 0`) has no live
coefficients anywhere in the group on that side; the descriptor's side flags carry `gr`
/ `gw` (whether ANY member token is read-/write-live on any FIXED level) alongside the
masks, and `is_read`/`is_write` fold that emptiness in, so a whole-visit skip is
possible even where the per-tile candidacy machinery of an earlier design could not see
it.

## <a id="sched-ring"></a>12. The schedule ring

The producer runs `kSchedDepth` groups ahead of the consumers (a `constexpr` dial,
`ROLA_SCHED_DEPTH`, never a template axis — the instantiation matrix does not gain a
coordinate). The ring holds `kSchedSlots = kSchedDepth + 1` descriptors: cooking group
`gidx + kSchedDepth` writes the slot that group `gidx - 1` vacated, which is never the
slot the current visit is reading — the ring must be AT LEAST 2 deep for that to hold,
which is why `kSchedSlots` is `depth + 1` rather than `depth`.

Publication rides the END-OF-VISIT `__syncthreads()` that already exists (every visit
closes on one): the producer's run-ahead cook for the NEXT slot runs concurrently with
the current visit's compute, at the bottom of the loop body, and the barrier that
already stands between one visit and the next is what makes the cooked descriptor
visible to every warp before the visit that consumes it starts. The ring therefore
carries no cursor word of its own and needs no barrier beyond the one the visit loop
already pays.

One descriptor (`kSchedSlotInts` ints) is:

```
[0 .. kGroupTokens)   the group's gathered GLOBAL token indices, ascending
[kSchedGN]            gN, the member count (0 is the stream-end sentinel)
[kSchedRMask]         the read atom mask
[kSchedWMask]         the write atom mask
[kSchedFlags]         side flags: bit 0 = read-live, bit 1 = write-live
```

Consumption multiplicity is `n_atoms + 1` (every atom-major consumer plus the group
scheduler itself), which is what earns the descriptor its own ring storage; the
scan's raw INGREDIENTS — the per-word AND results and the atom bitlist rows — are read
only by the producer and never materialize past its own cook.

## <a id="upper-product"></a>13. The per-owner upper product — [row scales](#row-scales), cooked a visit early

The fixed levels (`l < j`, `j = first_spanned_level`) contribute amplitudes that are
CONSTANT ALONG THE LEAF AXIS: `j` scalars per member token per side, common to every
atom's Kronecker coefficient. Their per-token product is therefore an OWNER-LEVEL fact,
governed by the same multiplicity rule membership is — computed ONCE PER OWNER, never
once per atom.

The product rides the ring slot beside the member word: `g_up[slot]` holds one pair of
`kGroupTokens` floats, `sr` (the read row scale) then `sw` (the write row scale), cooked
by the producer in the same pass that ballots the group. Because it is cooked with the
descriptor it arrives PRE-MULTIPLIED a whole visit early — the `2 * j` fp32 gathers this
would otherwise cost on the visit's critical path move into the run-ahead instead.

**A ragged group's padded slots carry EXACT ZERO row scales.** That is what makes their
coefficient rows inert through every GEMM without a membership test, and it is what
lets the [split-K emission](#split-k-emission)'s `out == 0` early-out skip a padded
slot without ever reading its (undefined) token index — the zero row scale is checked
before `t_idx` is touched.

## <a id="producer-state"></a>14. The producer's cursor rests in SMEM between cooks

The scan's cursor — the word cursor, the chunk base, the live-word mask, the selected
word, the remainder — and the cook's per-lane planes (the chunk's `fr`/`fw`/`member`
words and the selected word's two atom-bitlist rows) live in SMEM (`p_st`, `p_chunk`)
between cooks and are read at the point of use rather than held in registers across the
cook.

This is a MEASURED constraint, not a default: scheduler state held in registers is live
across the WHOLE visit body — the cursor is loaded once per cook and the cook runs
before the four GEMMs that dominate the visit's register pressure — and it competes
with their working set. Measured over all 32 arms on both architectures, holding it in
registers spills the `BC = 16, decay = 0` key, which carries the tightest register
budget in the matrix
([`smem_layout.md#reg-budget`](smem_layout.md#reg-budget)).

The chunk is still COOKED in registers — the fixed-level AND and the live-word ballot
happen there, one word per lane of the producer warp
([`smem_layout.md#sched-chunk`](smem_layout.md#sched-chunk)) — only the RESTING PLACE
between cooks is SMEM.

### <a id="producer-cursor"></a>HAZARD `producer-cursor` — the cursor crosses lanes at both ends of a cook

`p_st` is written by LANE 0 and read by ALL 32 lanes, so under independent thread
scheduling every crossing needs an explicit reconvergence; neither end can borrow the
visit's `__syncthreads()`. The LOAD's half: the cold start calls the cook twice in a row
with no barrier between, so the previous cook's store — and the once-per-CTA
initialization before it — are not covered by any CTA barrier in that window. The
STORE's half: on a stream-end cook (the scan is exhausted, the group is empty) the cook
returns without executing a single warp collective, so nothing separates the other
lanes' load of `p_st` from lane 0's store of it. A `__syncwarp()` stands at each end.

Both halves are detectable: `compute-sanitizer --tool racecheck` reports the shared
hazard when either `__syncwarp()` is removed and reports neither when both stand.

### <a id="producer-chunk"></a>HAZARD `producer-chunk` — the chunk planes are written per lane and read by index

At a refill, lane `w` writes word `w`'s `fr`/`fw`/`member` plane entries; word selection
then reads the SELECTED lane's entries FROM EVERY LANE, by the uniform index the ballot
picked. That is the same cross-lane crossing as the cursor and takes the same explicit
`__syncwarp()`, placed at the end of the refill body so it also covers the selection
that follows the refill loop. The ingredient planes need no such barrier: each lane
reads only its OWN slot.

## <a id="group-assembler"></a>15. The visit reads a cooked descriptor; it assembles nothing

Before the first visit, the producer (warp 0) fills the ring's first `kSchedDepth`
slots — the cold start — and a `__syncthreads()` publishes them. Every subsequent visit
then does no assembly at all: `gidx`'s slot is `g_ring + (gidx % kSchedSlots) *
kSchedSlotInts`, and `gN = R[kSchedGN] == 0` is the stream's end, as the producer saw
it. `t_idx` retargets to the slot directly (`R`'s first `kGroupTokens` ints) rather
than being copied, and `sr`/`sw` retarget to the matching `g_up` slot — every staging
loop reads the producer's published descriptor in place.

At the bottom of the same visit, the producer cooks the NEXT group ([the run-ahead —
above](#sched-ring)) concurrently with this visit's GEMMs, and the visit closes on the
same end-of-visit `__syncthreads()` that publishes it. One cooking producer against
eight reading warps is the whole point of the arrangement: the ballot ALU and the
atom-bitlist load issue are paid once per owner rather than once per warp.

## <a id="staging"></a>16. The staging — everything the visit's compute consumes but does not produce

`stage_group(is_read, is_write, gN, rmask, wmask)` runs, LIVE ATOMS ONLY, for every
leaf-axis loop:

- under `DECAY`, the clock prefix and entry/exit scale ([below](#decay-clock));
- the read panel `Rf` ([below](#read-panel));
- the write panel `WT` ([below](#write-panel));
- `Vw[t, ch] = sw[t] * v[token, ch]` — the write side folds the row scale into the
  value tile because there `t` is the CONTRACTION index and the scale cannot hoist out
  of the sum.

The [row scales](#row-scales) `sr`/`sw` are NOT computed here: they arrive
PRE-MULTIPLIED in the ring slot, [cooked by the producer](#upper-product) a whole visit
ahead, so `stage_group` only reads them (in the `Vw` fold above and at
[emission](#split-k-emission)).

Tokens are `t_idx[0..gN)`, globally ascending; a ragged group's padded slots
`[gN, kGT)` stage EXACT-ZERO coefficient rows and zero row scales, so they are inert
through every GEMM without any membership test.

`mask == kFullMask` selects the statically-unrolled dense arm at every one of these
loops — dense inputs pay zero mask instructions, because they self-heal to identity
grouping rather than taking a different code path.

## <a id="decay-clock"></a>17. `DECAY` steps (a) and (c) — the clock prefix, `P_ref`, `s0`, and the entry scale

Guarded by `is_write` structurally: a group with no write-live members has `c == 0` on
every leaf, so every factor this block computes is the identity and the plain path is
EXACT, not approximate.

The lowering is §7.4.1's decay-aware panels. `keep[t,s] = (1-rate[s])^c[t,s]`, so with
`P[t,s] = sum_{u<=t} c[u,s]` the survival from `j` to `t` is `keep^{P[t]-P[j]}`, and
that exponent SPLITS ACROSS THE TWO OPERANDS of a Gram because it is a difference of a
per-`t` term and a per-`j` term. With §12 rule 1's per-leaf reference
`P_ref = P[kGT-1] / 2`:

```
fr[t,s] = exp((P[t,s] - P_ref[s]) * log_keep[s])      folded into Rf
fw[j,s] = exp((P_ref[s] - P[j,s]) * log_keep[s])      folded into WT
s0[s]   = exp(P_ref[s] * log_keep[s])                 scales the STATE
fr[t,s] * fw[j,s] = survival(j -> t)                  (the identity)
```

so every term keeps its existing shape and the ONLY new arithmetic is the clock
prefix and two exponentials per (token, leaf). **One write panel serves both the Gram
and the fold**, which is what rule 1's half-mass shift buys.

The clock product runs over ALL levels, including the fixed ones (`l < j`): a fixed
level's factor is constant along the owner's leaves and hoists into the row scale, but
the clock is an EXPONENT and a row scale cannot carry it.

`panel_read_factor` / `panel_write_factor` are TAKEN from `intra_panel.cuh` rather
than reimplemented — the one clamp point and the one exponent sign convention.

**A dead write atom's leaves take `acc == 0`, hence `s0 == 1`.** This is the
architecture fact that makes the atom skip legal under decay: "ticks ride writes" —
the leaves the fold never touches this visit are exactly the leaves whose exit scale
is exactly 1, so skipping a dead atom's fold is the identity, not an approximation.

### <a id="clock-prefix-fp32"></a>HAZARD `clock-prefix-fp32` — the prefix accumulates in fp32

`acc` is fp32 and must STAY fp32. This is a requirement, not a default (spec §7.4): the
prefix multiplies `log_keep` and is then exponentiated, so a lost low bit here is an
EXPONENT error, not a mantissa one. One thread per leaf, serial over the group's `kGT`
tokens, because the prefix IS a serial dependence in `t` and `kGT <= 32` — a parallel
scan would cost more barriers than it saves steps. A ragged group's padded slots take
a FINITE zero prefix (`clk[t*CKP+i] = 0` for `t >= gN`) so both panel factors stay
finite (`keep^0 == 1`); their coefficient rows are exact zeros anyway.

## <a id="row-scales"></a>18. §3 step 3's row scales

`sr[t]`/`sw[t]`: the group's per-token read/write row scales — the read/write gain
times the FIXED levels' amplitudes at the owner's own digit, constant along the leaf
axis (which is why they are held out of the coefficient blocks entirely). They depend
on nothing the group's leaf-axis loops compute, only on `(bh, token)` and the owner's
`base[]` — an OWNER-LEVEL fact, cooked once per owner's group by the producer alongside
the membership ballot and read from the ring by every consumer warp
([`#upper-product`](#upper-product)). `global` fixes the read scale's multiplicative
identity at `1` (`g_read` does not exist as a field — raw normalization is dead), so
`sr` starts at `1` and multiplies in only the fixed-level read amplitudes; `sw` starts
at `p.g_write[bh, token]` and multiplies in the fixed-level write amplitudes. Padded
slots (`t >= gN`) take `0`.

## <a id="read-panel"></a>19. The read panel `Rf` (§3 steps 2-3), live atoms only

`rf_pair(t, k)` computes one bf16-packed coefficient PAIR (leaves `k`, `k+1`) for
token `t`: the product of the spanned levels' (`l >= j`) bf16 amplitudes at the
owner's digit, times — under `DECAY` when the group is also write-live —
`panel_read_factor`. A padded slot (`t >= gN`) returns the exact-zero pair.

**THE fp16 APPLICATION IS CARVED OUT** (U07-d built, adjudicated 2026-08-06): the
decay panel factors apply in fp32 against the bf16 STORAGE, never in fp16 — see [the
precision story](#one-precision-story) above.

The loop walks FRAGMENT SLOTS (one k-step = one atom = 128 words), so the store is
contiguous in `tid` and the buffer is atom-blocked by construction:
`rmask == kFullMask` unrolls dense over every slot; otherwise the loop walks only the
`128`-word blocks of atoms set in `rmask` (`__fns` picks the q-th live atom).

## <a id="write-panel"></a>20. The write panel `WT`, live atoms only

`wt_pairs(q)` fills one atom's 8 leaf-pairs x `kGT` tokens = 128 slots. Per slot:
the spanned levels' bf16 write amplitudes at the owner's digit, times — under
`DECAY` — `panel_write_factor` (fp32 against bf16 storage, the same carve as the read
panel). `wmask == kFullMask` unrolls dense over every atom (`#pragma unroll`); otherwise
the loop walks only the atoms set in `wmask`.

### <a id="summary-prequant"></a>HAZARD `summary-prequant` — pass 1's per-leaf inclusion answer is taken HERE, pre-quantization

Under `summary`, a leaf is included in the sparse written-leaf list iff SOME token's
staged coefficient for it is nonzero — tested on the COEFFICIENT, before it narrows to
bf16 storage, because a nonzero fp32 value can still round to a nonzero bf16 one but
the reverse direction (a value that rounds TO zero) must not silently exclude a leaf
that later reads as live. The `kGT` tokens of one leaf pair are one contiguous 16-lane
run, tested with `__any_sync` over that run.

## <a id="clk-dead"></a>21. HAZARD `clk-dead` — the `clk -> Vw` lifetime barrier

`if (DECAY && is_write) __syncthreads();` after the write panel and before the `Vw`
staging marks the exact point past which `Vw` may overwrite `clk`'s aliased bytes
([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)). Every reader of `clk`
(the read panel's `panel_read_factor`, the write panel's `panel_write_factor`, the
clock-prefix formation itself) runs before this barrier; the `Vw` staging store runs
after it. Moving either across the barrier is a silent aliasing hazard no compile-time
check can see.

---

# Part V — the four GEMMs

## <a id="gemm-readout"></a>22. GEMM 1 — the readout, `y_group = R . S`, live read atoms only

Runs when `is_read && warp < NT_DV`. `mask == kFullMask` unrolls the dense arm
(`mma_tile_fragA<BC, ...>` over the whole state); otherwise the loop walks only the
live atoms of `rmask`, contracting `kMmaKQuantum` leaves per atom
(`mma_tile_fragA<kMmaKQuantum, ...>`). The B operand (`S`) is packed out of SMEM on
the fly; state and accumulation stay fp32 throughout.

## <a id="mass-column"></a>23. The mass column's D fragment — `dm` is to the denominator what `y` is to the numerator

The mass column IS A MATRIX COLUMN, not a special case: all three of the
denominator's terms are matvecs against one column of the state (or, for the fold's
own term, one column being written into it), and an `n8` MMA tile carries a matvec by
BROADCASTING that column across the tile's eight `n` slots — every output column then
holds the same value, and the tile is read back at column 0. `dm0..dm3` is ONE
`[16 x 8]` register D fragment, carried by WARP 0, that both of the read-side terms
(TERM 1, [below](#mass-term1), and TERM 2, [phase C](#phase-c)) accumulate into and
that the [split-K emission](#split-k-emission) reads at the end. The fold's own term
(TERM 3, [`#gemm-fold`](#gemm-fold)) writes directly into `S[.][DV]` instead — it has
no reader downstream in the same visit, so it never needs a fragment of its own.

The scalar reduction this replaces ran on WARP 0 ALONE at half lane occupancy, a
64-long serial FADD chain — an `n8` tile's "wastes 7/8 of a fragment" cost prices the
fragment, not the instructions, and the instructions are what this kernel is bound on.

## <a id="mass-term1"></a>24. The mass column, TERM 1 — one more `n8` tile on WARP 0

`dm = R . S[.][DV]`, THE SAME GEMM AS THE READOUT: `mma_tile_fragA_massB` (WARP 0
only, live read atoms only) with `S`'s mass column BROADCAST across the tile's eight
`n` slots, so every output column holds the matvec and the emission reads it back at
column 0. Its placement is FORCED: the fold (GEMM 4, later in program order) WRITES
`S[i][DV]`, so term 1, which READS it, must already have run.

The B operand is COMPENSATED: it is a column of the fp32 STATE and the DENOMINATOR OF
A RATIO, so — unlike the numerator's columns — it has nothing to cancel a bf16
rounding against. `split_bf16x2` ([`mma_tile.md#split-bf16x2`](mma_tile.md#split-bf16x2))
sends it as a rounded value plus its residual across TWO `mma.sync`, ~16 mantissa bits
for one extra instruction per k-step.

### <a id="mass-split"></a>HAZARD `mass-split` — bit-identity of the split rests on same-thread program order

Term 1 (here, right after the readout) and term 2 ([phase C](#phase-c), after the
[visit barrier](#visit-barrier)) accumulate into the SAME `dm0..dm3` fragment, held by
the SAME thread (WARP 0) across two program points with no intervening reassignment of
which thread owns which lane — reordering either half relative to the fold, which
WRITES the state column term 1 reads, is a real numeric difference, not just a
reassociation.

## <a id="gemm-gram"></a>25. GEMM 2 — the causal effective-score block, `G += R_q . W_q^T`

Runs when `mixed = is_read && is_write` and `warp < kGT / 8`, over `gmask = rmask &
wmask` — atoms live on BOTH sides (either side dead makes the term exactly zero, and
the dead side's panel was never even staged). `gmask == kFullMask` unrolls dense over
every atom; otherwise the loop walks `gmask`'s live atoms. The completed accumulator
is written into `Gm` through §5's INCLUSIVE causal mask (`c <= _g` / `c <= _g + 8`
zeroing the strictly-future entries) via `stage_put` — the same store the staged
panels use, which is what licenses GEMM 3 to fetch `Gm` by `ldmatrix`
([below](#visit-barrier)).

## <a id="visit-barrier"></a>26. The visit's ONE internal barrier

`if (mixed) __syncthreads();`, placed after GEMM 2 and before phase C / the fold, and
it carries BOTH of the visit's intra-CTA orderings AT ONCE: the readout's reads of `S`
complete before the fold's writes, and the Gram's stores to `Gm` are published to
phase C. It is a HARDWARE rendezvous rather than a counter because both edges are
CTA-WIDE and CTA-UNIFORM, and every remaining item pool in the visit (the fold's,
[below](#gemm-fold)) is a STATIC PARTITION — there is no work item left for an early
warp to run instead of arriving, which is what made a counter-based rendezvous worth
its cost in the first place.

**The guard is `mixed`, because both edges are.** A read-only visit (`is_read &&
!is_write`) runs no fold and needs no readout-before-fold ordering; a write-only
visit reads no state and has no Gram, so there is no Gram-before-phase-C ordering
either — `mixed = is_read && is_write` is exactly the CTA-uniform predicate both edges
share, so a single-sided visit pays for the barrier not at all. This is what the
small topologies measure: their visits are frequently one-sided, where the dense cell
(always `mixed`) pays the barrier every visit regardless.

## <a id="phase-c"></a>27. Phase C — GEMM 3, the intra term, and the mass column's term 2

`y_group += G . Vw` into the SAME registers the readout used (`mma_tile_vwB<DV>`,
`warp < NT_DV`, `Gm` fetched by `ldmatrix` exactly as the fold fetches `WT`) — token-axis
contraction, so no atom mask applies here. The mass column's TERM 2, `dm += G . sw`
(`mma_tile_bcastB`, WARP 0 only), runs alongside under the same `mixed` guard (HAZARD
`mass-split`, [above](#mass-term1)): the causal mask is already IN `Gm` (the
strictly-future entries store zero), so the whole 16-token contraction is the `u <= t`
sum.

---

# Part VI — the epilogue

## <a id="split-k-emission"></a>28. The split-K emission

Guarded `is_read`, and IT RUNS BEFORE THE FOLD: the two are independent (the fold
writes `S`; the emission reads `y`/`dm` and writes global), and emitting first RETIRES
both output fragments before the fold's own accumulators go live, which is where the
visit's register high-water mark sits.

`emit(t, ch, val)` computes `out = sr[t] * val` and skips the `atomicAdd` on
`out == 0.0f` — a PERFORMANCE skip on a VALUE, never a membership decision: a ragged
group's padded slots have `sr == 0` by construction and never read `t_idx` at all
(members are `< T` by construction, so the `!= 0` guard cannot mask a real emission).
The numerator scatters through `t_idx[t]` per output channel, out of `y0..y3`.

The denominator emission is OUT OF THE MASS FRAGMENT (`dm0`, `dm2`), warp 0 only: a
broadcast B makes every output column of `dm` the same value, so column 0 — the lanes
with `_r == 0` — carries tokens `_g` and `_g + 8`, and the `!= 0` guard retires a
ragged group's padded slots on the same rule the numerator runs on.

## <a id="gemm-fold"></a>29. GEMM 4 — the boundary fold, `S += W^T . Vw`, live write atoms only, and the mass column's TERM 3

Runs when `is_write`. The visit's fold work is a STATIC STRIDED PARTITION —
`for (c = warp; c < n_fold_items; c += kWarps)`, `n_fold_items = popc(wmask) * NT_DV`
— no atomic, no claim cursor: fold items are uniform-cost and `n_fold_items` is
CTA-uniform, so a strided claim is balanced to within one item for any mask.

`mt = wmask == kFullMask ? mr : __fns(wmask, 0, mr+1)` maps a claimed dense index
`mr` to the `mr`-th LIVE atom. Skipping a dead atom's read-modify-write is exact: its
delta is zero and its exit scale is `s0 == 1` (ticks ride writes, [above](#decay-clock)).

**The item's RMW is ATOM-MAJOR, not per-window** (Change 1, `8675f95`): the claimed
atom's 4-value accumulator loads to registers ONCE (`a0..a3` from `S`) and stays
resident across an inner `#pragma unroll for (gs = 0; gs < kFoldStagedGroups;
++gs)` that walks every staged slot the atom must consume — each iteration computes
one `mma_fold_tile` delta and applies the exit-scale epilogue in registers — before
the ONE writeback back to `S`. `kFoldStagedGroups` is `smem_layout.cuh`'s
`kFoldStagedGroups`, PINNED to 1 today, so the loop is a single iteration in
practice; the per-atom fetch era that deepens staging is expected to raise only this
constant and its staging producer, not restructure the claim/writeback shape around
it.

### <a id="fold-epilogue"></a>`DECAY`'s exit scale, riding the fold's existing read-modify-write

```
for (gs = 0; gs < kFoldStagedGroups; ++gs) {
  a0 = e0 * (a0 + d0);   // e0 = DECAY ? s0[i0] : 1.0f, d0 from mma_fold_tile(gs)
  ...
}
S[i0][n0] = a0;          // the ONE writeback, after every staged slot is folded in
```

One extra multiply per staged slot on a value that stays in registers for the whole
inner loop — no separate scaling pass and no intermediate write to `S`.

**The mass column's TERM 3**, `S[.][DV] += W^T . sw`, is a SEPARATE loop after the main
fold, ON THE SAME TENSOR CORE AS THE FOLD ITSELF: `mma_tile_bcastB` against `WT`'s atom
block and a broadcast `sw`, one `mma.sync` per live atom, one atom per warp
(`for (mi = warp; mi < n_live; mi += kWarps)`). It is not folded into the main loop
above because it writes one state column (`DV`, `_r == 0` only) rather than an `n8`
tile of `DV` columns. A dead atom's rows: `acc == 0` and `s0 == 1` — the skip is the
identity, the same rule the main fold's atom skip runs on.

**THE RMW IS ONE LANE PER LEAF.** A broadcast-B tile's result lands in output column
0, so the accumulator is meaningful only on `_r == 0`; the READ of `S[i0 * SP + DV]` /
`S[i1 * SP + DV]` is therefore predicated `_r == 0` too (`0.0f` on the other three
lanes of the row), not just the writeback. An unpredicated read is safe only because
`mma.sync.aligned` reconverges the warp between the read and the write that follows
it — an argument `compute-sanitizer --tool racecheck` cannot check and nothing in the
source pins — and the other three lanes' value is discarded anyway, so predicating the
read removes the access rather than merely annotating it.

## <a id="pass1-epilogue"></a>30. Pass 1's per-leaf epilogue

Guarded `is_write && summary && DECAY`: `if (s0[i] != 1.0f) sum_slot[i] = 1;` catches
the leaves the [write-panel's coefficient-side test](#summary-prequant) could not —
decay alone can make a leaf's survival move even where its coefficient rows happened
to be zero on this visit. `a_state[i] *= s0[i] * s0[i]` accumulates the segment's
per-leaf survival product `A` (identity `1` under `decay = None`, since `s0` is never
formed there).

## <a id="pass1-publication"></a>31. Pass 1's publication — the sparse written-leaf summary

After the walk, under `summary`: warp 0 compacts `sum_slot[i] != 0` into an ascending
dense list via one `__ballot_sync` per 32-leaf block, writing the leaf's ordinal into
`p.ws_leaf[slot * BC + q]` and rewriting `sum_slot[i]` in place to its list position
(`-1` if excluded). `p.ws_count[slot]`, `p.ws_scale[slot*BC+q] = a_state[i]`, and
`p.ws_state[(slot*BC+q)*cols + c] = S[i*SP+c]` follow for every included leaf. Slot
`owner_slot0 + seg + 1` is one past the scan's entry stash, so pass 3's segment `q`
reads `owner_slot0 + q` uniformly.

## <a id="state-store"></a>32. The owner's state block, written to HBM once

Only on `!summary && last_seg`. Atom-wise, `#pragma unroll`ed over `kQPO <= 4`: an
atom whose `atom_row_base` is `< 0` (unpaged, no committed slot) is never stored back
— `last_seg` is §6.4's publication rule and is a deliberate numeric choice, not the
scan's job to derive.

---

# Hazard index

| slug | what it guards |
|---|---|
| [`clock-prefix-fp32`](#clock-prefix-fp32) | the clock prefix accumulator must stay fp32 |
| [`summary-prequant`](#summary-prequant) | pass 1's inclusion test runs on the coefficient, before bf16 narrowing |
| [`clk-dead`](#clk-dead) | the barrier between `clk`'s last read and `Vw`'s first write into the aliased bytes |
| [`mass-split`](#mass-split) | the mass column's two-term split relies on same-thread program order across the visit barrier |
