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
5. load the state's two bf16 planes `Shi`/`Slo` and its fp32 mass column `Smass`
   ([`smem_layout.md#state-planes`](smem_layout.md#state-planes)) — zeros in pass 1
   (the summary pass), from the §6.4 workspace on a split owner's non-entry segment,
   else atom-wise from `p.state` through the page table;
6. warp 0 [cooks](#producer) the schedule ring's first `W * kSchedDepth` group
   descriptors — the cold start — a `__syncthreads()` publishes them, the CTA issues
   window 0's [amplitude prefetch](#prefetch) and drains it, and a second
   `__syncthreads()` publishes the slab. Those are the ONLY two barriers the entry
   epoch pays and the only barrier the prefetch adds anywhere.

**The walk**, once per VISIT of a WINDOW of `W = GT / kGroupTokens` gathered-token
GROUPS (`kGroupTokens == 16`; every group is full except each segment's final ragged
flush and a window's trailing groups past stream end; visit count is exactly
`ceil(gathered / GT)`) — [`smem_layout.md#window`](smem_layout.md#window):

7. the visit [reads its `W` cooked descriptors](#group-assembler) straight off the
   [schedule ring](#sched-ring), one group each — the token indices, the read/write
   atom masks, the side flags, the pre-multiplied row scales `sr`/`sw` — and assembles
   nothing; concurrently, warp 0 cooks the ring's NEXT window of descriptors (the
   [run-ahead](#sched-ring)), published by this visit's own end-of-window barrier;
8. **STAGE** — under `DECAY`, the clock prefix (over the WHOLE window, one reference
   frame) and entry scale; the §3 step-3 row scales `sr`/`sw`; the fragment-ordered
   read panel `Rf` and write panel `WT`, `W` SLOTS of the group's own shape; the
   scale-folded value tile `Vw`, likewise `W` slots — every leaf-axis loop walking LIVE
   ATOMS ONLY, per group;
9. **GEMM 1, the readouts** — `y_a = R_a . S_entry` for every group `a` of the window,
   live read atoms only, accumulating into each warp's own registers, against the SAME
   entry state;
10. **GEMM 2, the window Gram** — the block LOWER TRIANGLE `G[a][b]`, `b <= a`: `b == a`
    is today's within-group masked block, `b < a` is a dense unmasked cross-group
    block (every token of group `b` precedes every token of group `a`), `b > a` is
    never formed — `W(W+1)/2` sub-blocks over `W^2`, published into the window Gram
    `Gm`;
11. the visit's [ONE internal barrier](#visit-barrier), guarded `mixed`, carrying both
    the readouts-before-fold edge on `S` and the Gram-before-phase-C edge on `Gm` at
    once;
12. **PHASE C**: **GEMM 3, the intra terms** — `y_a += sum_{b<=a} Gm[a][b] . Vw_b` for
    every group `a`, into the same registers the readouts used, plus the mass column's
    [TERM 2](#phase-c);
13. **EMIT** — per group, the split-K `atomicAdd` of `sr[t] * y` into `num`, and, out of
    the [mass column's D fragment](#mass-column), of `sr[t] * dm` into `den`;
14. **GEMM 4, the fold, ONCE PER WINDOW** — `S += sum_b W_b^T . Vw_b`, live write atoms
    only over the UNION write mask of the window's groups, walked as a STATIC STRIDED
    PARTITION over `(atom, n-tile)` items, the claimed atom's accumulator resident in
    registers across all `W` staged groups it must consume, with `DECAY`'s exit scale
    applying ONCE, after every group's delta has been summed, riding the ONE
    read-modify-write, and the mass column's [TERM 3](#gemm-fold) beside it.

**Per CTA, at exit:** under pass 1 (the summary pass), publish the sparse `(A, B)`
written-leaf summary; otherwise, if this is the owner's last segment, store `S` back
atom-wise.

Three facts a reader must not miss:

- **The state is SMEM-resident, not register-resident, as two bf16 planes plus an
  fp32 mass column.** `Shi`/`Slo` hold each value's bf16 rounding and its residual
  ([`smem_layout.md#state-planes`](smem_layout.md#state-planes)); `Smass` is the
  denominator column and stays fp32, outside the planes. That is what decouples the
  thread count from the state layout, frees the CTA to 8 warps, puts every per-visit
  term on the tensor cores as one of four GEMMs, and — the reason for the plane form
  specifically — lets GEMM 1's B operand arrive by `ldmatrix` instead of being packed
  from scalar shared loads at every use. Accumulation is still fp32: the fold reads
  the plane pair, sums in registers, and splits at the one writeback.
- **The one-copy invariant is a property of the LAUNCH SHAPE**, not of a policy the
  kernel implements: one CTA per owner, run to completion, one HBM read and one HBM
  write of the owner's state block. There is no wave machinery anywhere.
- **Every structural axis is a template parameter and no axis has a runtime
  fallback.** `BC in {16, 64}`, `DV in {32, 64}`, `D in {1..4}`, `DECAY in {0, 1}`,
  `GT in {16, 64}` — 64 arms total, see
  [`consumer.md`](consumer.md#instantiation-matrix). The endgame walk IS the path;
  dense (full-support) routing self-heals to identity grouping rather than taking a
  different code path.
- **The visit pays exactly TWO unconditional `__syncthreads()`** — the post-staging
  barrier (staging into GEMM 1/GEMM 2) and the end-of-window barrier that closes the
  loop and publishes the producer's run-ahead cook — plus the [visit barrier](#visit-barrier)
  (readouts before fold on `S`, Gram before phase C on `Gm`), which is CTA-UNIFORM but
  not unconditional: it is guarded `mixed`, because both of its edges are (a
  read-only window runs no fold, a write-only window reads no state, and neither has a
  Gram), so a single-sided window pays nothing for it. Every one of these three is
  PER WINDOW, not per group — the amortisation `GT` buys. `DECAY` arms pay two more,
  both inside staging: after the clock-prefix pass (before it scales `S`) and at
  [`clk-dead`](#clk-dead) (before `Vw` may overwrite `clk`'s aliased bytes). There is
  no separate group-entry barrier: the ring publishes
  a window's descriptors a whole window ahead of the visit that reads them.

## <a id="glossary"></a>2. Symbol glossary

| symbol | what it is |
|---|---|
| `BC` | leaves per owner's compute block; `{16, 64}` |
| `DV` | value width `d_v`; `{32, 64}` |
| `D` | level count of the topology; `1 <= D <= 4` (spec §1) |
| `j` | `first_spanned_level`. Levels `l < j` are FIXED for this owner |
| `kMmaKQuantum` | `16` — the hardware `mma.sync` k-extent on the sm80 bf16 path, and the sector invariant's other half ([`smem_layout.md#cute-pin`](smem_layout.md#cute-pin)). Defined once, in `arch_table.cuh` |
| `kGroupTokens` | `16` — the PRODUCER's work unit: one group of gathered tokens, == `kMmaKQuantum`. The cook, the ballot and the ring descriptor are sized by it and never by `GT` |
| `GT` | the CONSUMER's token quantum — the WINDOW, a template axis, `{16, 64}`. `kGT` is the kernel body's local alias for it |
| `W` | `window_groups(GT) = GT / kGroupTokens` — how many groups one VISIT consumes against one entry state ([`smem_layout.md#window`](smem_layout.md#window)) |
| `QPO` / `kQPO` | atoms per owner, `BC / kMmaKQuantum`. An ATOM is one `kMmaKQuantum`-leaf block in canonical leaf order, and also §4.5's page granule |
| `NT_DV` | `DV / 8` — the `n8` column tiles of a `DV`-wide output |
| `kWarps` | `8` — `kConsumerThreads / 32` |
| `kFullMask` | the atom masks' FULL value, `(1u << kQPO) - 1u` (or all-ones past 32 atoms). `mask == kFullMask` selects the statically-unrolled dense arm |
| `Shi`, `Slo` | the state's NUMERATOR, two bf16 SMEM planes, each `[BC][DV]`, resident for the owner's whole run — a value's bf16 rounding (`Shi`) and its residual (`Slo`) ([`smem_layout.md#state-planes`](smem_layout.md#state-planes)) |
| `Smass` | the state's DENOMINATOR column, `[BC]` fp32 SMEM, OUT of the planes ([`smem_layout.md#state-planes`](smem_layout.md#state-planes)) |
| `NP_DV` | `DV / 2` — channel PAIRS of a state row, the plane walk's unit |
| `cols` | `DV + 1`, UNCONDITIONAL — the mass column always exists (the HBM row width; the SMEM state's own shape is `Shi`/`Slo`/`Smass` above) |
| `WT` | the write coefficient panel, `W` SLOTS of `[BC][BTP]` bf16, LEAF-MAJOR per slot |
| `BTP` | `WT`'s row pitch, `kWtPitch = kGroupTokens + 8 = 24` ([`smem_layout.md#stage-format`](smem_layout.md#stage-format)) |
| `Vw` | the scale-folded value tile, `W` slots, `Vw[t,ch] = sw[t] * v[t0+t, ch]`, bf16, swizzled, no pad |
| `Gm` | the WINDOW Gram, ONE buffer, `[GT][GT+8]`, pitch `GMP = GT + 8`; only the block lower triangle is written or read |
| `Rf` | the read coefficients, `W` slots, MMA-FRAGMENT-ORDERED, `frag_a_words(BC)` words each, atom-blocked |
| `clk` | the WINDOW's per-leaf INCLUSIVE clock prefix `[GT][CKP]` fp32, `CKP = BC + 4`, ONE buffer over the whole window; ALIASES `Vw`'s bytes ([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)) |
| `log_keep`, `p_ref`, `s0` | `DECAY`'s three `[BC]` vectors — per-owner log retention, §12 rule 1's reference exponent (now over the window), and the entry/exit scale |
| `sr`, `sw` | the §3 step-3 per-token read/write ROW SCALES, `[kGroupTokens]` each, PER GROUP — retargeted every visit into `g_up`'s matching ring slot, [cooked by the producer](#upper-product) |
| `dm0..dm3` | the [mass column's D fragment](#mass-column), register-resident on WARP 0 — one `[16 x 8]` accumulator, to the denominator exactly what `y0..y3` is to the numerator |
| `dig` | the hoisted leaf digits, `[D][BC]` |
| `tk_of(a)` | group `a`'s `kGroupTokens` gathered GLOBAL token indices, ascending — addressed in place off the [schedule ring](#sched-ring)'s matching slot, never copied |
| `kSchedDepth` / `sched_slots(GT)` | the producer's run-ahead depth (`ROLA_SCHED_DEPTH`, a `constexpr` dial) and the ring's slot count, `window_groups(GT) * (kSchedDepth + 1)` — a WHOLE WINDOW of run-ahead ([`smem_layout.md#sched-ring`](smem_layout.md#sched-ring)) |
| `g_ring` | the [cooked schedule ring](#sched-ring), `sched_slots(GT)` group descriptors, `kSchedSlotInts` ints each, one group per slot |
| `g_up` | the ring's per-owner [upper-product](#upper-product) pairs, `sched_slots(GT) * 2 * kGroupTokens` floats |
| `p_st`, `p_chunk` | the [producer's cursor state](#producer-state): seven uniform scalars (`p_st`) and five 32-lane planes (`p_chunk`, the cooked chunk's `fr`/`fw`/`member` and the selected word's two atom-bitlist rows) |
| `A`, `B` | §6.4's per-segment summary: the per-leaf survival product and the accumulated deposit |
| `w_lo`, `w_hi` | this segment's word range over the scan stream, `[w_lo, w_hi)` |
| atom | one `kMmaKQuantum`-leaf block of the owner's leaf axis; also one 32-byte fetch sector ([`smem_layout.md#cute-pin`](smem_layout.md#cute-pin)) |
| group | the PRODUCER's work unit: `<= kGroupTokens` gathered, causally-ordered member tokens |
| window / visit | the CONSUMER's work unit: `W` groups (`<= GT` tokens), staged and consumed against one entry state and folded back in one writeback |

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

## <a id="state-planes"></a>4. The state as two bf16 planes and an fp32 mass column

`Shi`/`Slo` are `StageLayout` buffers at the state's own shape (`[BC][DV]`, no pad) —
the SAME layout `Vw` is stored under — so GEMM 1's B operand is fetched by the
`ldmatrix` atom that already serves `Vw` instead of packed from four scalar shared
loads, four clamps and two packs per `mma.sync` per lane. `Shi` holds each value's
bf16 rounding; `Slo` holds its residual, read together only where the pair is
reconstructed as a number (the fold's read-modify-write and the entry/exit
publication), never at the readout, whose B operand is `Shi` alone
([`mma_tile.md#mma-tile-fraga`](mma_tile.md#mma-tile-fraga)).

`Smass` stays fp32 and stays OUT of the planes: it is the denominator of a ratio, its
consumers are broadcast-B matvecs no `ldmatrix` can serve, and keeping it out is what
leaves the planes pad-free at exactly `BC * DV` — a swizzle's XOR field is only legal
over an unpadded tile. This is also why the carve is `BC*(DV+1)*4` rather than the
pre-plane `BC*(DV+4)*4`: the `+4` pad existed to decorrelate banks for a SCALAR
fragment map (`k = k0 + 2r` collided up to four ways at `SP == DV`, a multiple of 32),
and the planes need no such pad because they are read under a swizzle instead.

Five `constexpr` proofs answer the state planes' bank-conflict question at the
state's own shape, the same four questions [`smem_layout.md#stage-layout`](smem_layout.md#stage-layout)
answers for `Vw`, plus one more for the fold's paired access:
`stage_offset_matches_cute`, `stage_layout_is_bijection`, `stage_ldsm_banks_ok`,
`stage_pair_banks_ok`, `stage_store_banks_ok` — [below](#pitches).

## <a id="pitches"></a>5. Every operand pitch proves its own conflict-freedom

Each buffer's row pitch is pinned by a `static_assert` at the point the kernel body
carves it, so a future `BC`/`DV` that broke the proof fails the BUILD:

| assert | what it pins |
|---|---|
| `stage_offset_matches_cute<BC, DV>()` | the state planes' closed-form address is CuTe's layout |
| `stage_layout_is_bijection<BC, DV>()` | the swizzled state planes tile `[0, BC*DV)` exactly |
| `stage_ldsm_banks_ok<BC, DV>()` | GEMM 1's `ldmatrix.x2.trans` fetch of `Shi`/`Slo`, conflict- and alignment-free |
| `stage_pair_banks_ok<BC, DV>()` | the fold's paired read-modify-write of a channel pair, conflict-free ([`smem_layout.md#state-planes`](smem_layout.md#state-planes)) |
| `stage_store_banks_ok<BC, DV>()` | the entry-state staging store into `Shi`/`Slo`, conflict-free |
| `banks_distinct_u16(1, 2 * BTP)` | `WT` as GEMM 2's B operand |
| `banks_distinct_u16(BTP, 2)` | `WT` as GEMM 4's A operand |
| `pitched_ldsm_banks_ok<kMmaKQuantum, kGroupTokens, kWtPitch>()` | `WT`'s `ldmatrix` fetch conflict- and alignment-freedom at `kWtPitch`, per slot |
| `pitched_ldsm_banks_ok<kGroupTokens, kGroupTokens, GMP>()` | `Gm`'s `ldmatrix` fetch conflict- and alignment-freedom at `GMP`, as GEMM 3's A operand — a group-sized phase at the WINDOW's pitch |
| `stage_offset_matches_cute<kGroupTokens, DV>()` | `Vw`'s closed-form address IS CuTe's layout, per slot |
| `stage_layout_is_bijection<kGroupTokens, DV>()` | the swizzled `Vw` tiles `[0, kGroupTokens*DV)` exactly, per slot |
| `stage_ldsm_banks_ok<kGroupTokens, DV>()` | `Vw`'s `ldmatrix.x2.trans` fetch, per slot |
| `stage_store_banks_ok<kGroupTokens, DV>()` | `Vw`'s staging store, per slot |

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
`kMmaKQuantum * cols` contiguous HBM elements per atom. An atom with no committed slot
reads as ZEROS and is never stored back — the same arithmetic the dense path applies
to a never-written atom, which is what makes paging BITWISE INVISIBLE rather than
approximately equivalent.

`cols = DV + 1` is unconditional; there is no `raw`/`GLOBAL` axis left to branch it on.
This is the HBM row's own width, unrelated to the SMEM state's shape: the load SPLITS
as it lands, one walk over `NP_DV = DV / 2` channel PAIRS per row that narrows each fp32
pair into one `Shi`/`Slo` word via `split_bf16x2`, and a separate `[BC]` walk that
copies the row's mass element straight into `Smass` (no narrowing — the mass column
stays fp32 in SMEM exactly as it is in HBM).

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
ISSUE, not redundant latency. That distinction is the whole rationale: a prefetch that
buys stall movement with instructions loses on this kernel (the standing P11 verdict,
and the rule the [amplitude prefetch](#prefetch) was built to respect), so the win
available is cutting issued instructions, and a single producer warp cuts the scan's
ballot ALU and its `2 * QPO` atom-bitlist loads by 8x.

**Stability is the correctness requirement.** Words ascend and each word is consumed
LSB-first, so a descriptor's `kGroupTokens` slots are GLOBALLY ascending, and groups
cook in that same order — so a window's tokens are GLOBALLY ascending ACROSS its
groups as well as within one. The window Gram's block-triangular causal structure
([`smem_layout.md#window`](smem_layout.md#window)) is order-relative across groups
exactly as the diagonal block's mask is order-relative within one, and this ordering
IS what makes both correct.

A side whose visit mask is empty (`rmask == 0` or `wmask == 0`) has no live
coefficients anywhere in the group on that side; the descriptor's side flags carry `gr`
/ `gw` (whether ANY member token is read-/write-live on any FIXED level) alongside the
masks, and `is_read`/`is_write` fold that emptiness in, so a whole-visit skip is
possible even where the per-tile candidacy machinery of an earlier design could not see
it.

## <a id="sched-ring"></a>12. The schedule ring

The producer runs `kSchedDepth` GROUPS ahead of the consumers (a `constexpr` dial,
`ROLA_SCHED_DEPTH`, never a template axis — the instantiation matrix does not gain a
coordinate from it). The ring holds `sched_slots(GT) = window_groups(GT) * (kSchedDepth
+ 1)` descriptors — a WHOLE WINDOW's worth of run-ahead, because a visit reads
`window_groups(GT)` descriptors at once and the producer must still be cooking into
slots nobody is reading: cooking group `gidx + W * kSchedDepth` writes the slot that
group `gidx - W` vacated, which is never a slot the current window is reading — the
ring must be AT LEAST `2W` deep for that to hold, which is why `sched_slots` is `W *
(depth + 1)` rather than `W * depth`. `W` divides `sched_slots(GT)`, so a window's `W`
slots are the CONTIGUOUS run starting at `(widx * W) % sched_slots(GT)` — a group index
within a window is a pure offset, with no per-group modulo.

Publication rides the END-OF-WINDOW `__syncthreads()` that already exists (every
visit closes on one): the producer's run-ahead cook for the NEXT window's slots runs
concurrently with the current visit's compute, at the bottom of the loop body, and the
barrier that already stands between one visit and the next is what makes the cooked
descriptors visible to every warp before the visit that consumes them starts. The ring
therefore carries no cursor word of its own and needs no barrier beyond the one the
visit loop already pays.

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

## <a id="group-assembler"></a>15. The visit reads `W` cooked descriptors; it assembles nothing

Before the first visit, the producer (warp 0) fills the ring's first `W * kSchedDepth`
slots — the cold start, a WHOLE WINDOW of run-ahead — and a `__syncthreads()`
publishes them. Every subsequent visit then does no assembly at all: window `widx`'s
groups are the `W` CONTIGUOUS slots starting at `(widx * W) % sched_slots(GT)`, and
`tk_of(a)`, `gN_of(a)`, `rm_of(a)`, `wm_of(a)`, `sr_of(a)`, `sw_of(a)` address group
`a`'s fields in place off that slot range rather than copying them; `gN_of(0) == 0` is
the stream's end, as the producer saw it.

At the bottom of the same visit, the producer cooks the NEXT window's `W` groups ([the
run-ahead — above](#sched-ring)) concurrently with this visit's GEMMs, and the visit
closes on the same end-of-window `__syncthreads()` that publishes them. One cooking
producer against eight reading warps is the whole point of the arrangement: the
ballot ALU and the atom-bitlist load issue are paid once per owner's group rather than
once per warp.

## <a id="staged-write-mask"></a>15.1. The window's side flags, and the staged write mask

`isr_of(a)` / `isw_of(a)` are group `a`'s side flags AND its atom mask together (a
group can carry live write ATOMS while being side-dead, since the flag is the fixed
levels' rows and the mask is the atom bitlists — two independent bitmaps). The
window's sides are the OR over its groups: `w_read`, `w_write`, and `w_wmask = OR_a
wmk_of(a)`, where `wmk_of(a) = isw_of(a) ? wm_of(a) : 0` — **THE STAGED WRITE MASK,
the one mask every write-side consumer walks, never `wm_of(a)` alone.** A side-dead
group's `WT`/`Vw` slots are NOT STAGED, so reading its raw atom mask would send the
fold into a slot still holding a PREVIOUS window's panels; at `W == 1` this was
impossible (the fold ran under the group's own `is_write`), and at `W > 1` it is not,
which is exactly what `wmk_of` closes.

The window's row-scale publication is UNCONDITIONAL, not merely predicated fetch: a
side-dead group still writes exact zeros into its `g_up` slot (`rs = need_r ? 1 : 0`,
`ws` likewise), because a window's side flags are the OR over its groups, so a
side-dead group's `sw` row is still READ (as the fold's mass-term broadcast B) even
though its own `WT` rows are exact zeros — and `0 * NaN` is `NaN`, so leaving a stale
slot from a previous window unzeroed is not inert. Publishing the zero is what keeps
the 0-times-dead discipline exact.

An atom live in ANY group of the window folds once, with a per-group predicate inside
the accumulator's residency ([`#gemm-fold`](#gemm-fold)); an atom live in NONE
contributes an exactly-zero delta and (under decay) has `acc == 0` hence `s0 == 1`
([above](#decay-clock)), so skipping it entirely is the identity — ticks ride writes.

## <a id="staging"></a>16. The staging — everything the visit's compute consumes but does not produce

Per group `a` of the window, LIVE ATOMS OF THAT GROUP ONLY:

- under `DECAY`, the clock prefix and entry/exit scale, over the WHOLE WINDOW, not per
  group ([below](#decay-clock));
- the read panel `Rf`, into slot `a`, out of the [prefetched slab](#prefetch)
  ([below](#read-panel));
- the write panel `WT`, into slot `a`, likewise ([below](#write-panel));
- `Vw_a[t, ch] = sw_a[t] * v[token, ch]`, into slot `a` — the write side folds the row
  scale into the value tile because there `t` is the CONTRACTION index and the scale
  cannot hoist out of the sum.

The [row scales](#row-scales) `sr`/`sw` are NOT computed here: they arrive
PRE-MULTIPLIED in the ring slot, [cooked by the producer](#upper-product) a whole
window early, so staging only reads them (in the `Vw` fold above and at
[emission](#split-k-emission)).

Within one group, tokens are `tk_of(a)[0..gN)`, globally ascending, and ascend ACROSS
groups too ([above](#producer)); a ragged group's padded slots `[gN, kGroupTokens)` —
and every slot of a window's trailing dead groups past stream end — stage EXACT-ZERO
coefficient rows and zero row scales, so they are inert through every GEMM without any
membership test.

`mask == kFullMask` selects the statically-unrolled dense arm at every one of these
loops — dense inputs pay zero mask instructions, because they self-heal to identity
grouping rather than taking a different code path.

## <a id="prefetch"></a>16.1. The amplitude prefetch — the panels' gather, one visit early

Both panels' coefficients come out of a shared-memory SLAB
([`smem_layout.md#prefetch-slab`](smem_layout.md#prefetch-slab)) filled by `cp.async`
([`cp_async.md`](cp_async.md)) from inside the PREVIOUS window's visit. The panels
never touch global memory.

**The address, and why it is cheaper than the gather it replaces.** A slab row is a
window token; a level's sub-range sits at `pf_off[l]`, which is a REGISTER, not a
table, because every gather's level loop is `#pragma unroll` and `l` is a compile-time
constant at every use. So the read becomes

    pf[(t0 + t) * kPfPitch + pf_off[l] + dig[l*BC + k]]

where `kPfPitch` is a compile-time constant and `dig` is the same shared load the walk
always did. The direct form paid a 64-bit `tr * level_pitch` and a global load for the
same element. `pf_off` can be a register only because the alignment lead is a per-LEVEL
constant, which is a property of the padded amplitude pitch
([`consumer.md#amp-alignment`](consumer.md#amp-alignment)) — the earlier whole-slab
prefetch carried a per-`(level, token)` offset table instead, and that extra shared
load is most of what made it cost more instructions than it saved.

**The per-atom half is chunk-granular ZFILL.** Each 16-byte chunk carries `ch_atoms`,
the set of atoms whose digits it holds; the copy is issued with src-size 16 iff that
set intersects the token's group's STAGED side mask, and with src-size 0 otherwise.
This is EXACT rather than an approximation, and the argument is the self-masking
theorem read backwards: a leaf reading a zero-filled chunk belongs to an atom dead on
that side, whose staged coefficient must be exactly zero, and zeroing one of its level
factors delivers exactly that. It is STRONGER than what the direct gather relied on —
there the zero came from the amplitude tensor's own entmax zeros; here it is
structural. The ragged tail rides the same predicate, with the token index clamped so
no out-of-range address is ever formed.

**Where it overlaps, and the zero-barrier property.** The slab's last read is the panel
staging, so the refill may start at the staging barrier that already exists — that
barrier is there to publish the staged operands and equally guarantees every warp has
finished reading the slab. The drain sits immediately before the end-of-window
`__syncthreads()`, which is what publishes the slab cross-warp (`cp.async` completion
is visible to the issuing thread only). The copies are therefore in flight across both
Gram phases, the emission and the fold, and the steady state pays NO new barrier. The
one cost is `kSchedDepth = 2`: the issue needs the next window's token list one window
earlier than depth 1 publishes it.

**The delegated sm_86 choice (P25 contract item 7), discharged.** `cp.async` commit and
wait state is per WARP, so the question was warp-partitioned streams against
interleaved commit order. The answer is ONE COMMIT GROUP PER WARP PER VISIT on a FLAT
CTA-wide issue index: the visit consumes the whole slab at a single point, so there is
exactly one drain and a partial-fill ordering has no consumer that could use it, while
the flat stride is what makes a warp's 32 lanes take 32 CONSECUTIVE chunks — one
contiguous byte range per warp, which is the only property the memory system rewards.

**What it does and does not buy.** It removes the panel gather's latency: measured on
`sm_86`, `long_scoreboard` falls 46% on the dense cell and 26% at scale, DRAM traffic
falls 43% on the dense cell, and that cell's kernel duration falls 13%. It does NOT buy
occupancy freedom — the wide window's residency penalty is unchanged, so the CTAs were
hiding something else. And it is DISTRIBUTION-DEPENDENT: on the sparse at-scale cell
the 16-byte chunk granularity over-fetches (DRAM +12%, L1 hit rate 31.5% -> 20.2%),
shared-load bank conflicts rise 48%, and the cost lands on the existing barriers
(+22%), for +18% on that kernel. Evidence: `workflows/perf/p39_prefetch.md`.

## <a id="decay-clock"></a>17. `DECAY` steps (a) and (c) — the clock prefix, `P_ref`, `s0`, and the entry scale

Guarded by `w_write` structurally (the window's OR): a group with no write-live
members has `c == 0` on every leaf, so every factor this block computes is the
identity and the plain path is EXACT, not approximate.

**The prefix runs across the WHOLE WINDOW, not across a group** — an outer loop over
groups `b in [0, W)`, each contributing its `gN` (or fewer, for a ragged or
window-trailing dead group) tokens to one running `acc` per leaf, so every token of
the window shares ONE reference frame. That is what makes the Gram's cross-group
blocks ([`smem_layout.md#window`](smem_layout.md#window)) legal under decay: a
cross-group term `R_t . W_u` for `t` in group `a`, `u` in group `b < a` needs `t` and
`u` in the same frame to combine into `keep^{P[t]-P[u]}`.

The lowering is §7.4.1's decay-aware panels. `keep[t,s] = (1-rate[s])^c[t,s]`, so with
`P[t,s] = sum_{u<=t} c[u,s]` (the sum now over the WHOLE WINDOW's tokens) the survival
from `j` to `t` is `keep^{P[t]-P[j]}`, and that exponent SPLITS ACROSS THE TWO
OPERANDS of a Gram because it is a difference of a per-`t` term and a per-`j` term.
With §12 rule 1's per-leaf reference, now over the window, `P_ref = P[GT-1] / 2`:

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
EXPONENT error, not a mantissa one. One thread per leaf, serial over the window's `GT`
tokens (nested: `W` groups outer, each group's `<= kGroupTokens` tokens inner),
because the prefix IS a serial dependence in `t` and `GT <= 64` — a parallel scan
would cost more barriers than it saves steps. A ragged group's padded slots, and every
slot of the window's trailing dead groups, take a FINITE zero prefix (`clk[t*CKP+i] =
0`) so both panel factors stay finite (`keep^0 == 1`); their coefficient rows are
exact zeros anyway.

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

`wt_pairs(q)` fills one atom's 8 leaf-pairs x `kGroupTokens` tokens = 128 slots, per
group slot. Per slot:
the spanned levels' bf16 write amplitudes at the owner's digit, times — under
`DECAY` — `panel_write_factor` (fp32 against bf16 storage, the same carve as the read
panel). `wmask == kFullMask` unrolls dense over every atom (`#pragma unroll`); otherwise
the loop walks only the atoms set in `wmask`.

### <a id="summary-prequant"></a>HAZARD `summary-prequant` — pass 1's per-leaf inclusion answer is taken HERE, pre-quantization

Under `summary`, a leaf is included in the sparse written-leaf list iff SOME token's
staged coefficient for it is nonzero — tested on the COEFFICIENT, before it narrows to
bf16 storage, because a nonzero fp32 value can still round to a nonzero bf16 one but
the reverse direction (a value that rounds TO zero) must not silently exclude a leaf
that later reads as live. The `kGroupTokens` tokens of one leaf pair are one
contiguous 16-lane run, tested with `__any_sync` over that run.

## <a id="clk-dead"></a>21. HAZARD `clk-dead` — the `clk -> Vw` lifetime barrier

`if (DECAY && w_write) __syncthreads();` after the write panel and before the `Vw`
staging marks the exact point past which `Vw` may overwrite `clk`'s aliased bytes
([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)). Every reader of `clk`
(the read panel's `panel_read_factor`, the write panel's `panel_write_factor`, the
clock-prefix formation itself) runs before this barrier; the `Vw` staging store runs
after it. Moving either across the barrier is a silent aliasing hazard no compile-time
check can see.

---

# Part V — the four GEMMs

## <a id="gemm-readout"></a>22. GEMM 1 — the readouts, `y_a = R_a . S_entry` for every group of the window

An ITEM is one `(group, DV column tile)` pair, `W * NT_DV` of them, claimed by a
static strided partition over the CTA's `kWarps` warps (`kYItems =
ceil(W * NT_DV / kWarps)` fragments per warp) — at `W == 1, DV == 32` the old
`warp < NT_DV` predicate left half the CTA idle; the item partition keeps every warp
carrying work at every `(W, DV)`. Every group's readout reads the SAME entry state
(`Shi`/`Slo`), never a group-local copy. `mask == kFullMask` unrolls the dense arm
(`mma_tile_fragA_stateB<DV, BC>` over the whole state); otherwise the loop walks only
the live atoms of that group's `rmask`, contracting `kMmaKQuantum` leaves per atom
(`mma_tile_fragA_stateB<DV, kMmaKQuantum>`). The B operand is fetched by `ldmatrix`
from `Shi` alone — [`mma_tile.md#mma-tile-fraga`](mma_tile.md#mma-tile-fraga) has the
readout's own account of why `Slo` is not read here; accumulation stays fp32
throughout.

## <a id="mass-column"></a>23. The mass column's D fragment — `dm` is to the denominator what `y` is to the numerator

The mass column IS A MATRIX COLUMN, not a special case: all three of the
denominator's terms are matvecs against one column of the state (or, for the fold's
own term, one column being written into it), and an `n8` MMA tile carries a matvec by
BROADCASTING that column across the tile's eight `n` slots — every output column then
holds the same value, and the tile is read back at column 0. `dm` is now `kDmItems =
ceil(W / kWarps)` `[16 x 8]` register D fragments, ONE PER GROUP OF THE WINDOW rather
than one fixed to warp 0 — the item partition puts one group's fragment per warp where
the `W == 1` visit put every group's on warp 0 — that both of the read-side terms
(TERM 1, [below](#mass-term1), and TERM 2, [phase C](#phase-c)) accumulate into and
that the [split-K emission](#split-k-emission) reads at the end, per group. The fold's
own term (TERM 3, [`#gemm-fold`](#gemm-fold)) writes directly into `Smass` instead —
it has no reader downstream in the same visit, so it never needs a fragment of its
own.

The scalar reduction this replaces ran on WARP 0 ALONE at half lane occupancy, a
64-long serial FADD chain — an `n8` tile's "wastes 7/8 of a fragment" cost prices the
fragment, not the instructions, and the instructions are what this kernel is bound on.

## <a id="mass-term1"></a>24. The mass column, TERM 1 — one more `n8` tile per group

`dm_a = R_a . Smass`, THE SAME GEMM AS THE READOUT: `mma_tile_fragA_massB`, one
group's fragment per warp per the item partition, live read atoms only, with
`Smass` BROADCAST across the tile's eight `n` slots, so every output column holds
the matvec and the emission reads it back at column 0. Its placement is FORCED: the
fold (GEMM 4, later in program order) WRITES `Smass[i]`, so term 1, which READS it,
must already have run.

The B operand is COMPENSATED: `Smass` is the fp32 mass column and the DENOMINATOR OF
A RATIO, so — unlike the numerator's planes — it has nothing to cancel a bf16
rounding against, and it stays fp32 in storage rather than narrowing to a plane pair.
Its `[BC]` array is CONTIGUOUS, so the fetch is the mass tile's own broadcast map, not
a strided walk of a wider buffer. `split_bf16x2` ([`mma_tile.md#split-bf16x2`](mma_tile.md#split-bf16x2))
sends it as a rounded value plus its residual across TWO `mma.sync`, ~16 mantissa bits
for one extra instruction per k-step.

### <a id="mass-split"></a>HAZARD `mass-split` — bit-identity of the split rests on same-thread program order

Term 1 (here, right after the readout) and term 2 ([phase C](#phase-c), after the
[visit barrier](#visit-barrier)) accumulate into the SAME `dm0..dm3` fragment, held by
the SAME thread (WARP 0) across two program points with no intervening reassignment of
which thread owns which lane — reordering either half relative to the fold, which
WRITES the state column term 1 reads, is a real numeric difference, not just a
reassociation.

## <a id="gemm-gram"></a>25. GEMM 2 — the WINDOW GRAM, `G[a][b] += R_a . W_b^T` over the block lower triangle `b <= a`

An ITEM is one `(a, b, DV column tile)` triple over the block lower triangle
(`b <= a`, `W(W+1)/2` sub-blocks over `W^2`), claimed by a static strided partition
over the CTA's warps, and skipped where `gmask = rmask_a & wmask_b` is zero — atoms
live on BOTH sides (either side dead makes the term exactly zero, and the dead side's
panel was never even staged). This is FLA's sub-chunked-diagonal split
(`fla/ops/gla/chunk.py`'s `chunk_gla_fwd_A_kernel_intra_sub_inter` /
`..._intra_sub_intra`, transplanted with one divergence, [`smem_layout.md#window`](smem_layout.md#window)):
`b > a` is identically zero and NEVER FORMED — no item exists for it; `b < a` is a
fully DENSE sub-block (every token of group `b` strictly precedes every token of group
`a`) and stores UNMASKED; only the `W` DIAGONAL sub-blocks (`a == b`) carry §5's
INCLUSIVE causal mask (`c <= _g` / `c <= _g + 8` zeroing the strictly-future entries).
So the Gram's storage and instruction count are `O(W^2)` and the causal MASK stays
`O(W)`. The completed accumulator is written into `Gm`'s sub-block `(a, b)` via
`stage_put` — the same store the staged panels use, which is what licenses GEMM 3 to
fetch `Gm` by `ldmatrix` ([below](#visit-barrier)).

## <a id="visit-barrier"></a>26. The visit's ONE internal barrier

`if (mixed) __syncthreads();`, placed after GEMM 2 and before phase C / the fold, and
it carries BOTH of the visit's intra-CTA orderings AT ONCE: every group's readout
reads of `S` complete before the fold's writes, and the Gram's stores to `Gm` are
published to phase C. It is a HARDWARE rendezvous rather than a counter because both
edges are CTA-WIDE and CTA-UNIFORM, and every remaining item pool in the visit (the
fold's, [below](#gemm-fold)) is a STATIC PARTITION — there is no work item left for an
early warp to run instead of arriving, which is what made a counter-based rendezvous
worth its cost in the first place. **The barrier count is PER WINDOW, not per group**
— three for `W` groups where the group-quantum (`GT == kGroupTokens`) visit paid three
each, which is the amortisation the window buys ([`smem_layout.md#window`](smem_layout.md#window)).

**The guard is `mixed`, because both edges are.** A read-only window (`w_read &&
!w_write`, the OR over the window's groups) runs no fold and needs no
readout-before-fold ordering; a write-only window reads no state and has no Gram, so
there is no Gram-before-phase-C ordering either — `mixed = w_read && w_write` is
exactly the CTA-uniform predicate both edges share, so a single-sided window pays for
the barrier not at all. This is what the small topologies measure: their windows are
frequently one-sided, where the dense cell (always `mixed`) pays the barrier every
window regardless.

## <a id="phase-c"></a>27. Phase C — GEMM 3, the intra terms, and the mass column's term 2

`y_a += sum_{b <= a} Gm[a][b] . Vw_b` into the SAME fragments the readouts used
(`mma_tile_vwB<DV>`, the same item partition as GEMM 1, `Gm`'s sub-block fetched by
`ldmatrix` exactly as the fold fetches `WT`) — token-axis contraction, so no atom mask
applies here, and NO CAUSAL MASK EITHER: the zeros are already IN `Gm`. The mass
column's TERM 2, `dm_a += sum_{b <= a} Gm[a][b] . sw_b` (`mma_tile_bcastB`, one
group's fragment per warp), runs alongside under the same `mixed` guard (HAZARD
`mass-split`, [above](#mass-term1)): the causal mask is already in `Gm` (the
strictly-future entries of the diagonal block store zero and the strictly-future
blocks are never read), so the whole window's contraction is exactly the `u <= t` sum.

---

# Part VI — the epilogue

## <a id="split-k-emission"></a>28. The split-K emission

Guarded per group (`isr_of(a)`), and IT RUNS BEFORE THE FOLD: the two are independent
(the fold writes `S`; the emission reads `y`/`dm` and writes global), and emitting
first RETIRES every output fragment before the fold's own accumulators go live, which
is where the window's register high-water mark sits.

`emit(t, ch, val)` computes `out = sr_a[t] * val` and skips the `atomicAdd` on
`out == 0.0f` — a PERFORMANCE skip on a VALUE, never a membership decision: a ragged
group's padded slots have `sr == 0` by construction and never read the group's token
list at all (members are `< T` by construction, so the `!= 0` guard cannot mask a real
emission). The numerator scatters through `tk_of(a)[t]` per output channel, out of
`y`. **The emission CADENCE IS PER GROUP and is therefore UNCHANGED by the window**:
`y_a` is emitted for every group `a` of the window, so the `atomicAdd` count per token
is unmoved from the group-quantum visit.

The denominator emission is OUT OF THE MASS FRAGMENT, per group's warp: a
broadcast B makes every output column of `dm` the same value, so column 0 — the lanes
with `_r == 0` — carries tokens `_g` and `_g + 8`, and the `!= 0` guard retires a
ragged group's padded slots on the same rule the numerator runs on.

## <a id="gemm-fold"></a>29. GEMM 4 — the boundary fold, `Shi`/`Slo` += `sum_b W_b^T . Vw_b`, live UNION write atoms only, and the mass column's TERM 3

Runs when `w_write` (the OR of the window's write-live groups). The visit's fold work
is a STATIC STRIDED PARTITION — `for (c = warp; c < n_fold_items; c += kWarps)`,
`n_fold_items = popc(w_wmask) * NT_DV` — no atomic, no claim cursor: fold items are
uniform-cost and `n_fold_items` is CTA-uniform, so a strided claim is balanced to
within one item for any mask. `w_wmask` is the UNION over the window's `W` groups —
[the staged write mask](#staged-write-mask), never a group's own `wmask` — because an
atom live in ANY group of the window must fold once.

`mt = w_wmask == kFullMask ? mr : __fns(w_wmask, 0, mr+1)` maps a claimed dense index
`mr` to the `mr`-th LIVE atom. Skipping an atom dead in every group of the window is
exact: its delta is zero everywhere and its exit scale is `s0 == 1` (ticks ride
writes, [above](#decay-clock)).

**THIS IS WHERE THE WINDOW IS PAID FOR.** The claimed atom's 4-value accumulator
loads to registers ONCE (`a0..a3`, reconstructed by `merge_bf16x2` from the plane
pair at rows `i0`/`i1`, columns `n0`/`n0+1`) and stays resident while the atom
CONSUMES EVERY STAGED GROUP OF THE WINDOW it is live in — an inner loop over
`b in [0, W)` skipping groups where the atom is dead, each iteration computing one
`mma_fold_tile` delta into the tensor core's D fragment in ascending group order —
before the ONE writeback, which `split_bf16x2` re-narrows back into `Shi`/`Slo`. The
read-modify-write reads two words (`n0`, `n0+1` are one channel pair, hence one word
of each plane — `stage_pair_banks_ok`'s PROOF 4,
[`smem_layout.md#state-planes`](smem_layout.md#state-planes)) rather than four scalar
fp32 loads and stores, and is therefore amortised `W`-fold on top of that. The fp32
sum's rounding to the plane pair happens at this ONE writeback and nowhere else — the
accumulation itself stays fp32 across the whole inner loop. `kFoldStagedGroups` — the
constant this loop once counted, PINNED to 1 — is deleted; `W` is its replacement and
the loop trip count is the window's, not a fixed 1
([`smem_layout.md#values-shard`](smem_layout.md#values-shard)).

### <a id="fold-epilogue"></a>`DECAY`'s exit scale, riding the fold's existing read-modify-write, ONCE PER WINDOW

```
for (b = 0; b < W; ++b) {
  if atom dead in group b: continue
  d0 += mma_fold_tile(b)   // accumulates in the D fragment, ascending group order
  ...
}
a0 = e0 * (a0 + d0);   // e0 = DECAY ? s0[i0] : 1.0f -- ONE multiply, after the sum
split_bf16x2(a0, a1, Shi[i0][n0..n0+1], Slo[i0][n0..n0+1]);  // the ONE writeback
```

The exit scale applies ONCE, after all `W` groups' deltas have been summed, not once
per group — legal because every deposit is already in the window's ONE reference frame
(`keep^{P_ref - P[u]}`, [above](#decay-clock)), so the deposits simply add before the
frame is left. One multiply per window per accumulator, not one per staged group — no
separate scaling pass and no intermediate write to the planes.

**The mass column's TERM 3**, `Smass += sum_b W_b^T . sw_b`, is a SEPARATE loop
after the main fold, ON THE SAME TENSOR CORE AS THE FOLD ITSELF: `mma_tile_bcastB`
against `WT`'s atom block and a broadcast `sw`, one `mma.sync` per live atom per
staged group it is live in, one atom per warp (`for (mi = warp; mi < n_live; mi +=
kWarps)`). It is not folded into the main loop above because it writes one fp32
scalar per leaf (`_r == 0` only) rather than an `n8` tile of `DV` plane columns. A
dead atom's rows: `acc == 0` and `s0 == 1` — the skip is the identity, the same rule
the main fold's atom skip runs on.

**THE RMW IS ONE LANE PER LEAF.** A broadcast-B tile's result lands in output column
0, so the accumulator is meaningful only on `_r == 0`; the READ of `Smass[i0]` /
`Smass[i1]` is therefore predicated `_r == 0` too (`0.0f` on the other three
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
(`-1` if excluded). `p.ws_count[slot]`, `p.ws_scale[slot*BC+q] = a_state[i]`, and the
plane pair MERGED back to fp32 — `p.ws_state[(slot*BC+q)*cols + c..c+1] =
merge_bf16x2(Shi[i][c..c+1], Slo[i][c..c+1])` for the numerator's channel pairs, plus
`p.ws_state[(slot*BC+q)*cols + DV] = Smass[i]` for the mass column — follow for every
included leaf. Slot `owner_slot0 + seg + 1` is one past the scan's entry stash, so
pass 3's segment `q` reads `owner_slot0 + q` uniformly.

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
