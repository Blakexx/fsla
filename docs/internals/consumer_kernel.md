# `csrc/rola/src/consumer_kernel.cuh` — the RoLA V3 consumer kernel body

THE ENDGAME EXECUTOR, and the ONLY consumer. This doc owns the kernel BODY: the
streamed membership scan, the group assembler, the compact-into-registers atom walk,
the four GEMMs in their phase order, and the split-K epilogue.

**The body composes three layers and each has its own document:**

| document | source | owns |
|---|---|---|
| [`pipeline.md`](pipeline.md) | `pipeline.cuh` | the item stream — the control words, the release/acquire primitives, the edge DAG |
| [`smem_layout.md`](smem_layout.md) | `smem_layout.cuh` | the carve — address functions, conflict proofs, the register/residency budget, the byte ledger |
| [`mma_tile.md`](mma_tile.md) | `mma_tile.cuh` | the operand maps — fragment order, the `ldmatrix` atoms, the GEMMs' inner sequences |

Its sibling is [`consumer.md`](consumer.md), which owns `ConsumerParams` and the host
side and is NOT restated here: the leaf-digit decomposition, the paged state layout
and `kAbsentPage`, the §6.4 work list and its workspace, the three-pass driver, and
the instantiation matrix all live there. Two descriptions of one kernel is one that
can go stale.

## Reference conventions

The same in all four consumer-kernel documents — [`consumer_kernel.md`](consumer_kernel.md),
[`smem_layout.md`](smem_layout.md), [`mma_tile.md`](mma_tile.md),
[`pipeline.md`](pipeline.md):

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
6. reset the group scheduler's SMEM carry, `g_scan`.

**The walk**, once per GROUP of `kGroupTokens == 16` gathered tokens (every group is
full except each segment's final ragged flush; visit count is exactly
`ceil(gathered / 16)`):

7. the streamed membership scan advances the word cursor, ANDs the owner's fixed-level
   token-bit rows and the token-granular slice-union word, and ballots members into
   the group ascending, ORing in the per-token atom-union bytes as it goes;
8. **STAGE** — under `DECAY`, the clock prefix and entry scale; the §3 step-3 row
   scales `sr`/`sw`; the fragment-ordered read panel `Rf`; the write panel `WT`; the
   scale-folded value tile `Vw` — every leaf-axis loop walking LIVE ATOMS ONLY;
9. **GEMM 1, the readout** — `y_group = R . S`, live read atoms only, accumulating
   into each warp's own registers; the mass column's TERM 1 beside it;
10. retire `kCtlReadDone` (the readout READ `S`; the fold will WRITE it);
11. **GEMM 2, the Gram** — `G += R_q . W_q^T` over atoms live on BOTH sides, then §5's
    inclusive causal mask and the publication of `Gm`; **GEMM 4, the fold** —
    `S += W^T . Vw`, live write atoms only, run as a FLUID ITEM POOL, with `DECAY`'s
    exit scale riding its read-modify-write and the mass column's fold term beside it;
12. wait on `kCtlGramDone`, then **PHASE C**: **GEMM 3, the intra term** —
    `y_group += G . Vw` into the same registers the readout used, plus the mass
    column's TERM 2;
13. **EMIT** — the split-K `atomicAdd` of `sr[t] * y` into `num`, and of the
    denominator into `den`.

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
| `kGramWarps` | `min(kGT / 8, kWarps) == 2` — the warps that STORE `Gm`, and the narrow phase-C gate target |
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
| `sr`, `sw` | the §3 step-3 per-token read/write ROW SCALES, `[kGT]` each |
| `den` | the mass-column denominator accumulator, `[kGT]` |
| `dig` | the hoisted leaf digits, `[D][BC]` |
| `ctl` | the item-stream control block, `kCtlFold0 + 1` ints — ONE fold cursor ([`pipeline.md#item-stream`](pipeline.md#item-stream)) |
| `t_idx` | the group's `kGT` gathered GLOBAL token indices, ascending |
| `g_scan` | the group assembler's SMEM cross-visit carry (`fill`/`gword`/`g_fr`/`g_fw`/`member`/`gr,gw`) plus the `D` fixed-level row offsets — `[6 + D]` ints; the visit's read/write atom masks are plain per-thread registers, not carried here ([below](#group-assembler)) |
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
| `banks_distinct(GMP / 2, 1)` | `Gm` as GEMM 3's A operand |
| `banks_distinct_u16(BTP, 2)` | `WT` as GEMM 4's A operand |
| `pitched_ldsm_banks_ok<kMmaKQuantum, kGT, kWtPitch>()` | `WT`'s `ldmatrix` fetch conflict- and alignment-freedom at `kWtPitch` |
| `stage_offset_matches_cute<kGT, DV>()` | `Vw`'s closed-form address IS CuTe's layout |
| `stage_layout_is_bijection<kGT, DV>()` | the swizzled `Vw` tiles `[0, kGT*DV)` exactly |
| `stage_ldsm_banks_ok<kGT, DV>()` | `Vw`'s `ldmatrix.x2.trans` fetch |
| `stage_store_banks_ok<kGT, DV>()` | `Vw`'s staging store |

The predicates themselves, and the swizzle CuTe owns for `WT`/`Vw`, are
[`smem_layout.md#bank-predicates`](smem_layout.md#bank-predicates) and
[`smem_layout.md#stage-layout`](smem_layout.md#stage-layout).

## <a id="ldsm-alignment"></a>6. `ldmatrix` takes a 16-byte-aligned ROW ADDRESS

So both staged bases (`WT`, `Vw`) must be 16-byte aligned — a requirement `align4`
does NOT imply. Stated rather than trusted: `kWtOffsetBytes` and `kVwOffsetBytes` are
computed as explicit expressions from the carve and `static_assert`ed `% 16 == 0`,
because CUDA aligns the dynamic shared window itself to 16 B but says nothing about
what a mid-carve `align4` offset lands on.

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

## <a id="group-assembler"></a>11. The membership scan feeds the group assembler feeds one visit body

`w_lo`/`w_hi` bound this segment's range over the SCAN STREAM's word axis — the
32-token machine word is the metadata granule under §6.4's P > 1 partition now, not a
tile range. `cur`, the scan's word cursor, is a register (uniform by construction:
every thread evaluates it identically).

**Per iteration of the outer `for (;;)` loop, the scan fills a group to `kGT` members
(or the stream ends):**

- **word refill** (when the carried `member` bitmap is exhausted): AND the owner's
  fixed-level (`l < j`) `read_tok_bits` / `write_tok_bits` rows for this word, then
  form `member = (summary ? fw : (fr | fw)) & slice_union_tok[bh, slice, word]` — the
  token-granular slice-union word excludes slice-dead tokens AT BALLOT TIME, which is
  what keeps groups full. The summary pass ballots the WRITE side alone (union-visit
  semantics on one side);
- **ballot**: if `member` has more live tokens than the group has room for, it is cut
  at the `(room+1)`-th set bit (`__fns`) — the bits below the cut are exactly the
  lowest `room` live tokens, the remainder carries to the next group. Warp 0 writes
  `t_idx[fill + popc(taken & below-lane)] = token` for each taken bit;
- **atom masks, word-wise** (Change 1, `8675f95`): every thread redundantly ANDs
  `taken` against the `read_atom_bits` / `write_atom_bits` ROW for each of the
  `kQPO = BC / kMmaKQuantum` atoms at this ballot word (`ab_base + q * n_twords +
  gword`, `<= 2 * QPO` uniform loads) and ORs the nonzero result into `rmask` /
  `wmask` bit `q` — CTA-uniform by construction (uniform loads, uniform `taken`),
  no per-lane accumulator and no `__reduce_or_sync`. This is the atom-major
  contiguous layout's payoff: an atom's whole token-liveness row is one
  contiguous load per word, not a per-token byte gather;
- when the group fills (or the stream ends ragged), `rmask`/`wmask` are complete
  for the visit — plain per-thread REGISTERS, not SMEM: every thread rebuilt them
  redundantly from the same uniform loads/`taken` this iteration, so there is
  nothing to publish across the group-entry barrier for them (unlike `t_idx` and
  the `g_scan` cross-visit carry, which genuinely cross threads/iterations).

**Stability is the correctness requirement.** Words ascend and each word is consumed
LSB-first, so `t_idx` is GLOBALLY ascending — the intra Gram's inclusive causal mask
is order-relative, and this ordering IS what makes it correct.

**The cross-visit carry lives in `g_scan` (SMEM), not registers**, so no scheduler
state is register-live across the visit body. Every thread reloads the carry
redundantly from SMEM after the previous visit's `__syncthreads()`, and only `tid ==
0` stores the update, behind the group-entry barrier — the barrier both publishes
warp 0's `t_idx`/mask writes to every staging loop AND orders every thread's carry
LOADS (in the scan above) before `tid 0`'s carry STORES.

A side whose visit mask is empty (`rmask == 0` or `wmask == 0`) has no live
coefficients anywhere in the group on that side; `is_read`/`is_write` fold that
emptiness in alongside the group-level `gr`/`gw` flags, so a whole-visit skip is
possible even where the per-tile candidacy machinery of an earlier design could not
see it.

## <a id="staging"></a>12. The staging — everything the visit's compute consumes but does not produce

`stage_group(is_read, is_write, gN, rmask, wmask)` runs, LIVE ATOMS ONLY, for every
leaf-axis loop:

- under `DECAY`, the clock prefix and entry/exit scale ([below](#decay-clock));
- the §3 step-3 row scales `sr`/`sw` ([below](#row-scales));
- the read panel `Rf` ([below](#read-panel));
- the write panel `WT` ([below](#write-panel));
- `Vw[t, ch] = sw[t] * v[token, ch]` — the write side folds the row scale into the
  value tile because there `t` is the CONTRACTION index and the scale cannot hoist out
  of the sum.

Tokens are `t_idx[0..gN)`, globally ascending; a ragged group's padded slots
`[gN, kGT)` stage EXACT-ZERO coefficient rows and zero row scales, so they are inert
through every GEMM without any membership test.

`mask == kFullMask` selects the statically-unrolled dense arm at every one of these
loops — dense inputs pay zero mask instructions, because they self-heal to identity
grouping rather than taking a different code path.

## <a id="decay-clock"></a>13. `DECAY` steps (a) and (c) — the clock prefix, `P_ref`, `s0`, and the entry scale

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

## <a id="row-scales"></a>14. §3 step 3's row scales

`sr[t]`/`sw[t]`: the group's per-token read/write row scales — the read/write gain
times the FIXED levels' amplitudes at the owner's own digit, constant along the leaf
axis (which is why they are held out of the coefficient blocks entirely). They depend
on nothing the group's leaf-axis loops compute, only on `(bh, token)` and the owner's
`base[]`. `global` fixes the read scale's multiplicative identity at `1` (`g_read`
does not exist as a field — raw normalization is dead), so `sr` starts at `1` and
multiplies in only the fixed-level read amplitudes; `sw` starts at
`p.g_write[bh, token]` and multiplies in the fixed-level write amplitudes. Padded
slots (`t >= gN`) take `0`.

## <a id="read-panel"></a>15. The read panel `Rf` (§3 steps 2-3), live atoms only

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

## <a id="write-panel"></a>16. The write panel `WT`, live atoms only

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

## <a id="clk-dead"></a>17. HAZARD `clk-dead` — the `clk -> Vw` lifetime barrier

`if (DECAY && is_write) __syncthreads();` after the write panel and before the `Vw`
staging marks the exact point past which `Vw` may overwrite `clk`'s aliased bytes
([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)). Every reader of `clk`
(the read panel's `panel_read_factor`, the write panel's `panel_write_factor`, the
clock-prefix formation itself) runs before this barrier; the `Vw` staging store runs
after it. Moving either across the barrier is a silent aliasing hazard no compile-time
check can see.

---

# Part V — the four GEMMs

## <a id="gemm-readout"></a>18. GEMM 1 — the readout, `y_group = R . S`, live read atoms only

Runs when `is_read && warp < NT_DV`. `mask == kFullMask` unrolls the dense arm
(`mma_tile_fragA<BC, ...>` over the whole state); otherwise the loop walks only the
live atoms of `rmask`, contracting `kMmaKQuantum` leaves per atom
(`mma_tile_fragA<kMmaKQuantum, ...>`). The B operand (`S`) is packed out of SMEM on
the fly; state and accumulation stay fp32 throughout.

## <a id="mass-term1"></a>19. The mass column, TERM 1 — placement FORCED by phase order

`den[t] = sum_k Rf[t,k] * S[k, DV]`, live read atoms only, using `frag_a_get` to read
back an element of the fragment-ordered `Rf` (a `[kGT x BC] x [BC x 1]` matvec — an
`n8` MMA tile would waste 7/8 of a fragment, so this stays a scalar reduction). Its
placement is FORCED: the fold (GEMM 4, later in phase order) WRITES `S[i][DV]`, so
term 1, which READS it, must run before the fold does — this visit's fold writes only
apply to the state term 1 already consumed.

### <a id="mass-split"></a>HAZARD `mass-split` — bit-identity of the split rests on same-thread program order

Term 1 (here, before the fold) and term 2 (in phase C, after the fold) are added into
the same `den[t]` by the SAME thread across two program points with no intervening
reassignment of which thread owns which `t` — reordering either half relative to the
fold changes what `S[k, DV]` term 1 reads, which is a real numeric difference, not
just a reassociation.

## <a id="b6"></a>20. B6 — the visit's read-then-write ordering on `S`, as a counter

`retire_item(&ctl[kCtlReadDone], lane)` runs UNCONDITIONALLY at the end of the readout
phase — every warp retires once whether or not it did read work, because the counter
target (`kWarps`) counts warps, not readers. The fold below waits on this count before
its first read-modify-write of `S`, so only the fold waits, and it waits behind real
work rather than a rendezvous every warp must reach in lockstep.
[`pipeline.md#item-stream`](pipeline.md#item-stream) has the release/acquire
discipline and the deadlock argument this rests on.

## <a id="gemm-gram"></a>21. GEMM 2 — the causal effective-score block, `G += R_q . W_q^T`

Runs when `mixed = is_read && is_write` and `warp < kGT / 8`, over `gmask = rmask &
wmask` — atoms live on BOTH sides (either side dead makes the term exactly zero, and
the dead side's panel was never even staged). `gmask == kFullMask` unrolls dense over
every atom; otherwise the loop walks `gmask`'s live atoms. The completed accumulator
is written into `Gm` through §5's INCLUSIVE causal mask (`c <= _g` / `c <= _g + 8`
zeroing the strictly-future entries) via `gm_put`.

## <a id="gram-gate"></a>22. `kGramWarps` and the phase-C gate

`kGramWarps = min(kGT / 8, kWarps) == 2` narrows both the Gram's PUBLICATION
(`retire_item(&ctl[kCtlGramDone], lane)`, guarded `mixed && warp < kGramWarps`) and
phase C's WAIT (`await_count(&ctl[kCtlGramDone], kGramWarps)`, guarded `mixed`) to the
warps that actually write `Gm` — a measured scope, free on bf16 storage. `!mixed`
skips the gate entirely: a group with nothing live on one side has no Gram term to
wait for.

### <a id="gram-gate-guard"></a>HAZARD `gram-gate-guard` — the retirement predicate mirrors the STORE guard, never the GEMM guard

`retire_item`'s guard (`mixed && warp < kGramWarps`) must equal the SET OF WARPS THAT
STORE INTO `Gm`, which is `warp < kGT / 8` under `mixed` — the same condition GEMM 2
computes under, narrowed only by `kGramWarps <= kGT/8`. A retirement guard that
diverged from the store guard (e.g. matching the GEMM's accumulate condition instead
of its store condition) would let `kCtlGramDone` reach its target before every store
that target was supposed to count has landed.

## <a id="gemm-fold"></a>23. GEMM 4 — the boundary fold, `S += W^T . Vw`, live write atoms only

Runs when `is_write`, after `await_count(&ctl[kCtlReadDone], kWarps)`. The visit's
fold work is a FLUID ITEM POOL: `n_fold_items = popc(wmask) * NT_DV`, and every
participating lane claims the next item with a single WARP-UNIFORM `atomicAdd` on
`ctl[kCtlFold0]`, broadcast with `__shfl_sync` so every lane of the warp agrees on the
claimed index — the claim is one atomic per WARP per item, not per lane.

### <a id="fold-claim-uniform"></a>HAZARD `fold-claim-uniform` — the claim is warp-uniform and must never become per-lane

Only `lane == 0` issues the `atomicAdd`; every other lane's `c` comes from the
`__shfl_sync` broadcast. A per-lane claim would both explode the atomic traffic 32x
and break the item's `(atom, n-tile)` decomposition, which the whole warp must agree
on to compute the same `(mt, nt)` for the same MMA.

`mt = wmask == kFullMask ? mr : __fns(wmask, 0, mr+1)` maps the claimed dense index
`mr` to the `mr`-th LIVE atom. Skipping a dead atom's read-modify-write is exact: its
delta is zero and its exit scale is `s0 == 1` (ticks ride writes,
[above](#decay-clock)).

**The item's RMW is ATOM-MAJOR, not per-window** (Change 1, `8675f95`): the claimed
atom's 4-value accumulator loads to registers ONCE (`a0..a3` from `S`) and stays
resident across an inner `#pragma unroll for (gs = 0; gs < kFoldStagedGroups;
++gs)` that walks every staged slot the atom must consume — each iteration computes
one `mma_fold_tile` delta and applies the exit-scale epilogue in registers — before
the ONE writeback back to `S`. `kFoldStagedGroups` is `smem_layout.cuh`'s
`kFoldStagedGroups`, PINNED to 1 today, so the loop is a single iteration in
practice; the per-atom fetch era that deepens staging is expected to raise only this
constant and its staging producer, not restructure the claim/writeback shape around
it. The mass column's fold term (below) walks the identical `gs` nest over the same
constant.

### <a id="fold-epilogue"></a>`DECAY`'s exit scale, riding the fold's existing read-modify-write

```
for (gs = 0; gs < kFoldStagedGroups; ++gs) {
  a0 = e0 * (a0 + d0);   // e0 = DECAY ? s0[i0] : 1.0f, d0 from mma_fold_tile(gs)
  ...
}
S[i0][n0] = a0;          // the ONE writeback, after every staged slot is folded in
```

One extra multiply per staged slot on a value that stays in registers for the whole
inner loop — no separate scaling pass and no intermediate write to `S`. The mass
column's fold term (`S[row][DV]`, live write atoms only) takes the same `gs`-nested
epilogue: a dead atom's rows have `acc == 0` on every staged slot and `s0 == 1`, so
the skip is the identity there too.

## <a id="phase-c"></a>24. Phase C — GEMM 3, the intra term, and the mass column's term 2

After `await_count(&ctl[kCtlGramDone], kGramWarps)` (guarded `mixed`): `y_group +=
G . Vw` into the SAME registers the readout used (`mma_tile_vwB`, `warp < NT_DV`) —
token-axis contraction, so no atom mask applies here. The mass column's term 2,
`den[t] += sum_{u<=t} Gm[t,u] * sw[u]`, runs alongside under the same `mixed` guard
(HAZARD `mass-split`, [above](#mass-term1)).

---

# Part VI — the epilogue

## <a id="split-k-emission"></a>25. The split-K emission

Guarded `is_read`. `emit(t, ch, val)` computes `out = sr[t] * val` and skips the
`atomicAdd` on `out == 0.0f` — a PERFORMANCE skip on a VALUE, never a membership
decision: a ragged group's padded slots have `sr == 0` by construction and never read
`t_idx` at all (members are `< T` by construction, so the `!= 0` guard cannot mask a
real emission). The numerator scatters through `t_idx[t]` per output channel; the
denominator emission is the same pattern over `den[t]`.

## <a id="pass1-epilogue"></a>26. Pass 1's per-leaf epilogue

Guarded `is_write && summary && DECAY`: `if (s0[i] != 1.0f) sum_slot[i] = 1;` catches
the leaves the [write-panel's coefficient-side test](#summary-prequant) could not —
decay alone can make a leaf's survival move even where its coefficient rows happened
to be zero on this visit. `a_state[i] *= s0[i] * s0[i]` accumulates the segment's
per-leaf survival product `A` (identity `1` under `decay = None`, since `s0` is never
formed there).

## <a id="pass1-publication"></a>27. Pass 1's publication — the sparse written-leaf summary

After the walk, under `summary`: warp 0 compacts `sum_slot[i] != 0` into an ascending
dense list via one `__ballot_sync` per 32-leaf block, writing the leaf's ordinal into
`p.ws_leaf[slot * BC + q]` and rewriting `sum_slot[i]` in place to its list position
(`-1` if excluded). `p.ws_count[slot]`, `p.ws_scale[slot*BC+q] = a_state[i]`, and
`p.ws_state[(slot*BC+q)*cols + c] = S[i*SP+c]` follow for every included leaf. Slot
`owner_slot0 + seg + 1` is one past the scan's entry stash, so pass 3's segment `q`
reads `owner_slot0 + q` uniformly.

## <a id="state-store"></a>28. The owner's state block, written to HBM once

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
| [`mass-split`](#mass-split) | the mass column's two-term split relies on same-thread program order across the fold |
| [`gram-gate-guard`](#gram-gate-guard) | the Gram retirement guard must mirror the store guard, not the GEMM guard |
| [`fold-claim-uniform`](#fold-claim-uniform) | the fold's item claim is one atomic per warp, never per lane |
