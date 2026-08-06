# `csrc/rola/src/consumer_kernel.cuh` — the RoLA V3 consumer kernel body

The ONLY consumer. This doc owns the kernel BODY: the §3/§5 algorithm, the pointer
carve as the body spells it, the staging ring's candidate walk, the four GEMMs in
their phase order, the split-K epilogue, and the `DECAY` / `GLOBAL` arms.

**The body composes three layers and each has its own document:**

| document | source | owns |
|---|---|---|
| [`pipeline.md`](pipeline.md) | `pipeline.cuh` | the item stream — the control words, the release/acquire primitives, the edge DAG |
| [`smem_layout.md`](smem_layout.md) | `smem_layout.cuh` | the carve — address functions, conflict proofs, the register/residency budget, the ring depth, the byte ledger |
| [`mma_tile.md`](mma_tile.md) | `mma_tile.cuh` | the operand maps — fragment order, the `ldmatrix` atoms, the GEMMs' inner sequences |

Its sibling is [`consumer.md`](consumer.md), which owns `ConsumerParams` and the
host side and is NOT restated here: the leaf-digit decomposition, the paged state
layout and `kAbsentPage`, the §6.4 work list and its workspace, the three-pass
driver, the instantiation matrix, and the residency query's axes all live there.
Two descriptions of one kernel is one that can go stale.

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

One CTA per `(owner, segment)` work item, 8 warps, run to completion. The owner's
state block lives in SHARED memory for the CTA's whole life, is read from HBM
ONCE at entry and written ONCE at exit.

**Per CTA, once:**

1. resolve the work item (`bh`, `owner`, and §6.4's `seg` / `nseg` / `slot`);
2. decompose the owner into per-level digit bases `base[l]` (§3's rectangle);
3. build the candidate-tile bitmap by ANDing `read_union` / `write_union` with the
   FIXED levels' tile bitmaps and with this segment's tile range;
4. hoist the owner's leaf digits `dig[l][i]` — `D * BC` integer divisions, once,
   instead of once per (tile, token, leaf, level);
5. under `DECAY`, form the per-leaf `log_keep[i] = log1pf(-rate[i])`;
6. load `S[BC][SP]` — from the workspace under §6.4, from zeros in pass 1, else
   atom-wise from `p.state` through the page table.

**Per candidate tile** (the walk skips tiles with no read and no write bit):

7. **STAGE** — the `DECAY` clock prefix and
   entry scale, the §3 step-3 row scales `sr`/`sw`, the fragment-ordered read
   panel `Rf`, the write panel `WT`, and the scale-folded value tile `Vw`;
8. **GEMM 1, the readout** — `y_tile = R . S`, `[BT x BC] x [BC x DV]`,
   accumulating into each warp's own registers; plus the mass column's TERM 1
   under `GLOBAL`;
9. retire `kCtlReadDone` (the readout READ `S`; the fold will WRITE it);
10. **PHASE B**, once per `WT` window: **GEMM 2, the Gram** — `G += R_q . W_q^T`,
    then §5's inclusive causal mask and the publication of `Gm`; then **GEMM 4,
    the fold** — `S += W^T . Vw`, run as a FLUID ITEM POOL, with `DECAY`'s exit
    scale riding its read-modify-write and `GLOBAL`'s mass fold beside it;
11. wait on `kCtlGramDone`, then **PHASE C**: **GEMM 3, the intra term** —
    `y_tile += G . Vw` into the same registers the readout used, plus the mass
    column's TERM 2;
12. **EMIT** — the split-K `atomicAdd` of `sr[t] * y` into `num` (and of the
    denominator into `den` under `GLOBAL`).

**Per CTA, at exit:** under §6.4 pass 1, publish the sparse `(A, B)` summary;
otherwise, if this is the owner's last segment, store `S` back atom-wise.

Three facts a reader must not miss:

- **The state is SMEM-resident, not register-resident** (§10i.13). That is what
  decouples the thread count from the state layout, frees the CTA to 8 warps, and
  puts every per-tile term on the tensor cores.
- **The one-copy invariant is a property of the LAUNCH SHAPE**, not of a policy
  the kernel implements: one CTA per owner, run to completion, one read and one
  write of the state block.
- **Every structural axis is a template parameter and there is no runtime
  fallback on any of them.** `BT`, `BC`, `DV`, `D`, `READ_PREC`, `FOLD_PREC`,
  `DECAY`, `GLOBAL` — see [`consumer.md`](consumer.md#instantiation-matrix).

## <a id="origin"></a>2. The three structural moves, and the measurements that hold them

The kernel's shape is three decisions taken together — state SMEM-resident, every
per-tile term a tensor-core GEMM, and a WIDE CTA freed by the first two — and it is
**x2.32 at `L = 2048`** against a register-resident consumer (provenance: journal).
That is what a kernel of this shape buys over the latency-bound regime `ncu`
attributes to a register-resident, scalar-accumulate consumer: tensor pipe **0.60 %**
of peak, warps active **8.3 %**, **0.28 inst/cycle** (`harvest.md` §30.3).

Held FIXED across that comparison, and unchanged here: `ConsumerParams`, the
schedule, the candidate walk (`read_union` / `write_union` AND the fixed levels' tile
bitmaps, §4), the owner-stationary run-to-completion dataflow (ONE CTA PER OWNER,
`blockIdx.x` indexing the work list — traffic-optimal, §10i.12), and the split-K
atomic `num`/`den` accumulation with its `divide_pass`.

The three moves:

1. **STATE LIVES IN SMEM** (`S[BC][SP]`), not in `M[BC]` per thread. The register
   file is wrong for this object because it is thread-PRIVATE (so the state layout
   dictates the thread structure), NOT ADDRESSABLE, and it locks updates out of the
   tensor cores (§10i.13). A register-resident state pays this SMEM anyway, for the
   MMA readout's B operand — BOTH copies — so the move costs zero net shared memory
   and returns 64 registers per thread.
2. **EVERY PER-TILE TERM IS A GEMM** (§10i.8's coefficient domain, chunk-scoped):

   ```
   readout   y_tile  = R . S                 [BT x BC] x [BC x DV]
   intra     G       = R . W^T               [BT x BC] x [BC x BT], causal
             y_tile += G . Vw                [BT x BT] x [BT x DV]
   fold      S      += W^T . Vw              [BC x BT] x [BT x DV]
   ```

   The fold is §10i.8's "S += W_chunk^T V_chunk, one batched tensor-core GEMM per
   live owner", in place of per-visit scattered `d_v`-wide scalar accumulates — the
   0.28 inst/cycle serial chains §10i.5 measured.
3. **THE CTA IS 8 WARPS** (256 threads), not `DV` = 64 threads / 2 warps. This is
   the payoff of (1): once the state is shared rather than private, the thread
   count decouples from the state layout (§10i.13).

`decay`, `normalization`, `P > 1` / paging and `MmaPrec` are template axes with every
value instantiated, and `BT x BC x d_v` spans the planner's own `_SUPPORTED_*` sets
exactly. The supported surface is declared at `launch_consumer`
([`consumer.md`](consumer.md#plan-surface)) and deliberately not restated in the
kernel.

One scope note that is an ALGORITHM fact rather than a dispatch fact: under
`normalization = "global"` the mass column rides as column `DV` of the state and is
handled by scalar reductions rather than as an extra MMA `n` — one extra `n` would
waste 7/8 of an `n8` fragment ([`mma_tile.md#frag-a-get`](mma_tile.md#frag-a-get)).

## <a id="glossary"></a>3. Symbol glossary

| symbol | what it is |
|---|---|
| `BT` | tokens per compute tile; `{16, 32}` |
| `BC` | leafs per compute tile (the owner's leaf block); `{16, 64}` |
| `DV` | value width `d_v`; `{32, 64}` |
| `D` | level count of the topology; `1 <= D <= 4` (spec §1) |
| `j` | `first_spanned_level`. Levels `l < j` are FIXED for this owner |
| `kMmaKQuantum` | `16` — the hardware `mma.sync` k-extent on the sm80 bf16 path, and the named constant every quanta-shaped quantity is an expression in (spec §1 I1). Defined once, in `arch_table.cuh`: [arch_table.md#mma-quantum](arch_table.md#mma-quantum) |
| `QPO` / `kQPO` | quanta per owner, `BC / kMmaKQuantum`. An ATOM is one `kMmaKQuantum`-leaf block in canonical leaf order |
| `NM` | `BT / 16` — the tile's `m16` ROW BLOCKS |
| `NT_DV` | `DV / 8` — the `n8` column tiles of a `DV`-wide output |
| `kWarps` | `8` — `kConsumerThreads / 32` |
| `kYTiles` | output fragments per warp, `ceil(NM * NT_DV / kWarps)` |
| `kGTiles` | Gram fragments per warp, `ceil(NM * (BT/8) / kWarps)` |
| `kGramWarps` | `min(NM * (BT/8), kWarps)` — the warps that STORE `Gm`, and the narrow phase-C target |
| `kRing` | staging-ring depth, `1` or `2` ([`smem_layout.md#ring-depth`](smem_layout.md#ring-depth)) |
| `kWtWindow` | `WT` window size in ATOMS; `kWinRows = kWtWindow * 16` rows; `kNWin = kQPO / kWtWindow` windows per tile ([`smem_layout.md#wt-window`](smem_layout.md#wt-window)) |
| `S` | the STATE, `[BC][SP]` fp32 SMEM, resident for the owner's whole run |
| `SP` | `S`'s row pitch, `DV + 4` ([below](#state-pitch)) |
| `cols` | `DV + 1` under `GLOBAL` (the mass column), else `DV` |
| `WT` | the write coefficient panel, LEAF-MAJOR, `[kWinRows][BTP]` |
| `BTP` | `WT`'s row pitch; `BT + 8` narrow, `BT + 4` on `tf32x3` ([`smem_layout.md#stage-format`](smem_layout.md#stage-format)) |
| `Vw` | the scale-folded value tile, `Vw[t,ch] = sw[t] * v[t0+t, ch]` |
| `VWP` | `Vw`'s row pitch on the `tf32x3` arm, `DV + 8`. The narrow arm has NO pitch — it is `StageLayout` ([`smem_layout.md#stage-layout`](smem_layout.md#stage-layout)) |
| `Gm` | the causal effective-score block `[BT][BT]`, pitch `GMP` |
| `AmpRT`, `AmpWT` | the amplitude-class HBM storage types ([below](#amp-class-types)): bf16 on the `kBf16 && GLOBAL && !DECAY` arms (read side keyed to `READ_PREC`, write/value side to `FOLD_PREC`), `float` everywhere else |
| `kAmpSplit` | whether the amplitude storage is ROLE-SPLIT on this arm (`sizeof(AmpRT) == 2 \|\| sizeof(AmpWT) == 2`): spanned levels narrow, fixed levels fp32 |
| `Rf` | the read coefficients, MMA-FRAGMENT-ORDERED, `NM` blocks of `frag_a_words(BC)` words |
| `clk` | the tile's per-leaf INCLUSIVE clock prefix `[BT][CKP]` fp32, `CKP = BC + 4`; ALIASES `Vw`'s bytes |
| `log_keep`, `p_ref`, `s0` | `DECAY`'s three `[BC]` vectors — per-owner log retention, §12 rule 1's reference exponent, and the entry/exit scale |
| `sr`, `sw` | the §3 step-3 per-token read/write ROW SCALES, `[BT]` each |
| `den` | the `GLOBAL` denominator accumulator, `[BT]` |
| `dig` | the hoisted leaf digits, `[D][BC]` |
| `ctl` | the K3b control block, `2 + QPO` ints ([`pipeline.md#item-stream`](pipeline.md#item-stream)) |
| `kUnit` | the MMA's own k-step: `16` on bf16 (`m16n8k16`), `8` on tf32x3 (`m16n8k8`) |
| `kMaxWords` | candidate-tile words held in SMEM per CTA (`consumer.cuh`) |
| `A`, `B` | §6.4's per-segment summary: the per-leaf survival product and the accumulated deposit |

---

# Part I — the compile-time budget

## <a id="mma-precision"></a>6. MMA precision is ONE axis with two named points

```
PRODUCTION  = bf16   at BOTH sites   (Plan.mma_precision == "bf16")
EXACTNESS   = tf32x3 at BOTH sites   (Plan.mma_precision == "tf32x3")
```

The split is about WHAT EACH ARM IS FOR, not about which is numerically nicer: the
exactness arm exists so an oracle gate can see a STRUCTURAL bug through the numerics
(that needs mantissa), and the production arm exists to ship (that needs speed).
Gating the shipped arm on the exact arm's tolerance would be measuring a kernel
nobody runs (provenance: `de8567a0`).

**Why the fold is bf16 in production:**

- the accumulation hazard lives in the ACCUMULATOR, and the accumulator is fp32 SMEM
  state — unchanged by the operand format;
- the state deviation this costs was MEASURED FLAT IN L (3.447e-03 at `L=128`,
  3.441e-03 at `L=256`), i.e. per-deposit quantization with no growth pathway;
- `v` is bf16-quantized at the layer boundary already, so the fold's `V` operand
  carries that error whatever the MMA does with it;
- flash-attention's `P.V` and FLA's chunk kernels ship exactly this contract;
- and tf32x3 is "near-free" only AT 3.55 % TENSOR PIPE (measured). In the
  compute-bound regime this kernel runs in, 3x the fold's passes is ~1.7x the
  kernel's total MMA work.

`MmaPrec` is a PER-SITE template axis, but the two shipped POINTS on it set both
sites together. The per-site parameterization is what makes the contract STATEABLE
and testable at the call sites; no mixed arm ships.

The STATE deviation the production (bf16) arm carries is DERIVED, not observed: each
deposit rounds two bf16 operands, each to within `2^-8` relative, and the accumulation
is fp32, so the per-deposit relative error is bounded by `2^-7` and DOES NOT GROW WITH
THE NUMBER OF DEPOSITS. That is the bound the suite declares, and the flat-in-L
measurement above is its confirmation rather than its source.

## <a id="force-ctas"></a>9. `ROLA_FORCE_CTAS` — the forcing-census instrument

When defined, it replaces the DERIVED launch bound with a fixed CTA count on every
instantiation, so a candidate register budget can be censused over the whole 256-entry
dispatch matrix in one probe compile (`tools/ratify.py --census --force-ctas N`) with
no build and no overlay. It exists because `consumer_reg_budget` is a MEASURED constant
whose premise expires whenever the kernel's working set moves (D8-Q1), and the only way
to re-measure it is to ask `ptxas` what the kernel needs at each candidate bound.

**It is a census instrument, never a shipping knob:** the extension build does not
define it (`setup.py` passes no such flag), `tools/ratify.py`'s census mode is its only
legitimate definer, and a manifest is only ever ratified off the UNFORCED compile.
Forcing changes the REGISTER side only — `consumer_ring_depth` and the SMEM derivation
stay shipped, which is what the census wants (the register file is the axis under
measurement, the footprint is not).

---

# Part II — the carve, as the body spells it

## <a id="pitches"></a>15. Operand row pitches ARE bank-conflict padding, and they are load-bearing

Measured on the unpadded build at the at-scale cell: 65.6 M shared-load instructions
retired 266.1 M wavefronts — **4.06x amplification, i.e. 75 % of all SMEM load traffic was
bank conflict** — which is most of the 71.87 % memory-pipe wall the stage-1 `ncu`
attribution found.

The cause is the MMA fragment map, not the buffers: a fragment's lanes read DOWN a column
(`g = lane >> 2` selects the row for the A operand, the column for B), so an unpadded row
stride that is a multiple of 32 banks maps every one of those lanes onto the same bank.
Concretely, an unpadded fp32 coefficient panel addressed `[k * BC + m]` with `BC = 64`
floats puts the bank at `m % 32` for EVERY `k`, so the four threads of a group collide four
ways; an unpadded bf16 one addressed `[ra * BC + ca]` puts the 32-bit pair address at
`g * 32 + ...`, so all EIGHT `g` values collide.

One extra element per row decorrelates them. The bf16 buffers take `+2` elements rather
than `+1` because they are read as 32-bit PAIRS, so the stride must move by a whole 4-byte
bank. Cost is 256 B per CTA total and the residency is unchanged. Pitches are DERIVED from
the fragment maps and PROVEN by `banks_distinct`, not chosen: `+1` merely decorrelates
without making the 32 lanes distinct, and it measures 2.71 wavefronts per load.

### The `static_assert` ledger, and what each `stride_g` / `stride_r` pair reads off

```
A, bf16 pair-read : banks = g * (pitch/2) + r          (k steps by 2)
B, bf16           : banks = g * 1         + r * 2*pitch
A, tf32           : banks = g * pitch     + r * 1      (k steps by 1)
B, tf32           : banks = g * 1         + r * pitch
```

| assert | what it pins |
|---|---|
| `banks_distinct(1, 2 * SP)` | `S` as GEMM 1's B operand |
| `banks_distinct_u16(1, 2 * BTP)` / `banks_distinct(1, 2 * BTP)` | `WT` as GEMM 2's B, under the predicate its STORAGE FORMAT makes correct |
| `banks_distinct(GMP/2 or GMP, 1)` | `Gm` as GEMM 3's A |
| `banks_distinct_u16(BTP, 2)` / `banks_distinct(BTP, 1)` | `WT` as GEMM 4's A |
| `pitched_ldsm_banks_ok<kWtRows, BT, BT+8>()` | the narrow arm's `ldmatrix` fetch of `WT` — ADDED, not substituted: the two above are about SCALAR fragment maps at this pitch, and `ldmatrix`'s phase reads 16 BYTES per lane and requires that address 16-byte ALIGNED |
| `stage_offset_matches_cute<BT, DV>()` | `Vw`'s closed-form address IS CuTe's layout |
| `stage_layout_is_bijection<BT, DV>()` | the swizzled `Vw` tiles `[0, BT*DV)` exactly |
| `stage_ldsm_banks_ok<BT, DV>()` | `Vw`'s `ldmatrix.x2.trans` fetch |
| `stage_store_banks_ok<BT, DV>()` | `Vw`'s staging store |
| `banks_distinct(1, VWP)` | `Vw` as GEMM 4's B, `tf32x3` arm only |

**`WT` carries BOTH pitch predicates on both arms** — its `BT + 8` is a derived pad, not a
swizzle, and [`smem_layout.md#stage-format`](smem_layout.md#stage-format) gives the
measurement that keeps it that way. `Vw`'s three narrow-arm
proofs are STRICTLY MORE than a pitch predicate: a pitch predicate asks only whether 32
lanes striding by `2 * VWP` decorrelate, while these enumerate CuTe's own layout at the
addresses the two accesses actually issue — the `ldmatrix` phase (with its alignment
requirement no pitch predicate can see) and the staging store — and additionally require
the layout to be a bijection onto the bytes the footprint reserves for it.

### <a id="vw-conflict-trade"></a>The one operand that is not conflict-free, and the trade is explicit

On the `tf32x3` arm `Vw` is read by the fold under the tf32 map (`k` steps by 1) and by the
intra GEMM under the bf16 map (`k` steps by 2), and NO single pitch satisfies both — the two
conditions are `VWP = 8 (mod 32)` and `VWP = 4 (mod 16)`, which are disjoint. The fold is
chosen because it carries ~8x the operand traffic (4 warp-tiles x 2 k-steps against the intra
term's single k-step), so `Vw` is sized for it and the intra term keeps a 2-way conflict on
one of its two operands. Stated here rather than left for someone to rediscover with a
profiler, and pinned by `static_assert(kNarrowStage || !banks_distinct(1, 2 * VWP))` so the
note fails the build if it stops being true.

**On the narrow arm the trade DOES NOT ARISE**, and that is a consequence of the storage
format rather than a coincidence: with the operands in the MMA's own format the FOLD reads
`Vw` under the bf16 map too (`FOLD_PREC` is bf16 whenever `kNarrowStage`), so the fold and
the intra term do not want different pitches. **The swizzle then makes the question moot on
that arm rather than merely favourable**: both consumers issue the SAME `ldmatrix.x2.trans`
against the SAME swizzled layout, so there is ONE access pattern, not two to reconcile.

## <a id="ldsm-alignment"></a>16. `ldmatrix` takes a 16-byte-aligned ROW ADDRESS

So every ring slot base must be 16-byte aligned — a requirement the `align4` carve does NOT
imply and which `stage_ldsm_banks_ok` can only check RELATIVE to the base. Both terms are
stated rather than trusted: the offset of slot 0 (CUDA aligns the dynamic shared window
itself to 16 B) and the slot stride. `kVwOffsetBytes % 16`, `kVwSlotBytes % 16` and
`kWtOffsetBytes % 16` are `static_assert`ed on the narrow arm.

## <a id="state-pitch"></a>17. `SP = DV + 4` — the state's row pitch

`SP = DV + 4` is the kernel's ONE state pitch. The DERIVATION is stated because it is what
makes the value checkable:

`DV + 1` is what the tf32 MMA B map alone would ask for, where `k = k0 + r` steps by one — at
`SP == DV` (a multiple of 32) every lane in a group of four lands in the same bank and the
`+1` decorrelates them. The bf16 map steps `k = k0 + 2r`, so the bank index becomes
`(2r + g) % 32`, which COLLIDES up to four ways (`2r + g = 6` is hit by
`(r,g) = (0,6), (1,4), (2,2), (3,0)`). Since bf16 is the production default (`de8567a0`),
`DV + 1` would read the state through the conflicted map.

`DV + 4 = 68` makes the bf16 map conflict-free: `2r * 68 = 136r`, so the bank is
`(8r + g) % 32`, and `r in [0,4)`, `g in [0,8)` covers all 32 banks exactly once.

## <a id="gm-type"></a>18. `Gm`'s storage type follows `READ_PREC`, and that is a correctness requirement

`Gm` is an INTERMEDIATE of the `y` path (`G = R.W^T`, then `y += G.Vw`), so storing it bf16
on the tf32x3 arm quantizes a value the exactness arm exists to keep exact. That failure is
REAL rather than hypothetical: a bf16 `Gm` on the tf32x3 arm measures state 1.8e-07 (exact)
alongside `y` 1.0e-03 (not), and the oracle gate localizes it to the one object on the `y`
path that is narrow.

The pitch differs with the type because the FRAGMENT MAP does: the bf16 map reads 32-bit
pairs (`stride_g = GMP/2`), the tf32 map single floats (`stride_g = GMP`). `GMP` is `BT + 8`
on bf16 and `BT + 4` on tf32x3; both are proven by `banks_distinct` ([below](#pitches)).

## <a id="amp-class-types"></a>18b. `AmpRT` / `AmpWT` — the amplitude-class storage types, in the body

The P09 bf16-memory family narrows the HBM STORAGE of the spanned amplitude levels and the
value stream on the `kBf16 && GLOBAL && !DECAY` arms. The dtype rules, the per-stream table
and the measured carve rationale live with the parameter block they govern
([`consumer.md#amp-dtype`](consumer.md#amp-dtype)); what belongs HERE is how the body carries
the split without a new axis and without touching the hot loops:

- **The types are DERIVED, not new template parameters** — `AmpRT` keys off `READ_PREC`,
  `AmpWT` off `FOLD_PREC` (mirroring which GEMM consumes each stream), both gated on
  `!DECAY && GLOBAL`, exactly the way `StageT` and `GmT` are derived. Zero new axes; the
  dispatch matrix is unchanged.
- **One widening point, `amp_to_float`** — identity on `float`, the explicit
  `__bfloat162float` intrinsic on bf16. Overloads rather than a `static_cast`, so the
  conversion survives the build's `__CUDA_NO_BFLOAT16_CONVERSIONS__`. Every spanned gather
  (the read panel, `build_wt_rows`'s accessor, the `Vw` value load) widens through it; the
  arithmetic beyond the load is fp32 either way.
- **The tile bases keep the baseline's exact shape.** `amp_r[l]` / `amp_w[l]` are one typed
  pointer advance per level, no per-level role branch — a role ternary at the base measurably
  spilled the `D >= 3` decay arms. The cost of that shape is a stated caveat: on a split arm
  the entries `l < j` carry a WRONGLY-SCALED address (the byte stride is `AmpRT`/`AmpWT`'s,
  the fixed stream's is fp32) and are NEVER dereferenced there.
- **`kAmpSplit` is where the fixed levels go instead**: the row-scale loops index their
  always-fp32 streams directly off `p.amp_read[l]` / `p.amp_write[l]` under
  `if constexpr (kAmpSplit)`. Those loops run `BT` trips per tile, outside every hot loop, so
  the 64-bit re-derivation per element is paid where it does not price anything. On the
  uniform-fp32 arms the row scales read the tile bases exactly as they always did.

`v`'s load in the staging is typed `AmpWT` the same way (it feeds the fold's B operand, which
the narrow arm's MMA takes in bf16 regardless). The clock, `num`/`den`, and the state stay
fp32 unconditionally — accumulator class, never storage-narrowed.

## <a id="pass1-alias"></a>19. §6.4 pass 1's own state, aliased onto `Rf`

```
a_state[i]   the per-leaf survival product `A` over the segment. Identically 1 under
             decay = None; under decay it accumulates s0^2 per write tile ([below](#phase-c)).
sum_slot[i]  the written-leaf FLAG, rewritten in place by the summary's publication to
             the leaf's POSITION in the emitted list (-1 if not in it).
sum_count    the list length.
```

The alias costs ZERO shared memory, which is the point: pass 1 forces `is_read = false` on
EVERY tile (the candidate walk zeroes `cand_read`), so the read coefficients are neither
written nor read in that pass — while `P = 1` never runs pass 1 at all. A dedicated
allocation would have moved the `P = 1` launch's SMEM footprint, and `P = 1` is the primary
benchmarked configuration (§6.4): a perturbed baseline is not a baseline.

It names SLOT 0 of the ring, and that is sound at any depth: pass 1 forces `is_read = false`
on every tile, so no slot's read coefficients are ever built in the pass that owns this
alias — including the RUN-AHEAD's. `static_assert(frag_a_words(BC) >= 2 * BC + 1)` is what
holds the alias to the buffer.

---

# Part III — the paged address

## <a id="atom-row-base"></a>24. §4.5's paged address, re-keyed to the ATOM

A page is ONE `kMmaKQuantum`-leaf ATOM, not one owner (`page_states := BC` is DEAD — §1b's
ruling is that `BC` is a re-derivable COMPUTE tile while a page is a MEMORY-RESIDENCY granule,
and the thunder run makes `BC` per launch, so a `BC`-keyed page would have to be rebuilt
whenever the planner moved it). An owner therefore spans `QPO = BC / kMmaKQuantum`
INDEPENDENTLY PAGED atoms whose slots are not contiguous in general, and a CTA resolves `QPO`
bases at startup instead of one. `QPO <= 4` and both `BC` and the quantum are compile-time, so
every call sits inside a fully unrolled loop with a CONSTANT `q` — there is no runtime-indexed
register array here and nothing to spill.

`atom_row_base` returns the first STATE ROW of atom `q` of `owner`, or `-1` when the atom has
no committed slot. Under a plan the arena produced, an absent atom is one with no write
candidate at all: admission is TOTAL (the capped mode is deleted), so `page_table[atom] >= 0`
and "the routing writes this atom" are the same predicate, and an absent atom's dense state is
identically zero.

ONE EXPRESSION SERVES BOTH LAYOUTS. Unpaged is the identity slot map — atom `q` of owner `o` is
row `bh * N + o * BC + q * kMmaKQuantum` — so the paged and unpaged paths are the same
addressing strategy at different slot maps rather than two strategies to keep in sync.
`atoms_per_bh = N / kMmaKQuantum` is DERIVED rather than carried: the host sets
`p.N = owners_total * BC` under paging, so no ABI field is needed for it and there is no second
place for the table's stride to be wrong.

---

# Part IV — the tile walk

## <a id="cta-entry"></a>25. CTA entry — the work item, the owner rectangle, the candidate words

`blockIdx.x` indexes the flat `(owner, segment)` work list. `nullptr` in `work_seg` /
`work_nseg` / `work_slot` is `P = 1`, and it is `P = 1` WITHOUT READING MEMORY: `seg = 0`,
`nseg = 1`, `last_seg` true, and the tile range is the whole axis. That is what keeps `P = 1`
BIT-IDENTICAL rather than merely equivalent — the primary benchmarked configuration must not
acquire a load it did not have.

The owner decomposition is §3's rectangle: `slice = owner % n_slices`,
`rest = owner / n_slices`, and per level `base[l]` is the owner's fixed digit
(`l < j`), `slice * slice_digits` (`l == j`), or `0` (`l > j`).

The candidate walk is §4's word-AND chain: the union words ANDed with each FIXED
level's tile bitmap row, then ANDed with §6.4's per-segment tile-range mask
(`segment_word_mask`, one named definition in `segment_mask.cuh` because the host-side segment
partition at `P > 1` is pinned against exactly this arithmetic).

**Pass 1 walks WRITE tiles only** (`cand_read` is forced to zero). A read-only tile cannot move
the state — under `LeafMassDecay` its survival is `keep == 1` EXACTLY, §2's write-conditioned
quiescence, the same fact the fold's skip rests on — so it contributes nothing to the summary;
and a read emitted here would be counted a SECOND time by pass 3, which re-runs the whole
segment from the accumulated boundary.

## <a id="digit-hoist"></a>26. The leaf-digit hoist, and `log_keep`

§3's leaf-digit decomposition, HOISTED TO THE OWNER. `leaf_digit` is
`base + (i / stride) % modulus` with `stride` and `modulus` RUNTIME values, so it is a runtime
integer division AND modulo — and NVIDIA hardware has no integer divide, so each is a
~20-instruction sequence. It was being evaluated once per (tile, token, leaf, level) in both
panel builds, and it depends on NONE of the tile, the token or the side: only on
`(level, owner, leaf)`. Computed once per CTA it costs `D * BC` evaluations for the owner's
whole run instead of `n_tiles * BT * BC * (D - j) * 2`. `D * BC <= 4 * 64 = 256 =
kConsumerThreads`, so the hoist is at most one iteration per thread.

### §12 rule 2 — per-leaf log-retention, ONCE PER OWNER, IN LOG SPACE

`rate[s] = prod_l dial_l[d_l(s)]` (spec §2) accumulated in fp32, then
`log_keep = log1pf(-rate)`. Rule 2 is two prohibitions and both are STRUCTURAL here rather
than checked:

- the retention is never formed as a rounded linear-domain `1 - rate` and then logged —
  `log1pf` IS the composition, and it is what recovers the bits a near-unit `rate` cancels
  away;
- there is NO EPSILON FLOOR. `delta < 1` is strict in the fp32 dial this kernel reads
  (re-checked at the cast boundary by `schedule_v3.leaf_rate_dials`, because an fp64 dial
  within ~3e-8 of 1 is strictly `< 1` in fp64 and still rounds to exactly `1.0f`), so
  `rate < 1` and `keep > 0` structurally, and any later underflow of `expf` is ordinary
  final-power underflow — the correct limit, never fed back as a base.

It re-evaluates `leaf_digit` rather than reading the `dig` table computed two lines above, and
that is DELIBERATE: `dig` is written by a DIFFERENT thread than the one that would read it here
(`idx = l*BC + i` vs `i`), so using it would need a barrier. `D * BC` divisions once per owner
is exactly what the digit hoist itself costs; a barrier per CTA to avoid it would be the worse
trade.

## <a id="entry-state"></a>27. Where this segment's entry state comes from (§6.4)

```
pass 1                : ZERO. The summary is (A, B) with M_0 := 0, which is B's definition.
pass 3, SPLIT owner   : workspace slot owner_slot0 + seg, ALWAYS -- the scan's entry stash at
                        seg = 0, the accumulated boundary otherwise.
pass 3, UNSPLIT owner : the owner's own state, paged or dense, as at P = 1.
```

**A split owner's pass-3 CTA never reads `p.state`, and that is a CORRECTNESS requirement, not
a saving.** The owner's LAST segment WRITES those rows at exit, from a different CTA of the
SAME launch, and CUDA orders nothing between CTAs of one launch — so a segment-0 read of them
would be a race whose only protection is the work list happening to be ordered. The failure is
REAL rather than hypothetical: a permutation of the pass-3 work list that the contract PERMITS
(every owner's last segment before its first) produces `y` with relative error 1.0
(measured). Pass 2 stashes
`M_0` into `owner_slot0` in a DIFFERENT LAUNCH, hence ordered, so every pass-3 read is disjoint
from every pass-3 write BY CONSTRUCTION and the work list's order is free again. That is what
costs one extra slot per split owner (`P`, not `P - 1`). See
[`consumer.md`](consumer.md#workspace).

The dense read is ATOM-WISE, because the `kQPO` atoms of one owner are separately paged. The
outer loop is `#pragma unroll` over a compile-time `kQPO <= 4`, so each base is a SCALAR
resolved once — not an array a runtime `i >> 4` would index into local memory. The inner walk
covers `kMmaKQuantum * cols` elements, so the access pattern per atom is the same contiguous
run the owner-keyed load had. An atom with no committed slot reads as ZEROS and is never stored
back, which is exactly the dense path's arithmetic on a never-written atom — hence paging is
BITWISE INVISIBLE rather than approximately equivalent.

`GLOBAL` is the NORMALIZATION arm, and it is a template axis rather than a runtime
`p.den != nullptr` test because it changes the STATE'S SHAPE. Under `global` the state carries
a `+1` mass column (spec §4.2's `y = num / den` with `g_read := 1`); under `raw` there is no
mass column, no denominator stream and an explicit `g_read`. A runtime branch would make `cols`
a runtime value in every state-traffic address, and would keep the mass reduction's registers
alive on an arm that has no mass.

## <a id="candidate-walk"></a>28. The staging ring's candidate walk and its two lambdas

`next_cand(t)` is the tile loop's `continue` expressed as a FORWARD SCAN, and that inversion is
what makes a one-tile run-ahead expressible at all: a feeder has to know WHICH tile comes next,
and "the next candidate at or after `t`" is exactly that question. It visits the same tiles in
the same order the `continue` does, so at `kRing == 1` the loop is the plain one verbatim.

The candidate words are already masked to `[tile_lo, tile_hi)`, so restricting the loop bounds
is a SPEED change and not a semantic one — every tile it drops has no bit set and would hit the
`continue`. At `P = 1` the bounds are `0` and `p.n_tiles`, i.e. the whole axis.

### <a id="loop-induction"></a>HAZARD `loop-induction` — the loop keeps `++tile` and its `continue`

**And that is a MEASURED constraint.** Folding the candidate scan into the loop's INCREMENT
(`for (tile = next_cand(tile_lo); ...; tile = next_cand(tile + 1))`) is the same walk and it
costs, on the 7-entry probe against `2501eb5`, **12/28 -> 94/340 spill bytes on the `BC = 16`
decay arm and 36/72 -> 98/300 on the `BC = 64` decay raw arm** — on arms that take
`kRing == 1` and touch no run-ahead at all. So the scan stays where the run-ahead needs it and
NOWHERE else: inside the `if constexpr (kRing == 2)` blocks, which are not emitted at depth 1.

### <a id="no-captured-arrays"></a>HAZARD `no-captured-arrays` — the lambdas carry no captured mutable state

`build_wt_rows` is REACHED FROM TWO SCOPES — the staging itself at the plane, and the window
loop when the window is narrower — so everything it needs about the tile it is building arrives
as an ARGUMENT. It carries NO captured mutable state, and that is a MEASURED constraint rather
than a style choice.

Carrying that state (the `[D]` write-amplitude pointer array, or a scalar row base) at KERNEL
scope, captured by reference by two lambdas, puts it in LOCAL memory — §2.4's rejected
base-pointer hoist arriving through a closure instead of through a runtime `D`. Measured on
the 7-entry probe against the `2501eb5` baseline: an 8-byte stack frame and 61 -> 72 registers
on the freeze arm, and 12/28 -> 52/128 spill bytes on the `BC = 16` decay arm, with the DECAY
arms regressing even though they take `kRing == 1`. Parameters keep the array a plain local
with statically-indexed, fully unrolled accesses.

**The amplitude bases arrive as a CALLABLE, not as an array**, and that is the same measured
constraint stated once more. `const float* [D]` cannot cross a lambda boundary — by reference
it must be addressable, by value it is a copy with the same problem — and either form lands it
in LOCAL memory, which is §2.4's rejected hoist. An ACCESSOR crosses as CODE: the caller's
`amp_w[l]` is statically indexed IN THE CALLER'S FRAME, under the caller's `#pragma unroll`, so
it stays a register there and nothing about the function's shape can move it. Measured:
rebuilding the array inside instead costs the `BC = 16` decay arm 12/28 -> 40/80 spill bytes.

### `stage_tile(tile, slot, is_read, is_write, gN)`

Everything the tile's COMPUTE consumes but does not produce: the clock prefix, the
row scales, the read panel, the write panel at the plane, and the value tile. At `kRing == 1`
it is called at the top of the tile body and `slot` is identically 0, so the emitted code is
the un-ringed one with the braces moved. At `kRing == 2` it is called ONE VISIT EARLY, from
inside the previous tile's compute ([below](#row-scales)). `tile < 0` is a GATHERED visit
([below](#token-gather)): the staged tokens are `t_idx[0..gN)` and `slot` is 0.

The schedule bits arrive as ARGUMENTS rather than being re-derived from the candidate words: at
depth 1 the caller has just computed them, and a second SMEM read costs the
`BC = 64, BT = 16, DV = 32` global arm 2 registers — which is the 64-register allocation
boundary and therefore its 4th CTA.

### 64-bit address arithmetic, paid once per tile instead of per element

The amplitude streams are `[BH, n_tiles, BT, pitch]`, so the row base needs 64 bits, but the
offset WITHIN a tile does not — it is `t * pitch + digit`, bounded by `BT * pitch`. Folding the
64-bit part into a base pointer once per (tile, level) leaves the inner loop in 32-bit index
arithmetic. The bases are TYPED `AmpRT` / `AmpWT` ([above](#amp-class-types)): on a split arm
the `l < j` entries carry a wrongly-scaled address and are never dereferenced — the fixed
levels' consumers index the params directly instead.

**It stands ONLY because `D` is compile-time, and that dependence is measured (§2.4).** With a
RUNTIME `p.D` the level loop cannot unroll, so these arrays are dynamically indexed and `ptxas`
puts them in LOCAL memory — stack frame 16 -> 80 bytes, exactly the two 4-pointer arrays, and
the kernel is SLOWER for the hoist. `D` is compile-time here,
every level loop is `#pragma unroll`ed to a compile-time trip count, and the arrays are
therefore statically indexed and live in registers. This is also what `D` being a template axis
buys beyond the dispatch: it is a topology fact with four legal values (spec §1), not a
threshold, so branching on it is the standing "branch on structure, not thresholds" rule rather
than an exception to it.

## <a id="token-gather"></a>28b. The token-gather walk (P14, incremental form)

The kernel has TWO walks over one visit body, selected by a LAUNCH-UNIFORM structural fact:
`tg = p.token_gather && !summary`. `p.token_gather` is derived host-side from the activation
registry's `exact_zeros` column — set iff BOTH routing sides can produce exact zeros. When
either side is structurally full-support the membership word below is all-ones and the gather
would be the identity, so the aligned walk IS the path there and pays nothing; the summary
pass keeps the aligned walk because its per-leaf inclusion flags (`sum_slot`) are tile-framed.
This is a runtime fork on structure, not a template axis: zero new instantiations.

**The gathered walk.** The aligned candidate stream (`next_cand`, unchanged) feeds a
MEMBERSHIP scan: per candidate tile, `fr`/`fw` = the AND over the owner's `l < j` fixed-level
`read/write_tok_bits` rows, and `member = (fr | fw) & slice_union_tok[bh, slice, tile]` — the
token-granular slice union (schedule ABI v3), sides OR'd because the walk visits the
read|write union. Ballot-time exclusion of slice-dead tokens is what keeps groups full.
Every thread computes the scan REDUNDANTLY (all loads are CTA-uniform addresses), so group
boundaries are uniform with no broadcast and no per-tile barrier; only warp 0 stores `t_idx`
(lane = bit, slot = `fill + popc(below)`). A word with more live tokens than the group has
room for is split at the `(room+1)`-th set bit (`__fns`); the loop is bounded by
`ceil(32/BT) + 1` rounds per word, never a per-token scan. Each thread assembles from the
SMEM cross-visit carry `g_scan` (fill, split-word remainder, group flags, the hoisted
fixed-row indices) in registers; when `fill == BT` (or the stream ends ragged) the scheduler
yields `tile = -1` and the walk's ONE inline visit body runs it: the control block resets,
the group-entry `__syncthreads()` publishes warp 0's `t_idx` writes AND orders the carry
loads before tid 0 stores the updated carry, `stage_tile(-1, 0, ...)` stages the group, and
the SAME body an aligned tile runs follows.

**Stability is the correctness requirement, not a nicety.** Source tiles are visited
ascending and each word is consumed LSB-first, so `t_idx` is GLOBALLY ascending. The intra
Gram's inclusive causal mask is order-relative; an unstable gather produces a plausible wrong
answer that only the oracle gate catches (`tests/tier2_geometry/test_token_gather.py`, whose
teeth are mutant-proven against both a dropped-token and a permuted-order ballot).

**Why skipped tokens are exact identities** (the token-axis twin theorem): a token outside
the membership word has exact-zero coefficients on every leaf of this owner on both sides,
and under `DECAY` an exactly-zero clock product (`build_clock_stream` enforces
`stored != 0 => live != 0`, and membership exclusion implies a write-dead level for every
leaf), so it moves neither the state nor the output of THIS owner. Groups therefore preserve
the recurrence exactly; sparse results are tolerance-graded against the oracle (grouping
re-associates the intra term), while full-support routing is BIT-identical to the aligned
walk (gate-asserted).

**The ragged flush** (`fill < BT` at `tile_hi`): slots `[gN, BT)` stage EXACT-ZERO
coefficient rows (the gathered accessors return `0.0f` and never read `t_idx` for a padded
slot), zero row scales, and — under `DECAY` — a FINITE zero clock prefix (`clk` rows
zero-filled so both panel factors stay finite; `keep^0 == 1`). Padded slots are then inert
through every GEMM exactly the way the aligned walk's `token >= T` tail is, and the emission's
`out == 0` early-out never touches their addresses.

**Ring interaction.** A gathered visit does not ride the staging ring: it stages at the group
top into slot 0 on every arm (the next group's membership is not known until the scan reaches
it). On `kRing == 2` arms the aligned depth-2 write-panel build and the run-ahead are
`!ga`-guarded; the gathered write panel is built inside `stage_tile` behind the same pre-GEMM
barrier. Ringing the gathered walk (scan-ahead group assembly) is a recorded follow-on, not a
correctness need.

**Token addressing.** All four GEMMs are token-oblivious; token indices enter only through
the staging accessors and the emission/mass epilogue, as `t_idx[t]` against the `bh` stream
base (the aligned arm keeps its per-tile base and 32-bit inner offsets — its accessor bodies
are textually unchanged). The gathered split-K emission scatters through `t_idx`; the P14
scattered-emission microbench priced exactly this pattern at a 1.0009x DRAM-byte ratio
(banded), which is what licensed building the gather at all
(`workflows/perf/p14_microbench.md`).

**Cost of carrying both walks:** `t_idx` + the assembler's cross-visit carry `g_scan` =
`BT + 6 + D` ints of SMEM on every arm (smem_layout.md#smem-ledger), plus the register class
of the max-over-branches allocation — ratified per instantiation by the spill manifest like
every other body change. The carry lives in SMEM, not registers, and that is a MEASURED
placement, not a preference: register-resident carry (plus a register `srow[]`) cost the
pressure-critical dense arms up to +22 % wall clock (topo-D4-BC16); the SMEM form's residual
is ~+5 % on that one arm and inside the 3-sigma band on the other seven gated arms. The
handoff needs no extra barrier: every thread loads the carry after the previous visit's end
barrier, and tid 0 stores the update behind the group-entry barrier.

## <a id="decay-clock"></a>30. `DECAY` steps (a) and (c) — the clock prefix, `P_ref`, `s0`, and the entry scale

Guarded by `is_write` exactly as the `intra_panel` lowering is: a read-only tile has `c = 0` on
every leaf, so every factor this block would compute is the identity and the plain path is
EXACT, not approximate. `is_write` is a block-uniform SCHEDULE bit, not a density test — §2's
write-conditioned quiescence.

The lowering is §7.4.1's decay-aware panels and it does NOT add a fifth GEMM or change the shape
of any of the four. `keep[t,s] = (1-rate[s])^c[t,s]`, so with `P[t,s] = sum_{u <= t} c[u,s]` the
survival from `j` to `t` is `keep^{P[t] - P[j]}`, and THAT EXPONENT SPLITS ACROSS THE TWO
OPERANDS OF A GRAM because it is a difference of a per-`t` term and a per-`j` term. With §12
rule 1's per-leaf reference `P_ref = P[BT-1]/2`:

```
fr[t,s] = exp((P[t,s] - P_ref[s]) * log_keep[s])      folded into Rf
fw[j,s] = exp((P_ref[s] - P[j,s]) * log_keep[s])      folded into WT
s0[s]   = exp(P_ref[s] * log_keep[s])                 scales the STATE
fr[t,s] * fw[j,s] = survival(j -> t)                  (the identity)
```

so every term of the tile keeps its existing shape and the ONLY new arithmetic is the clock
prefix and two exponentials per (token, leaf):

```
readout    Sum_s (R[t,s] fr[t,s]) * (s0[s] M_entry[s])      Rf gains fr
intra Gram G = R~ . W~^T, causal                            unchanged
intra y    y += G . Vw                                      unchanged
fold       M_exit = s0 * (s0 M_entry + Sum_j W~[j,s] v[j])  WT gains fw
```

**One write panel serves both the Gram and the fold**, which is what rule 1's half-mass shift
buys and why decay does not need a second `[BT][BC]` buffer the SMEM budget cannot afford. The
mass column takes the identical treatment with `v := 1`, because it is the same recurrence on a
`v` of ones. The factorization, the three §12 rules, the measured cliff at the clamp and the
plan-time admission bound live in [`intra_panel.md`](intra_panel.md); the survival factors
themselves are TAKEN from `intra_panel.cuh` rather than reimplemented, because it is the one
clamp point and the one exponent sign convention, and a second copy of `expf(fminf(x, 88.0f))`
is a second thing that can drift.

### §12 rule 1 — `P_ref = P[BT-1]/2` IS the rule rather than an approximation of it

The largest exponent either panel forms is `max_t |P[t] - P_ref| * |log_keep|`; over a
non-decreasing `P` that is minimized at the midpoint of the range, and a per-`(t,j)` reference
cannot exist because a Gram contracts `i`, so the reference must be constant along the
contracted axis. It is the OPTIMAL choice of a free parameter. The half-mass shift also makes
the tile's ENTRY and EXIT scales the same object (`P[BT-1] - P_ref == P_ref`), which is what
lets one write panel serve both consumers.

`s0[i] = expf(pr * log_keep[i])` needs no clamp: `pr >= 0` and `log_keep <= 0`, so `s0 <= 1`.
Underflow to zero is rule 2's SANCTIONED underflow (a leaf whose tile-total survival is below
fp32 has genuinely lost its entry state), not a floor.

### The clock product runs over ALL levels

Including the FIXED ones (`l < j`), where the read and write panels' products start at `j`.
That is not an inconsistency: a fixed level's read/write factor is constant along the owner's
leafs and hoists into the §3 step-3 ROW SCALE, but the clock is an EXPONENT and a row scale
cannot carry it.

### <a id="clock-prefix-fp32"></a>HAZARD `clock-prefix-fp32` — the prefix accumulates in fp32

`acc` is fp32 and must STAY fp32. This is a REQUIREMENT, not a default (spec §7.4's V3-specific
rule): the prefix multiplies `log_keep` and is then exponentiated, so a lost low bit here is an
**exponent** error, not a mantissa one. Narrowing this accumulator is a silent accuracy
regression that no compile-time check catches.

One thread per leaf, SERIAL over the tile's `BT` tokens, because the prefix IS a serial
dependence in `t` and `BT <= 32`; a parallel scan would cost more barriers than it saves steps.
The store `clk[t * CKP + i]` is `i`-contiguous across consecutive threads, so it is
conflict-free.

### <a id="scatter-fill"></a>The scatter-fill identity — a THEOREM rather than a convention

A leaf with no live support on EITHER side has `c == 0` at every token of this tile, because
`build_clock_stream` enforces `stored != 0 => live != 0` per (level, digit) — so its clock
product is zero at every token, its prefix IS zero, and `keep^0 == 1` makes every factor this
block computes for it exactly `1`. **That is what makes the entry/exit scaling exact leaf by
leaf: the leaves the fold never touches are exactly the leaves whose `s0` is exactly 1**, so
the walk needs no membership test to leave them alone.

### Step (c) — the entry state enters the reference frame

The carry term is `R~[t,i] * (s0[i] * M_entry[i])`, so `s0` multiplies the STATE, not the read
panel. The mass column (`c == DV`) rides the same sweep because it is the same recurrence with
`v := 1`.

Row-per-warp rather than a flat `idx / cols` walk: under `global` `cols = DV + 1 = 65` is not a
power of two, so the flat form would pay an integer division per element — ~17 of them per
thread per tile, on hardware with no integer divide (§2.4 measures the sequence at 20
instructions a time).

## <a id="row-scales"></a>31. §3 step 3's row scales, computed above the panel builds

`sr[t]` and `sw[t]` are the tile's per-token row scales: the read/write gain times the FIXED
levels' amplitudes at the owner's own digit. They are constant along the leaf axis, which is why
they are held out of the coefficient blocks at all (`readout_mma.cuh`'s argument), and they
depend on NOTHING the tile computes: not the clock prefix, not either panel. Only on
`(tile, token)` and the owner's `base[]`. The fixed levels they read are ALWAYS fp32 (Option
B, [`consumer.md#amp-dtype`](consumer.md#amp-dtype)); on a `kAmpSplit` arm these loops index
that fp32 stream directly off the params, because the typed tile bases carry a wrongly-scaled
address at `l < j` ([above](#amp-class-types)).

**What their placement buys is an OVERLAP; the rendezvous count is the same either way.**
Computing them WITH the panels makes the pre-GEMM section `[panels + row scales] [Vw]`, whose
one barrier exists for one reason: to publish `sw[t]` to the `Vw` writers, which are different
threads (`sw`'s loop is `t = tid`, `Vw`'s is `t = tid / DV`). Computing and publishing the row
scales FIRST takes `Vw` off the panels and the section becomes `[row scales] [panels + Vw]` —
the same single rendezvous, but now the `Vw` staging's `BT * DV` GLOBAL loads issue ALONGSIDE
the panel builds' amplitude loads instead of behind them.

**The one arm that keeps the middle barrier is `DECAY`**, and it is a LIFETIME constraint rather
than a publication one: `clk` ALIASES `Vw`'s bytes, both panel builds READ `clk`, and the `Vw`
writers would clobber it. That barrier stands AT ITS REASON ([below](#build-wt)) rather than as an incidental
side effect of `sw`'s publication, which is what keeps it from being lost to an edit that only
looks at publication. On `DECAY = false` the rendezvous after the row scales
publishes `sw` to the `Vw` writers, which is what lets the staging run inside the SAME phase as
the two panel builds; under `DECAY` it is ABSENT, because the `clk -> Vw` lifetime barrier
already stands between the panels and the staging and publishes `sw` on its way past, so a
second rendezvous would order nothing.

## <a id="read-panel"></a>32. The read panel `Rf` (§3 steps 2-3)

`R[t, i] = scale_read[t] * prod_{l >= j} amp_read[l][t, digit_l(i)]`, with the row scale held
OUT of the block: it is constant along both `k` and `n`, so it applies AFTER the accumulation
(`readout_mma.cuh`'s own argument) and keeping it out of a bf16 operand keeps its precision.
Fixed levels (`l < j`) hoist into the row scale. The spanned gather widens through
`amp_to_float` ([above](#amp-class-types)) — `AmpRT` storage, fp32 product.

The loop walks FRAGMENT SLOTS, not `(token, leaf)` pairs: `frag_a_slot` is the FORWARD map, so
the store is `Rf[s]` with `s` contiguous in `tid` — conflict-free, one 32-bit store, no scatter.
Walking `(t, i)` and inverting would have been the same arithmetic backwards and a scattered
store. On bf16 one slot is an ADJACENT PAIR on `k`, which is why the map packs two leaves per
word and the loop runs half as many iterations.

The `m16` row blocks are an OUTER, COMPILE-TIME loop, so at `NM == 1` this is the single-block
loop verbatim — no division by a runtime word count, and no per-slot block arithmetic on the
cell the baseline was measured at.

**DECAY step (d):** `fr[t,i] = keep_i^{+(P[t,i] - P_ref[i])}`, folded into the READ coefficient.
§12 RULE 4 (zero-route annihilation) is satisfied BRANCHLESSLY and structurally by this
ordering: `a` is a product of amplitudes, so a structurally dead route is an exact `0.0f`; the
factor it multiplies is FINITE by rule 3's clamp; and `0 * finite == 0` exactly. No test on `a`,
and no `-inf` anywhere — the amplitude is never logged, which is the whole reason this lowering
takes the direct staged-factor form rather than the log-space one.

The clamp is on the READ side too, and it is LOAD-BEARING rather than a defensive copy of the
write side's: both panels' exponents range over `[-E, +E]` with
`E = max_i P_ref[i] * |log_keep[i]|`, so `fr` amplifies to `e^{+E}` at `P[t,i] = 0` (early `t`)
by exactly the amount `fw` amplifies at late `j`. See
[`intra_panel.md`](intra_panel.md#rule-3).

## <a id="build-wt"></a>33. `build_wt_rows` — the write panel, as an ATOM-RANGE function

`build_wt_rows(q_lo, q_hi, ...)` fills the write panel's rows `[q_lo, q_hi)`. At
`C_WT_WINDOW = QPO` it has ONE call site with `q_lo = 0`, so the range folds away and the
emitted loop is the plane pass verbatim; windowed, the same function is the per-atom fill.
Extracting it is what makes "the plane" a VALUE OF THE DIAL rather than a second code path (I4).
`WT`'s rows are WINDOW-LOCAL: the buffer holds `kWinRows` of them and `q_lo` is the window's
first leaf position, so the stored row is `q - q_lo`; at the plane `q_lo == 0` and this is `q`
exactly. The accessor's walk is SPANNED-ONLY (`l < j` continues), so every caller types its
amplitude bases `AmpWT` and the gather widens through `amp_to_float`
([above](#amp-class-types)).

**DECAY step (d):** `fw[t,i] = keep_i^{-(P[t,i] - P_ref[i])}`, folded into the WRITE
coefficient. THE SAME `WT` THEN SERVES BOTH GEMM 2 (the causal Gram, where it meets `Rf`'s `fr`
and the two reconstruct `survival(j -> t)` exactly) AND GEMM 4 (the fold, where it meets the
state's `s0`). One panel, two consumers — rule 1's half-mass shift is what makes that possible
and it is why decay needs no second `[BT][BC]` buffer.

This is the DIVISION by cumulative survival that §12 rule 3 exists for, and because the
production fold is bf16 ([above](#mma-precision)) rule 3's second site — the `clamp_bf16_finite` at the narrowing
point — is LIVE on the WRITE side of the shipped arm, which is rule 3's own motivating case.
Both sites are required and both are live: the exponent clamp bounds the value
at `e^88 = 1.65e38` (finite in fp32 AND in bf16, which shares fp32's 8-bit exponent), and the
format guard at the MMA catches anything the accumulated product does downstream.

### <a id="wt-row-t-fastest"></a>HAZARD `wt-row-t-fastest` — the loop's fastest axis is `t`, not `i`

Because `WT` is leaf-major: with `i` fastest the consecutive lanes would store at stride `BTP`,
which is a 4-way STORE conflict (`gcd(BTP, 32) = 4`). Walking `t` fastest makes the store
contiguous. The amplitude reads are the same set either way.

### <a id="summary-prequant"></a>HAZARD `summary-prequant` — §6.4 pass 1's emptiness answer is taken HERE, pre-quantization

The `P > 1` summary must include every leaf that can LEAVE `(A, B) = (1, 0)`, and the test is
taken IN THE PANEL BUILD, where the coefficient exists as the un-narrowed fp32 `a` and BEFORE
it is stored — never as a re-scan of the stored `WT`.

The invariant being protected is stated in the NEGATIVE, because that is the direction that is
not free: OVER-inclusion costs one exactly-`(1, 0)` entry in the emitted list and the scan's
fold returns the incoming value unchanged, while UNDER-inclusion **silently drops state**. A
bf16 `WT` can round a small-but-nonzero fp32 coefficient to exactly zero, so a
post-quantization scan is an under-inclusion hazard BY CONSTRUCTION.

### <a id="ballot-lane-run"></a>HAZARD `ballot-lane-run` — the reduction's `t`-group is a contiguous lane run

The loop's index map is `q = idx / BT`, `t = idx % BT`, walked at `kConsumerThreads` stride, so
the `BT` threads sharing a row `q` are lanes `[g0, g0 + BT)` of ONE warp (`BT` is 16 or 32, both
dividing 32 or equal to it). `kRowLanes` is that group's mask. It is a lane ARITHMETIC fact of
the index map, not a density or a threshold, and
`static_assert(BT == 16 || BT == 32)` is what makes a future `BT` fail the build rather than
silently mis-reduce.

`__any_sync` names ONLY this row's lane group, which is converged: the group's `BT` lanes
share one row `q` and take the loop's every iteration together. One lane per row publishes, so
`sum_slot` takes no intra-tile race — a row is one leaf and no two rows are the same leaf.

### <a id="wt-build-site"></a>HAZARD `wt-build-site` — where the plane's build may stand

The plane's call site is INSIDE `stage_tile` **only at depth 1**, and the reason is a RACE
rather than a preference. `WT` is single-buffered at every ring depth, and its readers — the
Gram and the fold — retire ASYNCHRONOUSLY: the fold is a claim pool, so a warp that exhausted
the pool can reach the staging point while another warp is still executing its last claimed
item. A run-ahead build there would overwrite rows that warp is reading.

At depth 2 the build therefore sits in the tile's OWN body, above the publication rendezvous,
which costs NO barrier at all: the PREVIOUS tile's end-of-tile barrier stands between the last
fold read of `WT` and this write, and the publication rendezvous immediately below stands
between this write and the Gram's read.

At depth 1 it must be inside the staging: `DECAY` pins the plane and its build reads `clk`,
which the lifetime barrier below kills. Hoisting it out of the window loop when the window IS
the plane is therefore not cosmetic — it is what keeps the `DECAY` arm's build ABOVE that
barrier, where `clk` is still live. `DECAY` pins `kWtPlane`, so the two facts are one fact.

## <a id="clk-dead"></a>34. HAZARD `clk-dead` — the `clk -> Vw` lifetime barrier, and it is DECAY's alone

`clk` (the tile's per-leaf inclusive clock prefix) ALIASES `Vw`'s bytes ([`smem_layout.md#vw-clk-union`](smem_layout.md#vw-clk-union)). Both panel builds
READ `clk` (the `panel_read_factor` / `panel_write_factor` calls), and the `Vw` staging WRITES
those same bytes, so the two lifetimes need a rendezvous between them. **Removing or moving this
`__syncthreads()` past a panel build silently corrupts the survival factors of the tile.**

Under `DECAY = false` there is no `clk` tenant, `Vw` is disjoint from `WT` and `Rf`, and the
staging joins the panel builds' phase: one fewer rendezvous and — the larger effect — the
staging's `BT * DV` fp32 GLOBAL loads issue alongside the panel builds' amplitude loads instead
of behind them.

`is_write` is CTA-uniform (the same per-tile schedule bit the surrounding blocks are guarded
on), and a read-only tile writes neither `clk` nor `Vw`.

### The `Vw` staging itself

The write side folds `scale_write[t]` into the VALUE tile instead of into a row scale, because
there `t` is the CONTRACTION index and the scale cannot hoist out of the sum.
`Vw[t, ch] = scale_write[t] * v[t0 + t, ch]` then serves BOTH the intra term and the boundary
fold — one staged object, two GEMMs. Out-of-range tokens (`t0 + t >= T`) contribute `0.0f`.
The `v` load is typed `AmpWT` and widens through `amp_to_float` ([above](#amp-class-types));
the scale-fold multiply is fp32, and the store narrows once at `stage_put` on the narrow arm.

## <a id="run-ahead"></a>35. The run-ahead, and why it sits exactly where it does

At `kRing == 2` the PROLOGUE stages the first candidate; thereafter each visit stages the NEXT
candidate from inside its own compute, at the one program point where this tile has no compute
left to protect and its warps would otherwise be sitting on the phase-C acquire. `visit` — not
`tile` — indexes the ring, because the ring's slots alternate over CANDIDATE VISITS while the
tile index skips.

Four facts license the placement, and all four are properties of the PROGRAM POINT rather than
of the schedule:

- **SLOT-FREE.** The slot it writes is `slot ^ 1`, which this CTA last READ in the previous
  visit's compute — and the previous visit's end-of-tile `__syncthreads()` stands between that
  read and this write.
- **SLOT-READY.** Nothing consumes `slot ^ 1` before the next visit's publication rendezvous,
  which this visit's end-of-tile barrier precedes.
- **`WT` is ringed by the SCHEDULE, not by storage.** The staging rebuilds the write panel at
  the plane, and its last reader — the Gram and the fold — retired with the window loop just
  above. So one `WT` buffer is enough at any ring depth, which is what keeps the slot at 4,480 B
  instead of 7,552.
- **No warp is spinning.** Every warp retires BOTH phase counters above this line (the
  `kCtlGramDone` retirement rides the last window), so the `__syncthreads()` inside the staging
  — `sw`'s publication to the `Vw` writers — is reached by the whole CTA with no warp blocked in
  `await_count`. That is the always-runnable clause of the deadlock argument
([`pipeline.md#ctl-reset`](pipeline.md#ctl-reset)), discharged
  for the one barrier that stands in the interior of a tile.

`nxt` is CTA-uniform (`next_cand` reads only SMEM candidate words), so the guard cannot diverge
across the barrier inside.

---

# Part V — the phase-C gate

## <a id="gram-gate"></a>38. `kGramWarps`, `kNarrowGate`, and the phase-C gate

`NM * (BT / 8)` is the number of `[16 x 8]` Gram warp tiles; a warp holds one iff its index is
below that, in BOTH forks — the `NM == 1` fork guards on `warp < BT / 8` (and `NM == 1` there),
and the `NM >= 2` fork breaks at `gt = warp + q * kWarps >= NM * (BT / 8)`, whose smallest `gt`
for warp `w` is `w` itself. So the storing set is EXACTLY `warp < kGramWarps`, not merely
contained in it. At the flagship (`BT = 16`) it is **2 of 8**.

`kGramWarps` is a TERNARY rather than `min()`: it must be a core-language constant expression
with no library dependence, and `<algorithm>`'s `min` is not `constexpr`-usable in device code
without pulling in `cuda/std`.

### Whether the gate is narrowed at all is a MEASURED scope, not a preference

The narrowing is bit-identical on every arm and free on the `bf16` ones, but on the `tf32x3` arms
it is not free: adding either guard at the two phase-C sites re-schedules `ptxas` on
instantiations that are ALREADY register-starved (128 and 255 registers, `Lb0` raw norm at
`BT = 32`), and the 1,216-entry ratchet measured **12 of them acquiring in-loop `LDL`/`STL`** —
`<32,64,32,1, tf32x3, raw>` going from ZERO spill to 32 st / 68 ld with 12 of the accesses inside
the tile loop. Attribution: `kGramWarps` alone is byte-identical SASS, 1a alone and 1b alone each
cause it independently and non-additively, so it is an ALLOCATOR outcome and there is no safe
half. Spill inside the tile loop is a memory round-trip per iteration; the ratchet refuses it and
re-ratifying would launder a refusal into a baseline.

**So the scope is the PRECISION axis**, which is a FORMAT axis the dispatch already forks on
(`consumer.cu` sets both GEMM sites together, and `stage_is_narrow` / `StageT` / `GmT` / `GMP`
all fork on it). It is NOT a size test and NOT a threshold: no `BT`, `BC`, `DV` or occupancy
number appears in it. The `tf32x3` arm takes the 8-wide gate, which costs it nothing it needs —
that arm is the EXACTNESS REFERENCE (`RTOL = 2e-5`), gated on correctness rather
than shipped for throughput, and the shipped default is `bf16`, so every shipping arm — the
parity flagship included — is inside the narrowed set.

### The retirement, both arms

**ON THE NARROW ARM:** retired by the WRITERS ONLY, and the wait's target is the same
`kGramWarps`. The phase fact the counter states is "no warp is still writing `Gm`", and the
writing set is EXACTLY `mixed && warp < kGramWarps` — equality, not containment, in both `NM`
forks. Counting `kGramWarps` retirements therefore counts every writer and nothing else, and the
deadlock argument stays two lines rather than becoming a case analysis.

**ON THE WIDE ARM:** every warp retires, target `kWarps`. The warps that hold no Gram tile retire
an empty phase, which is one atomic. Both forms satisfy the same release invariant — the retiring
set covers the storing set and the target equals the retiring count — and
`tests/unit/test_phase_c_gate_structure.py` checks BOTH ARMS of the fork separately rather than
whichever one the tree happens to compile.

**On the narrow arm `!mixed` skips the counter entirely**, at the retirement and at the wait. On
a `!mixed` visit nobody writes `Gm` (the stores are `mixed`-guarded) and nobody reads it (GEMM 3
and the mass column's term 2 are too), so the one edge this counter carries — `G -> C` of the
edge DAG — is VACUOUS there. `Gm` is born and dies inside one tile (it has no ring slot,
[`smem_layout.md#staging-set`](smem_layout.md#staging-set)) and
the next tile's stores are ordered behind the end-of-tile `__syncthreads()`, so nothing outside
the tile observes it either. The counter word is re-zeroed unconditionally by every warp
regardless of the branch taken, so the skip path leaves nothing inconsistent: it bypasses
`retire_item`, `eligible`, `await_count` and the word itself.

### <a id="gram-gate-guard"></a>HAZARD `gram-gate-guard` — the retirement predicate mirrors the STORE guard, never the GEMM guard

The GEMM carries an extra `kw > 0` — the k-extent guard — that the stores do NOT: a
warp with `kw <= 0` skips the multiply but still STORES ZEROS, so a retirement taken from the
GEMM guard would leave that warp unretired and **the wait would HANG**. Lowering the target
without restricting the retirement is the mirror-image bug and is a CORRECTNESS defect, not an
optimisation: a waiter could reach `kGramWarps` on retirements from warps
`kGramWarps..kWarps-1`, which store no `Gm`, and read the Gram before its producer published.
`tests/unit/test_phase_c_gate_structure.py` reads both off the source and rejects the defect as
mutant `M-A`; it is the class `LATENCY_COVERAGE_DESIGN.md` names — a lowered wait target with an
unrestricted retirement.

### <a id="gate-if-constexpr"></a>HAZARD `gate-if-constexpr` — the fork is `if constexpr`, so the wide arm emits the unguarded code

A single site written `if ((!kNarrowGate || mixed) && warp < kGateWarps)` would leave a
`warp < kWarps` comparison whose truth `ptxas` cannot see (`warp` is `tid >> 5`, a runtime
value), and the wide arm's byte-identity — the whole reason its scope is drawn where it is —
would be lost on the very arms the scope protects.

### The phase-C gate is free

`Gm` is published to GEMM 3 and to the mass column's term 2 by `kCtlGramDone`, retired BEFORE the
fold. So by the time a warp reaches the wait it has already run the whole fold pool, and every
other warp retired the Gram phase before entering that pool. The wait is REAL (it is what makes
the acquire fence stand between `Gm`'s stores and its loads) and it is never on the critical
path, which is exactly the trade a barrier here could not make: `__syncthreads` would have
ordered phase C against the FOLD as well, and the two are independent (`F` writes `S`; `C` reads
`Gm`, `Vw`, `sr`, `den`).

`kCtlReadDone` is NOT narrowable: its dependence is the readout reading `S` before the fold
writes it, and at `DV = 64` all eight warps run a readout tile.

---

# Part VI — the four GEMMs

## <a id="warp-partition"></a>39. The warp partition, and the `NM` fork

The tile's output is `NM x NT_DV` WARP TILES, and that PRODUCT — not `DV` alone — is what the 8
warps partition. `BT = 16, DV = 64` gives `1 x 8`, one tile per warp — the cell a bare
`warp < NT_DV` guard would express. The three other cells are `2 x 8`, `1 x 4` and `2 x 4`; the
partition is the same strided warp loop the FOLD uses, so no cell is a special case and none is
a fallback. `kYTiles = ceil(NM * NT_DV / kWarps)` output fragments per warp, a COMPILE-TIME
count: warp `w` owns tiles `w, w + kWarps, ...`. At `(BT, DV) = (16, 64)` the loop has trip
count 1, which is the shape the baseline is measured at; `DV = 32, BT = 16` leaves warps 4-7
with zero tiles — the same strided-loop answer the fold gives.

**The y-tile accumulator is REGISTERS, not SMEM**, and that is structural rather than a saving:
warp `w` owns output columns `[8w, 8w+8)` for BOTH the readout and the intra term, so the two
GEMMs write the SAME `m16n8` fragment of the SAME warp. No other warp ever reads it, so an SMEM
y-tile would be a private object routed through shared memory — 4 KiB and a round trip for
nothing. The fragment survives the `__syncthreads()` between the GEMMs because a barrier orders
MEMORY, not registers.

### <a id="nm-fork"></a>HAZARD `nm-fork` — the `NM == 1` fork is load-bearing at the ALLOCATOR, not at the source

`NM == 1` — the tile is ONE `m16` row block — means warp `w` owns column tile `w` outright and
every m-offset is identically zero. The general strided form COMPUTES that same partition, but
computing it is not free: **the collapse is exact at the source level and NOT at the register
allocator**, and the 256-entry ratified manifest is a property of the allocator. Measured: with
the general form alone, **143 of the 256 shipped instantiations move and 24 spill worse**; with
this fork, **0 move**. Both arms are LIVE — `NM == 1` is `BT = 16`, `NM == 2` is `BT = 32` — and
the predicate is a TILE COUNT, the same class as the `NT_DV` guard. The same fork stands at all
three GEMMs and at the emission, for the same measured reason.

## <a id="gemm-readout"></a>40. GEMM 1 — the readout, `y_tile = R . S`

`[BT x BC] x [BC x DV]`. One `m16` row block at `BT = 16`, `DV/8 = 8` `n8` column blocks —
exactly one per warp, which is why the CTA is 8 warps.

The state is fp32 in SMEM and the MMA wants bf16, so the B operand is packed on the fly out of
`S`. Kept as its own path rather than a third orientation flag on `mma_tile`, because this is the
ONE operand whose source dtype differs from its fragment dtype and hiding that in a template
would bury it. **STATE AND ACCUMULATION STAY fp32** (the authorized bf16 contract); only the
MMA's own operand narrows, exactly as `readout_mma.cuh`'s bf16 arm does.

The k-extent is `BC` on both `NM` arms — the panel's column `k` and the state's row `k` are the
SAME leaf index, so the B operand is read straight, with no index map between them.

## <a id="mass-term1"></a>41. The mass column, TERM 1 — and its placement is FORCED

```
den[t] = scale_read[t] * ( sum_i R[t,i] * S[i][DV]
                         + sum_{u <= t} G[t,u] * scale_write[u] )
```

— the same split the value term takes, with `v := 1`. It is ONE column out of `DV + 1`, so it is
~1.5 % of the tile's arithmetic; making it an MMA `n` would waste 7/8 of an `n8` fragment (the
same argument `frag_a_get` records,
[`mma_tile.md#frag-a-get`](mma_tile.md#frag-a-get)).

**The two terms STRADDLE the fold, and the windowed phase B is what forces it.** The fold's own
mass recurrence WRITES `S[i][DV]`; the fold shares a phase with the Gram, so term 1 — which
READS `S[·][DV]` — must already have run. Its operands
(`Rf`, `S`) are exactly the readout's, so it merges with a walk that exists rather than
adding one, and it needs no extra shared memory.

`Rf`'s block for token `t` is selected by `t >> 4`; at `NM == 1` the arithmetic folds away
entirely.

### <a id="mass-split"></a>HAZARD `mass-split` — bit-identity of the split rests on same-thread program order

Term 1 completes into `den[t]` and term 2 RESUMES from it, so the sequence of fp32 `+=`
operations is unchanged; `den` is fp32 shared memory, so the round trip is exact; and **both
loops are `t = tid; t < BT; t += kConsumerThreads`, so the SAME THREAD owns `t` on both sides and
the dependence is program order** — there is no barrier between them and none is needed. Changing
either loop's stride or its thread map turns a program-order dependence into an unsynchronized
cross-thread one.

## <a id="b6"></a>42. B6 — the tile's read-then-write ordering on `S`, as a counter

The readout (GEMM 1) and the mass column's term 1 consume the state at TILE ENTRY; phase B's
fold advances it. That IS a real read-then-write dependence and it is not weakened — it is
carried by `kCtlReadDone` instead of by a rendezvous.

**Why the counter is not a `__syncthreads` with extra steps**, which is the only thing that would
justify the machinery: a `__syncthreads` here orders the readout against the fold AND, as a side
effect, against the GRAM — and the Gram has no dependence on `S` at all (it reads `Rf` and `WT`).
Under a barrier every warp that finished its readout stands still until the slowest one arrives;
under the counter it goes and accumulates its Gram tile, and only the FOLD — which is on the far
side of the Gram in program order — waits, by which time the count is long since satisfied. The
wait therefore has REAL WORK standing in front of it, which is the entire mechanism.

The retirement is UNCONDITIONAL (a read-only tile retires too). That costs one atomic per warp
per tile and it is what makes the deadlock argument a two-line one instead of a case analysis.

The wait itself is taken ONCE, in the first window — the counter is monotone, so re-testing it in
later windows would order nothing — and the first window's Gram stands between the retirement and
it, so on a balanced tile the count is already satisfied when the first warp arrives. `is_write`
and `w` are both CTA-uniform, so either every warp takes this wait or none does.

## <a id="phase-b"></a>43. Phase B — the windowed `{build, Gram, publish, fold}` loop

The order INSIDE a window is `{build, Gram, [last window: publish Gm + retire], fold}`. `Gm`'s
publication sits at the LAST WINDOW'S Gram, ahead of that window's fold rather than after the
whole loop, and that placement is sound because the Gram's sum is complete at that point and the
fold touches neither `gd` nor `Gm`. It buys two things: `gd[kGTiles][4]` dies BEFORE the fold's
claim junction (the register dividend the `ptxas` probe reads), and the phase-C gate is satisfied
before any warp arrives at it.

`WT`'s whole-plane residency is a consequence of the PHASE STRUCTURE, not of the math. GEMM 2
contracts over the LEAF axis and accumulates into ONE `[BT x BT]` Gram; GEMM 4 has the leaf axis
as its OUTPUT-ROW axis and writes DISJOINT state rows. Neither revisits an atom, so `W_q` is dead
as soon as both have seen it, and the only cross-atom carrier is the Gram.

A window is `kWtWindow` ATOMS because `kMmaKQuantum = 16` is simultaneously the readout's k-step,
the Gram's k-step and the fold's m-block — the three extents align exactly, which is what makes
ONE loop drive all of them.

**At `kNWin == 1` (the shipped default, and the `DECAY` pin) this loop has ONE iteration**, the
build stands above it, no barrier is emitted inside it, and every window-local index expression
folds to its window-free form. That is the free strong gate.

### <a id="gram-pinned"></a>HAZARD `gram-pinned` — the Gram accumulator is WARP-PINNED across the whole loop

And that is a CORRECTNESS requirement rather than a scheduling preference: `gd` holds a partial
sum in REGISTERS, so ascending-quantum accumulation order — which is what makes the windowed Gram
bit-identical to the plane one — exists only as long as ONE WARP OWNS ONE GRAM TILE for the whole
tile. Any later fluid-role rung must treat the Gram items as PINNED.

### <a id="gemm-gram"></a>GEMM 2 — the causal effective-score block, `G += R_q . W_q^T`

`[BT x BC] x [BC x BT]`, then the §5 INCLUSIVE causal mask (`u <= t`), applied AFTER the loop
because the mask is on the COMPLETED Gram: a deposit at `t` is visible to the read at `t`, which
is what makes the tile's read-after-write inclusive.

`B_T = false`: with `WT` leaf-major, `G`'s B operand element `(k = leaf, n = token)` IS
`WT[leaf * BTP + token]`, i.e. already `[k][n]`. **The transpose that `W^T` names is now in the
STORAGE**, so the fragment does not perform one — which is the point of building `WT` leaf-major.

`G` is `[BT x BT]`: `NM` m-blocks x `BT/8` n-blocks of warp tiles, strided over the warps exactly
as the readout and the fold are. At `BT = 16` the grid is `1 x 2` and warps 0-1 take one tile
each, i.e. `warp < BT / 8`. Forked on `NM` for the same measured reason ([above](#warp-partition)).

The causal test is on the TILE-GLOBAL token indices, so the m-block's row offset enters both
sides of `u <= t`. At `NM == 1` the offset is zero and this is a plain `u <= t`.

### <a id="gemm-fold"></a>GEMM 4 — the boundary fold, `S += W^T . Vw`, and the tile's fluid item pool

`[BC x BT] x [BT x DV]` — §10i.8 (2): ONE batched tensor-core GEMM per live owner, in place of
per-visit scattered `d_v`-wide scalar accumulates. `W^T` is not a transpose at the fragment —
`WT` IS the transpose, built that way. We SYNTHESISE the write coefficients rather than receive them
in someone else's layout, so storing them leaf-major costs exactly nothing at the build (one
index expression) and buys the fold a conflict-free, token-contiguous A operand. The same buffer
serves GEMM 2 as a `[k][n]` B operand with no orientation flag at all.

The pool is PER WINDOW, because `WT` holds only the window's rows: the items of window `w` are the
`m`-tiles `[w * kWtWindow, min(n_mt, (w+1) * kWtWindow))` crossed with the `NT_DV` `n`-tiles, and
they read the buffer at WINDOW-LOCAL row `mt - mt_lo`. At the plane there is one window, `mt_lo`
is zero, and this is the whole `n_mt * NT_DV` grid with every offset folded away.

**Why the fold is the FLUID set and the other three GEMMs are not** — this is D0.2's requirement,
stated per class rather than as a preference:

- the READOUT and the INTRA term accumulate into the SAME warp's `y[kYTiles][4]` registers across
  the whole tile, so an output tile is bound to one warp from the readout to the emission. Not
  claimable.
- the GRAM accumulates `gd[kGTiles][4]` across the window loop, so it is WARP-PINNED by the same
  argument (D8-Q3).
- the FOLD's accumulators `d0..d3` are born and die INSIDE one item. The arbitration join
  therefore sits exactly where D0.2 requires it — at an item boundary with every accumulator dead
  — and `ptxas`'s max-over-paths at the claim is taken over a live set of the CTA/owner invariants
  plus the claimed index, with no accumulator in it.

Bit-identity is unaffected because the items write DISJOINT `S` elements: item `(mt, nt)` owns
rows `[16 mt, 16 mt + 16)` and columns `[8 nt, 8 nt + 8)` exactly once, so which warp runs it
cannot change a value, only a schedule.

### <a id="fold-claim-uniform"></a>HAZARD `fold-claim-uniform` — the claim is WARP-UNIFORM and must never become per-lane

Lane 0 takes the index with a single `atomicAdd` and broadcasts it with `__shfl_sync`, so all 32
lanes execute the SAME item and the MMA is issued ONCE per warp, not 32 times. A per-lane claim
would issue 32 different items per warp and consume the pool 32x as fast, with each lane holding
a fragment of a different tile.

### <a id="fold-epilogue"></a>`DECAY`'s exit scale, riding the fold's existing read-modify-write

`M_exit = s0 * (s0 * M_entry + Sum_j W~[j] v[j])`, and the inner `s0` was applied at entry ([above](#decay-clock)
step (c)), so what is left is `S <- s0 * (S + d)`. The fold ALREADY reads `S`, adds to it and
writes it back; folding the scale into that epilogue costs ONE EXTRA MULTIPLY on a value already
in a register and ZERO extra shared-memory traffic.

**What it costs to do instead as a separate `[BC x cols]` sweep is measured, and it is the
counters that say so:** two such `s0` sweeps are `2 x 4160` element read-modify-writes per WRITE
TILE, and at the at-scale cell they account for the bulk of decay's **+87.7 M shared-store
instructions (+219.6 % over the `decay = None` arm)** — 16x the clock prefix's traffic, which is
the term the shared-store excess is otherwise attributed to. A separate sweep also needs its own
barrier PAIR: the fold writes `S` under the `(m16, n8)` warp-tile map while a row-per-warp sweep
reads it back under a different one, so the two cannot share a barrier. Riding the fold's own
read-modify-write costs none of that.

The warp tiles cover `BC x DV` exactly once, so every element is scaled exactly once and none is
missed — and the WINDOWS PARTITION those tiles, so that stays true window by window.

### The mass state's own recurrence

`mass[i] += sum_t W[t, i] * scale_write[t]` — the `v := 1` column of the same fold. It exists only
under `global`, and it runs over the SAME leaf-position axis as the fold's `m`, so the two
together still cover `BC x (DV + 1)` exactly once. It is THREAD-strided rather than claimed: it is
a scalar reduction with no fragment and no accumulator to strand, so the 256-thread stride already
balances it and a claim would only add an atomic per row. `WT`'s row is WINDOW-LOCAL, the state's
row `i` is not. The mass column IS the fold's `v := 1` column, so it takes the same exit scale
in the same read-modify-write.

### <a id="slot-ready-free"></a>SLOT-READY and SLOT-FREE

SLOT-READY, at the head of the window loop, publishes the window's rows to the Gram and the fold.
SLOT-FREE, at the foot, is the other half of the handshake, and it exists BECAUSE the buffer is
narrowed to a window: window `w + 1`'s build OVERWRITES the rows window `w`'s Gram and fold have
been reading, so the two need a rendezvous between them. The pair is D8.2's `2 * QPO` sync tax,
and it is the price of the windowed footprint.

It is a full-CTA `__syncthreads` and not a counter, and that is a READING rather than a
concession: the participant set is the WHOLE CTA on both sides (every warp can claim a fold item,
every warp can hold a Gram tile), so there is no subset for a counter to let run ahead — a counter
here would be a `__syncthreads` with extra steps, which is the test the two phase counters pass
and this edge does not.

SLOT-FREE is NOT emitted at the last window (nothing overwrites those rows until the next tile,
which the end-of-tile barrier already orders) and NEITHER is emitted AT ALL at the plane, where
`kNWin == 1`.

## <a id="phase-c"></a>44. Phase C — GEMM 3, the intra term, and the mass column's term 2

`y_tile += G . Vw`. The SAME tile decomposition the readout used — including the same `NM` fork —
so the two GEMMs accumulate into the same warp's same fragment, which is what keeps the y-tile in
registers rather than in shared memory. It runs POST-LOOP because it needs the COMPLETED Gram,
which is what keeps `y` live across phase B (the register cost the `ptxas` probe gates).

On the narrow arm `Vw` is fetched by `ldmatrix` here too — the SAME atom against the SAME layout
the fold uses, which is what collapses the two access patterns the pitch had to choose between
([above](#pitches)) into one.

The mass column's term 2 resumes from term 1's `den[t]`: `acc = den[t]`, then
`acc += Gm[t][u] * sw[u]` over `u <= t`.

---

# Part VII — the epilogue

## <a id="emission-interval"></a>45. HAZARD `emission-interval` — no barrier between the mass reduction and the emission

There is NO `__syncthreads()` between the mass reduction and the emission, and what licenses that
is the SMEM read/write SETS of the interval, ENUMERATED rather than eyeballed:

- the emission READS `y[q][*]` (REGISTERS, so no barrier can order them and none is needed —
  they are the same warp's own accumulators), `sr[t]` (published three barriers back, at the end
  of the panel phase) and `den[t]`;
- the emission WRITES only GLOBAL memory, through `atomicAdd`;
- `den[t]` is written at `den[t] = acc` above and read at the denominator emit below **BY THE
  SAME THREAD** — both loops are `t = tid; t < BT; t += kConsumerThreads`, so the dependence is
  program order, not a cross-thread one.

So the interval contains NO SMEM object written by one thread and read by another, in EITHER
direction: nothing the emission writes is read by a lagging thread's mass reduction (it writes no
SMEM at all), and nothing it reads is written by one (`sr` and `den` are both already-ordered).
The read-then-write ordering on `S` is carried elsewhere entirely, by B6, which stands between
the readout/mass-term-1 and phase B ([above](#b6)) — so the emission sits on the far side of the fold and
the discharge is indifferent to that: the argument is about the interval's read/write SETS, never
about its position. **Any edit that puts a cross-thread SMEM dependence into this interval opens
a race with nothing to order it.**

## <a id="split-k-emission"></a>46. The split-K emission

Straight out of the D fragment: `d0,d1 -> (row g, col 2r, 2r+1)` and `d2,d3 -> (row g+8, ...)`,
with the §3 step-3 row scale `sr[t]` applied HERE because it is constant along both `k` and `n`.
Out-of-range tokens (`t0 + t >= T`) emit nothing. Under `GLOBAL` the denominator takes the same
shape: `sr[t] * den[t]`, atomically into `p.den`.

The `if (out != 0.0f)` guard is a PERFORMANCE SKIP ON A VALUE, not a membership decision — adding
`0.0f` is the identity, so including the term is a no-op and the test is over-inclusion-safe in
the direction that matters. See [`consumer.md`](consumer.md#split-k-emission) for its two
load-bearing consequences, including why it makes the intra term a SECOND atomic per
(token, channel) on a mixed tile — cause #1 in
[`intra_panel.md`](intra_panel.md#measured-status) for `panel` measuring slower than the walk.
Fusing the intra term into the readout epilogue is the standing follow-up for BOTH lowerings.

### <a id="emit-no-pointer-array"></a>HAZARD `emit-no-pointer-array` — never take the address of an accumulator

The emits stay WRITTEN OUT per fragment slot rather than looped over an ARRAY OF POINTERS to the
accumulators: **taking the address of a register-resident value is how an accumulator ends up in
local memory**, which is the one thing this fragment must not do. `y[q][*]` is indexed by an
unrolled compile-time `q`, which is a REGISTER SELECTION rather than an address. The OUTPUT side
does keep its hoist — one base pointer, no array, so no dynamic indexing and no local memory.
Forked on `NM` for the same measured reason as the three GEMMs
([above](#warp-partition)).

## <a id="pass1-epilogue"></a>47. §6.4 pass 1's per-leaf epilogue

Both blocks are PER-LEAF and window-independent, so they run once after phase B rather than being
partitioned by it: `s0` is a per-leaf tile constant and `a_state` is a per-leaf product, and
neither reads `WT`.

### The written-leaf list, accumulated across the segment

A leaf enters the list iff it can LEAVE `(A, B) = (1, 0)`, and the test is deliberately the
WEAKER of the two available ones. OVER-inclusion is free — the leaf is emitted at its exact
`(1, 0)` and the scan's fold returns the incoming value unchanged — while UNDER-inclusion
silently drops state. So the test is over-inclusion-safe by construction, which is the same
direction every `!= 0.0f` skip in this kernel takes.

**The COEFFICIENT half of the test is NOT here.** It runs in the write-panel build, on the fp32
value, before the store ([`summary-prequant`](#summary-prequant)), because a re-scan of `WT` reads a STORED
coefficient and a stored coefficient is a QUANTIZED one wherever the buffer's format is narrow.
What is left here is the term that has no pre-quantization site because it is not a coefficient
at all:

under decay `WT` carries `W * fw`, and "a nonzero write forces a nonzero coefficient" is an
argument about `W`, not about `W * fw`: `fw` can UNDERFLOW to zero on the early half of a tile at
an extreme survival range. That range is outside what `panel_exponent_bound` admits, but the
summary's inclusion test should not be the thing that depends on the admission check. `s0 != 1`
is the DIRECT statement that this leaf's clock moved in this tile, which is exactly the condition
under which `A` leaves `1`.

A leaf with no live support in this tile passes NEITHER term — its write coefficients are zero,
which the build's own test sees, and its `s0` is exactly `1` (the
[scatter-fill theorem](#scatter-fill)) — so the walk over all `BC` leaves admits exactly the
leaves that can leave `(A, B) = (1, 0)`.

### `A`, the segment's per-leaf survival product (§2.13)

One write tile's map is `M_out = s0^2 * M_in + s0 * (W~^T Vw)`, so the tile's `A` factor is
`s0 * s0` and the segment's is their product. Rule 1's half-mass reference is what makes this
expressible as `s0` SQUARED rather than as a second exponential to form and store:
`s0^2 = exp(2 * P_ref * log_keep) = keep^(tile's TOTAL clock)`, i.e. the genuine survival across
the tile. `A` therefore needs no separate reduction and no separate buffer.

It is a `[BC]` loop, not a `[BC x cols]` one — `A` is a per-leaf SCALAR, which is the whole reason
a segment summary is cheap. It runs over all `BC` leaves unconditionally: an inert leaf's `s0`
is exactly `1`, so the factor it contributes is the identity.

## <a id="pass1-publication"></a>48. §6.4 pass 1's publication — the SPARSE written-leaf summary

§6.4 makes the summary NORMATIVELY sparse: its traffic must scale with REALIZED writes, not with
`BC`.

Slot `owner_slot0` is the scan's entry-state stash, so segment `seg`'s summary lands ONE PAST it.
Pass 3's segment `q` then reads `owner_slot0 + q` UNIFORMLY: the stash at `q = 0`, and the
boundary the scan accumulated over this summary at `q = seg + 1`.

Warp 0 compacts the flags to an ASCENDING leaf list by ballot; ascending order is what makes
the emitted list a function of the flag SET and not of the ballot's scheduling.
`sum_slot[i]` is rewritten IN PLACE from "is written" to "position in the list"; only
lane `i` ever touches entry `i`, and it reads before it writes, so the rewrite is race-free.

`A` is identically `1` under `decay = None` (nothing multiplies `a_state` there), so the scan's
multiply is EXACT on that arm rather than skipped — one code path, no arm-specific fold.

**`B`'s emission is a plain runtime-indexed store**, and the SMEM-resident state is what allows
it: `S` is ADDRESSABLE, so a runtime list position `i` is just an index. A register-resident
state would need a compile-time `#pragma unroll` over leaves instead, because indexing it by a
runtime position spills the whole state to local memory and destroys §5's register invariant.

## <a id="state-store"></a>49. The owner's state block, written to HBM once

An atom with no committed slot is never stored back: it has no page, and its state was read as
zeros at entry. The skip is PER ATOM because RESIDENCY is per atom after §4.5's re-key — the same
granularity the read resolves at.

`last_seg` is §6.4's publication rule and is `true` at `P = 1`: the final state is published by
pass 3's LAST segment, running serially from the accumulated boundary, **NOT by the scan**. That
is a DELIBERATE NUMERIC CHOICE — it leaves the last segment's own arithmetic in serial order and
confines the fold's re-association to the `P - 1` boundaries.

---

## <a id="hazard-index"></a>50. Hazard index

Every `//: HAZARD <slug>` stub in the four consumer-kernel sources — `consumer_kernel.cuh`,
`smem_layout.cuh`, `mma_tile.cuh` and `pipeline.cuh` — and where its reasoning lives. The
index is kept here, in one place, because a hazard is a property of the kernel and not of
whichever layer happens to hold the line.

| slug | site | what a mechanical edit would silently change |
|---|---|---|
| [`frag-a-get`](mma_tile.md#frag-a-get) | `frag_a_get` | the hand-rolled inverse of `frag_a_slot`; nothing checks the pair |
| [`loop-induction`](#loop-induction) | the candidate walk | folding the scan into the loop increment: measured spills on arms at depth 1 |
| [`no-captured-arrays`](#no-captured-arrays) | `build_wt_rows` / the amplitude accessor | a captured pointer array lands in LOCAL memory |
| [`clock-prefix-fp32`](#clock-prefix-fp32) | the clock prefix `acc` | narrowing it is an EXPONENT error, not a mantissa one |
| [`wt-row-t-fastest`](#wt-row-t-fastest) | `build_wt_rows`' index map | `i`-fastest is a 4-way STORE conflict |
| [`summary-prequant`](#summary-prequant) | the `a != 0.0f` ballot | a post-quantization test UNDER-includes and drops state |
| [`ballot-lane-run`](#ballot-lane-run) | `kRowLanes` | the `t`-group must be a contiguous lane run inside one warp |
| [`wt-build-site`](#wt-build-site) | the plane's `build_wt_rows` call | a run-ahead build overwrites rows an async fold reader still holds |
| [`clk-dead`](#clk-dead) | the `clk -> Vw` barrier | `Vw`'s staging clobbers the aliased clock prefix |
| [`release-acquire`](pipeline.md#release-acquire) | `retire_item` / `eligible` | fence order, lane-0-only atomic, `volatile` load |
| [`gram-gate-guard`](#gram-gate-guard) | the `kCtlGramDone` retirement | mirroring the GEMM guard instead of the store guard HANGS |
| [`gate-if-constexpr`](#gate-if-constexpr) | the `kNarrowGate` fork | a folded predicate loses the wide arm's byte-identity |
| [`nm-fork`](#nm-fork) | the four `NM == 1` forks | 143 of 256 instantiations move, 24 spill worse |
| [`mass-split`](#mass-split) | mass terms 1 and 2 | the split is bit-identical only under the same thread map |
| [`gram-pinned`](#gram-pinned) | `gd` | a fluid Gram loses ascending-quantum accumulation order |
| [`fold-claim-uniform`](#fold-claim-uniform) | the fold's `atomicAdd` claim | a per-lane claim issues 32 different items per warp |
| [`emission-interval`](#emission-interval) | the removed barrier | the interval must stay free of cross-thread SMEM |
| [`emit-no-pointer-array`](#emit-no-pointer-array) | the `emit` lambdas | taking an accumulator's address spills it to local memory |
