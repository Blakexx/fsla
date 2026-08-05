# `csrc/rola/src/consumer_kernel.cuh` — the RoLA V3 consumer kernel body

The ONLY consumer. This doc owns the kernel BODY: the
§3/§5 algorithm, the compile-time budget, the shared-memory carve, the MMA
fragment maps, the bank-conflict proofs, the staging ring, the barrier and
retirement protocol, the split-K epilogue, and the `COMPACT` / `DECAY` / `GLOBAL`
arms.

Its sibling is [`consumer.md`](consumer.md), which owns `ConsumerParams` and the
host side and is NOT restated here: the leaf-digit decomposition, the paged state
layout and `kAbsentPage`, the §6.4 work list and its workspace, the three-pass
driver, the instantiation matrix, and the residency query's axes all live there.
Two descriptions of one kernel is one that can go stale.

Spec references are to `ROLA_V3_SPEC.md` unless named otherwise: §3
topology-agnostic kernel structure, §4 schedule, §5 consumer, §6.4 sequence
parallelism, §7.4.1 the decay lowering. `GLA_MASS_DECAY_SPEC.md` §12 is the decay
numerics contract.

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

7. **STAGE** — the `COMPACT` gather (§3 step 2), the `DECAY` clock prefix and
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
  `DECAY`, `GLOBAL`, `COMPACT` — see [`consumer.md`](consumer.md#instantiation-matrix).

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
waste 7/8 of an `n8` fragment (§21).

## <a id="glossary"></a>3. Symbol glossary

| symbol | what it is |
|---|---|
| `BT` | tokens per compute tile; `{16, 32}` |
| `BC` | leafs per compute tile (the owner's leaf block); `{16, 64}` |
| `DV` | value width `d_v`; `{32, 64}` |
| `D` | level count of the topology; `1 <= D <= 4` (spec §1) |
| `j` | `first_spanned_level`. Levels `l < j` are FIXED for this owner |
| `kMmaKQuantum` | `16` — the hardware `mma.sync` k-extent on the sm80 bf16 path, and the named constant every quanta-shaped quantity is an expression in (spec §1 I1) |
| `QPO` / `kQPO` | quanta per owner, `BC / kMmaKQuantum`. An ATOM is one `kMmaKQuantum`-leaf block in canonical leaf order |
| `NM` | `BT / 16` — the tile's `m16` ROW BLOCKS |
| `NT_DV` | `DV / 8` — the `n8` column tiles of a `DV`-wide output |
| `kWarps` | `8` — `kConsumerThreads / 32` |
| `kYTiles` | output fragments per warp, `ceil(NM * NT_DV / kWarps)` |
| `kGTiles` | Gram fragments per warp, `ceil(NM * (BT/8) / kWarps)` |
| `kGramWarps` | `min(NM * (BT/8), kWarps)` — the warps that STORE `Gm`, and the narrow phase-C target |
| `kRing` | staging-ring depth, `1` or `2` (§7) |
| `kWtWindow` | `WT` window size in ATOMS; `kWinRows = kWtWindow * 16` rows; `kNWin = kQPO / kWtWindow` windows per tile (§8) |
| `S` | the STATE, `[BC][SP]` fp32 SMEM, resident for the owner's whole run |
| `SP` | `S`'s row pitch, `DV + 4` (§17) |
| `cols` | `DV + 1` under `GLOBAL` (the mass column), else `DV` |
| `WT` | the write coefficient panel, LEAF-MAJOR, `[kWinRows][BTP]` |
| `BTP` | `WT`'s row pitch; `BT + 8` narrow, `BT + 4` on `tf32x3` (§13) |
| `Vw` | the scale-folded value tile, `Vw[t,ch] = sw[t] * v[t0+t, ch]` |
| `VWP` | `Vw`'s row pitch on the `tf32x3` arm, `DV + 8`. The narrow arm has NO pitch — it is `StageLayout` (§12) |
| `Gm` | the causal effective-score block `[BT][BT]`, pitch `GMP` |
| `Rf` | the read coefficients, MMA-FRAGMENT-ORDERED, `NM` blocks of `frag_a_words(BC)` words |
| `clk` | the tile's per-leaf INCLUSIVE clock prefix `[BT][CKP]` fp32, `CKP = BC + 4`; ALIASES `Vw`'s bytes |
| `log_keep`, `p_ref`, `s0` | `DECAY`'s three `[BC]` vectors — per-owner log retention, §12 rule 1's reference exponent, and the entry/exit scale |
| `sr`, `sw` | the §3 step-3 per-token read/write ROW SCALES, `[BT]` each |
| `den` | the `GLOBAL` denominator accumulator, `[BT]` |
| `dig` | the hoisted leaf digits, `[D][BC]` |
| `ctl` | the K3b control block, `2 + QPO` ints (§32) |
| `c_idx`, `c_slot`, `c_count` | `COMPACT`'s gathered leaf list, its inverse, and `{K, KI}` |
| `K` / `KI` | the gathered list length / its §5 causal PREFIX length |
| `Kpad`, `KIpad`, `KpadM` | the readout k-extent, the Gram k-extent and the fold m-extent, each rounded up to its quantum |
| `kUnit` | the MMA's own k-step: `16` on bf16 (`m16n8k16`), `8` on tf32x3 (`m16n8k8`) |
| `kMaxWords` | candidate-tile words held in SMEM per CTA (`consumer.cuh`) |
| `A`, `B` | §6.4's per-segment summary: the per-leaf survival product and the accumulated deposit |

---

# Part I — the compile-time budget

## <a id="reg-budget"></a>4. `consumer_reg_budget` — a MEASURED kernel fact, not an architecture fact and not a threshold

This is the register working set `ptxas` reports for THIS kernel's GEMM inner loop
when it is given slack — per `mma.m16n8k16`, 4 A fragments + 2 B fragments + 4
accumulators times the fold's warp tiles, plus the operand-pack temporaries and
loop state. It is keyed on `BC` AND on `DECAY`, the two plan axes the working set
actually varies with (the `BC = 64` arms carry four fold m-tiles where `BC = 16`
carries one; the decay arms carry two survival panels, the clock prefix and the two
coefficient folds on top of the same GEMM), and every number is READ OFF
`-Xptxas -v`, never chosen:

```
BC = 64             : 66 registers (decay = None), 80 (decay + compact)
BC = 16, decay=false: 60-64 registers, ZERO spills on all 32 entries
BC = 16, decay=true : 80 registers, and at a 64-register budget it SPILLS
                      (worst 66 st / 212 ld, §2.22)
```

The MAXIMUM per key is taken, because this constant DIVIDES: a budget that is too
LARGE only understates the residency the register file admits, while one that is
too small over-promises and the allocator buys the difference by SPILLING. The
conservative direction is therefore well defined, and it is the one taken.

### Why `DECAY` is a key and not an average — both sides of the trade are measured (§2.23)

A single `BC = 16` budget loses on one arm whichever value it takes, because the
two arms want opposite things and the loser is not small. One `ncu` launch per arm
at `topo-D4-w8x8x8x16-BC16` (1024 CTAs, ~3 waves), `--clock-control base`, which is
what makes cross-BUILD durations comparable on a contended box:

| arm | budget | regs | CTAs/SM | warps_active | duration | local ld sectors |
|---|---|---|---|---|---|---|
| `decay = None` | 64 | 64 | 4 | 63.9 % | 278-293 us | 11,262 |
| `decay = None` | 80 | 69 | 3 | 49.5 % | 324-336 us | 11,262 |
| `decay = warm` | 64 | 64 | 4 | 65.7 % | 626-635 us | 3,345,396 |
| `decay = warm` | 80 | 80 | 3 | 49.7 % | 588 us | 142,330 |

On the `decay = None` arm the budget of 80 buys NOTHING (the local traffic is
identical — it is local arrays, not spills) and costs a CTA: **+12 to +16 %**. On
the decay arm it removes 23x the local LOAD traffic and is **-6.5 %** even though it
gives up a quarter of the occupancy. Keying the constant is therefore not a tuning
knob, it is the two measurements written down.

`BC = 64` is NOT keyed, and that is also a measurement rather than an omission:
shared memory binds there at 3 CTAs/SM on every arm, so the register term is slack
and both 66 and 80 produce the same bound.

Reproduce the register/spill side with `python tools/ratify.py --census`, which
compiles the shipped instantiation shards and reports per structural key, and the
residency side with `rola_consumer_occupancy`; if the kernel's inner loop changes,
re-run both and update these numbers rather than the formula.

### <a id="reg-budget-nm"></a>The row-block term — measured over five candidate budgets, keyed on `NM` alone

A `BT = 32` tile is `NM = 2` `m16` row blocks, so the readout and the intra term
carry a second live y-fragment quad per warp wherever `DV = 64`, and a second
A-fragment block everywhere. `d_v` is NOT a key: the measurement shows no `d_v`
dependence, and a key nobody can point at a number for is an extrapolation.

The instrument is `python tools/ratify.py --census --force-ctas N` over all 1024
instantiations, once per candidate budget, with the ratified `BT = 16, DV = 64` matrix held at its landed
value as an IN-RUN CONTROL (byte-identical at all five sweep points, so the numbers
below are the new cells' own and nothing else's). Spill bytes summed over the 64
instantiations of each key:

| `(BT, DV, BC, decay)` | b=64 | b=80 | b=96 | b=128 | b=168 |
|---|---|---|---|---|---|
| (16, 32, 16, false) | 300 | 0 | 0 | 0 | 0 |
| (16, 32, 16, true ) | 3938 | 352 | 872 | 872 | 0 |
| (16, 32, 64, false) | 772 | 1056 | 728 | 728 | 0 |
| (16, 32, 64, true ) | 4480 | 1884 | 1732 | 1732 | 0 |
| (32, 32, 16, false) | 2528 | 1272 | 0 | 0 | 0 |
| (32, 32, 16, true ) | 5614 | 2532 | 1612 | 1612 | 1624 |
| (32, 32, 64, false) | 2666 | 1388 | 1276 | 1276 | 0 |
| (32, 32, 64, true ) | 3488 | 2388 | 1860 | 1860 | 1680 |
| (32, 64, 16, false) | 1860 | 1188 | 96 | 96 | 0 |
| (32, 64, 16, true ) | 5566 | 2316 | 344 | 344 | 384 |
| (32, 64, 64, false) | 568 | 568 | 416 | 416 | 0 |
| (32, 64, 64, true ) | 1768 | 1768 | 992 | 992 | 352 |

Three things the table decides, all three READINGS rather than choices:

1. **`b = 128` IS `b = 96`.** `by_regs = 65536 / (256 * b)` is 2 at both, so the
   two budgets state the SAME launch bound and `ptxas` emits the same code — every
   row above is identical between the two columns. 128 is not a candidate, it is a
   spelling of 96.
2. **`b = 168` IS NOT A CANDIDATE EITHER**: it states ONE CTA/SM, which is the
   unconstrained allocator, and it is in the table only to show what the working set
   costs when nothing bounds it. Three keys STILL spill there, at 255 registers —
   those are local ARRAYS, not pressure, which is §2.23's reading too.
3. **`d_v = 32` NEEDS NO NEW KEY.** At `BT = 16` the ratified `(BC, decay)` formula
   is already the best shipping value on three of the four `DV = 32` keys (80 at
   `BC = 16, decay`; 80 at both `BC = 64` arms is within 300 bytes of the best and 96
   costs a CTA there). The fourth, `(16, 32, 16, false)`, spills 300 bytes over 7 of
   its 64 instantiations at 64 where 80 spills none — and §2.23 MEASURED that trade
   on the same arm one `d_v` over: on a no-decay `BC = 16` arm the extra budget buys
   a CTA's worth of nothing and costs +12 to +16 %, against a spill smaller than the
   4 st / 8 ld §2.22 accepted as the optimum. The occupancy keeps it.

What is left is `NM`, and it is not close: at `NM = 2` the ratified formula's 64 and
80 leave 1860-5614 spill bytes on the `BC = 16` keys where 96 leaves 0-344, and 96 is
best-or-equal on ALL EIGHT `NM = 2` keys. `NM` is the same tile-count predicate the
warp partition forks on (§36), so the budget and the codegen state the same
structural fact once.

### <a id="reg-budget-session9"></a>The one class that clears 64, and why every other `NM = 2` key stays at 96

The keying is a 512-entry forcing census read back (`tools/ratify.py --census`, arms:
shipped / `--force-ctas 4` / `--force-ctas 3`, sm_86; provenance:
`workflows/20260802-thunder/trackK.md` session 9). The rule for lowering a key is
strict: **a key moves ONLY where the forced arm shows ZERO spill AND a bit-identical
in-loop local-access count on EVERY one of its 16 instantiations.**

Exactly ONE structural class passes: `NM = 2`, `decay = false`, bf16 at both MMA
sites, SINGLE y-quad per warp (`NM * NT_DV <= kWarps` — the warp partition's own tile
count; at `NM = 2` that is `d_v = 32`). Measured at the 64-register bound, both `BC`
arms, all 32 instantiations: **0 spill st / 0 spill ld, hot counts unchanged
entry-for-entry**; `BC = 16` gains a CTA on 9 of 16 (D1 raw 86 -> 52 registers,
2 -> 4 CTAs). The census clears the `BC = 64` half too; the DURATION gate declines it
(below), so only the register-bound `BC = 16` half ships.

**`BC < 64` is a DURATION verdict, not a census one (gate iii).** The `BC = 64` half
of this class is SMEM-capped at 3 CTAs, so a lower budget can never buy residency
there — it can only re-declare the bound 2 -> 3, which reschedules 80 -> 60 registers
of ILP for zero occupancy gain. Measured (`ncu --clock-control base`,
`topo-D1-w256-BC64 d_v=32 BT=32`, baseline vs forced): **+2.42 %** — declined on
measurement; 96 stands and the declared-2-vs-realized-3 under-declaration stays
recorded as the price. The `BC = 16` half is register-BOUND (SMEM admits 5-6) and the
same gate read **-6.97 %** (`topo-D4-w8x8x8x16-BC16 d_v=32 BT=32`: 66 -> 62
registers, 3 -> 4 CTAs, warps-active 49 -> 64 %), so it lands.

**Every other `NM = 2` key is held at 96, on measurement, not caution:** the
second-y-quad class (`d_v = 64`) ADDS in-loop local accesses at the 64-bound (max +7
per entry) AND at the 80-bound (max +3); every `decay` key adds them (bf16 max +15,
tf32x3 max +35..+63); every tf32x3 key spills at both forced bounds (140-380 B st per
key). An in-loop spill is a round-trip per tile-loop iteration — disqualifying
whatever the byte count, and the hot-loop ratchet in `tools/ratify.py` enforces
exactly that as a build failure. The `BT = 16` decay keys' candidate-64 is refused by
the same test (+3 in-loop max), which answers §2.23's duration question for them
without a second duration run: 80 stands. Full tables:
`workflows/20260802-thunder/trackK.md` session 9.

## <a id="residency"></a>5. `ctas_for_residency` and the launch bound

CTAs/SM to STATE IN THE LAUNCH BOUND — the smallest of what the three hardware
resources admit (shared memory, thread slots, registers). STRUCTURAL, computed per
instantiation rather than fixed: the decay arm carries ~4.9 KiB of panel machinery
the `decay = None` arm does not, so a single constant would either over-promise on
one arm or under-use the other.

**The THREAD term is what keeps the bound from being discarded outright, and its
absence is silent on every `BC = 16` instantiation — half the matrix (64 of 128).**
At `BC = 16` the footprint is ~13 KiB, so the shared-memory term ALONE returns 7, and
`7 x 256 = 1792 > 1536` thread slots. `ptxas` does not CLAMP an out-of-range bound,
it DISCARDS it (`Value of threads per SM ... is out of range. .minnctapersm will be
ignored`) — precisely the unconstrained-allocator state correction (d) exists to
prevent, and an arithmetic gap reaches it as surely as an omitted attribute does.

**And the thread term alone is not the fix**, which is why the register term is
there. Measured per candidate bound at `BC = 16` (§2.21):

| bound | regs | entries that SPILL | worst spill |
|---|---|---|---|
| absent | 57-64 | 1 of 6 | 20 st / 56 ld |
| 6 | 40 | **4 of 6** | **70 st / 284 ld** |
| 5 | 51 | 2 of 6 | 66 st |
| **4** | 64 | 1 of 6 | **4 st / 8 ld** ← the measured optimum |

Asking for the 6 CTAs the thread slots admit caps the allocator at
`65536 / (6 * 256) = 42` registers and it buys those CTAs by SPILLING. The register
term returns 4 at `BC = 16, decay = false`, which is both the register file's own
answer at that working set AND what the allocator converges to when left alone — and
`ncu` confirms the residency is 4 CTAs/SM either way, so on that arm the bound
converts a silent hazard into a stated invariant at zero measured occupancy cost.

**And on the `BC = 16` decay arm the term returns 3, by measurement** — see §4, which
carries both arms' `ncu` numbers. That arm's working set does not fit in 64
registers, so a budget of 64 there does not buy the fourth CTA, it buys 66 st / 212 ld
of SPILL; 3 CTAs at 80 registers is 6.5 % FASTER than 4 CTAs at 64 with the spill. The
table above is the `decay = false` arm's.

### Why this is stamped into `__launch_bounds__` and not left to the compiler

Meeting the shared-memory budget does NOT get you the occupancy. At
`__launch_bounds__(256, 1)` `ptxas` is free to spend up to 255 registers/thread — it
spent 96 — and `96 x 256` caps residency at 2 CTAs even where SMEM admits 3. Told the
real target it re-schedules to 66 registers with ZERO spills, so the constraint costs
nothing and its absence costs a third of the occupancy. This is coordinator correction
(d): the launch bound is a structural PLAN quantity carried through the instantiation
matrix.

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

## <a id="ring-depth"></a>7. §8 calibration placeholder `C_RING_DEPTH` — the staging ring's depth

How many TILES' worth of the STAGING SET the CTA holds at once. `1` is today's
structure exactly — stage tile `n`, rendezvous, compute tile `n` — so an unfilled or
over-large placeholder degrades to today's behaviour, never to a new one (§8's stated
principle). `2` lets the staging of tile `n + 1` be issued from INSIDE tile `n`'s
compute, at the one program point where the tile's work is finished and its warps
would otherwise be sitting on the phase-C acquire.

**`2` is the ceiling and it is STRUCTURAL, not a budget**: the run-ahead is ONE tile
(the CTA walks its candidate list serially and the staged slot is consumed by the very
next visit), so a third slot would have no producer. A larger value clamps.

**The depth is DERIVED, by one rule, applied per instantiation:** *the deepest staging
set the arm's headroom admits, with `Vw` excluded under `DECAY` by the alias.*
Mechanically, depth 2 is taken iff it costs NO CTA — `ctas_for_residency` returns the
same answer at both depths — i.e. "the ring pays exactly when it does not cost a CTA"
stated as arithmetic instead of as a recommendation (provenance: r1 §3.7). It is the
same derivation `__launch_bounds__` is, off the same two functions.

**And the shared-memory term is checked ON ITS OWN**, which is NOT implied by that
line. `ctas_for_residency` takes a MIN over three terms, so a `by_regs` floor can mask
an SMEM drop entirely — and the driver, which spends fewer registers than the budget,
then sees the drop the min hid. Measured: two shapes shipped a 3 -> 2 driver regression
through the min alone. The SMEM term is the only part of the driver's answer a
`constexpr` can see, so it is checked separately.

### <a id="decay-ring-pin"></a>`DECAY` is pinned to depth 1, and it is CORRECTNESS rather than footprint

D2.3 bars `Vw` from the ring because `clk` (the tile's clock prefix) ALIASES `Vw`'s
bytes, so slot `s`'s `Vw` and slot `s`'s `clk` are the same storage. That exclusion
PROPAGATES to the whole staging set rather than removing one member of it: BOTH panel
builds READ `clk` (`panel_read_factor` / `panel_write_factor`), so staging tile
`n + 1`'s `Rf` or `WT` requires tile `n + 1`'s CLOCK, which would have to be written
into the one region tile `n`'s fold and intra term are still reading. **There is no
non-empty subset of the staging set that can run ahead under `DECAY`.**

Un-aliasing `clk` would dissolve it and does not fit: `clk`'s own region is
`BT * (BC + 4) * 4` = 4,352 B at the flagship, against **2,941 B** of decay-arm
headroom there (§8 states that figure's arm and conditions). The trade is a
coordinator-level one, so the pin is asserted in the kernel as a correctness
`static_assert` and never left to the footprint arithmetic to enforce. Un-aliasing
`clk` is the only thing that changes this.

### <a id="ring-default"></a>The default is `1`, against §8's tabled `2`, and it is a measurement

§8 tables `2` on the principle that a wrong placeholder degrades to today's behaviour.
At depth 2 it does not: `rola_consumer_occupancy` — the DRIVER, which is spec §6.5's
sole residency authority — drops FOUR of the 32 `(decay, compact, BC, BT, DV)` shapes,
two by shared memory and two by registers:

```
!decay !compact BC16 BT32 DV32   4 -> 3   (registers)
!decay !compact BC64 BT16 DV32   4 -> 3   (registers)
!decay  compact BC16 BT32 DV64   3 -> 2   (shared memory)
!decay  compact BC64 BT32 DV32   3 -> 2   (shared memory)
```

The SMEM pair is caught by the derivation above. **The register pair CANNOT BE CAUGHT
AT COMPILE TIME**, and that is the load-bearing fact rather than a workaround: the
driver reports a residency HIGHER than `ctas_for_residency` states,
because `ptxas` spends fewer registers than `consumer_reg_budget` budgets, and the
ring's own addressing (+3 to +7 registers at the flagship, measured) eats that
undeclared surplus. No `constexpr` can see a `ptxas` output. Re-deriving
`consumer_reg_budget` — D8-Q1, already open — is the thing that would let this default
move to `2`.

## <a id="wt-window"></a>8. §8 calibration placeholder `C_WT_WINDOW` — the `WT` window

How many `kMmaKQuantum`-leaf ATOMS of the write panel `WT` the tile's second phase
holds at once. `WT`'s whole-plane residency is inherited from the PHASE BARRIER
STRUCTURE and not from the math: the Gram (`G += R_q . W_q`) contracts over the leaf
axis and the fold (`S[i,:] += Sum_t WT[i,t] Vw[t,:]`) has the leaf axis as its
OUTPUT-ROW axis, so neither ever revisits an atom and `W_q` is dead the moment both
have seen it. The cross-atom carrier is exactly one object, the Gram `[BT x BT]`.

`0` means `QPO`, **which IS the plane, i.e. today's structure exactly** — so a
wrong/unfilled placeholder degrades to today's behaviour. The terminal cycle (§9.2)
fills it from the producer/consumer variance measurement; the two ends of the dial are
one parameter, not two designs (I4).

**It is the ALLOCATION, not just the phase.** `consumer_smem_bytes` sizes `WT` from
`wt_window_rows`, so at `C_WT_WINDOW < QPO` the buffer really is `kNWin` times smaller
and the residency really moves; the rows are indexed WINDOW-LOCALLY and reused window
by window, which is what the SLOT-READY / SLOT-FREE `__syncthreads` pair in the tile's
second phase exists to make safe (§39). That pair is D8.2's `2 * QPO` sync tax, and it
is the price of the footprint.

`wt_window_rows` is THE definition — `consumer_smem_bytes` sizes the buffer from it and
the kernel body indexes with it, so the allocation and the addressing cannot disagree
about how many rows exist. `C_WT_WINDOW` must DIVIDE `QPO`: the window loop's trip
count is exact, never ragged (I1 — an expression in `BC` and `kMmaKQuantum`, never a
literal). At the plane `q_lo` and `mt_lo` are identically zero and every window-local
index expression folds back to the pre-window one, which is what makes the default
byte-identical for free.

**Pinned to the plane on the `DECAY` arm, and it is a CORRECTNESS pin rather than a
preference:** the windowed form moves the `WT` BUILD inside the second phase, and that
build reads `clk` (`panel_write_factor`) while `Vw` — which the fold reads every
window — ALIASES `clk`'s bytes by the `vw_union_bytes` lifetime argument. Windowing
would make two tenants of one region simultaneously live. Un-aliasing costs `clk` its
own `BT * (BC + 4) * 4` = 4,352 B, and the headroom is not there.

**The decay arm's headroom is 2,941 B**, and it is arithmetic off
`consumer_smem_bytes` rather than a measurement: at the flagship cell
(`BT = 16, BC = 64, DV = 64, D = 2`, `DECAY`, `GLOBAL`, bf16 staging, no `COMPACT`,
`kRing = 1`) that function returns 30,168 B, and 3 CTAs/SM admit
`102400 / 3 - kDriverSmemReservePerCta` = 33,109 B — a difference of 2,941 B, against
`clk`'s 4,352. Every term of both sides is compile-time, so the figure moves only when
the carve does.

## <a id="force-ctas"></a>9. `ROLA_FORCE_CTAS` — the forcing-census instrument

When defined, it replaces the DERIVED launch bound with a fixed CTA count on every
instantiation, so a candidate register budget can be censused over the whole 512-entry
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

# Part II — the shared-memory carve

## <a id="smem-ledger"></a>10. The ledger (the §10i.14 carve, at the at-scale cell's sizes)

```
S      [BC][SP]  fp32   16640 B   the STATE, resident for the owner's whole run
WT     [kWinRows][BTP]   5120 B   write coefficients, LEAF-MAJOR. fp32 on the
                                  tf32x3 arm, which is the arm the fold's full
                                  mantissa is for; bf16 and SWIZZLED on the narrow
                                  arm, kWinRows * BT bf16 = 3072 B
Vw     [BT][VWP]         4608 B   the scale-folded value tile, fp32 on the tf32x3
                                  arm; SWIZZLED and UNPADDED on the narrow arm,
                                  BT * DV bf16 = 2048 B
sr,sw  [BT]      fp32     128 B   the §3 step-3 row scales
den    [BT]      fp32      64 B
cand   2*kMaxWords u32   1024 B
dig    [4][BC]   int     1024 B   per-owner leaf digits (hoisted, §26)
Rf     frag-order       2048 B    read coefficients, MMA-FRAGMENT ORDERED
Gm     [BT][BT]  bf16     512 B   the causal effective-score block
                     = 32704 B = 31.94 KiB -> 3 CTAs/SM x 8 warps = 50 % occupancy
```

**The `WT` and `Vw` byte figures are the `tf32x3` (fp32-staging) arm's**, and the
narrow arm's are given beside them: those are the two rows that move with the staging
format (`stage_is_narrow` selects it, `wt_pitch` / `vw_elems` size them), so quoting
one arm alone under-states one build and over-states the other. Storing the `tf32x3`
arm's two buffers bf16 would save 4096 B and change no residency (`3 x 31.94 = 95.8`
KiB against the SM's 100 KiB — read off the driver, not assumed, because 3 CTAs/SM is
the whole occupancy result); it is refused because it is the arm whose fold precision
must be recoverable.

The carve is BYTE arithmetic, not pointer arithmetic, from `WT` onward: the staged
buffers' element size follows `StageT` while everything around them is fp32, so
`p + count` would silently mis-place every buffer after them. `align4` keeps each
subsequent fp32 buffer 4-byte aligned; every shipped `(BC, BTP)` product is already a
multiple of 2 elements, so it never pads.

The footprint is `constexpr` so the launch bound can be DERIVED from it rather than
restated next to it — two numbers that must agree are one number too many.

### The per-arm carves

- **`DECAY`** adds THREE buffers rather than four: `log_keep`, `p_ref`, `s0`, all
  `[BC]` — 768 B. The naive ledger is those three plus `clk[BT][BC]`, and at the
  at-scale cell `clk` alone is 4096 B against 1429 B of headroom — 36544 B/CTA, i.e.
  `3 * (36544 + 1024) = 112704 > 102400`, which would cost a CTA and a third of this
  redesign's entire occupancy result. It does not have to be paid, and the argument is
  a LIFETIME rather than a saving: `clk` exists only to build `Rf` and `WT`, it is DEAD
  before `Vw` is live, and `Vw` is the larger buffer. So `clk` ALIASES `Vw`'s storage
  and only the three `[BC]` vectors are new, leaving `3 * (32448 + 1024) = 100416 <=
  102400`. **Decay ships at 3 CTAs/SM, full residency.**
- **`COMPACT`** adds `c_idx` + `c_slot` + `c_count` AND NOTHING ELSE — 520 B at
  `BC = 64`, which is what leaves the `decay + compact` arm inside 3 CTAs/SM with
  ~140 B to spare (`3 * (32968 + 1024) = 101976 <= 102400`). READ OFF THE DRIVER on
  every build, never assumed: this is the tightest residency margin in the kernel. It
  is RINGED — the gathered list is a staging OUTPUT (the panel builds and the fold's
  scatter all read it), so a run-ahead gather must not overwrite the live one.

  The DESIGN (§2.18(e)) budgeted a FOURTH buffer, `supp_leaf_r/w` (`2 * BC` u32),
  aliased onto `WT` by a lifetime argument. It does not exist here, and the reason is
  that the support words are consumed by ONE WARP in the same instruction stream that
  produces them (§28), so they live in REGISTERS — which also removes the barrier that
  would have published them and the aliasing argument that would have licensed the
  storage. Same 520 B, one fewer barrier, no alias to keep sound.
- **`GLOBAL`** adds `den[BT]`. The `raw` arm has no denominator stream at all, so
  allocating its row would be dead storage rather than a rounding. `sr`/`sw` are
  STAGING outputs and carry the ring; `den` is not and does not.
- **`ctl`**, K3b's item-stream control block, is `2 + QPO` `int`s on EVERY arm (12 B at
  `BC = 16`, 24 B at `BC = 64`). It is the whole storage cost of the counter-gated
  phase ordering and of the fold's fluid item pool (the item table itself is DERIVED,
  never stored). It is named in the footprint function rather than folded into a
  neighbour because the launch bound is derived from that function and a footprint that
  is not written down is not checkable. READ OFF THE DRIVER, not asserted: the tightest
  3-CTA arm in the matrix is `decay + compact, BC = 64, BT = 32, DV = 32` at
  `3 * (32928 + 1024) = 101856` against the SM's `102400`, i.e. **544 B of total
  slack**. Every other arm has more. `rola_consumer_occupancy` returns the identical
  CTA count on all 32 `(decay, compact, BC, BT, DV)` shapes as the pre-K3b kernel,
  which is the check that matters — the byte count is only the reason to expect it.

  The counters are PACKED rather than bank-separated. Separation would cost `32 * 4`
  bytes PER COUNTER, and the `decay + compact` `BC = 64` arm carries the tightest
  per-CTA residency margin in the kernel — so padding the counters apart would cost a
  CTA to decorrelate a handful of `atomicAdd`s per tile. The traffic they carry is one
  atomic per warp per phase edge plus one per claimed fold item, which is not a
  bank-conflict-shaped problem.

### <a id="staging-set"></a>The STAGING SET, and why it is exactly these objects

The ring's members are the SMEM objects the tile's STAGING phase PRODUCES and the
tile's COMPUTE phase CONSUMES — `Rf`, `Vw`, the two row scales, and under `COMPACT`
the gathered list. Two per-tile objects are deliberately NOT in it, and both exclusions
are structural rather than budgetary:

- **`Gm`** is GEMM 2's OUTPUT, produced in phase B and consumed in phase C of the SAME
  tile. Nothing a feeder does one tile ahead can write it, so a second `Gm` slot would
  be storage with no producer — dead memory, not a smaller saving. (D1.3's table listed
  it because that table enumerated the per-tile SMEM objects, not the staging OUTPUTS;
  this is that distinction landing.)
- **`WT`** needs no slot because the run-ahead's PROGRAM POINT is already past its last
  reader: the staging of tile `n + 1` is issued after tile `n`'s window loop, where the
  Gram and the fold have both retired. `WT` is ringed by the SCHEDULE at depth 1, which
  is strictly cheaper than ringing it in storage.

`den` is written and read inside one tile's compute and is likewise excluded.

## <a id="vw-clk-union"></a>11. The `Vw` / `clk` union, and why it is sized in BYTES rather than elements

`clk` (the tile's per-leaf inclusive clock prefix, `[BT][BC + 4]`, fp32) is DEAD before
`Vw` (`[BT][DV + 8]` on the tf32x3 arm, `[BT][DV]` swizzled on the narrow one) is live,
so it ALIASES `Vw`'s storage and `vw_union_bytes` sizes the region as the MAX OVER ITS
LIVE TENANTS.

**The max is over BYTES, and that is load-bearing because the two tenants do not share
an element size.** `Vw` is bf16 on the narrow arm and `clk` is fp32 on both, so an
element-wise `max` would under-allocate by a factor of two, SILENTLY. The consequence
is a REAL limit on what narrowing `Vw` can save rather than an accounting detail: on
the DECAY arm `clk` is the larger tenant at every shipped shape
(`BT * (BC + 4) * 4` against `BT * DV * 2`), so **narrowing `Vw` frees no shared memory
at all there** and the decay arm's whole staging-format saving is `WT`'s. `Vw` is still
ADDRESSED at its own pitch; only the RESERVATION is the max.

The swizzled narrow `Vw` carries no pad, so its tenant term is `BT * DV` elements
rather than `BT * (DV + 8)` — `16 * BT` bytes per ring slot, on the arms where `clk` is
absent or smaller. On the DECAY arm it frees NOTHING, for the reason above: a `max`
does not move when its smaller argument shrinks.

`clk`'s row pitch is `BC + 4`, token-major (`clk[t * CKP + i]`), on the same
bank-decorrelation argument every other pitch here uses. The alias is sound because
`vw_union_bytes` sizes the region for whichever tenant is larger IN BYTES; the kernel
`static_assert`s that sizing against the use rather than assuming the two agree. `clk`
is declared AT THE CARVE rather than eighty lines later as `clk = Vw`: with two element
types that assignment does not type-check, and the alias is the load-bearing fact. It
names SLOT 0 and needs no ring index: `DECAY` pins `kRing == 1` (§7).

## <a id="bank-predicates"></a>12. The bank-conflict property, as a compile-time proof rather than a comment

Every MMA operand load has 32 lanes addressing SMEM as an affine function of the
fragment's two lane coordinates, `g = lane >> 2` and `r = lane & 3`. The access is
conflict-free exactly when those 32 addresses land on 32 DISTINCT banks. `banks_distinct`
ENUMERATES that condition instead of asserting a modular formula, so each operand proves
its own pitch at compile time and a future change to `BT`/`BC`/`DV` fails the build rather
than silently costing a 4x wavefront amplification — which is what the unpadded build did
measure (§2.2, and §14 below).

`stride_g` / `stride_r` are the address deltas in 4-BYTE BANKS per unit `g` and `r`, which
is why a bf16 operand read as 32-bit pairs contributes `pitch / 2`.

### `banks_distinct_u16` — the same proof for a 16-bit operand is NOT the same predicate

And the difference is not pedantry: `banks_distinct` would REJECT layouts that are
conflict-free and ACCEPT ones that are not, so a bf16 buffer proved with it is not proved.

Two things change. (i) The addresses are 2-BYTE, so the bank of element `e` is
`(e >> 1) % 32` rather than `e % 32`. (ii) Two lanes landing in the SAME 32-bit WORD do
not conflict — the SM services one word in one transaction — so the condition is not "32
distinct banks" but "any two lanes sharing a bank share a word". A 16-bit operand read by
32 lanes typically occupies 16 words in 16 banks with two lanes per word, and that is
conflict-FREE, which is exactly the case `banks_distinct` cannot express.

`e >> 1` is well defined here because every fragment's base offset is EVEN: the bases are
`k0 * pitch + n0` with `k0` a multiple of the MMA k-extent, `n0` a multiple of 8, and every
16-bit pitch this kernel uses even. `stride_g` and `stride_r` are the address deltas in
2-BYTE ELEMENTS per unit `g` and `r`.

## <a id="stage-layout"></a>13. The staged operands' SMEM layout, owned by CuTe

Both predicates in §12 are HAND-ROLLED PROOFS ABOUT A PADDED PITCH, and the argument
against leaving the staged operands on one is that the 32-bit predicate proves the wrong
proposition about a 16-bit buffer in BOTH directions — a class of error an enumeration
cannot warn you about, because you have to already suspect it.

On the narrow arm BOTH staged operands are therefore stored under a SWIZZLED layout the
library states, `composition(Swizzle<B,3,3>, (ROWS,COLS):(COLS,1))`, and the pad
DISAPPEARS: the swizzle decorrelates the banks by permuting 16-byte blocks WITHIN the tile
instead of by making the tile bigger, so the buffer is exactly `ROWS * COLS` elements. That
is a FOOTPRINT change as much as an addressing one — `16 * BT` bytes per ring slot for `Vw`
and `16 * kWinRows` bytes for `WT`.

**One layout serves `WT`'s two operand maps.** Satisfying GEMM 2's B map and GEMM 4's A
map at once with a PAD takes an enumeration (`BT + 8`, unique over
`pad in {4, 8, 12, 16}`); the swizzle satisfies both BY CONSTRUCTION, because `ldmatrix`'s ADDRESSING is the same for `.trans` and `.N` — 8 lanes,
8 rows, one 16-byte column block — and the transposition happens in the registers.
`stage_ldsm_banks_ok` quantifies over every `(8-row group, 16-byte block)` pair, so it
proves BOTH maps at once rather than one map per predicate.

`B` is chosen so the swizzle's XOR field covers the whole 128-byte bank cycle: a row is
`COLS` halves = `COLS / 8` blocks of 16 B, and `B = log2(COLS / 8)` capped at 3 (a 128 B
row already spans all 32 banks). `M = 3` keeps the 16-byte `ldmatrix` row intact; `S = 3`
puts the XOR source at the row index's low bits.

NOTHING HERE IS ASSERTED — the three properties the layout has to have are
`static_assert`ed by ENUMERATING THE LIBRARY'S OWN LAYOUT, which is what makes it more
verifiable than a hand-proved pitch rather than merely shorter.

### <a id="stage-offset"></a>The address function is a closed form, and CuTe is its specification

`StageLayout{}(row, col)` is the definition, but calling it at every access point is NOT
free, and the cost is a MEASUREMENT rather than a worry: with `WT` addressed through it,
`ptxas` spent **+15 to +18 registers** on the `BT = 32` instantiations and
`rola_consumer_occupancy` lost a CTA on 7 of the 32 shapes. The 8-second probe bisected it
exactly — a variant with the swizzled layout and NO `ldmatrix` at all carried the WHOLE
regression, and adding the copy atoms back cost 0-1 registers on top. So the atoms are free
and the generic layout call is not: it blocks the strength reduction the linear map
admitted, on the scalar walks (`WT`'s panel build, the mass fold) whose inner loop is over
a whole row.

`stage_offset` is that same map, in the two operations the swizzle actually is.
`offset = row * COLS + col`, and the swizzle XORs offset bits `[3, 3+B)` with bits
`[6, 6+B)`; the source bits come ENTIRELY from `row` (because `COLS >= 16`) and the target
bits lie ENTIRELY inside `col` (because `COLS >= 8 * 2^B`), so the XOR never carries between
the two fields and the map factors as `row * COLS + (col ^ mask(row))` — a row-invariant
mask the compiler CAN hoist out of a row walk.

`stage_offset_matches_cute` then enumerates EVERY `(row, col)` of the buffer and requires
the two to agree. That is what keeps the library as the authority: the layout is stated in
CuTe's algebra, the kernel's addressing is proven equal to it at every point, and a swizzle
parameter changed in one place and not the other fails the build.

### The three proofs

1. **`stage_layout_is_bijection`** — the layout is a BIJECTION onto `[0, ROWS * COLS)`.
   This is what licenses the reservation `ROWS * COLS * sizeof(bf16)`: a swizzle that
   overflowed its tile would alias the NEXT buffer, and one that under-covered would waste
   the bytes the pad was removed to save.
2. **`stage_ldsm_banks_ok`** — every `ldmatrix` FETCH is conflict-free. The instruction is
   serviced 8 lanes at a time and each lane reads a 16-byte row (4 consecutive banks), so
   the phase is conflict-free exactly when its 8 rows cover all 32 banks. This REPLACES
   `banks_distinct_u16` for the staged operands — it is the same question asked of the
   accesses the kernel actually issues, and it also checks the 16-byte ALIGNMENT
   `ldmatrix` requires, which no pitch predicate could see.
3. **`stage_store_banks_ok`** — the STAGING STORE stays conflict-free. Both staged buffers
   are filled by a flat `idx = row * COLS + col` loop at `kConsumerThreads` stride, so 32
   consecutive lanes write 32 consecutive positions, 2 bytes each, and the same-word rule
   `banks_distinct_u16` established applies — asked here of the swizzled addresses rather
   than of a stride.

**`pitched_ldsm_banks_ok`** is proof 2 asked of a PITCHED (unswizzled) buffer. `WT` keeps
its derived pad, and the fetch it issues is the same instruction `Vw`'s is — 8 lanes,
8 rows, one 16-byte column block per phase. This asks the identical question of
`row * PITCH + col`, and it is what LICENSES the atom at that pitch rather than assuming a
pad chosen for a scalar fragment map also serves a matrix load. `COLS` is the column extent
actually addressed (`BT` tokens), which is smaller than `PITCH`.

## <a id="stage-format"></a>14. The staged-operand storage format, and `wt_pitch`

`WT` (the write coefficients) and `Vw` (the scale-folded value tile) are the two PER-TILE
staged operands. They are stored in the MMA's own format on the bf16 production arm and
fp32 on the tf32x3 exactness arm. The full-mantissa requirement is the EXACTNESS arm's:
that is the arm that exists to be exact, and it is the arm the fp32 storage is for. On
the bf16 arm every MMA consumer of both buffers narrows at
the fragment (`pack_bf16x2(clamp_bf16_finite(x))`, and `pack_bf16x2` is
`__floats2bfloat162_rn` — the same round-to-nearest-even a store applies), so storing them
narrow MOVES THE NARROWING SITE and does not change a value.

`stage_is_narrow` is the CONJUNCTION of both sites, not `READ_PREC` alone, because the two
buffers feed both: `WT` is GEMM 2's B operand (`READ_PREC`) and GEMM 4's A (`FOLD_PREC`),
`Vw` is GEMM 3's B and GEMM 4's B. The two shipped POINTS set both sites together, so this
is the shipped surface stated exactly, not a new axis.

### `wt_pitch` — and the `+8` is the PROOF'S answer rather than a copy of `Gm`'s

`WT` serves two operand maps and the narrow pitch has to satisfy BOTH:

```
GEMM 2 B   B[k * ldb + n]   k = leaf steps 2 per r, n = token per g
GEMM 4 A   A[m * lda + k]   m = leaf per g,         k = token steps 2 per r
```

Enumerated over `pad in {4, 8, 12, 16}` at both `BT`, **`BT + 8` is the UNIQUE pad that
passes both** (`BT + 4` fails the GEMM 4 A map, `BT + 16` fails the GEMM 2 B map), and it
passes at `BT = 16` and `BT = 32` alike.

**`WT` stays PITCHED where `Vw` is swizzled, and that asymmetry is a MEASURED decline
rather than an omission.** `Vw`'s swizzle costs nothing because `Vw` has exactly two
addressing sites — a flat store loop and one `ldmatrix` row address per k-step. `WT` has
six, and two
of them are SCALAR WALKS ALONG A ROW (the panel build's `t` loop and the mass fold's `t`
reduction). A swizzle makes those walks non-affine, which costs the strength reduction the
linear map admits: measured **+15 to +18 registers** on the `BT = 32` instantiations and a
CTA lost on 7 of the 32 `rola_consumer_occupancy` shapes. The closed-form address recovers
most of it and still leaves TWO shapes crossing the 64-register boundary (`!decay !compact
BC16 BT32 DV32` and `!decay compact BC64 BT16 DV32`, both 4 CTAs -> 3), against a footprint
saving of `16 * BC` bytes that the same instrument shows moves NO shape's residency at all.
A rung that trades a real CTA for bytes nothing needs is a decline by the campaign's own
rule.

**What `WT` DOES take is the `ldmatrix` atoms**, and the same bisection is why: a variant
with the swizzle and NO `ldmatrix` carries the whole regression, and adding both copy atoms
back costs 0-1 registers. `ldmatrix` needs 16-byte-aligned 16-byte rows, which THIS PITCH
GIVES (`BTP * 2` is 48 B or 80 B, both multiples of 16), and `pitched_ldsm_banks_ok` proves
the phases conflict-free at it — the derived pitch LICENSES the atom rather than being
displaced by it.

`vw_elems` / `vw_index` and `wt_index` are one decision taken twice: the narrow arm stores
both under `StageLayout` and pays NO pad, the `tf32x3` arm keeps padded row-major buffers.
The 16-bit `ldmatrix` atoms do not apply to an fp32 buffer at all, so this is the
`kNarrowStage` storage-format split and not a second axis. `wt_index` is
row-major at `wt_pitch` on BOTH arms, and it is a FUNCTION rather than an open-coded
`q * BTP + t` because the `ldmatrix` fetches and the two scalar walks must not be able to
disagree about it.

`stage_put` stores a computed fp32 value into a buffer whose element type follows the
precision axis. It is OVERLOADED rather than a `static_cast` because the torch build defines
`__CUDA_NO_BFLOAT16_CONVERSIONS__` — the narrowing must be the explicit intrinsic. On the
narrow arm it is where §12 rule 3's format guard lives, because it is where the fp32 value
last exists. There is exactly ONE clamp on that path: `mma_tile`'s fragment-side clamp is a
no-op on an already-narrow operand (bf16 shares fp32's 8-bit exponent, so a finite bf16
cannot exceed the bound). `gm_put` is the same discipline for `Gm`.

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
swizzle, and §14 gives the measurement that keeps it that way. `Vw`'s three narrow-arm
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
on bf16 and `BT + 4` on tf32x3; both are proven by `banks_distinct` (§15).

## <a id="pass1-alias"></a>19. §6.4 pass 1's own state, aliased onto `Rf`

```
a_state[i]   the per-leaf survival product `A` over the segment. Identically 1 under
             decay = None; under decay it accumulates s0^2 per write tile (§44).
sum_slot[i]  the written-leaf FLAG, rewritten in place by the compaction to the leaf's
             POSITION in the emitted list (-1 if not in it).
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

# Part III — the operand maps

## <a id="frag-order"></a>20. Fragment-order operand synthesis

The generic `mma_tile` fetches each operand element with its own indexed SMEM read, so a
`[16 x 8]` output tile costs ~6 address computations and ~8 loads PER K-STEP, and the A
operand is re-fetched by every warp that shares it (all 8 in the readout). The instruction
mix says that arithmetic — not the loads and not the MMA — is the kernel's leading cost.

We SYNTHESISE the read coefficients in SMEM rather than receiving them, so we can store them
in the order the MMA's A fragment wants: word `w` of lane `lane` at k-step `ks` at
`[ks * 128 + lane * 4 + w]`. A warp then loads a whole fragment with ONE 128-bit `uint4` —
32 lanes x 16 B = 512 B, four full transactions, no bank conflict and no address arithmetic
at all.

BOTH precisions are laid out, because `READ_PREC` is a real axis (§6) and a layout that only
served bf16 would silently un-support the tf32x3 readout. They share the storage SHAPE
(`[ksteps][32][4]` 32-bit words) and differ only in the compile-time forward map, so there is
one path, parameterized — not a fast path with a fallback.

```
bf16 m16n8k16 : a0..a3 = (g, 2r|2r+1), (g+8, 2r|2r+1), (g, 2r+8|..), (g+8, ..)
                -> m = g + 8*(w&1),  k = 16*ks + 8*(w>>1) + 2r   (a PAIR)
tf32 m16n8k8  : a0..a3 = (g, r), (g+8, r), (g, r+4), (g+8, r+4)
                -> m = g + 8*(w&1),  k =  8*ks + 4*(w>>1) +  r   (a single)
```

The `m` map is identical; only the `k` step and the element count differ. `frag_a_slot` is
the FORWARD map (slot -> `(m, k)`), and `frag_a_words(K)` is the 32-bit word count of a
fragment-ordered A operand contracting over `K`, FOR ONE `m16` ROW BLOCK. A `BT`-token tile
holds `NM = BT / 16` of them, laid out block-major at a FIXED stride of `frag_a_words(BC)` —
fixed rather than packed to the compacted `Kpad`, so a block's base address is a compile-time
multiple and the MMA's `ks` offset is independent of the block count.

Because the coefficients are fragment-ordered they carry NO ROW PITCH at all — the layout is
conflict-free by construction rather than by padding, so there is no pitch to prove.

## <a id="frag-a-get"></a>21. `frag_a_get` — the inverse, and why the mass column pays it

Reads element `(m, k)` back OUT of a fragment-ordered A operand. The mass column's denominator
is a `[BT x BC] x [BC x 1]` matvec, one column out of `DV + 1`, and giving it its own `n8` MMA
tile would waste 7/8 of a fragment. So it stays a scalar reduction and pays this inverse
instead: ~10 integer ops on `BT * BC` elements spread over `BT` threads, once per tile.

**HAZARD `frag-a-get`.** `frag_a_get` is the hand-rolled INVERSE of `frag_a_slot`, and
nothing checks the pair against each other — the CuTe cross-checks in §23 pin the library's
maps, not this one. The two must be edited together: an inverse that disagrees with the
forward map reads the wrong element and produces a plausible, wrong denominator with no
diagnostic anywhere.

## <a id="mma-tile"></a>22. `mma_tile` — the generic `[16 x 8]` tile

`D[16x8] += A[16 x K] * B[K x 8]`, fp32 accumulate, one warp.

`TA` / `TB` are the operands' SMEM STORAGE types and are independent of `PREC`. Which types
they are is `stage_is_narrow`'s answer, not `mma_tile`'s, and it FORKS BY ARM: on the
`tf32x3` arm `WT` and `Vw` are stored fp32, and the same buffers feed the bf16 intra GEMMs by
narrowing at the fragment, which costs nothing extra (the pack was already a cast); on the
bf16 arm `stage_is_narrow` makes both buffers narrow and `mma_tile` receives them already in
the MMA's format (§14).

**The full-mantissa requirement is the `tf32x3` arm's alone.** Narrowing THAT arm's two
buffers would save 4 KiB and make its fold precision unrecoverable, which is the point of
having an exactness arm at all; it says nothing about the bf16 arm, where every consumer
narrows at the fragment regardless and storing narrow moves the narrowing site without
changing a value.

`A_T` / `B_T` select the operand's storage ORIENTATION, because the four GEMMs of a tile need
three different ones and a transposed COPY of a `[BT x BC]` panel would be pure SMEM traffic:

```
A_T == false : A element (m, k) at A[m * lda + k]
A_T == true  : A element (m, k) at A[k * lda + m]
```

and likewise for B with `(k, n)` and `ldb`. Fragment index maps are the PTX
`mma.sync.aligned` layouts, taken verbatim from `readout_mma.cuh` (`g = lane >> 2` is
`groupID`, `r = lane & 3` is `threadID_in_group`) rather than re-derived — that header's maps
are the ones the shipped oracle gates pass on.

§12 rule 3's clamp applies on the bf16 arm only, for the same reason `readout_mma.cuh` gives:
bf16 narrows the EXPONENT range, so the gap between bf16's max finite and fp32's rounds to
`inf`; tf32 carries fp32's 8-bit exponent and a clamp there would be dead code.

### <a id="mma-tile-fraga"></a>`mma_tile_fragA` — the same tile with a fragment-ordered A

`d += Af . B` with A FRAGMENT-ORDERED. The A operand costs one `uint4` per k-step and zero
address arithmetic; B keeps the generic indexed path, because B is `n`-indexed per warp and
each warp wants a different slice of it.

`B_GATHER` is COMPACTION's readout indirection: the A operand's `k` is a position in the
compacted leaf list while `B` (the state) stays LEAF-indexed and persists across tiles, so row
`k` of B is `b_row[k]`. It is a TEMPLATE FLAG rather than a `b_row != nullptr` test because
the non-compacted arm must not carry the load.

**The k-loop keeps a compile-time trip count AND an early exit, never a runtime bound.**
`KMAX` is the operand's allocated extent (`BC`, or `BT` for the intra term) and `K` is the
realized one; a runtime loop bound around an MMA sequence is un-unrollable and costs more than
the k-steps it skips (measured on a ragged-`K` loop; provenance: journal). With
`COMPACT = false` the caller passes `K = KMAX`, the compare folds away and the emitted code is
the uncompacted loop exactly.

The `[16 x 8]` D/C fragment's element map, shared by every writeback in the kernel, is
`d0, d1 -> (row g, col 2r, 2r+1)` and `d2, d3 -> (row g + 8, col 2r, 2r+1)`.

## <a id="ldsm-atoms"></a>23. The `ldmatrix` atoms

### `vw_frag_b` — `Vw` as B, fetched by one `ldmatrix.x2.trans`

`Vw` is B for BOTH of the tile's `Vw` consumers — GEMM 3 (the intra term, `y += G . Vw`) and
GEMM 4 (the fold, `S += W^T . Vw`) — and both want the same 16(k) x 8(n) block per k-step. The
generic path builds that block with FOUR scalar SMEM loads, four `mma_src` widenings, four
clamps and two packs PER LANE PER K-STEP; `ldmatrix.x2.trans` is ONE instruction that delivers
the same two registers, and the `.trans` is exactly right because `Vw` is stored `k`-major
(`n` contiguous) while the fragment wants `k`-adjacent pairs at a fixed `n`.

**Bit-identity, which is why this is a LAYOUT change and not a VALUE change.** `ldmatrix`
delivers the STORED 16-bit patterns. The generic path delivers
`pack_bf16x2(clamp_bf16_finite(__bfloat162float(x)))`, and those agree bit for bit iff `x` is
a finite bf16 — which the STORE guarantees: `stage_put` applies §12 rule 3's
clamp where the fp32 value last exists. A finite bf16 widens to fp32 exactly, passes the
clamp unchanged and rounds back to itself, so the composition is the identity and no summation
is reordered — the k-loop, its extent and its order are the generic path's, unrolled the same
way.

### `wt_frag_b` — `WT` as GEMM 2's B

The SAME atom against the SAME algebra: `WT` is stored leaf-major (`k` = leaf, `n` = token
contiguous) and the fragment wants `k`-adjacent pairs at a fixed `n`, which is `.trans` again.
The only difference from `Vw` is which axis is called what.

### `wt_frag_a` — `WT` as GEMM 4's A, and this is the atom's OTHER form

Here `WT` is `k`-ADJACENT (`m` = leaf per `g`, `k` = token, contiguous), which is the layout
`ldmatrix` reads WITHOUT transposing — so the fold's A operand is one `ldmatrix.x4` for the
whole `[16 m x 16 k]` block, against the generic path's eight scalar loads, eight widenings,
eight clamps and four packs per lane.

The addressing is the atom's own: `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8 MATRICES
STACKED (32 rows of 8 halves), not a 16x16 block, so lane `l` addresses matrix `l >> 3`, row
`l & 7` — and the `(m, k)` reading of the four matrices is the CALLER's: matrix `i` is
`m = 8 * (i & 1) + row`, `k = 8 * (i >> 1) + col`. That reading is what makes the delivered
registers `a0..a3` in `mma_tile`'s order, and `ldsm_n_dst_map_matches` checks it over all 256
slots.

### `mma_fold_tile`, `mma_tile_vwB`, `mma_tile_fragA_wtB`

- `mma_fold_tile` is GEMM 4 with BOTH operands fetched by `ldmatrix`: A = `WT` (`.N`),
  B = `Vw` (`.trans`). NARROW ARM ONLY.
- `mma_tile_vwB` is `d += A . Vw` for a generic (indexed) A and `Vw` as B — GEMM 3. NARROW ARM
  ONLY: the 16-bit atoms have no fp32 form, and `kNarrowStage` is precisely the arm on which
  `Vw` is 16-bit. `K` is `BT` at both call sites and the trip count stays compile-time.
- `mma_tile_fragA_wtB` is GEMM 2 with a fragment-ordered A and `WT` fetched by `ldmatrix`.
  `KMAX` is `WT`'s WINDOW row count and `K` the realized (causal, compacted) prefix, so the
  early exit, its compile-time trip count and the k-order are the generic path's, unchanged —
  the ONLY thing that moves is how the four B halves reach the registers.

  `static_assert(KMAX % 16 == 0)` is what makes the fetch legal: `ldmatrix` reads a whole
  16-row block, so a `KMAX` that was not a multiple of the k-quantum would read past the buffer
  on its last step. Every shipped window row count is a multiple of `kMmaKQuantum` and the
  assert holds a future one to it.

### <a id="cute-crosschecks"></a>The two library maps this file depends on, enumerated at compile time

`MMA_Traits::BLayout` is `(thr, val) -> n + 8k` over the atom's `[8 n x 16 k]` B tile;
`Copy_Traits<SM75_U16x4_LDSM_T>::DstLayout` is `(thr, val) -> BIT` over the same 16x8 half
tile. Both must equal `k = 2r + (v & 1) + 8 * (v >> 1)`, `n = g` — which is `mma_tile`'s
`b0`/`b1`, letter for letter. `lane & 15` is the `.trans` atom's own source map
(`Copy_Traits::SrcLayout` gives the first 16 lanes an address each, at stride one row, and
aliases lanes 16-31 onto them). For the `.N` atom the source map must be `row = lane`, so that
lane `l` addresses matrix `l >> 3`, row `l & 7`.

| assert | message |
|---|---|
| `mma_b_map_matches()` | CuTe's B fragment map is not `mma_tile`'s |
| `ldsm_t_dst_map_matches()` | `ldmatrix.x2.trans` does not deliver `b0`/`b1` |
| `ldsm_t_src_map_matches()` | `ldmatrix.x2.trans` row addresses moved |
| `ldsm_n_dst_map_matches()` | `ldmatrix.x4` does not deliver `a0..a3` |
| `ldsm_n_src_map_matches()` | `ldmatrix.x4` row addresses moved |

A library bump that moved either map fails the BUILD rather than the oracle.

### <a id="cute-pin"></a>The vendored CuTe pin

`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5. It is a CODEGEN INPUT, not a
packaging one — every line instantiated here is code `ptxas` assembles — which is why
`setup.py::_gate_vendored` hashes the subtree BEFORE the compile and `tools/ratify.py` carries
the same include path. It is used for exactly two things, and both of them are LAYOUT: `Vw`'s
swizzled SMEM layout and the `ldmatrix` copy atom that fetches its B fragment. The `mma.sync`
calls stay hand-rolled PTX; the correspondence between the two is `static_assert`ed against
`MMA_Traits::BLayout` rather than commented.

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

### `stage_tile(tile, slot, is_read, is_write)`

Everything the tile's COMPUTE consumes but does not produce: the gather, the clock prefix, the
row scales, the read panel, the write panel at the plane, and the value tile. At `kRing == 1`
it is called at the top of the tile body and `slot` is identically 0, so the emitted code is
the un-ringed one with the braces moved. At `kRing == 2` it is called ONE VISIT EARLY, from
inside the previous tile's compute (§31).

The schedule bits arrive as ARGUMENTS rather than being re-derived from the candidate words: at
depth 1 the caller has just computed them, and a second SMEM read costs the
`BC = 64, BT = 16, DV = 32` global arm 2 registers — which is the 64-register allocation
boundary and therefore its 4th CTA.

### 64-bit address arithmetic, paid once per tile instead of per element

The amplitude streams are `[BH, n_tiles, BT, pitch]`, so the row base needs 64 bits, but the
offset WITHIN a tile does not — it is `t * pitch + digit`, bounded by `BT * pitch`. Folding the
64-bit part into a base pointer once per (tile, level) leaves the inner loop in 32-bit index
arithmetic.

**It stands ONLY because `D` is compile-time, and that dependence is measured (§2.4).** With a
RUNTIME `p.D` the level loop cannot unroll, so these arrays are dynamically indexed and `ptxas`
puts them in LOCAL memory — stack frame 16 -> 80 bytes, exactly the two 4-pointer arrays, and
the kernel is SLOWER for the hoist. `D` is compile-time here,
every level loop is `#pragma unroll`ed to a compile-time trip count, and the arrays are
therefore statically indexed and live in registers. This is also what `D` being a template axis
buys beyond the dispatch: it is a topology fact with four legal values (spec §1), not a
threshold, so branching on it is the standing "branch on structure, not thresholds" rule rather
than an exception to it.

## <a id="compact-gather"></a>29. `COMPACT` — §3 step 2's support-driven gather

`COMPACT` is spec §6's sparsity quantum (iii), and in THIS kernel's shape it is a COMPUTE CUT
rather than a trip-count cut: the gathered leaf list shrinks the READOUT's k-extent, the intra
Gram's k-extent (through §5's causal sub-list) and the FOLD's m-extent. A cost model built for
a scalar-walk consumer prices compaction against terms this kernel does not have (a scalar
walk's trip count, a dense-panel-to-compacted COPY, a staging-ring probe, a third barrier), so
the axis is derived from THIS shape's arithmetic and not inherited.

Three steps, none of them a branch on a density:

- **(a')** the FIXED levels' AND, folded into TOKEN VALIDITY. A fixed level cannot distinguish
  two leafs of this owner — but a zero there kills the whole token's row scale, so ANDing it
  into `valid_tokens` shrinks every leaf's support word identically and is EXACT. Without it a
  `branches=[8,16], BC=16` owner's whole width-8 level is invisible to the compaction.
- **(b)** the per-leaf AND over the SPANNED levels -> `R_s`, `W_s`.
- **(c)** the two-pass ballot -> the ASCENDING list `c_idx[0..K)`, with §5's causal sub-list as
  its PREFIX `[0, KI)`.

**The support words never reach shared memory.** Warp 0 computes them into REGISTERS
(`kRounds = ceil(BC/32)` per lane, a compile-time count, so they are statically indexed — the
same requirement §2.4's rejected base-pointer hoist violates) and consumes them in the ballot
two instructions later. The design (§2.18(e)) budgeted them as a STAGED `supp_leaf_r/w` pair
aliased onto `WT` by a lifetime argument, for a CTA that computes them across the whole block.
ONE WARP computes them here, so neither that storage nor the barrier that would publish it
exists. Same 520 B of SMEM, one fewer barrier per candidate tile.

Cost is `(D - j) * BC` support-word loads on ONE warp per tile, against the panel builds'
`BT * BC * (D - j)` amplitude loads on all eight — ~1/16 of the build's traffic even before
compaction shrinks the build itself.

### Membership is the UNION, and the intra test is exact in ONE direction

Membership is `(R_s | W_s) != 0`, **never `W_s` alone**. The backward iterates
`support(W) UNION support(R)` per chunk, because an untouched-but-READ leaf has a zero forward
summary and a NONZERO backward covector — a write-only compaction would break the backward
while passing every forward gate on a zero initial state.

§5 (NORMATIVE): a leaf enters the intra gather only if it is read AT OR AFTER its first in-tile
write. `ww & -ww` is the first write token; negating it sets every bit at or above it. The test
is exact in ONE DIRECTION: over-inclusion is a pure performance loss (a leaf with no in-tile
write has a zero write panel; one written only after `t` is masked out by the Gram's `u <= t`),
while UNDER-inclusion drops state. That asymmetry is also what licenses the intra GEMM to run
over `ceil(KI/kUnit) * kUnit >= KI` k-steps: the over-run reaches leaves in `[KI, K)`, every one
of which contributes exactly zero.

The pad `c_idx[q >= K] = 0` keeps the gathered B-operand load ADDRESSABLE. The A operand's
columns `[K, Kpad)` are ZEROED at the panel build, and that — not the pad row — is what makes
the padded product zero.

`c_slot[i]` is the inverse, `-1` when leaf `i` is inert for the whole tile. It is read ONLY by
the decay clock prefix, which is where the scatter-fill identity is realized (§30).

With `COMPACT` compiled out, the gather's answer is its `K == BC` degeneracy, known at compile
time: every leaf is active, §5's causal sub-list is the whole ascending list, and
`c_idx[k] == k`. Every site resolves to the identity WITHOUT reading SMEM, which is what makes
`COMPACT = true` BIT-IDENTICAL to `COMPACT = false` on a tile whose gather IS dense, rather
than merely equal to it.

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

`COMPACT` walks the GATHERED LIST, so a dropped leaf's clock is never formed and its `clk` /
`p_ref` / `s0` are never written. Nothing reads them, because every consumer of those three is
itself compacted (both panels, the entry sweep, the fold's epilogue, the summary's test and
`a_state`).

A leaf the gather dropped has `c == 0` at every token of this tile, because
`build_clock_stream` enforces `stored != 0 => live != 0` per (level, digit) — so a leaf with no
live support on EITHER side has a zero clock product at every token, its true prefix IS zero,
and `keep^0 == 1` makes every factor this block would have computed exactly `1`. **That is what
makes the compacted entry/exit scaling exact: the leaves the fold never touches are exactly the
leaves whose `s0` is exactly 1.**

### <a id="compact-clock-list"></a>HAZARD `compact-clock-list` — it is a LIST RESTRICTION and NOT a predicate

Keeping the `[0, BC)` walk and PREDICATING the clock product on `c_slot[i] >= 0` is the same
arithmetic but NOT the same EXPRESSION: putting the product inside a branch blocks the
`acc += a0 * a1` FMA contraction the unpredicated form gets, so the compacted arm computes a
different (equally valid) `P[t,i]`. That divergence is REAL rather than hypothetical — the
dense-degeneracy gate measures it as a ~7e-09 state difference on the `D >= 2` decay cells and
NOT on `D = 1`, which is the signature, since at `D = 1` there is no product to contract.
Written as a LIST RESTRICTION the inner expression is textually identical on both arms and the
degeneracy is bit-identical.

### Step (c) — the entry state enters the reference frame

The carry term is `R~[t,i] * (s0[i] * M_entry[i])`, so `s0` multiplies the STATE, not the read
panel. The mass column (`c == DV`) rides the same sweep because it is the same recurrence with
`v := 1`.

Row-per-warp rather than a flat `idx / cols` walk: under `global` `cols = DV + 1 = 65` is not a
power of two, so the flat form would pay an integer division per element — ~17 of them per
thread per tile, on hardware with no integer divide (§2.4 measures the sequence at 20
instructions a time).

Under `COMPACT` the sweep runs over the GATHERED rows only, and that is EXACT rather than
approximate: a dropped leaf has `s0 == 1` exactly (the scatter-fill theorem), so the rows it
skips are the rows whose scale is the identity. It is also what keeps the entry and EXIT
scalings on the same row set — the exit scale rides the fold, which is compacted.

## <a id="row-scales"></a>31. §3 step 3's row scales, computed above the panel builds

`sr[t]` and `sw[t]` are the tile's per-token row scales: the read/write gain times the FIXED
levels' amplitudes at the owner's own digit. They are constant along the leaf axis, which is why
they are held out of the coefficient blocks at all (`readout_mma.cuh`'s argument), and they
depend on NOTHING the tile computes: not the gather, not the clock prefix, not either panel.
Only on `(tile, token)` and the owner's `base[]`.

**What their placement buys, stated as the PHASE count rather than as a barrier count**, because
the two differ per arm. Computing them WITH the panels makes the pre-GEMM section THREE phases —
`[gather] [panels + row scales] [Vw]` — with a middle barrier that exists for one reason: to
publish `sw[t]` to the `Vw` writers, which are different threads (`sw`'s loop is `t = tid`,
`Vw`'s is `t = tid / DV`). Computing and publishing the row scales FIRST takes `Vw` off the
panels and the section is TWO phases — `[gather + row scales] [panels + Vw]`. On the `COMPACT`
arms that is one barrier fewer outright; on the plain arm it is barrier-neutral and buys the
`Vw` staging's global loads an overlap with the panel builds' amplitude loads instead of a
rendezvous between them.

**The one arm that keeps the middle barrier is `DECAY`**, and it is a LIFETIME constraint rather
than a publication one: `clk` ALIASES `Vw`'s bytes, both panel builds READ `clk`, and the `Vw`
writers would clobber it. That barrier stands AT ITS REASON (§33) rather than as an incidental
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
Fixed levels (`l < j`) hoist into the row scale.

The loop walks FRAGMENT SLOTS, not `(token, leaf)` pairs: `frag_a_slot` is the FORWARD map, so
the store is `Rf[s]` with `s` contiguous in `tid` — conflict-free, one 32-bit store, no scatter.
Walking `(t, i)` and inverting would have been the same arithmetic backwards and a scattered
store. On bf16 one slot is an ADJACENT PAIR on `k`, which is why the map packs two leaves per
word and the loop runs half as many iterations.

The `m16` row blocks are an OUTER, COMPILE-TIME loop, so at `NM == 1` this is the single-block
loop verbatim — no division by a runtime word count, and no per-slot block arithmetic on the
cell the baseline was measured at.

**COMPACT: the panel's column `k` is a POSITION in the gathered list**, so the leaf it holds is
`c_idx[k]` — and columns `[K, Kpad)` are the RAGGED-K pad, ZEROED here. The A-side zero is what
makes the padded product zero; the B side only has to be addressable, which `c_idx[q >= K] = 0`
gives it. **This is the §2.6 dividend: we SYNTHESISE the coefficients, so we build them
compacted in the first place.** A consumer that RECEIVES its panel dense has to build a
`[BT x BC]` panel and COPY it into a compacted view — 8 KiB and an extra barrier, neither of
which exists here. `COMPACT` builds `frag_a_words(Kpad)` slots instead of `frag_a_words(BC)` — the
k-extent cut, realized at the BUILD as well as at the MMA. That bound is a runtime value, which
is fine HERE (it is a plain loop, not an MMA sequence); the MMA's own trip count stays
compile-time with an early exit.

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
exactly.

**DECAY step (d):** `fw[t,i] = keep_i^{-(P[t,i] - P_ref[i])}`, folded into the WRITE
coefficient. THE SAME `WT` THEN SERVES BOTH GEMM 2 (the causal Gram, where it meets `Rf`'s `fr`
and the two reconstruct `survival(j -> t)` exactly) AND GEMM 4 (the fold, where it meets the
state's `s0`). One panel, two consumers — rule 1's half-mass shift is what makes that possible
and it is why decay needs no second `[BT][BC]` buffer.

This is the DIVISION by cumulative survival that §12 rule 3 exists for, and because the
production fold is bf16 (§6) rule 3's second site — the `clamp_bf16_finite` at the narrowing
point — is LIVE on the WRITE side of the shipped arm, which is rule 3's own motivating case.
Both sites are required and both are live: the exponent clamp bounds the value
at `e^88 = 1.65e38` (finite in fp32 AND in bf16, which shares fp32's 8-bit exponent), and the
format guard at the MMA catches anything the accumulated product does downstream.

`COMPACT` builds only `KpadM` rows — `K` real ones plus the fold m-tile's ragged pad, ZEROED.
`WT` needs NO indirection at either consumer once it is built this way: GEMM 2 reads it as
`[k][n]` with `k` the compacted position, GEMM 4 as `[m][k]` with `m` the compacted position.
That is the second half of the §2.6 dividend: a separate compacted COPY is needed only by a
consumer that receives its panel dense.

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

`__any_sync` names ONLY this row's lane group: a `COMPACT` pad row `continue`s above, and `q` is
group-uniform, so the whole group leaves together and the group that stayed is converged. One
lane per row publishes, so `sum_slot` takes no intra-tile race (`c_idx` is injective, and
without `COMPACT` `i == q`).

### <a id="wt-build-site"></a>HAZARD `wt-build-site` — where the plane's build may stand

The plane's call site is INSIDE `stage_tile` **only at depth 1**, and the reason is a RACE
rather than a preference. `WT` is single-buffered at every ring depth, and its readers — the
Gram and the fold — retire ASYNCHRONOUSLY: the fold is a claim pool, so a warp that exhausted
the pool can reach the staging point while another warp is still executing its last claimed
item. A run-ahead build there would overwrite rows that warp is reading.

At depth 2 the build therefore sits in the tile's OWN body, above the publication rendezvous,
which costs NO barrier at all: the PREVIOUS tile's end-of-tile barrier stands between the last
fold read of `WT` and this write, and the publication rendezvous immediately below stands
between this write and the Gram's read. The operands it needs from the staging (`c_idx`, `K`)
were published by that same previous barrier.

At depth 1 it must be inside the staging: `DECAY` pins the plane and its build reads `clk`,
which the lifetime barrier below kills. Hoisting it out of the window loop when the window IS
the plane is therefore not cosmetic — it is what keeps the `DECAY` arm's build ABOVE that
barrier, where `clk` is still live. `DECAY` pins `kWtPlane`, so the two facts are one fact.

## <a id="clk-dead"></a>34. HAZARD `clk-dead` — the `clk -> Vw` lifetime barrier, and it is DECAY's alone

`clk` (the tile's per-leaf inclusive clock prefix) ALIASES `Vw`'s bytes (§11). Both panel builds
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
  `await_count`. That is the always-runnable clause of the deadlock argument (§37), discharged
  for the one barrier that stands in the interior of a tile.

`nxt` is CTA-uniform (`next_cand` reads only SMEM candidate words), so the guard cannot diverge
across the barrier inside.

The gathered list is READ before the publication rendezvous at BOTH depths, and for the same
reason each time: at depth 1 the gather's own `COMPACT` barrier stands behind it, and at depth 2
the PREVIOUS tile's end-of-tile barrier does (this slot was gathered one visit ago).

---

# Part V — the item work stream and the barrier protocol

## <a id="item-stream"></a>36. The control block and its two primitives

The tile's intra-CTA phase ordering is carried by COUNTERS rather than by `__syncthreads()`. The
point is not to spell a barrier differently: a warp that reaches a phase edge early goes and
executes another WORK ITEM instead of stalling on a rendezvous, which is what makes the fold a
fluid pool. A rendezvous there leaves 6 of 8 warps idle through the Gram (measured; provenance:
r1 finding A).

**A WORK ITEM is one `(GEMM, warp-tile)` pair with a WHOLE `k` extent (design D1.1).** Full-`k`
is a BIT-IDENTITY requirement and not a convenience: an item fixes its own accumulation order
internally, so WHICH warp runs it is arithmetically irrelevant, while splitting a `k` extent
would be split-K INSIDE the tile and would move `y`'s summation order. Nothing here splits one.

**The item TABLE IS DERIVED, NOT STORED (design D1.2):** the fold's item count is
`(m-tiles in the window) * NT_DV` and a claimed index decodes to `(m-tile, n-tile)` by the same
strided arithmetic a warp loop walks. Cost is one `atomicAdd` per claim and ZERO shared memory
beyond the counters.

```
kCtlReadDone   warps that have retired the READOUT phase, i.e. finished every read of S this
               tile makes. Gates the fold, which WRITES S. (This is B6, a counter and not a
               barrier.)
kCtlGramDone   warps that have retired the GRAM phase. Gates phase C, which READS Gm. (B5.)
kCtlFold0 + w  window w's fold item-claim cursor -- the pool's whole state. One per window.
```

Both PHASE counters count WARPS and every warp retires both UNCONDITIONALLY, which is what makes
the deadlock argument (§37) a two-line one.

The fold's claim cursors are ONE PER WINDOW, because a narrowed `WT` makes the fold's item pool
per-window: window `w`'s pool must start at zero, and resetting a shared cursor mid-tile would
need a rendezvous to do it safely — which is the thing this whole rung is removing. `QPO` of
them is the compile-time MAXIMUM (`kNWin <= kQPO`), it is `BC / kMmaKQuantum <= 4`, and the
unused tail costs 4 bytes each on the arms that do not window. I1-generic: an expression in `BC`
and `kMmaKQuantum`, never a literal.

### <a id="release-acquire"></a>HAZARD `release-acquire` — the fence order in `retire_item` / `eligible`

**RELEASE (`retire_item`).** The fence comes FIRST, so every shared store the warp made for the
phase it is leaving is visible CTA-wide before the counter that advertises them moves. The
`atomicAdd` is executed by lane 0 ALONE because the counters count WARPS — a per-lane count would
make the target a lane count and the phase fact would stop being a phase fact.

**ACQUIRE (`eligible`).** The load is `volatile` because the counter is written by OTHER warps
and a cached load would spin forever; the fence comes AFTER a successful test, so no subsequent
shared load of the published object can be hoisted above the observation that published it.
WARP-UNIFORM by construction — every lane loads the same address, so the predicate is identical
in all 32 lanes with no shuffle.

`await_count` is the spin, written in terms of `eligible` so that the release/acquire discipline
lives in exactly TWO functions and can be audited there rather than at every phase edge.
`__nanosleep` backs the polling warp off the issue port so a spinning warp does not compete with
the warps it is waiting for.

## <a id="ctl-reset"></a>37. The tile's control-block reset, and the deadlock argument

The reset needs no barrier of its own on EITHER side, and both halves of that are PROGRAM-ORDER
facts rather than hopes:

- BEHIND it stands the previous tile's end-of-tile `__syncthreads()` (or, for the first tile,
  the CTA-init one), so no warp can still be reading last tile's counters;
- AHEAD of it stands the staging publication `__syncthreads()`, which is unconditional and sits
  between this store and the first `retire_item`. It publishes the zeroed counters on the same
  trip it publishes `Rf`, `WT` and `Vw`.

### <a id="edge-dag"></a>Deadlock freedom, as the edge DAG and the always-runnable claim

NODES (per tile): `STAGE` (gather / clocks / row scales / the three panel builds), `A` (readout +
mass term 1), `G` (Gram + `Gm` publication), `F` (fold items + mass fold), `C` (intra + mass
term 2 + emission), `B7` (end of tile).

| edge | mechanism |
|---|---|
| `STAGE -> A`, `STAGE -> G_0` | the staging `__syncthreads()` (hardware) |
| `G_w -> F_w`, `F_w -> G_{w+1}` | the window's SLOT-READY / SLOT-FREE `__syncthreads()` pair (`!kWtPlane` only; window `w+1`'s build overwrites `WT`'s rows) |
| `A -> F` | `kCtlReadDone` reaching `kWarps` (readout READS `S`, the fold WRITES it). Taken ONCE, in window 0 |
| `G -> C` | `kCtlGramDone` (`G` writes `Gm`, `C` reads it). `G`'s publication rides the LAST window. The mechanism FORKS on `kNarrowGate` and both forms carry the SAME edge (§38) |
| `F -> B7`, `C -> B7` | program order into the end-of-tile barrier |

The relation is a TOTAL ORDER on `{STAGE, {A, G_0}, F_0, G_1, F_1, ..., C, B7}` and therefore
ACYCLIC. (`A` is unordered with respect to every `G_w` — the Gram reads `Rf` and `WT` and touches
neither `S` nor `Gm`'s readers — and so are `F` and `C`: `F` writes `S`, `C` reads `Gm`, `Vw`,
`sr`, `den`, and the sets are disjoint. That is the overlap this rung exists to expose.)

**ALWAYS-RUNNABLE**, and it is two facts:

1. **Every increment a waiter needs is issued, in program order, before any warp can reach the
   wait that needs it.** For `kCtlReadDone` that is the unconditional retirement every warp
   executes above the window loop, target `kWarps`. For `kCtlGramDone` it holds on BOTH ARMS of
   the `kNarrowGate` fork, because on each the retiring set and the target are derived from ONE
   expression rather than chosen separately — wide: every warp retires, target `kWarps`; narrow:
   `mixed && warp < kGramWarps` retires, target `kGramWarps`. Taking the narrow arm, which is
   the one with a guard to discharge:
   - both guards are CTA-UNIFORM in `mixed` (it is `is_read && is_write`, the tile's schedule
     bits) and compile-time-comparable in `warp`, so either the whole CTA takes the counter or
     none of it does. A tile cannot have waiters without retirers;
   - on a `mixed` tile warps `0..kGramWarps-1` execute the retirement inside `if (w + 1 ==
     kNWin)`, which every warp reaches because `kNWin` is CTA-uniform, and that is strictly
     before the wait in program order. The counter is monotone with target `kGramWarps`, so the
     wait terminates after exactly `kGramWarps` retirements;
   - on a `!mixed` tile nobody retires, nobody waits, and the word stays at the reset value —
     which is re-established next tile by the same unconditional store, on a path every warp
     reaches whichever branch it took.
2. **No `__syncthreads` is ever reached while another warp spins.** Every warp executes the same
   `kNWin` iterations and the same barriers within them (`is_write`, `w` and `kNWin` are all
   CTA-uniform), and the only wait inside the loop — `kCtlReadDone` in window 0 — is on a counter
   every warp has already retired, ABOVE the loop. So a warp sitting at a SLOT-READY or SLOT-FREE
   barrier is never waiting on a warp that is spinning on something that warp itself must supply.

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
edge DAG — is VACUOUS there. `Gm` is born and dies inside one tile (it has no ring slot, §10) and
the next tile's stores are ordered behind the end-of-tile `__syncthreads()`, so nothing outside
the tile observes it either. The counter word is re-zeroed unconditionally by every warp
regardless of the branch taken, so the skip path leaves nothing inconsistent: it bypasses
`retire_item`, `eligible`, `await_count` and the word itself.

### <a id="gram-gate-guard"></a>HAZARD `gram-gate-guard` — the retirement predicate mirrors the STORE guard, never the GEMM guard

The GEMM carries an extra `kw > 0` — the causal-prefix early exit — that the stores do NOT: a
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

`COMPACT` cuts the K EXTENT on both `NM` arms — `ceil(K/kUnit)` k-steps instead of `BC/kUnit` —
and gathers the B operand's rows through `c_idx`, because the state is LEAF-indexed and persists
across tiles while the panel's `k` is a per-tile list position.

## <a id="mass-term1"></a>41. The mass column, TERM 1 — and its placement is FORCED

```
den[t] = scale_read[t] * ( sum_i R[t,i] * S[i][DV]
                         + sum_{u <= t} G[t,u] * scale_write[u] )
```

— the same split the value term takes, with `v := 1`. It is ONE column out of `DV + 1`, so it is
~1.5 % of the tile's arithmetic; making it an MMA `n` would waste 7/8 of an `n8` fragment (the
same argument `frag_a_get` records, §21).

**The two terms STRADDLE the fold, and the windowed phase B is what forces it.** The fold's own
mass recurrence WRITES `S[i][DV]`; the fold shares a phase with the Gram, so term 1 — which
READS `S[·][DV]` — must already have run. Its operands
(`Rf`, `S`, `c_idx`) are exactly the readout's, so it merges with a walk that exists rather than
adding one, and it needs no extra shared memory.

`COMPACT`: `K` terms instead of `BC`, gathering the mass column through `c_idx` exactly as the
readout's B operand does. `Rf`'s block for token `t` is selected by `t >> 4`; at `NM == 1` the
arithmetic folds away entirely.

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

`COMPACT` cuts this k-extent to §5's CAUSAL PREFIX, `ceil(KI/kUnit)` k-steps. No gather: `WT`'s
rows are already the compacted positions. The pad rows `[KI, KIpad)` are leaves in `[KI, K)`
(real, but non-causal, so exactly zero under the `u <= t` mask) or the zeroed rows
`[K, KpadM)` — over-inclusion, which is the direction §5's test is exact in. Windowed, that same
prefix simply STOPS the Gram some windows before the fold stops: `kw <= 0` on the tail windows
and `mma_tile_fragA`'s own early exit does the rest.

`G` is `[BT x BT]`: `NM` m-blocks x `BT/8` n-blocks of warp tiles, strided over the warps exactly
as the readout and the fold are. At `BT = 16` the grid is `1 x 2` and warps 0-1 take one tile
each, i.e. `warp < BT / 8`. Forked on `NM` for the same measured reason (§39).

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

`COMPACT` cuts the M EXTENT to `ceil(K/16)` warp m-tiles instead of `BC/16`, the third of the
three extents compaction shrinks, and on the flagship cell's density
(`mean_K_candidate = 0.9665` of `BC = 64`) it is a **4 -> 1 cut on the kernel's most expensive
GEMM**.

### <a id="fold-claim-uniform"></a>HAZARD `fold-claim-uniform` — the claim is WARP-UNIFORM and must never become per-lane

Lane 0 takes the index with a single `atomicAdd` and broadcasts it with `__shfl_sync`, so all 32
lanes execute the SAME item and the MMA is issued ONCE per warp, not 32 times. A per-lane claim
would issue 32 different items per warp and consume the pool 32x as fast, with each lane holding
a fragment of a different tile.

### <a id="fold-epilogue"></a>`DECAY`'s exit scale, riding the fold's existing read-modify-write

`M_exit = s0 * (s0 * M_entry + Sum_j W~[j] v[j])`, and the inner `s0` was applied at entry (§30
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

### <a id="ragged-m-guard"></a>HAZARD `ragged-m-guard` — `COMPACT`'s output scatter needs the STORE predicated

The fold's `m` is a compacted position while the state is leaf-indexed and persists across tiles,
so the store scatters through `c_idx`. **The guard is NOT redundant with the zeroed pad rows:** a
pad row's DELTA is zero, but `c_idx[q >= K] = 0` aims it at LEAF 0, and under decay the
epilogue's `s0` multiply would then be applied to leaf 0 a SECOND time. Predicating the STORE is
what keeps the pad inert; zeroing the A operand alone is not enough.

### The mass state's own recurrence

`mass[i] += sum_t W[t, i] * scale_write[t]` — the `v := 1` column of the same fold. It exists only
under `global`, and it runs over the SAME leaf-position axis as the fold's `m`, so the two
together still cover `BC x (DV + 1)` exactly once. It is THREAD-strided rather than claimed: it is
a scalar reduction with no fragment and no accumulator to strand, so the 256-thread stride already
balances it and a claim would only add an atomic per row. `WT`'s row is WINDOW-LOCAL, the state's
row `i` is not. `COMPACT`: the reduction runs over the gathered rows and the row it writes is
scattered through `c_idx` — the mass column IS the fold's `v := 1` column, so it takes the same
compaction and the same scatter, and the same exit scale in the same read-modify-write.

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
(§15) into one.

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
the readout/mass-term-1 and phase B (§42) — so the emission sits on the far side of the fold and
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
Forked on `NM` for the same measured reason as the three GEMMs (§39).

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
value, before the store (§33's `summary-prequant`), because a re-scan of `WT` reads a STORED
coefficient and a stored coefficient is a QUANTIZED one wherever the buffer's format is narrow.
What is left here is the term that has no pre-quantization site because it is not a coefficient
at all:

under decay `WT` carries `W * fw`, and "a nonzero write forces a nonzero coefficient" is an
argument about `W`, not about `W * fw`: `fw` can UNDERFLOW to zero on the early half of a tile at
an extreme survival range. That range is outside what `panel_exponent_bound` admits, but the
summary's inclusion test should not be the thing that depends on the admission check. `s0 != 1`
is the DIRECT statement that this leaf's clock moved in this tile, which is exactly the condition
under which `A` leaves `1`.

`COMPACT`: a leaf OUTSIDE the gathered list cannot enter the summary, and that is exact on BOTH
terms — it has no live write in this tile (so its write coefficients are zero, which the build's
own test sees) and its `s0` is exactly `1` (the scatter-fill theorem, §30), so it cannot leave
`(A, B) = (1, 0)`.

### `A`, the segment's per-leaf survival product (§2.13)

One write tile's map is `M_out = s0^2 * M_in + s0 * (W~^T Vw)`, so the tile's `A` factor is
`s0 * s0` and the segment's is their product. Rule 1's half-mass reference is what makes this
expressible as `s0` SQUARED rather than as a second exponential to form and store:
`s0^2 = exp(2 * P_ref * log_keep) = keep^(tile's TOTAL clock)`, i.e. the genuine survival across
the tile. `A` therefore needs no separate reduction and no separate buffer.

It is a `[BC]` loop, not a `[BC x cols]` one — `A` is a per-leaf SCALAR, which is the whole reason
a segment summary is cheap. `COMPACT`: the gathered rows only. A dropped leaf's `s0` is exactly
`1`, so the factor it would contribute is the identity — and under `COMPACT` its `s0` is not even
written, which is why this loop MUST be restricted rather than merely being free to be.

## <a id="pass1-publication"></a>48. §6.4 pass 1's publication — the SPARSE written-leaf summary

§6.4 makes the summary NORMATIVELY sparse: its traffic must scale with REALIZED writes, not with
`BC`.

Slot `owner_slot0` is the scan's entry-state stash, so segment `seg`'s summary lands ONE PAST it.
Pass 3's segment `q` then reads `owner_slot0 + q` UNIFORMLY: the stash at `q = 0`, and the
boundary the scan accumulated over this summary at `q = seg + 1`.

Warp 0 compacts the flags to an ASCENDING leaf list — the same ballot compaction §3 step 2's
gather uses, for the same reason (an ascending list is what makes the dense degeneracy the
identity). `sum_slot[i]` is rewritten IN PLACE from "is written" to "position in the list"; only
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

Every `//: HAZARD <slug>` stub in `consumer_kernel.cuh`, and where its reasoning lives.

| slug | site | what a mechanical edit would silently change |
|---|---|---|
| [`frag-a-get`](#frag-a-get) | `frag_a_get` | the hand-rolled inverse of `frag_a_slot`; nothing checks the pair |
| [`loop-induction`](#loop-induction) | the candidate walk | folding the scan into the loop increment: measured spills on arms at depth 1 |
| [`no-captured-arrays`](#no-captured-arrays) | `build_wt_rows` / the amplitude accessor | a captured pointer array lands in LOCAL memory |
| [`clock-prefix-fp32`](#clock-prefix-fp32) | the clock prefix `acc` | narrowing it is an EXPONENT error, not a mantissa one |
| [`compact-clock-list`](#compact-clock-list) | the compacted clock walk | a predicate instead of a list restriction blocks the FMA contraction (~7e-09) |
| [`wt-row-t-fastest`](#wt-row-t-fastest) | `build_wt_rows`' index map | `i`-fastest is a 4-way STORE conflict |
| [`summary-prequant`](#summary-prequant) | the `a != 0.0f` ballot | a post-quantization test UNDER-includes and drops state |
| [`ballot-lane-run`](#ballot-lane-run) | `kRowLanes` | the `t`-group must be a contiguous lane run inside one warp |
| [`wt-build-site`](#wt-build-site) | the plane's `build_wt_rows` call | a run-ahead build overwrites rows an async fold reader still holds |
| [`clk-dead`](#clk-dead) | the `clk -> Vw` barrier | `Vw`'s staging clobbers the aliased clock prefix |
| [`release-acquire`](#release-acquire) | `retire_item` / `eligible` | fence order, lane-0-only atomic, `volatile` load |
| [`gram-gate-guard`](#gram-gate-guard) | the `kCtlGramDone` retirement | mirroring the GEMM guard instead of the store guard HANGS |
| [`gate-if-constexpr`](#gate-if-constexpr) | the `kNarrowGate` fork | a folded predicate loses the wide arm's byte-identity |
| [`nm-fork`](#nm-fork) | the four `NM == 1` forks | 143 of 256 instantiations move, 24 spill worse |
| [`mass-split`](#mass-split) | mass terms 1 and 2 | the split is bit-identical only under the same thread map |
| [`gram-pinned`](#gram-pinned) | `gd` | a fluid Gram loses ascending-quantum accumulation order |
| [`fold-claim-uniform`](#fold-claim-uniform) | the fold's `atomicAdd` claim | a per-lane claim issues 32 different items per warp |
| [`ragged-m-guard`](#ragged-m-guard) | the fold's scatter store | an unpredicated pad row applies `s0` to leaf 0 twice |
| [`emission-interval`](#emission-interval) | the removed barrier | the interval must stay free of cross-thread SMEM |
| [`emit-no-pointer-array`](#emit-no-pointer-array) | the `emit` lambdas | taking an accumulator's address spills it to local memory |
