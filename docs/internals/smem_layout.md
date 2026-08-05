# `csrc/rola/src/smem_layout.cuh` — the consumer's shared-memory layout layer

What one consumer CTA's shared memory looks like, and what it costs: the address
functions the staged operands are stored under, the `constexpr` proofs that make each
one conflict-free, the measured register budget and residency arithmetic the launch
bound is derived from, and the byte ledger `consumer_smem_bytes_at` computes — which
is what the launch actually reserves.

The kernel body's pointer carve (`consumer_kernel.cuh`) follows this ledger term for
term, in the same order. The `ldmatrix` atoms that READ these layouts are
[`mma_tile.md`](mma_tile.md); the control words the ledger budgets for are
[`pipeline.md`](pipeline.md).

**Section numbers are shared across the four consumer-kernel documents** —
`consumer_kernel.md`, this file, `mma_tile.md` and `pipeline.md` — so a `§N`
reference in any of them names exactly one section, wherever it lives.

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

# Part II — the shared-memory carve

## <a id="smem-ledger"></a>10. The ledger (the §10i.14 carve, term by term)

Every term of `consumer_smem_bytes_at`, at the FLAGSHIP CELL — `BT = 16, BC = 64,
DV = 64, D = 2`, `GLOBAL`, no `DECAY`, no `COMPACT`, `kRing = 1` — on both precision
arms. The staging format is the only axis that moves a byte figure without moving a
template parameter, so both arms are given: `stage_is_narrow` selects it and
`wt_pitch` / `vw_elems` size it, and quoting one arm alone under-states one build
while over-stating the other.

| region | extent | bf16 (narrow) | tf32x3 | what it is |
|---|---|---|---|---|
| `S` | `[BC][SP]` fp32 | 17,408 | 17,408 | the STATE, resident for the owner's whole run. `SP = DV + 4` (§17), not `DV + 1` |
| `sr`, `sw` | `RING * 2 * [BT]` fp32 | 128 | 128 | the §3 step-3 row scales |
| `den` | `[BT]` fp32 | 64 | 64 | `GLOBAL` only |
| `cand` | `2 * kMaxWords` u32 | 1,024 | 1,024 | the candidate-tile words |
| `dig` | `[D][BC]` int | 512 | 512 | per-owner leaf digits (hoisted, §26) |
| `WT` | `[kWinRows][BTP]` | 3,072 | 5,120 | write coefficients, LEAF-MAJOR; swizzled bf16 narrow, padded fp32 on `tf32x3` |
| `Vw` (∪ `clk`) | `RING *` the union | 2,048 | 4,608 | the scale-folded value tile; swizzled and UNPADDED narrow |
| `Rf` | `RING * NM * frag_a_words(BC)` | 2,048 | 4,096 | read coefficients, MMA-FRAGMENT ORDERED |
| `Gm` | `[BT][BT + 8]` bf16 / `[BT][BT + 4]` fp32 | 768 | 1,280 | the causal effective-score block |
| `ctl` | `2 + QPO` int | 24 | 24 | the item-stream control block ([`pipeline.md#item-stream`](pipeline.md#item-stream)) |
| **total** | | **27,096** | **34,264** | |

**This table is CHECKED, not transcribed.** `rola_consumer_smem_bytes` reports the
compiled `consumer_smem_bytes` of a shipped instantiation, and the row arithmetic above
reproduces it EXACTLY on all 32 driver-queryable cells (`decay` x `compact` x `BC` x
`BT` x `DV`, at `D = 2`, bf16, `GLOBAL` — the axes `query_row` pins). The flagship's
27,096 B is one of those 32.

### The four arms of the flagship cell, and what the driver says

| arm | bytes | CTAs/SM (driver) |
|---|---|---|
| raw of both | 27,096 | 3 |
| `COMPACT` | 27,616 | 3 |
| `DECAY` | 30,168 | 3 |
| `DECAY` + `COMPACT` | 30,688 | 3 |

`DECAY` adds 3,072 B here and not 768: the three `[BC]` vectors are 768, and the other
2,304 is `clk` RAISING the `Vw` union from 2,048 to 4,352 (§11). `COMPACT` adds 520 B
flat. CTAs/SM is `rola_consumer_occupancy` — the driver, on the built binary — not the
launch bound's own arithmetic, which is a floor rather than an answer (§5).

**The tightest shared-memory margin in the shipped matrix is 544 B**, at
`DECAY + COMPACT, BC = 64, BT = 32, DV = 32`: 32,928 B/CTA, so
`3 * (32928 + 1024) = 101,856` against the SM's `102,400`. The driver returns 3 CTAs
there. Every other arm has more room; the largest footprint in the matrix is
`DECAY + COMPACT, BC = 64, BT = 32, DV = 64` at 41,120 B, which the driver runs at 2.

Storing the `tf32x3` arm's `WT` and `Vw` as bf16 would save 4,608 B at the flagship and
change no residency; it is refused because that is the arm whose fold precision must be
recoverable.

The carve is BYTE arithmetic, not pointer arithmetic, from `WT` onward: the staged
buffers' element size follows `StageT` while everything around them is fp32, so
`p + count` would silently mis-place every buffer after them. `align4` keeps each
subsequent fp32 buffer 4-byte aligned; every shipped `(BC, BTP)` product is already a
multiple of 2 elements, so it never pads.

The footprint is `constexpr` so the launch bound can be DERIVED from it rather than
restated next to it — two numbers that must agree are one number too many.

### The per-arm carves

- **`DECAY`** adds THREE buffers rather than four: `log_keep`, `p_ref`, `s0`, all
  `[BC]` — 768 B. The naive ledger is those three plus `clk[BT][BC + 4]`, which is
  4,352 B at the flagship cell and 8,704 B at `BT = 32`. It does not have to be paid,
  and the argument is a LIFETIME rather than a saving: `clk` exists only to build `Rf`
  and `WT`, it is DEAD before `Vw` is live, so it ALIASES `Vw`'s storage and only the
  raise of that union is charged (§11) — 3,072 B in total at the flagship against
  5,120 B un-aliased. **Decay ships at 3 CTAs/SM on the flagship cell, full
  residency** (30,168 B, driver-confirmed). Un-aliasing costs a CTA on the deepest arm
  in the matrix — `D = 4, BT = 16, BC = 64, DV = 64`, `DECAY + COMPACT`: 33,248 B
  against 31,200, i.e. `3 * (33248 + 1024) = 102,816 > 102,400`. That figure is
  arithmetic off `consumer_smem_bytes_at`, not a measurement: no un-aliased arm exists
  to ask the driver about.
- **`COMPACT`** adds `c_idx` + `c_slot` + `c_count` AND NOTHING ELSE — 520 B at
  `BC = 64` (136 B at `BC = 16`), which is what leaves `decay + compact` inside 3 CTAs/SM
  at every `DV = 32` and flagship shape. READ OFF THE DRIVER on every build, never
  assumed: this is where the tightest residency margin in the kernel sits. It
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
  slack** — both numbers confirmed against the built binary (§10). Every other 3-CTA
  arm has more. `rola_consumer_occupancy` returns the identical CTA count on all 32
  `(decay, compact, BC, BT, DV)` shapes as the pre-K3b kernel, which is the check that
  matters — the byte count is only the reason to expect it.

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

### <a id="cute-pin"></a>The vendored CuTe pin, as this layer uses it

`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5. It is a CODEGEN INPUT, not a
packaging one — every line instantiated here is code `ptxas` assembles — which is why
`setup.py::_gate_vendored` hashes the subtree BEFORE the compile and `tools/ratify.py`
carries the same include path.

This layer uses it for exactly ONE thing: `StageLayout`, the swizzled layout `Vw` is
stored under, together with the three predicates above that interrogate it. The
`ldmatrix` copy atoms that read that layout are the tile layer's
([`mma_tile.md#cute-pin`](mma_tile.md#cute-pin)), and the split is the point — the
library owns the address function in one file and the fetch in the other, and
`stage_offset_matches_cute` is what keeps the closed form honest to it.

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

# Part III — the fragment-ordered operand

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
