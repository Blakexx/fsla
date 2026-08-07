# `csrc/rola/src/smem_layout.cuh` — the consumer's shared-memory layout layer

What one consumer CTA's shared memory looks like, and what it costs: the address
functions the staged operands (`WT`, `Vw`) are stored under, the `constexpr` proofs
that make every operand conflict-free, the measured register budget and residency
arithmetic the launch bound is derived from, and `consumer_smem_bytes` — the byte
ledger the launch actually reserves.

The kernel body's pointer carve (`consumer_kernel.cuh`) follows this ledger term for
term, in the same order — see [`consumer_kernel.md`](consumer_kernel.md). The
`ldmatrix` atoms that READ these layouts are [`mma_tile.md`](mma_tile.md).

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

## The group is the tile

The executor's work unit is a GROUP of `kGroupTokens == kMmaKQuantum == 16` gathered
tokens — one MMA m-quantum's worth. Groups are always full; only a segment's final
flush is ragged. Every quantity below that would once have been `BT`-shaped is
`kGroupTokens`-shaped instead: there is no separate tile-token axis.

## The one precision story

bf16 storage everywhere the staged operands and amplitude streams live, fp32
accumulation/state/clock/row-scales, fp64 torch reference as the oracle. There is no
precision axis — no `MmaPrec`, no `PREC` template parameter, no tf32x3 arm. Every
layout and proof below is stated for bf16 alone.

## <a id="cute-pin"></a>The sector invariant, and the vendored CuTe pin

```
static_assert(kMmaKQuantum * sizeof(__nv_bfloat16) == kSectorBytes)   // 16 * 2 == 32
```

One atom of bf16 digits — `kMmaKQuantum == 16` leaves — IS one 32-byte memory sector.
This is THE central invariant tying together the atom (one `kMmaKQuantum`-leaf block
of the owner's leaf axis), the MMA k-quantum (`mma.sync`'s hardware k-extent on the
sm80 bf16 path), and the fetch/union machinery's granule: `read_atom_bits` /
`write_atom_bits` (host ABI `[BH, n_slices, QPO, n_twords]` int32 —
[`consumer.md#atom-union-bytes`](consumer.md#atom-union-bytes)) devote one whole
CONTIGUOUS row of `n_twords` token-liveness words to each atom, because one atom is
exactly one fetch sector — the ballot ANDs a row's word against the taken token word
to produce that atom's single bit in `rmask`/`wmask`. The walk assumes this
unconditionally; there is no fp32-sector side path, because fp32 amplitude storage is
dead.

`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5. It is a CODEGEN INPUT, not a
packaging one — every line instantiated here is code `ptxas` assembles — which is why
`setup.py::_gate_vendored` hashes the subtree before the compile and `tools/ratify.py`
carries the same include path. This layer uses it for exactly ONE thing: the
swizzled `StageLayout` both staged operands (`WT`, `Vw`) are stored under, together
with the three proofs below that interrogate it. The `ldmatrix` copy atoms that read
that layout are the tile layer's ([`mma_tile.md#cute-pin`](mma_tile.md#cute-pin)),
and the split is the point — the library owns the address function in one file and
the fetch in the other, and `stage_offset_matches_cute` ([below](#stage-offset)) is
what keeps the closed form honest to it.

---

# Part I — the compile-time budget

## <a id="reg-budget"></a>4. `consumer_reg_budget` — a MEASURED kernel fact, not an architecture fact and not a threshold

```
consumer_reg_budget(BC, decay, DV, D) = (BC >= 64 || decay || D >= 4) ? 80 : 64
```

This is the register working set `ptxas` reports for the kernel's GEMM inner loop
when it is given slack, keyed on the axes the working set MEASURABLY varies with
— `BC`, `DECAY` and `D` — with every number read off `-Xptxas -v` and the MAXIMUM per
key taken, because a too-small budget is bought back by SPILLING (the conservative
direction is well defined: over-stating a budget only understates the residency the
register file admits, under-stating it makes the allocator spill to buy the
difference). `DV` does NOT key it — the parameter is accepted and ignored
(`(void)DV;`), stated explicitly rather than silently dropped, because the working
set the atom walk touches per visit does not scale with the value width the way it
once did under the deleted `NM`/row-block axis: every group is one `m16` row block
now, so there is no second row-block term to key on.

`BC >= 64` takes the wide budget unconditionally (the `BC = 64` arms are shared-memory
bound at every occupancy the register term could buy, so a slack budget costs
nothing); at `BC = 16` the decay arms carry the clock prefix, the survival panels and
the two coefficient folds on top of the same GEMM and need the wide budget too, while
the no-decay arms clear the working set at 64.

`D` joined the key with the schedule-ring producer: level count is a structural
template axis, and the producer's cook working set grows with it — one fixed-level
bit row per level, per side, per refill. Measured over all 32 arms on both
architectures: at `D <= 3` the `BC = 16, decay = 0` arms fit 64 with zero spill; at
`D = 4` they do not, and moving the cook's carried state into
[SMEM](../internals/consumer_kernel.md#producer-state) does not close it further —
the cook is already resting there. The cost is recorded rather than hidden: 80 puts
`by_regs` at `65536 / (256 * 80) = 3`, so the `D = 4, BC = 16, decay = 0` arms run 3
CTAs/SM where 64 gave 4, which is worth about -7% on the `topo-D4-w8x8x8x16-BC16`
wall.

Reproduce with `python tools/ratify.py --census`, which compiles the shipped
instantiation shards and reports per structural key; if the kernel's inner loop
changes, re-run it and update the formula, not this prose.

## <a id="residency"></a>5. `ctas_for_residency` and the launch bound

```
ctas_for_residency(smem, reg_budget) = min(by_smem, by_threads, by_regs)
  by_smem    = kSm.smem_per_sm / (smem + kDriverSmemReservePerCta)
  by_threads = kSm.max_threads_per_sm / kConsumerThreads     // kConsumerThreads == 256
  by_regs    = kSm.regs_per_sm / (kConsumerThreads * reg_budget)
```

CTAs/SM to STATE IN THE LAUNCH BOUND — the smallest of what the three hardware
resources admit, computed per instantiation rather than fixed. All three terms are
load-bearing: `ptxas` does not CLAMP an out-of-range `__launch_bounds__` argument, it
DISCARDS it, so a missing thread term is silent everywhere the smem term alone would
claim more CTAs than 256-wide blocks fit; and a missing register term buys the extra
CTAs the other two terms admit by SPILLING rather than by residency.

`kSm` is the arch table's per-architecture row ([`arch_table.md#sm-resources`](arch_table.md#sm-resources));
on sm86/87/89 it is `{smem_per_sm = 102400, max_threads_per_sm = 1536, regs_per_sm =
65536}`. Two worked cells, both derived straight from `consumer_smem_bytes`
([below](#smem-ledger)) and `consumer_reg_budget` — no number here is a measurement,
every one is the formula evaluated:

```
BC=16, DV=32, D=1, DECAY=false   smem = 6,528 B   reg_budget = 64
  by_smem    = 102400 / (6528 + 1024)  = 13
  by_threads = 1536 / 256              = 6
  by_regs    = 65536 / (256 * 64)      = 4
  ctas_for_residency = min(13, 6, 4) = 4

BC=64, DV=64, D=2, DECAY=true    smem = 30,016 B  reg_budget = 80
  by_smem    = 102400 / (30016 + 1024) = 3
  by_threads = 1536 / 256              = 6
  by_regs    = 65536 / (256 * 80)      = 3
  ctas_for_residency = min(3, 6, 3) = 3
```

At `BC = 16` the register term is what keeps the bound honest: the thread term alone
would claim 6 CTAs, which is more than the register file supports at the measured
working set, and asking for 6 caps the allocator at `65536 / (6 * 256) = 42`
registers, buying those CTAs by spilling. At `BC = 64` the shared-memory term is what
binds — both the register term and the thread term have slack — so a slack register
budget (`80` rather than a tighter fit) costs nothing there, which is the other half
of [why `reg_budget` keys the way it does](#reg-budget).

This is stamped into `__launch_bounds__` (`ROLA_CONSUMER_CTAS_EXPR` in
`consumer_kernel.cuh`) rather than left to the compiler: meeting the shared-memory
budget does not get you the occupancy on its own — an unconstrained `ptxas` is free to
spend up to 255 registers/thread, which caps residency below what the register file
would otherwise admit. Told the real target, `ptxas` re-schedules to fit it. The
`ROLA_FORCE_CTAS` override ([`consumer_kernel.md#force-ctas`](consumer_kernel.md#force-ctas))
replaces this whole expression with a fixed literal for exactly one purpose: censusing
a candidate register budget over the dispatch matrix without touching the shipped
formula.

---

# Part II — the shared-memory carve

## <a id="smem-ledger"></a>10. The ledger — `consumer_smem_bytes`, term by term

Every term of `consumer_smem_bytes<BC, DV, D, DECAY>()`, in carve order (the kernel
body's pointer arithmetic follows this exact sequence):

| region | extent | bytes | what it is |
|---|---|---|---|
| `S` | `[BC][DV+4]` fp32 | `BC*(DV+4)*4` | the STATE, resident for the owner's whole run. Pitch `SP = DV+4` ([`consumer_kernel.md#state-pitch`](consumer_kernel.md#state-pitch)) |
| `dig` | `[D][BC]` int | `D*BC*4` | per-owner hoisted leaf digits ([`consumer_kernel.md#digit-hoist`](consumer_kernel.md#digit-hoist)) |
| `WT` | `[BC][kWtPitch]` bf16, `align4` | `align4(BC*24*2)` | write coefficients, LEAF-MAJOR, pitch `kWtPitch = kGT+8 = 24` ([below](#stage-format)) |
| `Vw` ∪ `clk` | the union, `align4` | `vw_union_bytes<BC,DV,DECAY>()` | the scale-folded value tile; on `DECAY` ALIASES the clock prefix ([below](#vw-clk-union)) |
| `Rf` | `frag_a_words(BC)` u32 | `(BC/16)*128*4` | read coefficients, MMA-FRAGMENT ORDERED ([below](#frag-order)) |
| `Gm` | `[kGT][kGT+8]` bf16 | `16*24*2 = 768` | the causal effective-score block, pitch `GMP = kGT+8 = 24` — an `ldmatrix` SOURCE (GEMM 3's A and the intra mass term's), so its base carries the same 16-byte alignment `WT`/`Vw` do |
| `log_keep`, `p_ref`, `s0` | `3 * [BC]` fp32, `DECAY` only | `DECAY ? 3*BC*4 : 0` | `DECAY`'s three per-leaf vectors |
| `g_ring` | `kSchedSlots * kSchedSlotInts` int | `kSchedSlots*(kGT+4)*4` | the [cooked schedule ring](#sched-ring): `kSchedSlots` group descriptors, each the group's `kGT` ascending token indices plus `gN`/read-mask/write-mask/flags |
| `g_up` | `kSchedSlots * 2 * kGT` fp32 | `kSchedSlots*2*16*4` | the ring's per-owner [upper-product](consumer_kernel.md#upper-product) pairs — `sr`/`sw`, cooked beside the member word |
| producer cursor state | `(8 + 5*32)` int | `168*4 = 672` | the [producer's cursor](consumer_kernel.md#producer-state) (7 uniform scalars, 1 pad) and its 5 32-lane cook planes — rests here between cooks so no scheduler state is register-live across the visit body |

`cols = DV + 1` is unconditional (the mass column always exists — normalization is no
longer a template axis), which is why `S`'s row EXTENT above is `DV + 4` regardless:
the pitch derivation ([`consumer_kernel.md#state-pitch`](consumer_kernel.md#state-pitch))
is chosen to keep the bf16 MMA B map conflict-free and already covers the mass
column's one extra live column within it.

**Two worked totals**, computed by hand from the table above (both independently
cross-checked against `consumer_smem_bytes` in Python):

```
BC=16, DV=32, D=1, DECAY=false (kSchedDepth=1, kSchedSlots=2):
  S=2304  dig=64  WT=768  Vw∪clk=1024
  Rf=512  Gm=768  decay-extra=0
  g_ring=160  g_up=256  producer-state=672
  total = 6,528 B  (~6.4 KiB)

BC=64, DV=64, D=2, DECAY=true (kSchedDepth=1, kSchedSlots=2):
  S=17408  dig=512  WT=3072  Vw∪clk=4352
  Rf=2048  Gm=768  decay-extra=768
  g_ring=160  g_up=256  producer-state=672
  total = 30,016 B  (~29.3 KiB)
```

These are the same two shapes [the residency section](#residency) plugs into
`ctas_for_residency`. No other shape's byte total is stated here; plug the formula
above into `consumer_smem_bytes` for any other `(BC, DV, D, DECAY)` — that function,
not this prose, is the authority a future carve change must agree with.

### <a id="mass-col"></a>The mass column is unconditional

`cols = DV + 1` on every arm: there is no `raw`/`GLOBAL` axis left to make it
conditional. A run that never needed the mass column still pays 4 bytes/leaf-row for
it — deleting the axis deleted the branch, not the column. The column's denominator
accumulator is a register-resident MMA D fragment (`dm0..dm3`, WARP 0,
[`consumer_kernel.md#mass-column`](consumer_kernel.md#mass-column)), not an SMEM
array — there is nothing here to carve for it.

### <a id="sched-chunk"></a><a id="sched-ring"></a>The schedule ring's SMEM block

Warp 0 (the [producer](consumer_kernel.md#producer)) is the ONLY warp that touches
`read_tok_bits`/`write_tok_bits`/`read_atom_bits`/`write_atom_bits`: it cooks one
`kSchedChunkWords`-word chunk of the membership scan at a time — one 32-token machine
word per lane of the producer warp, `kSchedChunkWords == 32` — entirely in registers
(the AND of the owner's fixed-level rows and the ballot that finds the chunk's live
words), then ballots members into a group and publishes a COOKED GROUP DESCRIPTOR
into `g_ring`, one of `kSchedSlots = kSchedDepth + 1` slots
([`consumer_kernel.md#sched-ring`](consumer_kernel.md#sched-ring)). Every consumer
warp reads a descriptor rather than deriving it: this SMEM block is bought by the
consumption multiplicity that earns it, one writer against `n_atoms + 1` readers.

`g_ring[slot]` is `kSchedSlotInts = kGT + 4` ints: the group's `kGT` gathered token
indices ascending, then `gN` (member count, `0` = stream end), the read atom mask, the
write atom mask, and the side flags. `g_up[slot]` is the matching pair of `kGT`-float
[upper products](consumer_kernel.md#upper-product), `sr` then `sw`, cooked in the same
pass.

The producer's OWN cursor (word cursor, chunk base, live-word mask, selected word,
remainder — 7 scalars, uniform across the producer warp) and its cook planes (5
32-lane arrays: the cooked chunk's `fr`/`fw`/`member` words and the selected word's
read/write atom-bitlist rows) rest in a third SMEM block between cooks
([`consumer_kernel.md#producer-state`](consumer_kernel.md#producer-state)) rather than
in registers, for the measured reason that governs all scheduler state here: none of
it may be register-live across the visit body, where it would compete with the four
GEMMs' working set. Every reload happens at the point
of use inside the next cook, and the ring's publication rides the end-of-visit
`__syncthreads()` that already exists — there is no separate group-entry barrier.

## <a id="vw-clk-union"></a>11. The `Vw` / `clk` union, and why it is sized in BYTES rather than elements

```
vw_union_bytes<BC,DV,DECAY>() = align4(max(vw_bytes, clk_bytes))
  vw_bytes  = kGT * DV * sizeof(bf16)
  clk_bytes = DECAY ? kGT * (BC+4) * sizeof(float) : 0
```

`clk` (the visit's per-leaf inclusive clock prefix, `[kGT][BC+4]` fp32, token-major)
is DEAD before `Vw` (the scale-folded value tile) is live within one visit — the clock
prefix is fully consumed building `Rf` and `WT` before the value staging that fills
`Vw` runs, and a `__syncthreads()` (HAZARD `clk-dead`,
[`consumer_kernel.md#clk-dead`](consumer_kernel.md#clk-dead)) marks the exact point
past which `Vw` may overwrite `clk`'s bytes. So `clk` ALIASES `Vw`'s storage rather
than getting its own region.

**The max is over BYTES, and that is load-bearing** because the two tenants do not
share an element size: `Vw` is bf16, `clk` is fp32, so an element-wise max would
under-allocate by 2x, silently. `clk` is the larger tenant at every shipped `DECAY`
shape (`kGT*(BC+4)*4` against `kGT*DV*2`; at `BC=64,DV=64` that is 4,352 against
2,048), so on the decay arms narrowing `Vw` frees no shared memory at all — the
reservation is the max, whatever the smaller tenant does. `Vw` is still ADDRESSED at
its own layout; only the RESERVATION is the max.

`clk` is declared AT THE CARVE (`float* clk = reinterpret_cast<float*>(carve);`)
rather than assigned from `Vw` later, because the two element types make that
assignment not type-check — the alias is the load-bearing fact, stated where the
pointer is born. The kernel body carries a `static_assert` pinning
`kGT * CKP * sizeof(float) <= vw_union_bytes<BC,DV,DECAY>()` so the sizing is checked
against the actual use, never assumed to agree.

## <a id="bank-predicates"></a>12. The bank-conflict proofs, as compile-time predicates rather than comments

Every MMA operand load has 32 lanes addressing SMEM as an affine function of the
fragment's two lane coordinates, `g = lane >> 2` and `r = lane & 3`. The access is
conflict-free exactly when those 32 addresses land on 32 DISTINCT banks.
`banks_distinct` ENUMERATES that condition for a 32-bit (fp32) operand instead of
asserting a modular formula, so each operand proves its own pitch at compile time and
a future change to `BC`/`DV` fails the BUILD rather than silently costing bank-conflict
wavefront amplification. `stride_g` / `stride_r` are the address deltas in 4-BYTE
BANKS per unit `g` and `r`.

`banks_distinct_u16` is the SAME question for a 16-bit (bf16) operand, and it is NOT
the same predicate: the bank of element `e` is `(e >> 1) % 32` rather than `e % 32`,
and two lanes landing in the SAME 32-bit word do not conflict — the SM services one
word in one transaction — so the condition is "any two lanes sharing a bank share a
word", which `banks_distinct` cannot express. Applying the 32-bit predicate to a
16-bit buffer proves the wrong proposition in BOTH directions (rejects conflict-free
layouts, accepts conflicting ones). `stride_g` / `stride_r` here are deltas in
2-BYTE ELEMENTS.

`pitched_ldsm_banks_ok<ROWS, COLS, PITCH>` asks the SAME `ldmatrix` phase question
(below) of a PITCHED, unswizzled buffer: `WT` keeps its derived pad rather than a
swizzle, so this licenses the atom at that pitch instead of assuming a pad chosen for
a scalar fragment map also serves a matrix load. `COLS` is the column extent actually
addressed (`kGroupTokens`), smaller than `PITCH`.

## <a id="stage-layout"></a>13. The staged operands' SMEM layout, owned by CuTe

Both `WT` and `Vw` are stored under the library's own swizzled layout,
`composition(Swizzle<B,3,3>, (ROWS,COLS):(COLS,1))`, and the pad DISAPPEARS: the
swizzle decorrelates banks by permuting 16-byte blocks WITHIN the tile instead of by
making it bigger, so the buffer is exactly `ROWS * COLS` elements. `B` is chosen so
the XOR field covers the whole 128-byte bank cycle (`stage_swizzle_b<COLS>()`); `M = 3`
keeps the 16-byte `ldmatrix` row intact; `S = 3` puts the XOR source at the row
index's low bits.

Nothing here is asserted — the properties are `static_assert`ed by ENUMERATING the
library's layout:

1. **`stage_layout_is_bijection<ROWS,COLS>()`** — the layout is a BIJECTION onto
   `[0, ROWS*COLS)`. This licenses the reservation `ROWS*COLS*sizeof(bf16)`: a swizzle
   that overflowed its tile would alias the next buffer.
2. **`stage_ldsm_banks_ok<ROWS,COLS>()`** — every `ldmatrix` FETCH is conflict-free.
   The instruction is serviced 8 lanes at a time, each lane reading a 16-byte row, so
   the phase is conflict-free exactly when its 8 rows cover all 32 banks. It also
   checks the 16-byte ALIGNMENT `ldmatrix` requires, which no pitch predicate alone
   could see.
3. **`stage_store_banks_ok<ROWS,COLS>()`** — the STAGING STORE stays conflict-free.
   Both staged buffers are filled by a flat `idx = row*COLS + col` loop at
   `kConsumerThreads` stride, so the same-word rule from
   [`banks_distinct_u16`](#bank-predicates) applies, asked of the swizzled addresses.

### <a id="stage-offset"></a>The address function is a closed form, and CuTe is its specification

Calling `StageLayout{}(row, col)` at every access point measurably blocks the strength
reduction the linear map admits on the scalar row walks `WT` has (its panel build's
token loop, the fold's mass-column reduction). `stage_offset<COLS>(row, col)` is the
SAME map, in the two operations the swizzle actually is:
`row * COLS + (col ^ mask(row))`. The XOR source bits come ENTIRELY from `row`
(`COLS >= 16`) and the target bits lie ENTIRELY inside one row (`COLS >= 8 << B`), so
the XOR never carries between the two fields and the mask is ROW-INVARIANT — the
compiler can hoist it out of a row walk.

`stage_offset_matches_cute<ROWS,COLS>()` enumerates every `(row, col)` of the buffer
and requires the closed form to agree with `StageLayout{}(row, col)`. That is what
keeps the library the authority: the layout is stated in CuTe's algebra, the kernel's
closed-form addressing is proven equal to it at every point, and a swizzle parameter
changed in one place and not the other fails the build.

## <a id="stage-format"></a>14. The staged-operand storage format, and `kWtPitch`

`WT` (the write coefficient panel) and `Vw` (the scale-folded value tile) are the two
per-visit staged operands, and both are bf16 UNCONDITIONALLY — there is no precision
axis to select a format from.

`Vw` is stored under `StageLayout` with NO pad — its element count is
`vw_elems<DV>() = kGroupTokens * DV`, addressed by `vw_index<DV>(t, ch) =
stage_offset<DV>(t, ch)`.

`WT` stays PITCHED rather than swizzled, and that asymmetry is a MEASURED decline, not
an omission: `Vw` has exactly two addressing sites (a flat store loop and one
`ldmatrix` row address per k-step) where `WT` has more, including the scalar panel
build's row walk, and a swizzle makes that walk non-affine — a cost the closed-form
address only partly recovers. `WT` serves TWO operand maps and its pitch has to
satisfy both:

```
GEMM 2 (Gram), WT as B    B[k*ldb + n]    k = leaf steps 2 per r, n = token per g
GEMM 4 (fold), WT as A    A[m*lda + k]    m = leaf per g,         k = token steps 2 per r
```

Enumerated over `pad in {4, 8, 12, 16}`, **`kWtPitch = kGroupTokens + 8 = 24` is the
UNIQUE pad that passes both** — pinned by `pitched_ldsm_banks_ok<kMmaKQuantum, kGT,
kWtPitch>()` (the `ldmatrix` phase, including the 16-byte alignment `WT`'s row bytes
give: `kWtPitch * 2 = 48`, a multiple of 16) and by the two `static_assert`s in the
kernel body pinning `WT`'s two scalar-fragment-map readings
([`consumer_kernel.md#pitches`](consumer_kernel.md#pitches)).

`wt_index(q, t) = q * kWtPitch + t` is a FUNCTION, row-major at `kWtPitch`, rather
than an open-coded product — the `ldmatrix` fetches (`mma_tile.cuh`'s `wt_frag_a`,
`wt_frag_b`) and the scalar walks must not be able to disagree about it.

`stage_put` (`mma_tile.cuh`) stores a computed fp32 value into `WT`/`Vw`'s bf16
storage through the explicit `__float2bfloat16` intrinsic (never `static_cast`,
because the torch build defines `__CUDA_NO_BFLOAT16_CONVERSIONS__`), and it is where
§12 rule 3's finite-bf16 format guard (`clamp_bf16_finite`) lives — the fp32 value's
last existence before the narrowing.

---

# Part III — the fragment-ordered operand

## <a id="frag-order"></a>20. Fragment-order operand synthesis

The read coefficients `Rf` are SYNTHESISED in SMEM rather than fetched with a generic
indexed read, in the exact order the MMA's A fragment wants: word `w` of lane `lane`
at k-step `ks` lives at `[ks * 128 + lane * 4 + w]`. A warp then loads a whole
fragment with ONE 128-bit `uint4` — 32 lanes x 16 B = 512 B, four full transactions,
no bank conflict and no address arithmetic at all.

`frag_a_words(K) = (K / kMmaKQuantum) * 128` is the 32-bit word count of a
fragment-ordered A operand contracting over `K`. ONE k-step covers ONE ATOM
(`kMmaKQuantum` leaves), so the buffer is atom-blocked BY CONSTRUCTION — which is what
lets the atom walk index a live atom's 128-word block affinely (`Rf + q * 128`) instead
of needing a second address map for "the q-th live atom".

Because the coefficients are fragment-ordered they carry NO ROW PITCH at all — the
layout is conflict-free by construction rather than by padding, so there is nothing
to prove a `banks_distinct` predicate about here.

`frag_a_slot` (the forward map, slot -> `(m, k)`) is `mma_tile.cuh`'s — see
[`mma_tile.md`](mma_tile.md) for the reading and its `static_assert`ed cross-check
against CuTe's own fragment maps. It has no inverse in this design: the mass column
reads the state through its own `n8` tile against a broadcast B
([`mma_tile.md#mass-column`](mma_tile.md#mass-column)) rather than reading an element
back out of the fragment.
