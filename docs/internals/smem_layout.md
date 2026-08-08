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

The PRODUCER's work unit is a GROUP of `kGroupTokens == kMmaKQuantum == 16` gathered
tokens — one MMA m-quantum's worth. Groups are always full; only a segment's final
flush is ragged. Every quantity below that would once have been `BT`-shaped is
`kGroupTokens`-shaped instead: there is no separate tile-token axis. The cook, the
ballot and the ring descriptor are all sized by it, and none of them varies with the
window below.

## <a id="window"></a>The window is the consumer's token quantum

One VISIT does not consume one group; it consumes a WINDOW of `W = GT / kGroupTokens`
groups against ONE entry state and folds them back in ONE writeback. `GT` — the
window's token extent — is a fifth structural template axis, `{16, 64}`
([`consumer.md#instantiation-matrix`](consumer.md#instantiation-matrix)), and
`window_groups(GT)` / `sched_slots(GT)` (`smem_layout.cuh`) are its two derived
quantities: the group count and the schedule ring's slot count (`W * (kSchedDepth +
1)`, since a whole window of run-ahead must be simultaneously live).

**Why `W > 1` needs a Gram cross-term, not just a wider stage.** Pacing the fold `W`
groups behind means all `W` groups read the SAME entry state, so the deposits the
un-paced fold would already have written are missing from it and must arrive as Gram
cross-terms — the Gram has to span the window. In `kGroupTokens x kGroupTokens` blocks
indexed by group `(a, b)` (`a` = read group, `b` = write group), the window Gram is
block LOWER TRIANGULAR: `b > a` is identically zero and never formed or stored, `b ==
a` is today's within-group masked block, and `b < a` — every token of group `b`
strictly precedes every token of group `a` — is unmasked and dense (FLA's
sub-chunked-diagonal split, `fla/ops/gla/chunk.py`'s `chunk_gla_fwd_A_kernel_intra_sub_inter`
/ `..._intra_sub_intra`, transplanted with the diagonal kept as a masked GEMM rather
than their scalar row loop). The causal mask therefore stays `O(W)` — only the `W`
diagonal blocks carry a predicate — while the Gram's storage grows `O(W^2)`.

What the window buys against that price: every PER-VISIT term amortises `W`-fold —
the three `__syncthreads()`, the fold's SMEM read-modify-write of `S`, the descriptor
decode, the atom-mask decode. The producer's cook cadence and the emission cadence
are unchanged: the producer still cooks one group at a time, and `y` is still emitted
per group.

**Multi-slot, not wide-tile.** `WT`, `Vw` and `Rf` become `W` INDEPENDENT SLOTS of
their existing per-group shape, never one `W`-times-wider buffer — every one of their
layout proofs is therefore a statement about ONE slot and stands unchanged across the
window. `clk` is the one exception among the group-shaped buffers: the clock prefix
runs across the WHOLE window (not per group, because the window is one reference
frame under decay), so it is `[GT][BC+4]` fp32, one buffer. `Gm`'s shape is the only
buffer whose address function moves: `[GT][GT+8]`, square, though only the block lower
triangle is ever written or read — a packed triangular layout would make a block's
base a non-affine function of `(a, b)`, and one closed-form address function per
buffer is the standard every other operand here is held to. The waste is the strict
upper triangle.

**The budget, flagship `<BC=64, DV=64, D=2>` on `sm_86` (`smem_per_sm = 102400`)**,
as a function of `W`:

```
nodecay:  512 W^2 + 7840 W + 18592 bytes
decay:    512 W^2 + 10144 W + 19360 bytes
```

`W = 4` (`GT = 64`) lands at ONE CTA/SM there; `W = 8` (`GT = 128`) does not fit a CTA
on `sm_86` at all (the `512 W^2` term), which is why `128` is not in the instantiated
set. Residency at `GT = 64` falls to one CTA/SM at `BC = 64` (SMEM-bound) and to two at
`BC = 16`; register budget for `GT > kGroupTokens` is 112 — censused 87..117 on `sm_86`
(113 max on `sm_80`) across the 32 window arms, zero spill everywhere
([below](#reg-budget)).
Per the standing register/SMEM policy, the residency loss at `BC = 64, GT = 64` is a
PURCHASE: occupancy buys only latency hiding, and this kernel hides latency
explicitly (the producer ring, the resident accumulators).

`GT` is DISTRIBUTION-DEPENDENT and therefore a PLAN-SELECTED axis, never a raised
constant: a wide window is free at dense gathering and wastes tile at low
gathered-token counts (a window's only waste is a ragged tail of dead groups), and
under decay the panel's admission bound scales linearly with the window
(`rola/planner/schedule.py`'s `panel_exponent_bound`, `group_tokens` parameter) —
a `W = 4` window admits a quarter of the routing a `W = 1` window admits. Both facts
are why `select_window` (`rola/planner/schedule.py`) picks `GT` per routing rather
than the kernel picking a fixed one.

### <a id="values-shard"></a>The values-shard dial stays inert at every `GT`

The former `kFoldStagedGroups` constant — PINNED to 1, with a `static_assert` tripwire
naming the P25 contract's values-shard dial (item 4) as the thing that would need
building the day the fold held more than one staged group per atom residency — is
gone. `GT`/`window_groups` is its CAPABILITY REPLACEMENT, not a workaround: the fold
now DOES hold `W` staged groups per atom residency at `GT = 64`, which is exactly the
condition the tripwire named — and the dial is STILL inert, because the tripwire's
premise was about the cursor SPREAD, not the staged depth as such. All `W` slots of a
window are staged in the SAME phase of the SAME visit and released by the SAME
end-of-window `__syncthreads()`, so every atom holds every tile for exactly the
window's duration and the cursor spread stays zero. `k > 1` would still be `k`
byte-identical copies of tiles no code path distinguishes. The dial becomes live only
if the window's staging stops being one barrier-delimited phase; until then `k` is a
PLAN-TIME quantity, pinned at 1 by the host. Revival of the old constant, if the
tripwire's premise is ever restated: git `2decba5^`.

## The one precision story

bf16 storage everywhere — the staged operands, the amplitude streams, and, since P40,
the state's own numerator (`Shi`/`Slo`, [below](#state-planes)) — fp32 accumulation
throughout (the state's mass column, the clock, the row scales, and every
`mma.sync`'s D fragment including the fold's read-modify-write of the state planes),
fp64 torch reference as the oracle. There is no precision axis — no `MmaPrec`, no
`PREC` template parameter, no tf32x3 arm. Every layout and proof below is stated for
bf16 alone.

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
swizzled `StageLayout` the staged operands and the state planes (`Vw`, `Shi`, `Slo` —
`WT` stays pitched, [below](#stage-format)) are stored under, together with the
proofs below that interrogate it. The `ldmatrix` copy atoms that read
that layout are the tile layer's ([`mma_tile.md#cute-pin`](mma_tile.md#cute-pin)),
and the split is the point — the library owns the address function in one file and
the fetch in the other, and `stage_offset_matches_cute` ([below](#stage-offset)) is
what keeps the closed form honest to it.

---

# Part I — the compile-time budget

## <a id="reg-budget"></a>4. `consumer_reg_budget` — a MEASURED kernel fact, not an architecture fact and not a threshold

```
consumer_reg_budget(BC, decay, DV, D) = 128
```

It is ONE number for every key, and the reason it is flat is the reason it is a
measurement rather than a dial: **the budget's only channel to codegen is the CTA count
in `__launch_bounds__`.** `ptxas` is given `regs_per_sm / (kConsumerThreads * CTAs)`
registers, so the reachable allocations on a 65,536-register SM at 256 threads are the
four the divisor admits — and the spill-vs-budget curve HAS NO INTERIOR POINT. Measured
on the flagship entry `consumer_forward_kernel<64, 64, 2, false>`, single translation
unit, `ptxas -v`, the census instrument `-DROLA_FORCE_CTAS=N` supplying the bound:

| CTAs | the budgets that yield it | regs/thread `ptxas` takes | spill stores | spill loads |
|---:|---|---:|---:|---:|
| 4 | <= 64 | 64 | 144 | 336 |
| 3 | 65..85 | 80 | 52 | 140 |
| 2 | 86..128 | 121 | 0 | 0 |
| 1 | >= 129 | 120 | 0 | 0 |

**The kernel's true register working set is 121.** At three CTAs the cap is 85 and the
allocator spills 52/140 bytes to reach it; at two it spills nothing, and a fourth
budget rung buys nothing beyond the second. `128` is the smallest clean value that
yields the two-CTA bound, and it is what ships. Every parameter is accepted and
ignored (`(void)BC;` and the rest), stated explicitly rather than silently dropped: no
axis of the matrix moves the working set enough to change which rung it lands on.

**What the third CTA was costing, and what its shared memory bought.** Dropping to two
CTAs frees roughly 20 KB of shared memory per CTA that the residency arithmetic was
holding back. A quarter of it pays for [the visit scratch's parity pair](../internals/consumer_kernel.md#visit-parity),
which is what deletes the end-of-visit rendezvous; the rest goes to
[staging depth](#pf-stages), which rises on every arm in the matrix (`S = 3 -> 4` at the
flagship, matrix minimum 3). The occupancy loss and the register/barrier/depth gain are
one decision, and the acceptance measurement is what adjudicates it — this budget is
not a free raise and is not recorded as one.

Reproduce with `python tools/ratify.py --census`, which compiles the shipped
instantiation shards and reports per structural key; if the kernel's inner loop
changes, re-run it and update this table, not prose about it. The per-arm SPILL RATCHET
lives in `tools/manifests/sm_{80,86}.json` and is a ratchet rather than a spill ban: at
this budget every consumer arm holds zero spill bytes except
`<64, 64, 4, true>`, the widest decay carve, which holds 4/4.

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
BC=16, DV=32, D=1, DECAY=false          smem = 17,008 B  reg_budget = 128
  by_smem    = 102400 / (17008 + 1024) = 5
  by_threads = 1536 / 256              = 6
  by_regs    = 65536 / (256 * 128)     = 2
  ctas_for_residency = min(5, 6, 2) = 2

BC=64, DV=64, D=2, DECAY=true           smem = 49,920 B  reg_budget = 128
  by_smem    = 102400 / (49920 + 1024) = 2
  by_threads = 1536 / 256              = 6
  by_regs    = 65536 / (256 * 128)     = 2
  ctas_for_residency = min(2, 6, 2) = 2
```

At `BC = 16` the REGISTER term binds and the shared-memory term has slack; at
`BC = 64` the two agree exactly, which is not a coincidence — the depth derivation
spends shared memory until the carve's CTA count falls to the register term's, so the
two rails meet by construction ([`pf_stages`](#pf-stages)). The thread term never
binds at 256-wide blocks, and it is still load-bearing: `ptxas` DISCARDS an
out-of-range `__launch_bounds__` rather than clamping it, so a missing thread term is
silent everywhere the shared-memory term alone would claim more CTAs than 256-wide
blocks fit.

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

Every term of `consumer_smem_bytes<BC, DV, D, DECAY, GT>()`, in carve order (the
kernel body's pointer arithmetic follows this exact sequence). `W = window_groups(GT)`
is the window's group count; every region marked "PER SLOT" is `W` independent slots
of the group's own shape ([`#window`](#window)), and `Gm` is the one exception — a
single buffer at the WINDOW's extent:

| region | extent | bytes | what it is |
|---|---|---|---|
| `Shi`, `Slo`, `Smass` | two `[BC][DV]` bf16 planes + `[BC]` fp32 | `BC*(DV+1)*4` | the STATE: `Shi`/`Slo` are the numerator (a value's bf16 rounding and its residual), `Smass` the fp32 mass column, kept out of the planes ([below](#state-planes)) |
| `dig` | `[D][BC]` int | `D*BC*4` | per-owner hoisted leaf digits ([`consumer_kernel.md#digit-hoist`](consumer_kernel.md#digit-hoist)) |
| `WT` | `W` slots of `[BC][kWtPitch]` bf16, `align4` | `align4(W*BC*24*2)` | write coefficients, LEAF-MAJOR, pitch `kWtPitch = kGroupTokens+8 = 24` ([below](#stage-format)) |
| `Vw` ∪ `clk` | the union, `align4` | `vw_union_bytes<BC,DV,D,DECAY,GT>()` | `W` slots of the scale-folded value tile; on `DECAY` ALIASES the WINDOW's clock prefix ([below](#vw-clk-union)) |
| `Rf` | `W` slots of `frag_a_words(BC)` u32 | `W*(BC/16)*128*4` | read coefficients, MMA-FRAGMENT ORDERED, per slot ([below](#frag-order)) |
| `Gm` | `[GT][GT+8]` bf16, ONE buffer | `GT*(GT+8)*2` | the WINDOW Gram, pitch `GMP = GT+8` — an `ldmatrix` SOURCE (GEMM 3's A and the intra mass term's), so its base carries the same 16-byte alignment `WT`/`Vw` do. Only the block lower triangle is written or read ([`#window`](#window)) |
| `log_keep`, `p_ref`, `s0` | `3 * [BC]` fp32, `DECAY` only | `DECAY ? 3*BC*4 : 0` | `DECAY`'s three per-leaf vectors |
| `g_ring` | `sched_slots(GT) * kSchedSlotInts` int | `sched_slots(GT)*(kGroupTokens+4)*4` | the [cooked schedule ring](#sched-ring): `sched_slots(GT) = W*(kSchedDepth+1)` group descriptors, each ONE GROUP's `kGroupTokens` ascending token indices plus `gN`/read-mask/write-mask/flags |
| `g_up` | `sched_slots(GT) * 2 * kGroupTokens` fp32 | `sched_slots(GT)*2*16*4` | the ring's per-owner [upper-product](consumer_kernel.md#upper-product) pairs — `sr`/`sw`, cooked beside the member word |
| producer cursor state | `(8 + 5*32)` int | `168*4 = 672` | the [producer's cursor](consumer_kernel.md#producer-state) (7 uniform scalars, 1 pad) and its 5 32-lane cook planes — rests here between cooks so no scheduler state is register-live across the visit body |
| chunk descriptors | `3 * amp_slab_chunks(BC,D)` int | `3*chunks*4` | the prefetch's per-owner `ch_lvl`/`ch_src`/`ch_atoms` ([below](#prefetch-slab)) |
| `pfR`, `pfW` | `2 * [GT][amp_slab_pitch(BC,D)]` bf16, `align16` base | `2*GT*pitch*2` | the AMPLITUDE PREFETCH SLABS — one row per window token per side, filled by `cp.async` a visit ahead ([below](#prefetch-slab)) |

`cols = DV + 1` is unconditional (the mass column always exists — normalization is no
longer a template axis): it is the HBM row width `p.state`/`p.ws_state` carry, and the
SMEM state's own carve (`Shi`/`Slo`/`Smass` above) is exactly `BC*(DV+1)*4` bytes —
two pad-free bf16 planes plus one fp32 column, no bank-decorrelation pad, because the
planes are read by `ldmatrix` under a swizzle rather than by a scalar fragment map
([below](#state-planes)).

**Two worked totals**, computed by hand from the table above (both independently
cross-checked against `consumer_smem_bytes` in Python):

```
BC=16, DV=32, D=1, DECAY=false, GT=16 (W=1, kSchedDepth=2, sched_slots=3):
  Shi/Slo/Smass=2112  dig=64  WT=768  Vw∪clk=1024
  Rf=512  Gm=768  decay-extra=0
  g_ring=240  g_up=384  producer-state=672  chunk-desc=36
  core (align16) = 6,592   slabs = 2*16*24*2 = 1,536
  total = 8,128 B  (~7.9 KiB)

BC=64, DV=64, D=2, DECAY=true, GT=16 (W=1, kSchedDepth=2, sched_slots=3):
  Shi/Slo/Smass=16640  dig=512  WT=3072  Vw∪clk=4352
  Rf=2048  Gm=768  decay-extra=768
  g_ring=240  g_up=384  producer-state=672  chunk-desc=120
  core (align16) = 29,584   slabs = 2*16*80*2 = 5,120
  total = 34,704 B  (~33.9 KiB)
```

The `BC = 64` decay arm is the one whose residency the prefetch slab costs (3 CTAs/SM
to 2); the `BC = 64` NODECAY arm keeps its three (31,632 B). At `GT = 64` (`W = 4`)
the slab's contribution is linear in `W` where the Gram's is quadratic, and the
`BC = 64` window arms stay at one CTA/SM.

These are the same two `GT=16` shapes [the residency section](#residency) plugs into
`ctas_for_residency`. No other shape's byte total is stated here; plug the formula
above into `consumer_smem_bytes` for any other `(BC, DV, D, DECAY, GT)` — that function,
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
into `g_ring`, one of `sched_slots(GT) = window_groups(GT) * (kSchedDepth + 1)` slots
([`consumer_kernel.md#sched-ring`](consumer_kernel.md#sched-ring)) — a WHOLE WINDOW of
run-ahead, because a visit reads `window_groups(GT)` descriptors at once and the
producer must still be cooking into slots nobody is reading. Every consumer
warp reads a descriptor rather than deriving it: this SMEM block is bought by the
consumption multiplicity that earns it, one writer against `n_atoms + 1` readers.

`g_ring[slot]` is `kSchedSlotInts = kGroupTokens + 4` ints, ONE GROUP per slot: the
group's `kGroupTokens` gathered token indices ascending, then `gN` (member count, `0` =
stream end), the read atom mask, the write atom mask, and the side flags. `g_up[slot]`
is the matching pair of `kGroupTokens`-float
[upper products](consumer_kernel.md#upper-product), `sr` then `sw`, cooked in the same
pass. The ring stays group-shaped even though the visit spans a window — a window's
slots are the `window_groups(GT)` CONTIGUOUS slots starting at its first, since
`window_groups(GT)` divides `sched_slots(GT)` by construction.

**`kSchedDepth` is 2, and the prefetch is why.** The amplitude slab for window `w + 1`
is issued from inside window `w`'s visit — that overlap IS the mechanism — so window
`w + 1`'s token list must already be published while window `w` is staging. At depth 1
it is not: the producer cooks `w + 1` DURING visit `w` and publishes it at that
visit's end. Depth 2 costs one more window of descriptor slots (208 B at the group
quantum) and is the null configuration from here; `ROLA_SCHED_DEPTH` survives as the
sweep knob it always was, with 2 as its floor in practice.

### <a id="prefetch-slab"></a>The amplitude prefetch slab

Two slabs, `pfR` and `pfW`, one row per WINDOW TOKEN, each row holding every SPANNED
level's contiguous digit range concatenated. They are the LAST regions of the carve so
that `cp.async`'s 16-byte destination alignment is one `align16` rather than a
constraint every region before them would have to carry. Both take the SAME layout: the
digit map is side-independent, and `Plan.amplitude_pitch` gives both streams the same
alignment quantum.

Level `l` contributes `align8(lead_l + span_l)` elements, where
`span_l = min(modulus_l, (BC-1)/stride_l + 1)` is the owner's contiguous digit range at
that level and `lead_l = base_l & 7` is the alignment slack the 16-byte copy leaves in
front of it. The spanned spans multiply to `BC` (the owner rectangle's invariant), and
`sum_l align8(7 + f_l)` over `prod_l f_l == BC` is maximised by ONE span of `BC` with
the rest at 1 — so the compile-time reservation is

    amp_slab_pitch(BC, D) = align8(7 + BC) + 8 * (D - 1)      elements

and `amp_slab_chunks = amp_slab_pitch / 8` is the 16-byte chunk count a row holds. The
kernel ADDRESSES at this compile-time pitch and fills only the runtime prefix, which is
what makes the per-token offset a constant multiply.

Three per-owner descriptors accompany them, one entry per chunk: `ch_lvl` (which
spanned level the chunk belongs to), `ch_src` (its element offset inside that level's
amplitude row, already aligned down), and `ch_atoms` (the SET OF ATOMS whose digits it
carries). The last is what makes the copy per-atom: a chunk is fetched iff some atom
live on that side in that group needs it, and zero-filled otherwise
([`consumer_kernel.md#prefetch`](consumer_kernel.md#prefetch)).

**Per-atom PACKING is refused, and the derivation is short.** An owner's leaf set is
the CARTESIAN PRODUCT of the levels' digit ranges, so giving each atom its own copy of
its sub-range duplicates every factor an atom shares with its neighbours — at
`<BC = 64, D = 2>` that is 4-8x the shared memory for exactly the same DRAM sectors.
The liveness, which is what per-atom was for, is bought at chunk granularity instead
and costs nothing.

The producer's OWN cursor (word cursor, chunk base, live-word mask, selected word,
remainder — 7 scalars, uniform across the producer warp) and its cook planes (5
32-lane arrays: the cooked chunk's `fr`/`fw`/`member` words and the selected word's
read/write atom-bitlist rows) rest in a third SMEM block between cooks
([`consumer_kernel.md#producer-state`](consumer_kernel.md#producer-state)) rather than
in registers, for the measured reason that governs all scheduler state here: none of
it may be register-live across the visit body, where it would compete with the four
GEMMs' working set. Every reload happens at the point
of use inside the next cook, and the ring's publication is the cook's own monotone
ready cursor ([`consumer_kernel.md#cook-cursor`](consumer_kernel.md#cook-cursor)) —
no barrier is involved on that path, and there is no group-entry barrier.

## <a id="vw-clk-union"></a>11. The `Vw` / `clk` union, and why it is sized in BYTES rather than elements

```
vw_union_bytes<BC,DV,D,DECAY,GT>() = align4(max(vw_bytes, clk_bytes))
  vw_bytes  = window_groups(GT) * kGroupTokens * DV * sizeof(bf16)   // W slots
  clk_bytes = DECAY ? GT * (BC+4) * sizeof(float) : 0                // ONE window buffer
```

`Vw` is `W = window_groups(GT)` SLOTS of the group's own `kGroupTokens * DV` shape —
one tenant of the union scales with the window like every other staged operand. `clk`
(the visit's per-leaf inclusive clock prefix, `[GT][BC+4]` fp32, token-major) does NOT:
it is ONE buffer over the whole window, because the clock prefix runs across the
window's tokens — the reference frame the window shares — not per group
([`#window`](#window)).

`clk` is DEAD before `Vw` is live within one visit — the clock prefix is fully
consumed building `Rf` and `WT` before the value staging that fills `Vw` runs, and a
`__syncthreads()` (HAZARD `clk-dead`,
[`consumer_kernel.md#clk-dead`](consumer_kernel.md#clk-dead)) marks the exact point
past which `Vw` may overwrite `clk`'s bytes. So `clk` ALIASES `Vw`'s storage rather
than getting its own region.

**The max is over BYTES, and that is load-bearing** because the two tenants do not
share an element size: `Vw` is bf16, `clk` is fp32, so an element-wise max would
under-allocate by 2x, silently. `clk` is the larger tenant at every shipped `DECAY`
shape (`GT*(BC+4)*4` against `W*kGroupTokens*DV*2`; at `GT=16,BC=64,DV=64` that is
4,352 against 2,048), so on the decay arms narrowing `Vw` frees no shared memory at
all — the reservation is the max, whatever the smaller tenant does. `Vw` is still
ADDRESSED at its own layout; only the RESERVATION is the max.

`clk` is declared AT THE CARVE (`float* clk = reinterpret_cast<float*>(carve);`)
rather than assigned from `Vw` later, because the two element types make that
assignment not type-check — the alias is the load-bearing fact, stated where the
pointer is born. The kernel body carries a `static_assert` pinning
`GT * CKP * sizeof(float) <= vw_union_bytes<BC,DV,D,DECAY,GT>()` so the sizing is
checked against the actual use, never assumed to agree.

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

## <a id="stage-layout"></a>13. The swizzled `StageLayout`, owned by CuTe

`Vw` and the state's two bf16 planes (`Shi`/`Slo`, [below](#state-planes)) are stored
under the library's own swizzled layout,
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
UNIQUE pad that passes both** — pinned by `pitched_ldsm_banks_ok<kMmaKQuantum,
kGroupTokens, kWtPitch>()` (the `ldmatrix` phase, including the 16-byte alignment `WT`'s row bytes
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

## <a id="state-planes"></a>14.1. The state's two bf16 planes, and the fp32 mass column kept out of them

The state's NUMERATOR is `Shi`/`Slo`, two `StageLayout` buffers at the state's own
shape (`[BC][DV]`, no pad) — the SAME layout `Vw` is stored under, at `state_index<DV>(i,
c) = stage_offset<DV>(i, c)`. `Shi` holds each value's bf16 rounding, `Slo` its
residual: together the pair carries ~16 mantissa bits where a single bf16 narrowing
carries ~8, and that is the precision the STATE ITSELF is stored at across the
recurrence — the fold reads the pair, sums in fp32 registers and re-splits at its one
writeback.

**GEMM 1 READS `Shi` ALONE**, and that is the shipped contract rather than a
concession: a fp32-resident state narrowed its B operand to plain bf16 at the fragment,
and `Shi` holds exactly that narrowing, so the readout's numerics are unmoved and its
`mma.sync` count is unmoved with them. `Slo` is not a readout operand — it is what
keeps the state's own storage from decaying to 8 mantissa bits between visits, and its
cost is the merge and split at the fold's read-modify-write, not an extra multiply. The
plane form is what lets GEMM 1 fetch its operand with the `ldmatrix` atom that already
serves `Vw` ([`mma_tile.md#mma-tile-fraga`](mma_tile.md#mma-tile-fraga)) instead of
packing it from scalar shared loads at every use. The operand that genuinely needs
compensation is the DENOMINATOR, and it gets it in fp32 — below.

`state_plane_elems<BC, DV>() = BC * DV` is the element count of ONE plane — no pad,
which is what makes the swizzle's XOR field legal.

The state's DENOMINATOR, `Smass`, stays fp32 and stays OUT of the planes, in its own
`[BC]` array: it is the denominator of a ratio, its consumers
([`mma_tile.md#mass-column`](mma_tile.md#mass-column)) are broadcast-B matvecs no
`ldmatrix` can serve, and keeping it out is what leaves the planes exactly `BC * DV`
elements rather than `BC * (DV + 1)`. The carve is therefore `BC*(DV+1)*4` bytes total
— two bf16 planes (`2 * BC * DV * 2` bytes) plus one fp32 column (`BC * 4` bytes) —
where a single fp32-resident state with a bank-decorrelation pad cost `BC*(DV+4)*4`:
the `+4` pad existed to decorrelate banks for a SCALAR fragment map (`k = k0 + 2r`
collided up to four ways at a row extent of `DV`, a multiple of 32), and the planes
need no such pad because they are read by `ldmatrix` under a swizzle instead.

**The state planes answer the same four questions [`#stage-layout`](#stage-layout)
answers for `Vw`, at the state's shape, plus one more for the fold's paired access:**

1. **`stage_offset_matches_cute<BC,DV>()`** — the closed-form address IS CuTe's layout.
2. **`stage_layout_is_bijection<BC,DV>()`** — the swizzled layout tiles `[0, BC*DV)`
   exactly.
3. **`stage_ldsm_banks_ok<BC,DV>()`** — GEMM 1's `ldmatrix.x2.trans` fetch is
   conflict-free and 16-byte aligned.
4. **`stage_pair_banks_ok<BC,DV>()`** — PROOF 4, new for the state: a CHANNEL PAIR
   (`(i, c)` and `(i, c+1)` for even `c`) is one 32-bit word of one plane, and every
   paired access is conflict-free. The swizzle XORs bit 3 and above of the column, so
   it cannot separate an even/odd pair — the two elements stay adjacent under the
   swizzle exactly as they are unswizzled. This enumerates the two access patterns
   that read or write a pair at once — the fold's `(g = lane >> 2, r = lane & 3)` map
   at every column tile, and the flat pair loops the entry-state staging and the
   exit-state publication both run — and requires 32 distinct banks in each. It is
   what keeps the fold's read-modify-write at two loads and two stores per row rather
   than four.
5. **`stage_store_banks_ok<BC,DV>()`** — the entry-state staging store into `Shi`/`Slo`
   is conflict-free.

`merge_bf16x2` is `split_bf16x2`'s inverse (`mma_tile.md#split-bf16x2`): the two
planes' words read back as the fp32 pair they were split from, to ~16 mantissa bits.
It is the ONE place a plane pair is read as a NUMBER rather than as an MMA operand —
the fold's read half and the entry/exit publication's, never the readout, whose B
operand is `Shi` alone
([`consumer_kernel.md#gemm-readout`](consumer_kernel.md#gemm-readout)). `state_pair`
/ `state_pair_load` address a channel pair as one 32-bit word (`unsigned&` / `unsigned`
respectively) for exactly this reconstruction and for the fold's writeback.

Accumulation is unmoved by any of this: the fold reads the pair, sums in fp32
registers, and splits back to `Shi`/`Slo` at the one writeback — the semantic change
is the state's own STORAGE precision between visits, not what it accumulates in
([`docs/architecture/fp32-accumulators.md`](../architecture/fp32-accumulators.md)
distinguishes the two and is the authority on which is which).

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
