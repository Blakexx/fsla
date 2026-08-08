# `csrc/rola/src/mma_tile.cuh` — the consumer's MMA tile layer

The fragment machinery and the four GEMMs' inner sequences: the fragment-ordered A
operand's element map, the `ldmatrix`-fed operand fetches, the GEMM assemblies built
from them, the MASS COLUMN's broadcast-B forms, and the compile-time cross-checks that
keep CuTe — not this code — the authority on what a fragment contains.

**The mass column is a matrix column.** Its denominator terms are matvecs against a
single column, and an `n8` tile carries a matvec by BROADCASTING that column across
all eight of its `n` slots: every output column then holds the same value and the
tile is read at `n = 0`. Seven eighths of the fragment is redundant and that is the
whole price — one `mma.sync` against the scalar reduction it replaces.

**bf16 is the only operand format.** One precision story: bf16 storage, fp32
accumulation, fp64 torch reference as oracle. There is no `PREC` axis and no `tf32x3`
arm — both died with the P26 endgame rewrite (`git show 52a67d8^:csrc/rola/src/mma_tile.cuh`
has the last version that carried them); every GEMM below is the bf16 pass,
unconditionally.

**The group is the tile every GEMM atom fetches.** The PRODUCER's work unit is a GROUP
of `kGroupTokens == 16` gathered tokens — the same constant as `kMmaKQuantum`, the MMA
k-quantum, by construction. There is no `BT` axis: a group is always full except a
segment's final flush, and every contraction below runs over one atom (`kMmaKQuantum`
leaves) or one group (`kGroupTokens` tokens) at a time — that stays true even though
the CONSUMER's own quantum, the WINDOW (`GT in {16, 64}`, [`smem_layout.md#window`](smem_layout.md#window)),
now spans `W = GT / kGroupTokens` groups per visit: `WT`/`Vw`/`Rf` are `W`
INDEPENDENT SLOTS of the group's own shape, so every fetch and every GEMM atom below
still operates on one 16-token group's worth per invocation, called once per slot the
kernel body's loop indexes ([`consumer_kernel.md`](consumer_kernel.md)). Only the
window Gram `Gm` is addressed at the window's own pitch (`GMP = GT + 8`), and its
`ldmatrix` atom (`ldsm_frag_a` at `GMP`, [below](#ldsm-atoms-a)) still fetches one
group-sized `[16 x 16]` sub-block at a time.

Where an operand LIVES is [`smem_layout.md`](smem_layout.md) (`vw_index`, `wt_index`,
`stage_offset`) — every address this file's routines take comes from there, which is
what makes it impossible for an atom fetch and a scalar walk over the same buffer to
disagree. Which warp runs which GEMM, and in what phase order, is
[`consumer_kernel.md`](consumer_kernel.md).

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

# Part III — the operand maps

## <a id="frag-a-slot"></a>21. `frag_a_slot` — the fragment-ordered A operand's element map

The read panel `R` (the readout's A operand, and the Gram's) is synthesized directly
into fragment order in SMEM rather than stored row-major and indexed: word `w` of lane
`lane` at k-step `ks` lives at `Af[ks * 128 + lane * 4 + w]`, so a warp loads a whole
16x16 k-step's worth of A with one 128-bit `uint4` per lane and zero address
arithmetic. `frag_a_slot(s, m, k)` is the FORWARD map — the `(m, k)` pair that storage
slot `s` holds, one k-step per `kMmaKQuantum` leaves, 128 words per step:

```cpp
ks = s >> 7;  rem = s & 127;  lane = rem >> 2;  w = rem & 3;
m = (lane >> 2) + ((w & 1) << 3);
r = lane & 3;
k = 16 * ks + ((w >> 1) << 3) + 2 * r;
```

(`k` names the low element of the packed `bf16x2` pair the word holds; the high
element is `k + 1`.) `frag_a_slot` has no inverse in this file: the mass column no
longer reads an element back out of the fragment — [below](#mass-column) — so there is
nothing left to keep in sync with the forward map.

## <a id="stage-put"></a>22. Storage writes: `stage_put`, `stage_put2`

`stage_put(d, x)` stores a bf16 OPERAND BUFFER's element, and is where
[`readout_mma.md`](readout_mma.md#rule-3-bf16)'s §12 rule 3 clamp lives:
`d = __float2bfloat16(clamp_bf16_finite(x))`, via the intrinsic rather than
`static_cast` because the torch build defines `__CUDA_NO_BFLOAT16_CONVERSIONS__`.
THE CLAMP BELONGS AT THE WRITE: every bf16 operand buffer (`WT`, `Vw`, `Gm`) is
written once per element per visit and read by an `ldmatrix` many times, so a
read-side clamp would be a per-consumer repetition of a per-producer fact — and an
`ldmatrix` has no place to put one. It is what makes the `ldmatrix` fetches below
bit-identical to a generic indexed read of the same buffer.

`stage_put2(d, x, y)` is the PAIRED form, and the panels' write sites take it wherever
the two destinations are adjacent and the first is at an even offset — which is every
fragment store in this file, because a D fragment's two output columns are `c0` and
`c0 + 1` and every pitch in the carve is even (asserted, not assumed). It narrows the
pair through
[`pack_clamp_bf16x2`](readout_mma.md#rule-3-bf16) and issues ONE 32-bit shared store
where the scalar form issued two 16-bit ones, at bit-identical results. The load/store
unit is this kernel's busiest pipe; the width is where its instruction count is bought
back.

## <a id="mma-tile-fraga"></a>23. `mma_tile_fragA_stateB` — GEMM 1's readout, the state fetched by `ldmatrix`

`y += Af . S`, fp32 accumulate, one warp, for A FRAGMENT-ORDERED (one `uint4` per
k-step) and B the state's numerator plane `Shi` — a `StageLayout` buffer at the
state's own shape, fetched by `stage_frag_b<DV>` exactly as `Vw` is
([below](#ldsm-atoms)).

**The readout reads `Shi` alone, and that is the shipped contract, not a
concession.** The fp32 form this replaced narrowed its B to plain bf16 at the
fragment (`pack_bf16x2(clamp_bf16_finite(x))`), so `Shi` — which holds exactly that
narrowing — gives the numerator the SAME operand it always had. `Slo` exists to carry
the STATE's own storage precision across the recurrence (the fold's
read-modify-write, [`consumer_kernel.md#gemm-fold`](consumer_kernel.md#gemm-fold)
reads the pair, sums in fp32 and re-splits); reading it here as well would be an
ADDED correction term the family budget never asked for, and it was measured at
+34.4% `mma.sync` and +16.6% instructions on the dense cell for no wall (P40,
`workflows/perf/p40_pipeline.md`). The denominator is the operand that genuinely
needs compensation, and it gets it — in fp32, from `Smass`
([below](#mass-column)).

**The k-loop is a two-stage software pipeline** ([below](#two-stage-pipeline)), and
that is what the plane form is FOR: k-step `ks + 1`'s fragments are issued before
k-step `ks`'s `mma.sync`, covering the shared-memory fetch latency with the multiply
of its predecessor. An fp32 B could not be pipelined this way — its "fetch" was four
scalar loads and six ALU instructions with nothing for the multiply to hide behind.

**The k-loop's trip count is COMPILE-TIME, with no runtime bound.** `KMAX` is the
contraction extent at both call sites (`BC` for the dense arm, `kMmaKQuantum` for a
per-atom call), so the fetch of the loop's final stage is simply never emitted — there
is no `K`/early-exit pair to fold away, unlike the deleted generic form this
succeeded (`docs/internals/DELETIONS.md`).

The `[16 x 8]` D/C fragment's element map, shared by every writeback in this file:

```
d0, d1 -> (row g,     col 2r, 2r+1)
d2, d3 -> (row g + 8, col 2r, 2r+1)
```

for `g = lane >> 2`, `r = lane & 3` — the PTX `mma.sync.aligned.m16n8k16` layout,
taken verbatim rather than re-derived.

## <a id="ldsm-atoms"></a>24. The `ldmatrix` atoms

### `stage_frag_b` — a `StageLayout` buffer as B, fetched by one `ldmatrix.x2.trans`

Two buffers are fetched this way — `Vw` (B for GEMM 3, the intra term `y += G . Vw`,
and GEMM 4, the fold `S += WT^T . Vw`) and the state's `Shi` plane (B for GEMM 1,
[above](#mma-tile-fraga)) — and both want the same 16(k) x 8(n) block per k-step, so
ONE routine (`stage_frag_b<COLS>`) serves both rather than one written per buffer.
`Slo` is a `StageLayout` buffer too and could be fetched the same way, but nothing
fetches it as an MMA operand: it is read only as a scalar pair by the fold
(`state_pair_load`, [below](#state-pair)). A generic indexed path would build the
block with four scalar SMEM loads, four widenings, four clamps and two packs per lane
per k-step; `ldmatrix.x2.trans` is one instruction that delivers the same two
registers, and `.trans` is exactly right because both buffers are stored `k`-major
(`n`/column contiguous) while the fragment wants `k`-adjacent pairs at a fixed `n`.

**Bit-identity, which is why this is a LAYOUT change and not a VALUE change.**
`ldmatrix` delivers the STORED 16-bit patterns. A generic indexed path delivers
`pack_bf16x2(clamp_bf16_finite(__bfloat162float(x)))`, and those agree bit for bit iff
`x` is a finite bf16 — which the store guarantees: `stage_put` applies the clamp where
the fp32 value last exists. A finite bf16 widens to fp32 exactly, passes the clamp
unchanged and rounds back to itself, so the composition is the identity and no
summation is reordered.

### `wt_frag_b` — `WT` as GEMM 2's B

The SAME atom against the SAME algebra: `WT` is stored leaf-major (`k` = leaf, `n` =
token contiguous) and the fragment wants `k`-adjacent pairs at a fixed `n`, which is
`.trans` again. `WT16` points at the 16-row block (one atom) being contracted; the
only difference from `Vw` is which axis is called what.

### <a id="ldsm-atoms-a"></a>`ldsm_frag_a` — a `[16 m x 16 k]` bf16 A operand, one `ldmatrix.x4`, at a RUNTIME PITCH

The shared primitive behind every `ldmatrix`-fetched A in this file: A `k`-adjacent —
the layout `ldmatrix` reads WITHOUT transposing — so a `[16 m x 16 k]` block is one
`ldmatrix.x4`, against a generic path's eight scalar loads, eight widenings, eight
clamps and four packs per lane. It takes the row pitch (`lda`) as an argument rather
than assuming one, which is what lets both of its callers share it: `wt_frag_a` is a
one-line wrapper at `kWtPitch` (`WT`'s atom block, GEMM 4's A), and [the mass column's
tile](#mass-column) calls it directly at `GMP` (the Gram, GEMM 3's A) — `Gm` sits at a
pitch `pitched_ldsm_banks_ok` licenses for exactly this fetch, which is what makes the
Gram's `ldmatrix` legal at all.

The addressing is the atom's own: `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8
matrices stacked (32 rows of 8 halves), not a 16x16 block, so lane `l` addresses
matrix `l >> 3`, row `l & 7` — and the `(m, k)` reading of the four matrices is the
CALLER's: matrix `i` is `m = 8 * (i & 1) + row`, `k = 8 * (i >> 1) + col`. That reading
is what makes the delivered registers `a0..a3` land in the same order the fragment-
ordered A consumers expect, and `ldsm_n_dst_map_matches` checks it over all 256 slots.

## <a id="two-stage-pipeline"></a>24.1. The two-stage software pipeline, as a form

Three loops whose operand sequence is known at compile time run this same shape: GEMM
1's readout k-loop ([above](#mma-tile-fraga)), the mass column's readout k-loop
([below](#mass-column)), and GEMM 2's DENSE atom arm (`mma_tile_fragA_wtB_atoms`,
[below](#gemm-assemblies)). Stage `i + 1`'s fragments (an `ldmatrix` and/or a fetch)
are issued BEFORE stage `i`'s `mma.sync`, and the stage in flight is copied to a
second set of registers — that copy IS the second buffer, spent from the register
budget rather than from a second `ldmatrix` slot:

```cpp
fetch(0);
for (int i = 0; i < N; ++i) {
  const auto staged = current;       // the copy IS the second buffer
  if (i + 1 < N) fetch(i + 1);       // stage i+1's fetch, issued before...
  mma_m16n8k16(..., staged, ...);    // ...stage i's multiply
}
```

The motivation is a specific issue-gap pair this kernel pays into on every
`ldmatrix -> mma.sync` step: a fixed-latency dependency (`stall_wait`) behind a
shared-memory one (`short_scoreboard`). Pipelining hides the fetch of stage `i + 1`
behind the multiply of stage `i` instead of behind a sibling warp's progress. An fp32
B could never run this form — its "fetch" was four scalar loads and six ALU
instructions with no single latency to hide, which is exactly what the state's plane
form ([`consumer_kernel.md#state-planes`](consumer_kernel.md#state-planes)) replaced
with one `ldmatrix`.

GEMM 2's SPARSE arm (`mma_tile_fragA_wtB`, [below](#gemm-assemblies)) does NOT run this
form, and the reason is structural, not an omission: its live-atom walk is a runtime
loop over set bits, so the next atom's address does not exist until the current one
retires — there is no "stage `i + 1`" to issue early.

## <a id="split-bf16x2"></a>25. `split_bf16x2` — an fp32 pair delivered as two bf16 words

A fp32 pair narrowed as a ROUNDED VALUE plus its RESIDUAL, packed into two separate
`bf16x2` words. Two `mma.sync` against the same A fragment then carry ~16 mantissa
bits of the B operand instead of 8 — one extra instruction per k-step for a B operand
that has nothing to cancel its rounding against, which is exactly [the mass column's
readout term](#mass-column)'s situation. §12 rule 3's guard applies to BOTH words: the
residual of a saturated value is itself out of range and must be narrowed the same way.

`split_bf16x2` has a SECOND role that is not an MMA operand at all: it is how the
state's two planes are WRITTEN. Every value that enters `Shi`/`Slo` — the entry state,
the decay rescale, the fold's writeback — goes through it, which is what makes the
plane pair a storage format rather than a fetch trick.

## <a id="state-pair"></a>25.1. `merge_bf16x2`, `state_pair`, `state_pair_load` — the state as a number

The plane pair's non-MMA face. `merge_bf16x2` is `split_bf16x2`'s INVERSE: it reads
the two planes' words back as the fp32 pair they were split from, to ~16 mantissa
bits. It is the fold's read half and the publication's, and it is the ONE place the
pair is read as a NUMBER rather than as an operand — the readout never calls it,
because the readout wants `Shi`'s bits, not their value ([above](#mma-tile-fraga)).

`state_pair<DV>` / `state_pair_load<DV>` address a CHANNEL PAIR of one plane as a
single 32-bit word. The swizzle XORs bit 3 and above of the column, so it cannot
separate `(i, c)` from `(i, c + 1)` for even `c`: the two elements stay adjacent, and
the fold's read-modify-write moves a pair per instruction rather than two scalars.
`stage_pair_banks_ok<ROWS, COLS>` ([`smem_layout.md#state-planes`](smem_layout.md#state-planes))
is the compile-time proof of both properties — the pair-in-one-word claim and the
conflict-freedom of the two access patterns that rely on it (the fold's `(g, r)` lane
map, and the flat pair loops of the staging and the writeback).

**The cost this face carries is the batch's own measured finding**, recorded because
it bounds what the plane form is worth: the merge and the split at the state's format
boundary are ~94% of P40's instruction delta on the dense cell, because the fold
REWRITES THE WHOLE STATE EVERY VISIT and every rewrite pays a conversion
(`workflows/perf/p40_pipeline.md` §3).

## <a id="mass-column"></a>26. The mass column's tiles

Two forms, both against a BROADCAST B — the column repeated across the tile's eight
`n` slots rather than varying with it, which is what turns a `[16 x K] x [K x 1]`
matvec into an `n8` MMA tile instead of wasting seven eighths of one:

- `mma_tile_fragA_massB` — the READOUT's term, `d += Af . m`, for a fragment-ordered
  A and `m` the CONTIGUOUS `[BC]` fp32 mass column `Smass`, COMPENSATED: `Smass` is
  the denominator of a ratio, so it rides [`split_bf16x2`](#split-bf16x2) instead of a
  single narrowing, and the tile is two `mma.sync`. Its k-loop is the same
  [two-stage pipeline](#two-stage-pipeline) the readout's is, for the same reason: the
  column's four scalar loads are a shared dependency the previous k-step's two
  multiplies can cover.
- `mma_tile_bcastB` — the intra term's and the fold's term, `d += A . v` for an
  `ldmatrix`-fetched A (`Gm` or `WT`'s atom block) and `v` a register-side vector (the
  row scale `sw`). <a id="bcast-frag-b"></a>`bcast_frag_b(v, lane, b0, b1)` packs it: the B fragment's `k` map
  is `2r + (w & 1) + 8 * (w >> 1)` and its `n` map is `g`, so a broadcast B is
  `g`-INDEPENDENT — four scalar loads and two packs for the whole fragment, whatever
  the tile's `n0`. §12 rule 3's guard applies here too, at the point the fp32 row
  scale narrows.

Output column `0` (lanes with `r == 0`) carries the matvec, rows `g` and `g + 8`: the
split-K emission reads both forms back at exactly that slot.

### <a id="gemm-assemblies"></a>The GEMM assemblies

- `mma_fold_tile` — GEMM 4, both operands `ldmatrix`-fetched: A = `WT` (`.N`), B =
  `Vw` (`.trans`). One atom's 16-row block; the contraction axis is the group's
  `kGroupTokens == 16` tokens, exactly one k-step.
- `mma_tile_vwB` — GEMM 3, `d += A . Vw` for `A` fetched by `ldmatrix` (`Gm`, exactly
  as the fold fetches `WT`) and `Vw` as B. The contraction is the group's 16 tokens:
  one k-step. Legal because `Gm`'s §12 rule 3 guard is at ITS store (`stage_put`),
  which is what makes an `ldmatrix` fetch of it bit-identical to a generic indexed
  read.
- `mma_tile_fragA_wtB` — GEMM 2's SPARSE arm, ONE ATOM: a fragment-ordered A (`Af16`,
  one atom's 128-word block) against `WT` fetched by `ldmatrix` (`WT16`, one atom's
  16-row block). The only thing this adds over `wt_frag_a`/`stage_frag_b` is issuing
  the single `mma_m16n8k16` once both operands are in registers. This is the tile the
  live-atom walk calls once per set bit — the runtime address sequence that keeps it
  out of the [two-stage pipeline](#two-stage-pipeline).
- `mma_tile_fragA_wtB_atoms<QPO>` — GEMM 2's DENSE arm, the WHOLE atom sequence in one
  call: every atom is live, so the sequence is known at compile time and runs the
  [two-stage pipeline](#two-stage-pipeline) over `QPO` atoms — atom `q + 1`'s
  fragments issued before atom `q`'s multiply. This is the arm `mask == kFullMask`
  selects, and it is where `BC = 64`'s four atoms would otherwise be four chained
  `ldmatrix -> mma.sync` dependencies in program order.

### <a id="cute-crosschecks"></a>The two library maps this file depends on, enumerated at compile time

`MMA_Traits::BLayout` is `(thr, val) -> n + 8k` over the atom's `[8 n x 16 k]` B tile;
`Copy_Traits<SM75_U16x4_LDSM_T>::DstLayout` is `(thr, val) -> BIT` over the same 16x8
half tile. Both must equal `k = 2r + (v & 1) + 8 * (v >> 1)`, `n = g` — which is the
hand-rolled `b0`/`b1` packing above, letter for letter. `lane & 15` is the `.trans`
atom's own source map (`Copy_Traits::SrcLayout` gives the first 16 lanes an address
each, at stride one row, and aliases lanes 16-31 onto them). For the `.N` atom the
source map must be `row = lane`, so that lane `l` addresses matrix `l >> 3`, row
`l & 7`.

| assert | message |
|---|---|
| `mma_b_map_matches()` | CuTe's B fragment map is not the hand-rolled one |
| `ldsm_t_dst_map_matches()` | `ldmatrix.x2.trans` does not deliver `b0`/`b1` |
| `ldsm_t_src_map_matches()` | `ldmatrix.x2.trans` row addresses moved |
| `ldsm_n_dst_map_matches()` | `ldmatrix.x4` does not deliver `a0..a3` |
| `ldsm_n_src_map_matches()` | `ldmatrix.x4` row addresses moved |

A library bump that moved either map fails the BUILD rather than the oracle.

### <a id="cute-pin"></a>The vendored CuTe pin

`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5. It is a CODEGEN INPUT, not a
packaging one — every line instantiated here is code `ptxas` assembles — which is why
`setup.py::_gate_vendored` hashes the subtree BEFORE the compile and `tools/ratify.py`
carries the same include path. It is used for exactly two things, and both of them are
LAYOUT: the `ldmatrix` COPY ATOMS (`SM75_U16x4_LDSM_T`, `SM75_U32x4_LDSM_N`) this file
issues, and the fragment maps (`MMA_Traits`) the cross-checks above enumerate. The
`mma.sync` calls stay hand-rolled PTX (`mma.cuh`); the correspondence between the two
is `static_assert`ed against the library's own maps rather than assumed.
