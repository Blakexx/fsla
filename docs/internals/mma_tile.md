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

## 22. Storage writes: `stage_put`

`stage_put(d, x)` stores a bf16 OPERAND BUFFER's element, and is where
[`readout_mma.md`](readout_mma.md#rule-3-bf16)'s §12 rule 3 clamp lives:
`d = __float2bfloat16(clamp_bf16_finite(x))`, via the intrinsic rather than
`static_cast` because the torch build defines `__CUDA_NO_BFLOAT16_CONVERSIONS__`.
THE CLAMP BELONGS AT THE WRITE: every bf16 operand buffer (`WT`, `Vw`, `Gm`) is
written once per element per visit and read by an `ldmatrix` many times, so a
read-side clamp would be a per-consumer repetition of a per-producer fact — and an
`ldmatrix` has no place to put one. It is what makes the `ldmatrix` fetches below
bit-identical to a generic indexed read of the same buffer.

## <a id="mma-tile-fraga"></a>23. `mma_tile_fragA` — a fragment-ordered A against a generic B

`d += Af . B`, fp32 accumulate, one warp, for A FRAGMENT-ORDERED (one `uint4` per
k-step) and B a generic indexed operand — B keeps the indexed path because each warp
wants a different `n`-slice of it and every use of this template picks B off `S`, the
state, which has no fragment-ordered form. `TB` is B's SMEM storage type (`float` for
the fp32 state) and `B_T` selects its orientation:

```
B_T == false : B element (k, n) at B[k * ldb + n]
B_T == true  : B element (k, n) at B[n * ldb + k]
```

**The k-loop keeps a compile-time trip count and an early exit, never a runtime
bound.** `KMAX` is the operand's allocated extent (`BC`, or `kMmaKQuantum` for a
per-atom call) and `K` is the realized one; a runtime loop bound around an MMA
sequence is un-unrollable and costs more than the k-steps it skips. Where the caller
passes `K = KMAX` the compare folds away and the emitted code is the plain unrolled
loop.

The `[16 x 8]` D/C fragment's element map, shared by every writeback in this file:

```
d0, d1 -> (row g,     col 2r, 2r+1)
d2, d3 -> (row g + 8, col 2r, 2r+1)
```

for `g = lane >> 2`, `r = lane & 3` — the PTX `mma.sync.aligned.m16n8k16` layout,
taken verbatim rather than re-derived.

## <a id="ldsm-atoms"></a>24. The `ldmatrix` atoms

### `vw_frag_b` — `Vw` as B, fetched by one `ldmatrix.x2.trans`

`Vw` is B for BOTH of its consumers — GEMM 3 (the intra term, `y += G . Vw`) and
GEMM 4 (the fold, `S += WT^T . Vw`) — and both want the same 16(k) x 8(n) block per
k-step. The generic path builds that block with four scalar SMEM loads, four widenings,
four clamps and two packs per lane per k-step; `ldmatrix.x2.trans` is one instruction
that delivers the same two registers, and `.trans` is exactly right because `Vw` is
stored `k`-major (`n` contiguous) while the fragment wants `k`-adjacent pairs at a
fixed `n`.

**Bit-identity, which is why this is a LAYOUT change and not a VALUE change.**
`ldmatrix` delivers the STORED 16-bit patterns. The generic path delivers
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
is what makes the delivered registers `a0..a3` land in the same order `mma_tile_fragA`
expects, and `ldsm_n_dst_map_matches` checks it over all 256 slots.

## <a id="split-bf16x2"></a>25. `split_bf16x2` — an fp32 pair delivered as two bf16 words

A fp32 pair narrowed as a ROUNDED VALUE plus its RESIDUAL, packed into two separate
`bf16x2` words. Two `mma.sync` against the same A fragment then carry ~16 mantissa
bits of the B operand instead of 8 — one extra instruction per k-step for a B operand
that has nothing to cancel its rounding against, which is exactly [the mass column's
readout term](#mass-column)'s situation. §12 rule 3's guard applies to BOTH words: the
residual of a saturated value is itself out of range and must be narrowed the same way.

## <a id="mass-column"></a>26. The mass column's tiles

Two forms, both against a BROADCAST B — the column repeated across the tile's eight
`n` slots rather than varying with it, which is what turns a `[16 x K] x [K x 1]`
matvec into an `n8` MMA tile instead of wasting seven eighths of one:

- `mma_tile_fragA_massB` — the READOUT's term, `d += Af . B[.][n0]`, for a
  fragment-ordered A and `B` a column of the fp32 STATE, COMPENSATED: the state
  column is the denominator of a ratio, so it rides [`split_bf16x2`](#split-bf16x2)
  instead of a single narrowing, and the tile is two `mma.sync`.
- `mma_tile_bcastB` — the intra term's and the fold's term, `d += A . v` for an
  `ldmatrix`-fetched A (`Gm` or `WT`'s atom block) and `v` a register-side vector (the
  row scale `sw`). <a id="bcast-frag-b"></a>`bcast_frag_b(v, lane, b0, b1)` packs it: the B fragment's `k` map
  is `2r + (w & 1) + 8 * (w >> 1)` and its `n` map is `g`, so a broadcast B is
  `g`-INDEPENDENT — four scalar loads and two packs for the whole fragment, whatever
  the tile's `n0`. §12 rule 3's guard applies here too, at the point the fp32 row
  scale narrows.

Output column `0` (lanes with `r == 0`) carries the matvec, rows `g` and `g + 8`: the
split-K emission reads both forms back at exactly that slot.

### The three GEMM assemblies

- `mma_fold_tile` — GEMM 4, both operands `ldmatrix`-fetched: A = `WT` (`.N`), B =
  `Vw` (`.trans`). One atom's 16-row block; the contraction axis is the group's
  `kGroupTokens == 16` tokens, exactly one k-step.
- `mma_tile_vwB` — GEMM 3, `d += A . Vw` for `A` fetched by `ldmatrix` (`Gm`, exactly
  as the fold fetches `WT`) and `Vw` as B. The contraction is the group's 16 tokens:
  one k-step. Legal because `Gm`'s §12 rule 3 guard is at ITS store (`stage_put`),
  which is what makes an `ldmatrix` fetch of it bit-identical to a generic indexed
  read.
- `mma_tile_fragA_wtB` — GEMM 2, a fragment-ordered A (`Af16`, one atom's 128-word
  block) against `WT` fetched by `ldmatrix` (`WT16`, one atom's 16-row block). The
  only thing this adds over `wt_frag_a`/`vw_frag_b` is issuing the single `mma_m16n8k16`
  once both operands are in registers.

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
