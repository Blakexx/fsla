# `csrc/rola/src/mma_tile.cuh` — the consumer's MMA tile layer

The fragment machinery and the four GEMMs' inner sequences: the fragment-ordered A
operand's element maps, the generic `[16 x 8]` tile, its `ldmatrix`-fed
specializations, and the compile-time cross-checks that keep CuTe — not this code —
the authority on what a fragment contains.

Where an operand LIVES is [`smem_layout.md`](smem_layout.md); which warp runs which
tile, and in what phase order, is [`consumer_kernel.md`](consumer_kernel.md).

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

# Part III — the operand maps

## <a id="frag-a-get"></a>21. `frag_a_get` — the inverse, and why the mass column pays it

Reads element `(m, k)` back OUT of a fragment-ordered A operand. The mass column's denominator
is a `[BT x BC] x [BC x 1]` matvec, one column out of `DV + 1`, and giving it its own `n8` MMA
tile would waste 7/8 of a fragment. So it stays a scalar reduction and pays this inverse
instead: ~10 integer ops on `BT * BC` elements spread over `BT` threads, once per tile.

**HAZARD `frag-a-get`.** `frag_a_get` is the hand-rolled INVERSE of `frag_a_slot`, and
nothing checks the pair against each other — the CuTe cross-checks in [the atom section](#ldsm-atoms) pin the library's
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
the MMA's format ([`smem_layout.md#stage-format`](smem_layout.md#stage-format)).

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
