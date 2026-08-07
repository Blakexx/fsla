# `csrc/rola/src/mma.cuh` — bare tensor-core PTX primitives

Instruction wrappers, nothing more: `m16n8k16` bf16, and the `bf16x2` pack. **No RoLA
semantics live here.**

The callers are `readout_mma.cuh` and `mma_tile.cuh`, and there are no others.

The source file carries a short header and is already at the mirror's target comment
density; this doc is its reference rather than a relocation of its prose.

Symbols: `m`, `n`, `k` are the MMA tile extents in the PTX `mma.sync.aligned` naming;
`A` is the `m x k` operand, `B` the `k x n` operand, `C` the accumulator input and `D`
the accumulator output; a *fragment* is the per-lane slice of one of those matrices.

---

## <a id="provenance"></a>1. Provenance — a V2-lineage header in a V3 tree

These wrappers are the V2 shared-Gram MMA primitives, carried into the V3 tree
verbatim: the arithmetic is V2's, and it is V2's because a `mma.sync.aligned` wrapper
has no version — it has an instruction. They are **the only V2-lineage header the V3
consumer `#include`s**, and they are load-bearing: `consumer.cu` does not compile
without them.

They live in `namespace rola::mma`, under a name that describes the file's contents,
so nothing in the tree carries a prefix naming a kernel family that is not here.

## <a id="no-semantics"></a>2. The layering rule this file exists to keep

Everything here is a one-to-one wrapper over a PTX instruction or a hardware
conversion. There is no masking, no scaling, no clamping, no gather, and no knowledge
of leaves, tokens, panels or state.

That matters in two directions:

- the readout's `clamp_bf16_finite` (§12 rule 3) is applied by the CALLER, before
  packing — see [`readout_mma.md`](readout_mma.md#rule-3-bf16). Pushing it down into
  `pack_bf16x2` would silently apply it on paths that must not narrow;
- the fragment INDEX MAPS — which lane holds which `(row, col)` — are the caller's
  knowledge too, and are documented at the call site in
  [`mma_tile.md`](mma_tile.md#mma-tile-fraga). This file only requires that the
  caller hand it operands already in fragment order.

`namespace rola::mma` exists so those two responsibilities stay visibly separate.

## <a id="shapes"></a>3. The shape

| wrapper | PTX instruction | A/B format | accumulate |
|---|---|---|---|
| `mma_m16n8k16` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | bf16 (packed `bf16x2`) | fp32 |

`.row.col`: A is row-major, B is column-major, in the PTX sense of the fragment map —
not a property of any array in memory.

`D = A*B + C`, and the wrapper takes `c0..c3` and writes `d0..d3`, so chaining passes
(as every GEMM in `mma_tile.cuh` does, across k-steps) means passing the previous `d`
back in as `c`.

`pack_bf16x2` converts a pair of floats with `__floats2bfloat162_rn` and type-puns the
result to `unsigned`, which is the register class the PTX operand constraint (`"r"`)
requires.

## 4. The one precision story

bf16 storage, fp32 accumulation, fp64 torch reference as oracle. There is no
precision AXIS: `mma_m16n8k16` is the only shape this file wraps, and every caller
narrows to it unconditionally. A caller that needs more precision than the bf16 pass
delivers does not select a different kernel arm — it runs the fp64 reference path,
which is also the correctness truth every gate compares the bf16 kernel against.

An error-compensated tf32 split (`tf32x3`: three `m16n8k8` tf32 passes recovering most
of tf32's lost mantissa) and a `MmaPrec` template axis selecting between it and this
file's bf16 pass existed through the P26 endgame rewrite and were deleted with it —
the consumer kernel that chose between them is gone, not merely this file's half of
it. The construction, if a future rung wants it back: `git show 52a67d8^:csrc/rola/src/mma.cuh`.
