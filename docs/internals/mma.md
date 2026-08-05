# `csrc/rola/src/mma.cuh` — bare tensor-core PTX primitives

Instruction wrappers, nothing more: `m16n8k16` bf16, `m16n8k8` tf32 and its three-pass
extended-precision form, and the `bf16x2` pack. **No RoLA semantics live
here.**

The callers are `readout_mma.cuh` and `mma_tile.cuh`, and there are no others.

The source file carries a short header and one inline note and is already at the
mirror's target comment density; this doc is its reference rather than a relocation
of its prose.

Symbols: `m`, `n`, `k` are the MMA tile extents in the PTX `mma.sync.aligned` naming;
`A` is the `m x k` operand, `B` the `k x n` operand, `C` the accumulator input and `D`
the accumulator output; a *fragment* is the per-lane slice of one of those matrices;
`rna` is round-to-nearest with the tf32 mantissa.

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
  [`readout_mma.md`](readout_mma.md#fragment-maps). This file only requires that the
  caller hand it operands already in fragment order.

`namespace rola::mma` exists so those two responsibilities stay visibly separate.

## <a id="shapes"></a>3. The shapes

| wrapper | PTX instruction | A/B format | accumulate |
|---|---|---|---|
| `mma_m16n8k16` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | bf16 (packed `bf16x2`) | fp32 |
| `mma_m16n8k8_tf32` | `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` | tf32 | fp32 |

Both are `.row.col`: A is row-major, B is column-major, in the PTX sense of the
fragment map — not a property of any array in memory.

`D = A*B + C` in every case, and every wrapper takes `c0..c3` and writes `d0..d3`, so
chaining passes means passing the previous `d` back in as `c`. The extended-precision
wrappers below do exactly that.

`pack_bf16x2` converts a pair of floats with `__floats2bfloat162_rn` and type-puns the
result to `unsigned`, which is the register class the PTX operand constraint (`"r"`)
requires. `f2tf32` is the same idea for tf32: `cvt.rna.tf32.f32` produces the rounded
value already in a 32-bit register, and `tf32_round` reinterprets it back to `float` so
the residual can be computed in fp32.

## <a id="tf32-splitting"></a>4. `tf32x3` — error-compensated tf32

tf32 keeps fp32's 8-bit exponent but only a 10-bit mantissa. Splitting each operand
into a rounded high part and its exact fp32 residual,

```
a_hi = tf32_round(a)      a_lo = a - a_hi      (exact in fp32)
b_hi = tf32_round(b)      b_lo = b - b_hi
```

recovers most of the lost mantissa by issuing more passes:

| wrapper | passes | terms accumulated | dropped |
|---|---|---|---|
| `mma_m16n8k8_tf32x3` | 3 | `a_hi*b_hi`, `a_hi*b_lo`, `a_lo*b_hi` | `a_lo*b_lo` |

`tf32x3` is the Karatsuba-style split the readout calls "near-fp32": the dropped
`a_lo*b_lo` term is the product of two residuals and is below the fp32 accumulator's
resolution for operands of comparable magnitude. It is what the exactness gate against
the fp64 oracle runs.

There is no two-pass variant: the readout's plan-time precision axis exposes
`kTf32x3` and `kBf16` only, so a wrapper splitting A alone would have no caller
([`DELETIONS.md`](DELETIONS.md) records its removal). A cheaper correction, if a rung
ever needs one, is the construction above with `b_hi := b` and the third pass
dropped.

**Ordering is load-bearing.** Each pass feeds `d0..d3` into the next pass's
`c0..c3`, so the accumulation order is fixed by the call sequence. Reordering the
passes, or hoisting one out, changes the rounding and therefore the numbers — with no
diagnostic.
