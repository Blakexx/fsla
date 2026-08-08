# `csrc/rola/src/readout_mma.cuh` — the bf16 format guard

`GLA_MASS_DECAY_SPEC.md` §12 rule 3, lowered onto the one storage format the consumer
kernel uses. What lives here is exactly one thing: the READ-side clamp that keeps
every bf16 narrowing finite. It is not a GEMM, not a contraction, and not a fragment
map — those live in [`mma_tile.md`](mma_tile.md), which imports this file's clamp
(`using rola::clamp_bf16_finite;`) rather than reimplementing it.

---

## <a id="rule-3-bf16"></a>1. HAZARD `rule-3-bf16` — the clamp, and why it is the only precision story

**bf16 storage, fp32 accumulation, fp64 torch reference as oracle.** There is no
precision axis in this kernel: every MMA operand narrows to bf16 unconditionally, and
a caller that needs more precision than that delivers runs the fp64 reference path —
which is also the correctness truth every gate compares the kernel against, not a
second kernel arm. (A `tf32x3` extended-precision arm and the `MmaPrec` axis that
selected it existed through the P26 endgame rewrite and died with it:
`git show 52a67d8^:csrc/rola/src/readout_mma.cuh` has the last version that carried
them.)

Because bf16 is the only format, every operand that reaches an MMA instruction is
clamped to bf16's **max finite magnitude**,

```
kBf16MaxFinite = 3.38953139e38f
```

— never to fp32 `FLT_MAX`, because the gap between the two rounds to bf16 `inf` on
the narrowing cast. `clamp_bf16_finite` is that clamp:

```cpp
float clamp_bf16_finite(float x) {
  return fmaxf(-kBf16MaxFinite, fminf(kBf16MaxFinite, x));
}
```

It is branchless, and it is **NOT NaN-propagating**: `fminf` / `fmaxf` return the
NON-NaN operand when exactly one operand is NaN, so a NaN `x` maps to
`+kBf16MaxFinite` rather than passing through. That is the defined behaviour of the
guard, not an accident of it — a non-finite value must not reach an MMA operand in any
form. §12 rule 4's annihilation is the WRITE-side guard against non-finite content
reaching state at all; this is the READ-side format guard that keeps a
finite-but-oversized value from rounding to `inf` on the narrowing cast, and the two
are independent — neither substitutes for the other.

**THE SAME GUARD AT PAIR WIDTH.** `pack_clamp_bf16x2(lo, hi)` is the form every MMA
operand word actually takes. It narrows the pair FIRST and clamps the packed
`bf16x2` AFTER, so an operand word costs one `cvt.rn.bf16x2` and two pair-wide
`min`/`max` where the scalar order costs four scalar `min`/`max` and a narrowing —
5 instructions to 3 per pair, and the format guard's share of the kernel's
instructions halves with it.

```cpp
unsigned pack_clamp_bf16x2(float lo, float hi) {   // __hmax2(-M, __hmin2(M, cvt(lo, hi)))
```

**The two orders agree BIT FOR BIT, which is what makes this a WIDTH change and not a
precision one**, enumerated over the whole input space:

| `x` | clamp-then-narrow | narrow-then-clamp |
|---|---|---|
| `|x| <= M` | identity, then RN narrowing | RN narrowing cannot exceed `M` (it is exactly representable), then identity |
| `M < x <` midpoint(`M`, 2^128) | clamps to `M`, narrows to `M` | narrows to `M`, clamp is the identity |
| `x >=` midpoint, incl. `+inf` | clamps to `M` | narrows to `+inf`, `__hmin2(M, inf) = M` |
| NaN | `fminf(M, NaN) = M` | `__hmin2` is the NON-propagating variant (`__hmin2_nan` is the propagating one), so `M` |
| `+/-0`, subnormals | RN `f32 -> bf16`, no `.ftz` | the same convert at pair width |

`M = kBf16MaxFinite`. The claim is not left as an argument: an exhaustive device sweep
over all 2^32 fp32 bit patterns, in both lanes, reports **0 mismatches** against the
scalar order on sm_86 under the shipped toolchain (`workflows/perf/p50_review.md`,
gate 2).

The hazard: removing a `clamp_bf16_finite` from any of `mma_tile.cuh`'s pack sites
changes results and compiles clean. Every site that packs a value into a bf16 MMA
operand — `stage_put`, `split_bf16x2`, `bcast_frag_b` — routes through this one
function, so an audit of the guard is an audit of this file's one caller list, not of
scattered inline clamps. An `ldmatrix`-fetched operand (`WT` as GEMM 2/4's A, `Vw`,
`Gm` as GEMM 3's A, the state planes `Shi`/`Slo` as GEMM 1's B) carries no clamp of
its own at the fetch — the value was already clamped where it was written (`stage_put`
for `WT`/`Vw`/`Gm`, `split_bf16x2` for the state planes), which is what makes the
`ldmatrix` fetch bit-identical to a generic indexed read.

## 2. §12 rule 1 does NOT go live here

Mid-tile clock normalization. Rule 1 guards the two SEPARATELY EXPONENTIATED survival
panels, and this file forms no panel — it is a scalar clamp, not a contraction. Rule 1
becomes live with the `panel` intra lowering, the one construct in this design that
exponentiates `keep^{+P}` and `keep^{-P}` separately. See
[`intra_panel.md`](intra_panel.md#rule-1).
