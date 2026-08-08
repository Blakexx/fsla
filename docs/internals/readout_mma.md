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

It is branchless and NaN-preserving by construction: `fminf` / `fmaxf` return the
non-NaN operand only when exactly one operand is NaN, so a NaN `x` propagates through
unchanged. §12 rule 4's annihilation is the WRITE-side guard against non-finite
content reaching state at all; this is the READ-side format guard that keeps a
finite-but-oversized value from rounding to `inf` on the narrowing cast, and the two
are independent — neither substitutes for the other.

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
