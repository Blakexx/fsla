# `csrc/rola/src/readout_mma.cuh` — the tensor-core carry readout

`ROLA_V3_SPEC.md` §5 ("Warp tiling") and §3 step 4 ("MMA over `ceil(K/8)` k-steps;
row-scale; emit the partial"), lowered onto the `mma.sync.aligned` shapes wrapped in
`mma.cuh`.

Symbols: `t` is a token of the tile, `i` the owner's resident LEAF index, `ch` a value
channel; `BT` is the token tile, `BC` the leaf tile, `DV` the value width, `SP` the
padded SMEM row stride of the published state; `M` is the state; `K` is a tile's
support size and `Kpad` that size rounded up to the MMA shape's `k`; `lane` is the
lane id within the warp, `g = lane >> 2` the PTX `groupID` and `r = lane & 3` its
`threadID_in_group`.

---

## <a id="contraction"></a>1. The contraction this file lowers

The CARRY readout:

```
num[t, ch] += scale_read[t] * sum_{i < BC} panel_read[t, i] * M[i, ch]
```

i.e. `[BT x BC] * [BC x d_v]` with `t` on `m`, `ch` on `n`, and the owner's resident
leaf index `i` on `k`.

`scale_read[t]` is the §3 step-3 row scale (the product over the owner's FIXED
levels, times `g_read`). It is applied AFTER the accumulation because it is constant
along `k` and along `n` — so it can never be folded into a fragment, and moving it
inside the k-loop would only multiply the same value in `ksteps` times.

## <a id="b-from-smem"></a>2. Why the B operand comes from SMEM

`M` is owner-resident in REGISTERS with one thread per channel (`M[BC]` per thread),
which is the layout the deposit wants: it indexes `M` by (leaf, own channel). An MMA B fragment wants (leaf,
fragment-determined channel) instead, and those two maps differ by a **cross-thread
transpose**.

Rather than pick one layout and pay a shuffle on the other path, the registers stay
canonical and the readout *publishes* a copy of `M` into SMEM.

The publish is **write-conditioned**: it happens once, lazily, and is invalidated only
by a deposit — which is §2's write-conditioned quiescence reused, not a new argument.
Across a run of read-only tiles (the common case at `fill > 1`) the state is published
once and read by every tile.

## <a id="precision-axis"></a>3. Precision is a compile-time axis, not a tolerance knob

Findings D4.3. `MmaPrec` is stamped by `Plan.mma_precision` at plan time:

| arm | shape and passes | role |
|---|---|---|
| `kTf32x3` | three `m16n8k8` tf32 passes (Karatsuba-style split), near-fp32 | what the exactness gate against the fp64 oracle runs — bf16 A/B would contribute ~1e-2 relative error and swamp any structural disagreement |
| `kBf16` | one `m16n8k16` bf16 pass, fp32 accumulate | the performance configuration, reported against its own honestly looser tolerance |

This is a STRUCTURAL branch (a numeric format chosen once at plan time), not a density
branch, so it is admissible under §3.

## <a id="rule-3-bf16"></a>4. HAZARD — §12 rule 3 goes live here, on the bf16 arm only

`GLA_MASS_DECAY_SPEC.md` §12 rule 3. On the bf16 arm the A and B operands NARROW to
bf16, so every one of them is clamped to bf16's **max finite magnitude**,
`kBf16MaxFinite = 3.38953139e38f` — never to fp32 `FLT_MAX`, because the gap between
the two rounds to bf16 `inf`.

The tf32 arm does not narrow the exponent range (tf32 has fp32's 8-bit exponent), so
no clamp applies there and none is inserted: **a clamp that cannot fire is dead code,
and one that fires on the wrong arm would alter values the format represents exactly.**

`clamp_bf16_finite` is branchless and NaN-preserving by construction: `fminf` / `fmaxf`
return the non-NaN operand only when exactly one is NaN, so a NaN `x` propagates. §12
rule 4's annihilation is the WRITE-side guard against non-finite content; this is the
READ-side format guard, and the two are independent — neither substitutes for the
other.

The hazard: removing a `clamp_bf16_finite` from any of the six pack sites, or applying
one on the tf32 arm, changes results and compiles clean.

## <a id="rule-1-not-here"></a>5. §12 rule 1 does NOT go live here

Mid-tile clock normalization. That is a derived fact rather than an omission: rule 1
guards the two SEPARATELY EXPONENTIATED survival panels, and this readout forms no
panel.

Under `decay = None` there is no survival at all. Rule 1 becomes live with the
`panel` intra lowering, the one construct in this design that exponentiates
`keep^{+P}` and `keep^{-P}` separately — and nowhere earlier. See
[`intra_panel.md`](intra_panel.md#rule-1).

## <a id="fragment-maps"></a>6. Fragment index maps

PTX ISA `mma.sync.aligned` operand layouts. For every shape here, `g = lane >> 2` and
`r = lane & 3`. The D/C fragment is shared by both shapes:

```
d0,d1 -> (row g,     col 2r, 2r+1)
d2,d3 -> (row g + 8, col 2r, 2r+1)
```

`m16n8k8`, tf32:

```
A: a0..a3 -> (g, r), (g+8, r), (g, r+4), (g+8, r+4)
B: b0,b1  -> (row r, col g), (row r+4, col g)
```

`m16n8k16`, bf16 (operands are `bf16x2` packs):

```
A: a0..a3 pack (g, 2r|2r+1), (g+8, 2r|2r+1), (g, 2r+8|2r+9), (g+8, 2r+8|2r+9)
B: b0,b1  pack (row 2r|2r+1, col g), (row 2r+8|2r+9, col g)
```

## <a id="readout-fragment"></a>7. `readout_fragment` — one `[16 x 8]` output tile

Accumulated over the COMPACTED contraction.

- `A` is `panel_read`, row stride `BC`, whose column `k` holds the amplitude of leaf
  `kidx[k]` (spec §3 step 2's gathered leaf list);
- `B` is the published state `[BC][SP]` row-major in the LEAF index, `SP` its padded
  row stride, so the B operand is gathered through `kidx`;
- `mt` selects the 16-token row block, `ntg` the 8-channel column block.

`ksteps` is §3 step 4's "MMA over `ceil(K/8)` k-steps" made literal: the caller passes
`Kpad / 8` (tf32x3) or `Kpad / 16` (bf16), with `Kpad` the compacted count rounded up
to the shape's `k`. **It is a runtime bound because `K` is a per-tile support size, and
that is the ONLY runtime quantity here** — the fragment maps, the shape and the format
stay compile-time. At the degeneracy `K == BC` this is a fully-unrolled `BC/8` loop
with `kidx[k] == k`, bit-identically.

## <a id="ksteps-early-exit"></a>8. Why the k-loop is a compile-time trip count with an early exit

Not a runtime bound. A runtime loop bound around an MMA sequence is un-unrollable and
costs more than the k-steps it skips — MEASURED on a runtime-bounded MMA level loop
(provenance: findings §5.3 point 2). So the loop is
`#pragma unroll` over `BC/8` (or `BC/16`) with `if (ks >= ksteps) break;` inside.

## <a id="compact"></a>9. `COMPACT` — the plan-time gather axis

Spec §6, ratified rev `2026-07-29.5` Q1. When the planner's measured tile unions
saturate, the gather is compiled OUT and the B row index is the loop index itself
(`brow` resolves at compile time on both arms).

That is **not an approximation** of the gathered path, it is its `K == BC` degeneracy
written without the indirection: `kidx[k] == k` there by the ascending-order property
of the compaction, so the two paths issue identical loads in identical order.

`kidx` is `nullptr` under `!COMPACT` and is never dereferenced.
