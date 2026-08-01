#pragma once

// RoLA V3 consumer -- tensor-core readout (ROLA_V3_SPEC.md §5, "Warp tiling"
// and §3 step 4: "MMA over ceil(K/8) k-steps; row-scale; emit the partial").
//
// The contraction this file lowers is the CARRY readout:
//
//   num[t, ch] += scale_read[t] * sum_{i < BC} panel_read[t, i] * M[i, ch]
//
// i.e. `[BT x BC] * [BC x d_v]` with `t` on `m`, `ch` on `n`, and the owner's
// resident leaf index `i` on `k`. `scale_read[t]` is the §3 step-3 row scale (the
// product over the owner's FIXED levels, times `g_read`), applied after the
// accumulation because it is constant along `k` and along `n`.
//
// WHY THE B OPERAND COMES FROM SMEM. `M` is owner-resident in REGISTERS with one
// thread per channel (`M[BC]` per thread), which is the layout the deposit and the
// LeafMassDecay token-sequential walk want: both index `M` by (leaf, own channel).
// An MMA B fragment wants (leaf, fragment-determined channel) instead, and those
// two maps differ by a cross-thread transpose. Rather than pick one layout and pay
// a shuffle on the other path, the registers stay canonical and the readout
// *publishes* a copy of `M` into SMEM. The publish is write-conditioned: it
// happens once, lazily, and is invalidated only by a deposit -- which is §2's
// write-conditioned quiescence reused, not a new argument. Across a run of
// read-only tiles (the common case at `fill > 1`) the state is published once and
// read by every tile.
//
// PRECISION IS A COMPILE-TIME AXIS (findings D4.3), not a tolerance knob:
//   - `kTf32x3` -- three m16n8k8 tf32 passes (Karatsuba-style split), near-fp32.
//     This is what the exactness gate against the fp64 oracle runs, because bf16
//     A/B would contribute ~1e-2 relative error and swamp any structural
//     disagreement.
//   - `kBf16`   -- one m16n8k16 bf16 pass, fp32 accumulate. The performance
//     configuration, reported against its own honestly looser tolerance.
// This is a structural branch (a numeric format chosen once at plan time), not a
// density branch, so it is admissible under §3.
//
// `GLA_MASS_DECAY_SPEC.md` §12 RULE 3 GOES LIVE HERE, on the bf16 arm only: A/B
// operands narrow to bf16, so they are clamped to bf16's max FINITE magnitude
// (3.38953139e38), never to fp32 `FLT_MAX` -- the gap between the two rounds to
// bf16 `inf`. The tf32 arm does not narrow the exponent range (tf32 has fp32's
// 8-bit exponent), so no clamp applies there and none is inserted: a clamp that
// cannot fire is dead code, and one that fires on the wrong arm would alter values
// the format represents exactly.
//
// §12 RULE 1 (mid-tile clock normalization) DOES NOT go live here, and that is a
// derived fact rather than an omission: rule 1 guards the two SEPARATELY
// EXPONENTIATED survival panels, and this readout forms no panel -- under
// `decay=None` there is no survival at all, and under LeafMassDecay the intra term
// is the token-sequential walk, which carries survival as a running product. Rule 1
// becomes live with the `panel` intra lowering, which is the first construct in
// this design that exponentiates `keep^{+P}` and `keep^{-P}` separately. See
// consumer.md §1 (the per-lowering audit) and §10 consequence 1.

#include <cuda_bf16.h>

#include "v3_mma.cuh"

namespace fla_rola::v3 {

using fla_rola::v3::mma::mma_m16n8k16;
using fla_rola::v3::mma::mma_m16n8k8_tf32x3;
using fla_rola::v3::mma::pack_bf16x2;

//: MMA operand precision. Compile-time; `V3Plan.mma_precision` stamps it.
enum class MmaPrec { kTf32x3 = 0, kBf16 = 1 };

//: bf16's largest finite magnitude. Anything above it rounds to `inf` on the
//: narrowing cast, which is what §12 rule 3 exists to prevent.
constexpr float kBf16MaxFinite = 3.38953139e38f;

__device__ __forceinline__ float clamp_bf16_finite(float x) {
  // Branchless, and NaN-preserving by construction: `fminf`/`fmaxf` return the
  // non-NaN operand only when exactly one is NaN, so a NaN `x` propagates. §12
  // rule 4's annihilation is the write-side guard against non-finite content; this
  // is the read-side format guard, and the two are independent.
  return fmaxf(-kBf16MaxFinite, fminf(kBf16MaxFinite, x));
}

// ---------------------------------------------------------------------------
// Fragment index maps (PTX ISA `mma.sync.aligned` operand layouts).
//
// For every shape below: `g = lane >> 2` is the PTX `groupID`, `r = lane & 3` is
// `threadID_in_group`. The D/C fragment is shared by both shapes:
//     d0,d1 -> (row g,     col 2r, 2r+1)
//     d2,d3 -> (row g + 8, col 2r, 2r+1)
// ---------------------------------------------------------------------------

//: One `[16 x 8]` output tile of the readout, accumulated over the COMPACTED
//: contraction. `A` is `panel_read`, row stride `BC`, whose column `k` holds the
//: amplitude of leaf `kidx[k]` (spec §3 step 2's gathered leaf list); `B` is the
//: published state `[BC][SP]` row-major in the LEAF index, `SP` its padded row
//: stride, so the B operand is gathered through `kidx`. `mt` selects the 16-token
//: row block, `ntg` the 8-channel column block.
//:
//: `ksteps` is §3 step 4's "MMA over ceil(K/8) k-steps" made literal: the caller
//: passes `Kpad / 8` (tf32x3) or `Kpad / 16` (bf16), with `Kpad` the compacted
//: count rounded up to the shape's `k`. It is a runtime bound because `K` is a
//: per-tile support size, and that is the ONLY runtime quantity here -- the
//: fragment maps, the shape and the format stay compile-time. At the degeneracy
//: `K == BC` this reduces to the previous fully-unrolled `BC/8` loop with
//: `kidx[k] == k`, bit-identically.
//:
//: `COMPACT` is the PLAN-TIME axis (spec §6, ratified rev `2026-07-29.5` Q1): when
//: the planner's measured tile unions saturate, the gather is compiled OUT and the
//: B row index is the loop index itself. That is not an approximation of the
//: gathered path, it is its `K == BC` degeneracy written without the indirection:
//: `kidx[k] == k` there by the ascending-order property of the compaction, so the
//: two paths issue identical loads in identical order. `kidx` is `nullptr` under
//: `!COMPACT` and is never dereferenced.
template <int BT, int BC, int SP, MmaPrec PREC, bool COMPACT>
__device__ __forceinline__ void readout_fragment(
    const float* __restrict__ A, const float* __restrict__ B, const int* __restrict__ kidx,
    int ksteps, int mt, int ntg, int lane,
    float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  d0 = d1 = d2 = d3 = 0.0f;
  //: B's row for compacted column `k`. Compile-time-resolved on both arms.
  auto brow = [kidx](int k) { return COMPACT ? kidx[k] : k; };

  if (PREC == MmaPrec::kTf32x3) {
    // m16n8k8, tf32. A: a0..a3 -> (g, r), (g+8, r), (g, r+4), (g+8, r+4).
    //                B: b0,b1 -> (row r, col g), (row r+4, col g).
    // Compile-time trip count with an early exit, not a runtime bound: a runtime
    // loop bound around an MMA sequence is un-unrollable and costs more than the
    // k-steps it skips (the cost the deleted `gram` lowering's runtime-bounded
    // level loop was measured to pay, findings §5.3 point 2).
#pragma unroll
    for (int ks = 0; ks < BC / 8; ++ks) {
      if (ks >= ksteps) break;
      const int ka = ks * 8;
      const int ra = mt * 16 + g;
      mma_m16n8k8_tf32x3(
          d0, d1, d2, d3,
          A[ra * BC + ka + r], A[(ra + 8) * BC + ka + r],
          A[ra * BC + ka + r + 4], A[(ra + 8) * BC + ka + r + 4],
          B[brow(ka + r) * SP + ntg * 8 + g], B[brow(ka + r + 4) * SP + ntg * 8 + g],
          d0, d1, d2, d3);
    }
  } else {
    // m16n8k16, bf16. A: a0..a3 pack (g, 2r|2r+1), (g+8, 2r|2r+1),
    //                              (g, 2r+8|2r+9), (g+8, 2r+8|2r+9).
    //                 B: b0,b1 pack (row 2r|2r+1, col g), (row 2r+8|2r+9, col g).
    // §12 rule 3: every operand is clamped to bf16's finite range before packing.
#pragma unroll
    for (int ks = 0; ks < BC / 16; ++ks) {
      if (ks >= ksteps) break;
      const int ka = ks * 16;
      const int ra = mt * 16 + g;
      const int ca = ka + r * 2;
      const unsigned a0 = pack_bf16x2(clamp_bf16_finite(A[ra * BC + ca]),
                                      clamp_bf16_finite(A[ra * BC + ca + 1]));
      const unsigned a1 = pack_bf16x2(clamp_bf16_finite(A[(ra + 8) * BC + ca]),
                                      clamp_bf16_finite(A[(ra + 8) * BC + ca + 1]));
      const unsigned a2 = pack_bf16x2(clamp_bf16_finite(A[ra * BC + ca + 8]),
                                      clamp_bf16_finite(A[ra * BC + ca + 9]));
      const unsigned a3 = pack_bf16x2(clamp_bf16_finite(A[(ra + 8) * BC + ca + 8]),
                                      clamp_bf16_finite(A[(ra + 8) * BC + ca + 9]));
      const int nb = ntg * 8 + g;
      const unsigned b0 = pack_bf16x2(clamp_bf16_finite(B[brow(ca) * SP + nb]),
                                      clamp_bf16_finite(B[brow(ca + 1) * SP + nb]));
      const unsigned b1 = pack_bf16x2(clamp_bf16_finite(B[brow(ca + 8) * SP + nb]),
                                      clamp_bf16_finite(B[brow(ca + 9) * SP + nb]));
      mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
  }
}

}  // namespace fla_rola::v3
