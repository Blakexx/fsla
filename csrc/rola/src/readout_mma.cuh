#pragma once

// RoLA V3 consumer -- tensor-core readout (ROLA_V3_SPEC.md §5, "Warp tiling"
// and §3 step 4: "MMA over ceil(K/8) k-steps; row-scale; emit the partial").
//
// The contraction this file lowers is the CARRY readout:
//
//   num[t, ch] += scale_read[t] * sum_{i < BC} panel_read[t, i] * M[i, ch]
//
// i.e. `[BT x BC] * [BC x d_v]` with `t` on `m`, `ch` on `n`, and the owner's
// resident leaf index `i` on `k`. `scale_read[t]` is the §3 step-3 row scale and is
// applied AFTER the accumulation, being constant along both `k` and `n`.
//
// Three facts a reader must not miss. The B operand comes from SMEM because `M` is
// register-resident in a (leaf, own channel) map and a B fragment wants
// (leaf, fragment-determined channel) -- so the readout PUBLISHES `M`, once, lazily,
// invalidated only by a deposit. Precision is a COMPILE-TIME axis stamped by
// `Plan.mma_precision`, not a tolerance knob: tf32x3 is what the fp64-oracle gate
// runs, bf16 is the performance arm. And `GLA_MASS_DECAY_SPEC.md` §12 rule 3 goes
// live here on the BF16 ARM ONLY, while rule 1 does not go live here at all.
//
// The full derivation of each, the PTX fragment index maps, and the `COMPACT` and
// `ksteps` degeneracy arguments: docs/internals/readout_mma.md

#include <cuda_bf16.h>

#include "mma.cuh"

namespace rola {

using rola::mma::mma_m16n8k16;
using rola::mma::mma_m16n8k8_tf32x3;
using rola::mma::pack_bf16x2;

//: MMA operand precision. Compile-time; `Plan.mma_precision` stamps it.
enum class MmaPrec { kTf32x3 = 0, kBf16 = 1 };

//: bf16's largest finite magnitude. Anything above it rounds to `inf` on the
//: narrowing cast, which is what §12 rule 3 exists to prevent.
constexpr float kBf16MaxFinite = 3.38953139e38f;

//: HAZARD rule-3-bf16 -- docs/internals/readout_mma.md#rule-3-bf16
//: The READ-side format guard: branchless, NaN-preserving, and independent of §12
//: rule 4's write-side annihilation. Belongs on the bf16 arm and on no other.
__device__ __forceinline__ float clamp_bf16_finite(float x) {
  return fmaxf(-kBf16MaxFinite, fminf(kBf16MaxFinite, x));
}

//: One `[16 x 8]` output tile of the readout, accumulated over the COMPACTED
//: contraction. `A` is `panel_read` (row stride `BC`), `B` the published state
//: `[BC][SP]` row-major in the LEAF index, `mt` the 16-token row block, `ntg` the
//: 8-channel column block, and `ksteps` the §3 step-4 k-step count -- the ONLY
//: runtime quantity here. `COMPACT` is the plan-time gather axis; under `!COMPACT`
//: `kidx` is `nullptr` and is never dereferenced.
//: The PTX fragment index maps, the `K == BC` bit-identity and the reason the k-loop
//: keeps a compile-time trip count: docs/internals/readout_mma.md#readout-fragment
template <int BT, int BC, int SP, MmaPrec PREC, bool COMPACT>
__device__ __forceinline__ void readout_fragment(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 const int* __restrict__ kidx, int ksteps, int mt,
                                                 int ntg, int lane, float& d0, float& d1, float& d2,
                                                 float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  d0 = d1 = d2 = d3 = 0.0f;
  //: B's row for compacted column `k`. Compile-time-resolved on both arms.
  auto brow = [kidx](int k) { return COMPACT ? kidx[k] : k; };

  if (PREC == MmaPrec::kTf32x3) {
    // m16n8k8, tf32 -- fragment maps: docs/internals/readout_mma.md#fragment-maps
#pragma unroll
    for (int ks = 0; ks < BC / 8; ++ks) {
      if (ks >= ksteps) break;
      const int ka = ks * 8;
      const int ra = mt * 16 + g;
      mma_m16n8k8_tf32x3(d0, d1, d2, d3, A[ra * BC + ka + r], A[(ra + 8) * BC + ka + r],
                         A[ra * BC + ka + r + 4], A[(ra + 8) * BC + ka + r + 4],
                         B[brow(ka + r) * SP + ntg * 8 + g], B[brow(ka + r + 4) * SP + ntg * 8 + g],
                         d0, d1, d2, d3);
    }
  } else {
    // m16n8k16, bf16 -- fragment maps: docs/internals/readout_mma.md#fragment-maps
    // §12 rule 3: every operand is clamped to bf16's finite range before packing.
#pragma unroll
    for (int ks = 0; ks < BC / 16; ++ks) {
      if (ks >= ksteps) break;
      const int ka = ks * 16;
      const int ra = mt * 16 + g;
      const int ca = ka + r * 2;
      const unsigned a0 =
          pack_bf16x2(clamp_bf16_finite(A[ra * BC + ca]), clamp_bf16_finite(A[ra * BC + ca + 1]));
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

}  // namespace rola
