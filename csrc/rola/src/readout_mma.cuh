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
// The full derivation of each and the PTX fragment index maps:
// docs/internals/readout_mma.md

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

}  // namespace rola
