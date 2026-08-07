#pragma once

// RoLA V3 consumer -- tensor-core readout format facts (ROLA_V3_SPEC.md §5).
//
// THE ONE PRECISION STORY: bf16 storage, fp32 accumulation, fp64 reference as
// oracle. There is no precision axis -- the tf32x3 arm died with it (revival =
// git); a caller needing more precision than the bf16 arm delivers runs the fp64
// torch reference path, which is also the correctness truth every gate compares
// against. What lives here is the format guard that keeps every bf16 narrowing
// finite.
//
// `GLA_MASS_DECAY_SPEC.md` §12 rule 3 goes live here; rule 1 does not go live
// here at all. The derivation: docs/internals/readout_mma.md

#include <cuda_bf16.h>

#include "mma.cuh"

namespace rola {

using rola::mma::mma_m16n8k16;
using rola::mma::pack_bf16x2;

//: bf16's largest finite magnitude. Anything above it rounds to `inf` on the
//: narrowing cast, which is what §12 rule 3 exists to prevent.
constexpr float kBf16MaxFinite = 3.38953139e38f;

//: HAZARD rule-3-bf16 -- docs/internals/readout_mma.md#rule-3-bf16
//: The READ-side format guard: branchless, NaN-preserving, and independent of §12
//: rule 4's write-side annihilation.
__device__ __forceinline__ float clamp_bf16_finite(float x) {
  return fmaxf(-kBf16MaxFinite, fminf(kBf16MaxFinite, x));
}

}  // namespace rola
