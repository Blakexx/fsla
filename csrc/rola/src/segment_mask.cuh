#pragma once

// §6.4's per-segment tile-range mask, and nothing else: the arithmetic that decides
// which tiles a SEGMENT owns under sequence parallelism.
//
// It is a named single definition rather than kernel-body arithmetic because at
// `P > 1` it must agree BIT FOR BIT with the host-side segment partition the schedule
// publishes, and that agreement is what the sequence-parallel gates pin.
//
// The mask arithmetic and its agreement contract: docs/internals/segment_mask.md

#include <cuda_runtime.h>
#include <cstdint>

namespace rola {
namespace detail {

//: §6.4: the bits of candidate word `w` that lie in the tile range `[lo, hi)`.
//: A segment owns a CONTIGUOUS tile range, so restricting the candidate walk to it is
//: one more word-AND, of exactly the shape §4's `j+1` candidate ANDs already are. At
//: `nseg == 1` every word comes back all-ones and the AND is the identity, which is one
//: half of why `P = 1` is bit-identical to the pre-`P` kernel
//: -- docs/internals/segment_mask.md#word-mask
__device__ __forceinline__ uint32_t segment_word_mask(int w, int lo, int hi) {
  const int base = w * 32;
  int a = lo - base;
  int b = hi - base;
  a = (a < 0) ? 0 : ((a > 32) ? 32 : a);
  b = (b < 0) ? 0 : ((b > 32) ? 32 : b);
  if (b <= a) return 0u;  // also the only way `a == 32` could reach the shift below
  const uint32_t upto = (b == 32) ? 0xffffffffu : ((1u << b) - 1u);
  return upto & ~((1u << a) - 1u);
}

}  // namespace detail
}  // namespace rola
