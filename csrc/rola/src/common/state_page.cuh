#pragma once

// THE STATE PAGE: 16 leaves as [16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32].
// See docs/internals/common/state_page.md

#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"

namespace rola::state_page {

//: The page's byte geometry: a leaf-channel is the s24 `hi << 8 | lo` at its group's exponent. -- see docs/internals/common/state_page.md#geometry
template <int DV>
struct Blocks {
  static constexpr int kRows = 16;
  static constexpr int kGroups = DV / 32;
  static constexpr uint32_t kHi = 0;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kExp = kLo + (uint32_t)(kRows * DV);
  static constexpr uint32_t kMass = (kExp + (uint32_t)(kRows * kGroups) + 15u) & ~15u;
  static constexpr uint32_t kBytes = kMass + (uint32_t)(kRows * 4);
  static_assert(DV % 32 == 0, "a channel group is 32 channels");
};

//: The s24 mantissa's bound and the smallest group exponent. -- see docs/internals/common/state_page.md#codec
constexpr int kMantissaMax = (1 << 23) - 1;
constexpr int kExpMin = -126;

//: `2^e` for `kExpMin <= e <= 127`, as the float's exponent field.
__device__ __forceinline__ float pow2(int e) { return __int_as_float((e + 127) << 23); }

//: The exponent a group stores at, from its largest magnitude. -- see docs/internals/common/state_page.md#codec
__device__ __forceinline__ int group_exponent(float amax) {
  int k = 0;
  frexpf(amax, &k);
  return amax == 0.0f ? 0 : max(k - 23, kExpMin);
}

//: `v`'s s24 mantissa at exponent `e`: one rounding, held inside s24.
__device__ __forceinline__ int mantissa(float v, int e) {
  const int m = __float2int_rn(v * pow2(-e));
  return min(max(m, -kMantissaMax), kMantissaMax);
}

__device__ __forceinline__ float value(int hi, uint32_t lo, int e) {
  return (float)((hi << 8) | (int)lo) * pow2(e);
}

//: The hi plane alone, the s16 operand: the value truncated to 15 bits of its group.
__device__ __forceinline__ float value_hi(int hi, int e) { return (float)hi * pow2(e + 8); }

//: THE UNPORTED CARRY'S PAGE, split bf16 planes; the carry port deletes it. -- see docs/internals/common/state_page.md#split
template <int DV>
struct SplitBlocks {
  static constexpr int kRows = 16;
  static constexpr int kPairs = kRows * DV / 2;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kMassHi = (uint32_t)(2 * kRows * DV * 2);
  static constexpr uint32_t kMassLo = kMassHi + (uint32_t)(kRows * 2);
  static constexpr uint32_t kBytes = kMassLo + (uint32_t)(kRows * 2);
  static_assert(DV % 2 == 0, "a 32-bit access carries a PAIR of value columns");
};

}  // namespace rola::state_page
