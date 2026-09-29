#pragma once

// THE STATE PAGE'S SPLIT bf16 PLANES: [16 x DV hi][16 x DV lo][16 mass hi][16 mass lo].
// See docs/internals/common/state_page.md

#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"

namespace rola::state_page {

//: The page's leaf rectangle, and the block offsets it implies, in BYTES. -- see docs/internals/common/state_page.md#geometry
template <int DV>
struct Blocks {
  static constexpr int kRows = 16;
  static constexpr int kPairs = kRows * DV / 2;
  static constexpr uint32_t kLo = (uint32_t)(kRows * DV * 2);
  static constexpr uint32_t kMassHi = (uint32_t)(2 * kRows * DV * 2);
  static constexpr uint32_t kMassLo = kMassHi + (uint32_t)(kRows * 2);
  static constexpr uint32_t kBytes = kMassLo + (uint32_t)(kRows * 2);
  static_assert(DV % 2 == 0, "a 32-bit access carries a PAIR of value columns");
  static_assert(kBytes == (uint32_t)(kRows * (DV + 1) * (int)sizeof(float)),
                "the split layout is a re-layout of the fp32 page, not a resize");
};

}  // namespace rola::state_page
