// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREPARED MASKS' PLACE (docs/internals/common/plan/mask_prepare_layout.md): where the mask-preparation pass writes and the gather and the carry read. Pure, constexpr, total.
#pragma once

#include <cstdint>

#include "cta_carve.h"

namespace fsla::plan {

inline constexpr int kMaskWindowWords = 32;

inline constexpr int kLevelSides = 2 * kCarveMaxLevels;

//: A WINDOW'S BLOCK of a level-side `width_bits` wide, in bytes, whatever its group: a total and 32 counts a digit, 2 bytes each
FSLA_CARVE_FN constexpr std::int64_t prepared_block_bytes(int width_bits) {
  return std::int64_t{2 * (kMaskWindowWords + 1)} << width_bits;
}

//: in a block of `groups` groups: the totals at 0 (uint16 a group), then the counts (uint16, word-major), then the words (uint32, word-major; none at a group of one digit)
FSLA_CARVE_FN constexpr std::int64_t prepared_counts_at(int groups) {
  return std::int64_t{2} * groups;
}

FSLA_CARVE_FN constexpr std::int64_t prepared_words_at(int groups) {
  return (std::int64_t{2 * (kMaskWindowWords + 1)} * groups + 3) & ~std::int64_t{3};
}

struct PreparedLayout {
  std::int64_t offset[kLevelSides];
  std::int64_t bytes;
};

//: THE PLACE of a call's prepared masks: `width_bits[s]` and `heads[s]` (batch times the side's heads) of each level-side, 0 heads where it has no mask, over `windows` windows
constexpr PreparedLayout prepared_layout(const int* width_bits, const std::int64_t* heads,
                                         std::int64_t windows) {
  PreparedLayout p{};
  for (int s = 0; s < kLevelSides; ++s) {
    p.offset[s] = p.bytes;
    p.bytes += heads[s] * windows * prepared_block_bytes(width_bits[s]);
  }
  return p;
}

}  // namespace fsla::plan
