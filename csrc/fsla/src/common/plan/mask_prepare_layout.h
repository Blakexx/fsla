// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREPARED MASKS' PLACE (docs/internals/common/plan/mask_prepare_layout.md): where the mask-preparation pass writes and the carry reads. Pure, constexpr, total.
#pragma once

#include <cstdint>

#include "cta_carve.h"

namespace fsla::plan {

inline constexpr int kLevelSides = 2 * kCarveMaxLevels;

//: in a block of `groups` groups: the totals at 0 (uint16 a group), then the words (uint32, word-major; none at a group of one digit, whose words are the mask's)
FSLA_CARVE_FN constexpr std::int64_t prepared_words_at(int groups) {
  return (std::int64_t{2} * groups + 3) & ~std::int64_t{3};
}

//: A WINDOW'S BLOCK of a level-side `width_bits` wide, in bytes, whatever its group, a window `words` mask words: the larger of a total a digit and, at groups of 2 digits, the totals and a word a mask word a group
FSLA_CARVE_FN constexpr std::int64_t prepared_block_bytes(int width_bits, int words) {
  const std::int64_t ones = prepared_words_at(1 << width_bits);
  if (width_bits < 2) return ones;
  const int groups = 1 << (width_bits - 1);
  const std::int64_t twos = prepared_words_at(groups) + std::int64_t{4} * words * groups;
  return twos > ones ? twos : ones;
}

struct PreparedLayout {
  std::int64_t offset[kLevelSides];
  std::int64_t bytes;
};

//: THE PLACE of a call's prepared masks: `width_bits[s]` and `heads[s]` (batch times the side's heads) of each level-side, 0 heads where it has no mask, over `windows` windows of `words` mask words
constexpr PreparedLayout prepared_layout(const int* width_bits, const std::int64_t* heads,
                                         std::int64_t windows, int words) {
  PreparedLayout p{};
  for (int s = 0; s < kLevelSides; ++s) {
    p.offset[s] = p.bytes;
    p.bytes += heads[s] * windows * prepared_block_bytes(width_bits[s], words);
  }
  return p;
}

}  // namespace fsla::plan
