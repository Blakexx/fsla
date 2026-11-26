// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE LEAF GROUPS' MASKS' PLACE (docs/internals/common/plan/mask_prepare_layout.md): where the mask-preparation pass writes a side's leaf groups' mask words and counts, and where the carry reads them. Pure, constexpr, total.
#pragma once

#include <cstdint>

#include "cta_carve.h"
#include "leaf_groups.h"

namespace fsla::plan {

//: FIXED by the op's mathematics: a call has a read side and a write side, the read side first
inline constexpr int kCarveSides = 2;
inline constexpr int kLevelSides = kCarveSides * kCarveMaxLevels;

//: A LEAF GROUP'S LINE of a window of `words` mask words, in bytes: a word a mask word, a bit a token of the window
FSLA_CARVE_FN constexpr std::int64_t leaf_line_bytes(int words) { return std::int64_t{4} * words; }

//: in a block of `1 << bits` leaf groups: the counts at 0 (uint16 a leaf group), then the lines, on a line's boundary
FSLA_CARVE_FN constexpr std::int64_t leaf_lines_at(int bits, int words) {
  const std::int64_t line = leaf_line_bytes(words), counts = std::int64_t{2} << bits;
  return (counts + line - 1) / line * line;
}

//: A WINDOW'S BLOCK of a side at `1 << bits` leaf groups, in bytes: its counts and a line a leaf group
FSLA_CARVE_FN constexpr std::int64_t leaf_block_bytes(int bits, int words) {
  return leaf_lines_at(bits, words) + (leaf_line_bytes(words) << bits);
}

struct PreparedLayout {
  std::int64_t offset[kCarveSides];
  std::int64_t bytes;
};

//: THE PLACE of a call's leaf groups' masks, sized at the most leaf groups its shape allows: `width_bits` of its `levels` levels, and of each side `masked` (a bit a level it has a mask on)
//: and `heads` (batch times the side's heads), over `windows` windows of `words` mask words
constexpr PreparedLayout prepared_layout(int levels, const int* width_bits, const int* masked,
                                         const std::int64_t* heads, std::int64_t windows,
                                         int words) {
  PreparedLayout p{};
  for (int side = 0; side < kCarveSides; ++side) {
    const int most = leaf_group_bits_most(levels, width_bits, masked[side]);
    p.offset[side] = p.bytes;
    p.bytes += heads[side] * windows * leaf_block_bytes(most, words);
  }
  return p;
}

namespace layout_proof {

static_assert(leaf_lines_at(0, 32) == 128 && leaf_block_bytes(0, 32) == 256,
              "one leaf group: a line of counts and its line");
static_assert(leaf_lines_at(6, 32) == 128 && leaf_block_bytes(6, 32) == 128 + 64 * 128,
              "the cell's 64 leaf groups: their counts fill a line");
static_assert(leaf_lines_at(12, 32) == 8192 && leaf_block_bytes(12, 32) == 8192 + 4096 * 128,
              "4,096 leaf groups: 8,192 bytes of counts, then 4,096 lines");

}  // namespace layout_proof

}  // namespace fsla::plan
