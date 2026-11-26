// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE SEGMENTS OF A CALL (docs/internals/common/plan/segments.md): how a long prefill is cut into op calls, from its length, a maximum of tokens a step and the card's window. Pure, constexpr, no device.
#pragma once

#include <climits>
#include <cstdint>

#include "mask_prepare_layout.h"

namespace fsla::plan {

//: FIXED by the entry's contract, no card's fact: a maximum is a token or more, so 0 is the one that is no limit
inline constexpr std::int64_t kNoMaximum = 0;

//: `steps` op calls of `tokens` tokens each, the last holding the rest; `steps == 0`: the call has no plan
struct SegmentPlan {
  std::int64_t steps;
  std::int64_t tokens;
};

//: THE PLAN of `length` tokens under `maximum` a step on a card of `window` tokens: one step where the call fits, else the most whole windows under the maximum a step; none where a split is needed and the maximum is under a window
constexpr SegmentPlan segment_plan(std::int64_t length, std::int64_t maximum, int window) {
  if (maximum == kNoMaximum || length <= maximum) return {1, length};
  const std::int64_t tokens = maximum / window * window;
  if (tokens == 0) return {0, 0};
  return {(length + tokens - 1) / tokens, tokens};
}

//: THE OP'S OPERAND FORMATS (docs/api.md section 2), each the size of its type: a route digit a uint8, a scale an fp32, a value channel a bf16, a mask word an int32, a bit a token of one digit
inline constexpr int kRouteDigitBytes = sizeof(std::uint8_t);
inline constexpr int kScaleBytes = sizeof(float);
inline constexpr int kValueChannelBytes = sizeof(std::uint16_t);
inline constexpr int kMaskWordBytes = sizeof(std::int32_t);
inline constexpr int kMaskWordTokens = kMaskWordBytes * CHAR_BIT;

//: A STEP'S SHAPE: its levels' widths as log2, each side's masked levels (a bit a level) and batch-heads (the read side first), and the values' channels
struct StepShape {
  int levels;
  int width_bits[kCarveMaxLevels];
  int masked[kCarveSides];
  std::int64_t heads[kCarveSides];
  int value_width;
};

//: A STEP'S SHAPE from a call's numbers (`levels` widths; two masks and two head counts, the read side first), the one place it is made; `levels == 0` where a part is outside the op's
constexpr StepShape step_shape(const std::int64_t* widths, std::int64_t levels,
                               const std::int64_t* masked, const std::int64_t* heads,
                               std::int64_t value_width) {
  StepShape s{};
  if (levels < 1 || levels > kCarveMaxLevels || value_width < 1) return s;
  for (int l = 0; l < levels; ++l) {
    s.width_bits[l] = carve_width_bits(widths[l]);
    if (s.width_bits[l] < 0) return StepShape{};
  }
  for (int side = 0; side < kCarveSides; ++side) {
    if (heads[side] < 1) return StepShape{};
    s.masked[side] = static_cast<int>(masked[side]);
    s.heads[side] = heads[side];
  }
  s.value_width = static_cast<int>(value_width);
  s.levels = static_cast<int>(levels);
  return s;
}

//: THE PROJECTED OPERANDS of `windows` windows of `window` tokens, in bytes: each side's rows and scale a level, each masked level-side's mask words, and the write side's values and their scale
constexpr std::int64_t operand_bytes(const StepShape& s, std::int64_t windows, int window) {
  std::int64_t bytes = 0;
  for (int side = 0; side < kCarveSides; ++side) {
    std::int64_t token = 0, word = 0;
    for (int l = 0; l < s.levels; ++l) {
      const std::int64_t width = std::int64_t{1} << s.width_bits[l];
      token += width * kRouteDigitBytes + kScaleBytes;
      if ((s.masked[side] >> l & 1) != 0) word += width * kMaskWordBytes;
    }
    if (side == kWriteSide) token += s.value_width * kValueChannelBytes + kScaleBytes;
    bytes += s.heads[side] * windows * (token * window + word * (window / kMaskWordTokens));
  }
  return bytes;
}

//: THE OP'S WORKSPACE of those windows, in bytes, at the most leaf groups the shape allows: the prepared layout, the function its allocation is sized by
constexpr std::int64_t workspace_bytes(const StepShape& s, std::int64_t windows, int window) {
  return prepared_layout(s.levels, s.width_bits, s.masked, s.heads, windows,
                         window / kMaskWordTokens)
      .bytes;
}

//: A STEP'S BYTES: every part a step of `windows` windows holds beside the state, each from the function that sizes it; a window's bytes a window
constexpr std::int64_t step_bytes(const StepShape& s, std::int64_t windows, int window) {
  return workspace_bytes(s, windows, window) + operand_bytes(s, windows, window);
}

//: THE MOST TOKENS, whole windows, a step may hold under `limit` bytes; 0 where one window does not fit
constexpr std::int64_t step_tokens_within(const StepShape& s, std::int64_t limit, int window) {
  return limit / step_bytes(s, 1, window) * window;
}

namespace segments_proof {

constexpr bool same(SegmentPlan p, std::int64_t steps, std::int64_t tokens) {
  return p.steps == steps && p.tokens == tokens;
}

static_assert(same(segment_plan(300, kNoMaximum, 1024), 1, 300), "no limit: one step");
static_assert(same(segment_plan(300, 500, 1024), 1, 300), "a call that fits is one step");
static_assert(same(segment_plan(10000, 4000, 1024), 4, 3072), "the most whole windows a step");
static_assert(same(segment_plan(2049, 2048, 1024), 2, 2048), "the last step holds the rest");
static_assert(segment_plan(10000, 500, 1024).steps == 0, "under a window: no plan");
static_assert(segment_plan(800, 500, 1024).steps == 0, "under a window: no plan");

//: two levels of 256 digits, the read side masked on both and the write side on the first, 4 and 6 batch-heads, 64 channels
constexpr StepShape kShape = {2, {8, 8}, {3, 1}, {4, 6}, 64};
constexpr std::int64_t kOne = step_bytes(kShape, 1, 1024);
static_assert(operand_bytes(kShape, 1, 32) == 4 * (32 * 520 + 2048) + 6 * (32 * 652 + 1024),
              "rows, scales and mask words a side, and the write side's values");
static_assert(step_bytes(kShape, 7, 1024) == 7 * kOne, "a step is a window's bytes a window");
static_assert(step_tokens_within(kShape, 3 * kOne + 1, 1024) == 3 * 1024, "those that fit");
static_assert(step_tokens_within(kShape, kOne - 1, 1024) == 0, "none where one does not fit");

}  // namespace segments_proof

}  // namespace fsla::plan
