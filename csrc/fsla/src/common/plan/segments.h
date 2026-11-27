// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE SEGMENTS OF A CALL (docs/internals/common/plan/segments.md): how a long prefill is cut into op calls, head blocks of token steps, from its length, its batch and write heads, a policy's maxima and the card's window. Pure, constexpr, no device.
#pragma once

#include <climits>
#include <cstdint>

#include "mask_prepare_layout.h"

namespace fsla::plan {

//: FIXED by the entries' contract, no card's fact: a maximum is a token or a write head or more, so 0 is the one that is no limit
inline constexpr std::int64_t kNoMaximum = 0;

//: IN EACH HEAD BLOCK of `sequences` whole sequences, or of `heads` write heads of one sequence (`sequences == 1`), `steps` op calls of `tokens` tokens each, the last holding the rest; `steps == 0`: the call has no plan
struct SegmentPlan {
  std::int64_t steps;
  std::int64_t tokens;
  std::int64_t sequences;
  std::int64_t heads;
};

//: THE PLAN of `length` tokens of `batch` sequences of `heads` write heads under `max_tokens` a step and `max_heads` write heads a step, on a card of `window` tokens. Heads: whole sequences a block where the maximum holds one sequence's write heads, else a range of one sequence's. Tokens: one step where the call fits, else the most whole windows under the maximum; no plan where a split is needed and the maximum is under a window
constexpr SegmentPlan segment_plan(std::int64_t length, std::int64_t max_tokens,
                                   std::int64_t max_heads, std::int64_t batch, std::int64_t heads,
                                   int window) {
  SegmentPlan p{1, length, batch, heads};
  if (max_heads != kNoMaximum && max_heads < batch * heads) {
    if (max_heads >= heads) {
      p.sequences = max_heads / heads;
    } else {
      p.sequences = 1;
      p.heads = max_heads;
    }
  }
  if (max_tokens == kNoMaximum || length <= max_tokens) return p;
  p.tokens = max_tokens / window * window;
  if (p.tokens == 0) return {0, 0, 0, 0};
  p.steps = (length + p.tokens - 1) / p.tokens;
  return p;
}

//: THE OP'S OPERAND FORMATS (docs/api.md section 2), each the size of its type: a route digit a uint8, a scale an fp32, a value channel a bf16, a mask word an int32, a bit a token of one digit
inline constexpr int kRouteDigitBytes = sizeof(std::uint8_t);
inline constexpr int kScaleBytes = sizeof(float);
inline constexpr int kValueChannelBytes = sizeof(std::uint16_t);
inline constexpr int kMaskWordBytes = sizeof(std::int32_t);
inline constexpr int kMaskWordTokens = kMaskWordBytes * CHAR_BIT;

//: A STEP'S SHAPE: its levels' widths as log2, each side's masked levels (a bit a level) and batch-heads (the read side first: the query side, a whole number of the write side's), and the values' channels
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
  if (heads[0] % heads[kWriteSide] != 0) return StepShape{};
  s.value_width = static_cast<int>(value_width);
  s.levels = static_cast<int>(levels);
  return s;
}

//: ONE WRITE HEAD of a step's shape, with every query head that reads it: the unit a step's bytes are a multiple of
constexpr StepShape write_head_of(const StepShape& s) {
  StepShape one = s;
  one.heads[0] = s.heads[0] / s.heads[kWriteSide];
  one.heads[kWriteSide] = 1;
  return one;
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

//: A STEP'S BYTES: every part a step of `windows` windows holds beside the state, each from the function that sizes it; a window's bytes a window, a write head's a write head
constexpr std::int64_t step_bytes(const StepShape& s, std::int64_t windows, int window) {
  return workspace_bytes(s, windows, window) + operand_bytes(s, windows, window);
}

//: THE MOST TOKENS, whole windows, a step of every head may hold under `limit` bytes; 0 where one window does not fit
constexpr std::int64_t step_tokens_within(const StepShape& s, std::int64_t limit, int window) {
  return limit / step_bytes(s, 1, window) * window;
}

//: A STEP'S POLICY: the most tokens and the most write heads a step may hold (`heads` kNoMaximum: every head); `tokens == 0`: no step fits
struct StepPolicy {
  std::int64_t tokens;
  std::int64_t heads;
};

//: THE POLICY under `limit` bytes, tokens first: the most whole windows of every head; only where one window of every head does not fit, one window and the most write heads; none where one window of one write head does not fit
constexpr StepPolicy step_policy_within(const StepShape& s, std::int64_t limit, int window) {
  const std::int64_t tokens = step_tokens_within(s, limit, window);
  if (tokens > 0) return {tokens, kNoMaximum};
  const std::int64_t heads = limit / step_bytes(write_head_of(s), 1, window);
  if (heads == 0) return {0, 0};
  return {window, heads};
}

namespace segments_proof {

constexpr bool same(SegmentPlan p, std::int64_t steps, std::int64_t tokens, std::int64_t sequences,
                    std::int64_t heads) {
  return p.steps == steps && p.tokens == tokens && p.sequences == sequences && p.heads == heads;
}

static_assert(same(segment_plan(300, kNoMaximum, kNoMaximum, 2, 8, 1024), 1, 300, 2, 8),
              "no limit: one step of every head");
static_assert(same(segment_plan(300, 500, kNoMaximum, 2, 8, 1024), 1, 300, 2, 8),
              "a call that fits is one step");
static_assert(same(segment_plan(10000, 4000, kNoMaximum, 2, 8, 1024), 4, 3072, 2, 8),
              "the most whole windows a step");
static_assert(same(segment_plan(2049, 2048, kNoMaximum, 2, 8, 1024), 2, 2048, 2, 8),
              "the last step holds the rest");
static_assert(segment_plan(10000, 500, kNoMaximum, 2, 8, 1024).steps == 0,
              "under a window: no plan");
static_assert(segment_plan(800, 500, kNoMaximum, 2, 8, 1024).steps == 0, "under a window: no plan");
static_assert(same(segment_plan(300, kNoMaximum, 32, 4, 8, 1024), 1, 300, 4, 8),
              "a maximum of every write head: one block");
static_assert(same(segment_plan(300, kNoMaximum, 20, 4, 8, 1024), 1, 300, 2, 8),
              "whole sequences a block where the maximum holds a sequence's write heads");
static_assert(same(segment_plan(300, kNoMaximum, 8, 4, 8, 1024), 1, 300, 1, 8),
              "a sequence a block");
static_assert(same(segment_plan(300, kNoMaximum, 3, 4, 8, 1024), 1, 300, 1, 3),
              "a range of one sequence's write heads under one sequence's");
static_assert(same(segment_plan(10000, 4000, 3, 2, 8, 1024), 4, 3072, 1, 3),
              "the two axes: head blocks, and token steps within each");
static_assert(segment_plan(10000, 500, 3, 2, 8, 1024).steps == 0,
              "the head axis lifts no token refusal");

//: two levels of 256 digits, the read side masked on both and the write side on the first, 12 and 6 batch-heads (two query heads a write head), 64 channels
constexpr StepShape kShape = {2, {8, 8}, {3, 1}, {12, 6}, 64};
constexpr std::int64_t kOne = step_bytes(kShape, 1, 1024);
constexpr std::int64_t kHead = step_bytes(write_head_of(kShape), 1, 1024);
static_assert(operand_bytes(kShape, 1, 32) == 12 * (32 * 520 + 2048) + 6 * (32 * 652 + 1024),
              "rows, scales and mask words a side, and the write side's values");
static_assert(step_bytes(kShape, 7, 1024) == 7 * kOne, "a step is a window's bytes a window");
static_assert(kOne == 6 * kHead, "a step is a write head's bytes a write head");
static_assert(step_tokens_within(kShape, 3 * kOne + 1, 1024) == 3 * 1024, "those that fit");
static_assert(step_tokens_within(kShape, kOne - 1, 1024) == 0, "none where one does not fit");
static_assert(step_policy_within(kShape, 3 * kOne + 1, 1024).tokens == 3 * 1024
                  && step_policy_within(kShape, 3 * kOne + 1, 1024).heads == kNoMaximum,
              "tokens first: whole windows of every head where one fits");
static_assert(step_policy_within(kShape, 4 * kHead + 5, 1024).tokens == 1024
                  && step_policy_within(kShape, 4 * kHead + 5, 1024).heads == 4,
              "heads only where one window of every head does not fit");
static_assert(step_policy_within(kShape, kHead - 1, 1024).tokens == 0,
              "none where one window of one write head does not fit");

}  // namespace segments_proof

}  // namespace fsla::plan
