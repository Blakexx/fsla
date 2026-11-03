// csrc/rola/src/scale/scale.cuh -- THE SCALE PASS's constants and operand block: each token's scale byte for the
// carry's f16 sum across CTAs from the write-aligned bound. see docs/internals/scale/scale.md
#pragma once

#include <cstdint>

#include "carry/carry.cuh"
#include "facts/liveness_contract.cuh"

namespace rola::scale {

using carry::Plane;

//: THE GRAINS: 8 warps, a lane 8 digits of a chunk, 32 tokens a group. -- see docs/internals/scale/scale.md#constants
constexpr int kWarps = 8, kThreads = kWarps * 32;
constexpr int kLaneDigits = 8, kChunk = 32 * kLaneDigits, kGroup = 32, kAhead = 4;

constexpr float kInvDigit = 1.0f / facts::kRowMaxDigit;
constexpr float kFloor = facts::kDenseFloor;

//: THE OPERAND BLOCK: the rows, `(F, m)`, the masses, the bytes. -- see docs/internals/scale/scale.md#params
struct ScaleParams {
  Plane write_s, write_r, read_s, read_r;
  const float2* fm;
  float* mass;
  int8_t* bytes;
  float inv_factor;
  int L, window, windows, heads, write_heads, width_s, width_r, grid;
};

}  // namespace rola::scale
