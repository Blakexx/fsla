// THE CARRY'S VALUE LINES, their fetch -- docs/internals/carry/carry_values.md
#pragma once

#include <cstdint>

#include "common/atoms/copy.cuh"
#include "common/atoms/warp.cuh"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: its K is 16 tokens
constexpr int kStepTokens = 16;
//: FIXED by the step's operand: a brick is 8 channels and a line 16 bytes, a token's 8 channels of a brick
constexpr int kBrickChannels = 8;
constexpr int kLineBytes = 2 * kBrickChannels;
//: A STEP'S VALUE TILE: a brick's 16 token lines, the bricks a line more than that apart -- see docs/internals/carry/carry_values.md#tile
constexpr int kBrickPitch = kStepTokens + 1;

template <int CHANNELS>
constexpr int kValueTileBytes = CHANNELS / kBrickChannels * kBrickPitch * kLineBytes;

//: THE COPIES OF A ROUND'S VALUE LINES into its steps' tiles at `tiles`, one committed group -- see docs/internals/carry/carry_values.md#fetch
template <int CHANNELS>
__device__ __forceinline__ void fetch_round(ops::SmemAddr tiles, const std::uint8_t* values,
                                            int token) {
  constexpr int kLanes = ops::kBallotLanes, kPieces = CHANNELS / kBrickChannels;
  constexpr int kTokens = kLanes / kPieces;
  static_assert(CHANNELS % kBrickChannels == 0 && kLanes % kPieces == 0,
                "a token's pieces fill lanes");
  const int lane = static_cast<int>(threadIdx.x) & (kLanes - 1);
  const int piece = lane % kPieces, which = lane / kPieces;
#pragma unroll
  for (int i = 0; i < kPieces; ++i) {
    const int place = kTokens * i + which;
    const int from = ops::lane_broadcast(token, place);
    const int line = piece * kBrickPitch + place % kStepTokens;
    const ops::SmemAddr to =
        tiles + (place / kStepTokens) * kValueTileBytes<CHANNELS> + line * kLineBytes;
    const std::uint8_t* const at = values + static_cast<std::int64_t>(from) * (2 * CHANNELS);
    ops::stage_tile_if(to, at + piece * kLineBytes, from >= 0);
  }
  ops::stage_commit();
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
