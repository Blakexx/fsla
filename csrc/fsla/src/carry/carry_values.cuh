// THE CARRY'S VALUE LINES, their fetch -- docs/internals/carry/carry_values.md
#pragma once

#include <cstdint>

#include "common/atoms/copy.cuh"
#include "common/atoms/tensor.cuh"
#include "common/atoms/warp.cuh"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: its K is 16 tokens
constexpr int kStepTokens = 16;
//: FIXED by the step's operand: a row's piece is 16 bytes, 8 channels
constexpr int kPieceChannels = 8;
constexpr int kPieceBytes = 2 * kPieceChannels;
//: A ROUND'S TILE: its 32 tokens' value lines, a row each as they lie in global memory -- see docs/internals/carry/carry_values.md#tile
template <int CHANNELS>
constexpr int kRowBytes = 2 * CHANNELS;

template <int CHANNELS>
constexpr int kRoundBytes = ops::kBallotLanes * kRowBytes<CHANNELS>;

//: THE COPIES OF A ROUND'S VALUE LINES into its rows at `tiles`, one committed group; a token's line is `stride` bytes on from the token before's -- see docs/internals/carry/carry_values.md#fetch
template <int CHANNELS>
__device__ __forceinline__ void fetch_round(ops::SmemAddr tiles, const std::uint8_t* values,
                                            std::int64_t stride, int token) {
  constexpr int kLanes = ops::kBallotLanes, kPieces = CHANNELS / kPieceChannels;
  constexpr int kTokens = kLanes / kPieces;
  static_assert(kRowBytes<CHANNELS> == 128, "a row is the 128 bytes the step's swizzle turns");
  const int lane = static_cast<int>(threadIdx.x) & (kLanes - 1);
  const int piece = lane % kPieces, which = lane / kPieces;
#pragma unroll
  for (int i = 0; i < kPieces; ++i) {
    const int place = kTokens * i + which;
    const int from = ops::lane_broadcast(token, place);
    const ops::SmemAddr row = tiles + static_cast<ops::SmemAddr>(place * kRowBytes<CHANNELS>);
    const std::uint8_t* const at = values + static_cast<std::int64_t>(from) * stride;
    const ops::SmemAddr to = ops::step_row_piece(row, static_cast<std::uint32_t>(piece));
    ops::stage_tile_if(to, at + piece * kPieceBytes, from >= 0);
  }
  ops::stage_commit();
}

//: STEP `step` OF A ROUND'S TILE as the tensor step's operand: its 16 rows; `tiles` is on a 1,024-byte boundary
template <int CHANNELS>
__device__ __forceinline__ std::uint64_t round_step(ops::SmemAddr tiles, int step) {
  const int first = step * kStepTokens * kRowBytes<CHANNELS>;
  return ops::step_operand_rows(tiles + static_cast<ops::SmemAddr>(first));
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
