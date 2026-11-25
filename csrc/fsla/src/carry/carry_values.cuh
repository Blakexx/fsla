// THE CARRY'S VALUE LINES, their fetch -- docs/internals/carry/carry_values.md
#pragma once

#include <cstdint>

#include "carry/carry_list.cuh"
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

//: THE COPIES OF A ROUND'S VALUE LINES into its rows at `tiles`, one committed group: `tokens` the round's 32 places of the warp's list, a token's line `stride` bytes on from the token before's; `tiles` on a 1,024-byte boundary -- see docs/internals/carry/carry_values.md#fetch
template <int CHANNELS>
__device__ __forceinline__ void fetch_round(ops::SmemAddr tiles, const std::uint8_t* values,
                                            std::int64_t stride, const std::uint16_t* tokens,
                                            bool live) {
  constexpr int kLanes = ops::kBallotLanes, kPieces = CHANNELS / kPieceChannels;
  constexpr int kTokens = kLanes / kPieces;
  static_assert(kRowBytes<CHANNELS> == 128, "a row is the 128 bytes the step's swizzle turns");
  const int lane = static_cast<int>(threadIdx.x) & (kLanes - 1);
  const int piece = lane % kPieces, which = lane / kPieces;
  //: THE ROUND'S TOKENS ARE LOADED FIRST, all of them: a token loaded where it is used holds the warp until it lands
  int listed[kPieces];
#pragma unroll
  for (int i = 0; i < kPieces; ++i) listed[i] = tokens[kTokens * i + which];

  //: a copy's row is `kTokens` rows on from the copy before's, so the rows of every other copy turn their pieces alike: two places a lane, formed once
  const auto turned = static_cast<std::uint32_t>(piece);
  const ops::SmemAddr even =
      ops::step_row_piece(tiles + static_cast<ops::SmemAddr>(which * kRowBytes<CHANNELS>), turned);
  const ops::SmemAddr odd = ops::step_row_piece(
      tiles + static_cast<ops::SmemAddr>((kTokens + which) * kRowBytes<CHANNELS>), turned);
  const std::uint8_t* const lines = values + piece * kPieceBytes;
  //: a line's place in a window fits 32 bits
  const int line = static_cast<int>(stride);
#pragma unroll
  for (int i = 0; i < kPieces; ++i) {
    const int from = listed[i];
    const auto on = static_cast<ops::SmemAddr>((i / 2) * 2 * kTokens * kRowBytes<CHANNELS>);
    const ops::SmemAddr to = ((i & 1) != 0 ? odd : even) + on;
    ops::stage_tile_if(to, lines + static_cast<std::uint32_t>(from * line),
                       live && from != kNoToken);
  }
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
