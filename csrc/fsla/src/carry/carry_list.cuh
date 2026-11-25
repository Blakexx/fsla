// THE CARRY'S LISTING, its device code -- docs/internals/carry/carry_list.md
#pragma once

#include <cstdint>

#include "common/atoms/warp.cuh"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: A GROUP'S WINDOW AS A WARP HOLDS IT, a lane a mask word -- see docs/internals/carry/carry_list.md#listing
struct Listing {
  std::uint32_t word;
  int before;
  int through;
  int total;
};

//: THE LISTING OF A WINDOW from a lane's word of it (the side's masked levels' words ANDed) -- see docs/internals/carry/carry_list.md#listing
template <int WORDS>
__device__ __forceinline__ Listing listing_of(std::uint32_t word) {
  static_assert(WORDS == ops::kBallotLanes, "a window's mask words are a warp's lanes");
  const int live = __popc(word);
  const int through = ops::lane_prefix_incl<WORDS>(live);
  return {word, through - live, through, ops::lane_broadcast(through, WORDS - 1)};
}

//: FIXED by the mask: a word is 32 tokens
constexpr int kWordTokens = 32;

//: A PLACE OF A WARP'S LIST that holds no token
constexpr std::uint16_t kNoToken = 0xFFFF;

//: A WARP'S LIST of a set's tokens: a place a token of a window, and a spare place a lane -- see docs/internals/carry/carry_list.md#lay
template <int WORDS>
constexpr int kListPlaces = WORDS * kWordTokens + ops::kBallotLanes;

//: THE PART OF A WORD'S TOKENS a warp takes of a set that `1 << split_bits` warps share, the warp their `turn`th: that share of the word's bits -- see docs/internals/carry/carry_list.md#share
__device__ __forceinline__ std::uint32_t shared_part(int split_bits, int turn) {
  const int bits = kWordTokens >> split_bits;
  return split_bits == 0 ? ~0u : ((1u << bits) - 1u) << (turn * bits);
}

//: THE LISTING'S TOKENS LAID into `list`: the token of rank `r` at place `r`, and no token from the total to the end of its round -- see docs/internals/carry/carry_list.md#lay
template <int WORDS>
__device__ __forceinline__ void lay_tokens(const Listing& s, std::uint16_t* list) {
  constexpr int kLanes = ops::kBallotLanes;
  const int lane = static_cast<int>(threadIdx.x & (kLanes - 1));
  const int spare = WORDS * kWordTokens + lane;
  //: the trips are the fullest word's tokens, one warp reduction, a value every lane holds: no vote a trip
  const int most = __reduce_max_sync(0xFFFFFFFFu, __popc(s.word));

  std::uint32_t rest = s.word;
  int rank = s.before;
#pragma unroll 1
  for (int trip = 0; trip < most; ++trip) {
    //: a store every trip and no branch: a lane with no token left stores at its spare place
    const int bit = __ffs(static_cast<int>(rest)) - 1;
    list[rest != 0u ? rank : spare] = static_cast<std::uint16_t>(kWordTokens * lane + bit);
    rank += rest != 0u ? 1 : 0;
    rest &= rest - 1u;
  }

  const int tail = s.total & (kLanes - 1);
  list[tail != 0 && lane >= tail ? s.total - tail + lane : spare] = kNoToken;
  //: every lane's stores stand before any lane's loads of the list
  __syncwarp();
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
