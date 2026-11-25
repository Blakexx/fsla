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

//: THE TOKEN AT A PLACE of the listing: lane `p` of round `r` takes place `WORDS r + p`; -1 past the total -- see docs/internals/carry/carry_list.md#select
template <int WORDS>
__device__ __forceinline__ int listed_token(const Listing& s, int round) {
  const int place = WORDS * round + static_cast<int>(threadIdx.x & (WORDS - 1));
  //: the lane whose tokens hold the place: the count of lanes that end at or before it
  int lane = 0;
#pragma unroll
  for (int step = WORDS / 2; step >= 1; step >>= 1) {
    const int ends = ops::lane_broadcast(s.through, lane + step - 1);
    lane += ends <= place ? step : 0;
  }
  const auto word = static_cast<std::uint32_t>(ops::lane_broadcast(static_cast<int>(s.word), lane));
  const int before = ops::lane_broadcast(s.before, lane);
  const int token = WORDS * lane + ops::nth_set_bit(word, place - before);
  return place < s.total ? token : -1;
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
