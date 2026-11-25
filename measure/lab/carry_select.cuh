// measure/lab/carry_select.cuh -- THE SELECT: the token at a place of a window's listing, each lane finding its own
// place's token. The library's supply lays its tokens in a list (docs/internals/carry/carry_list.md#lay); the
// experiments `carry_list` and `carry_fetch` measure this form, the one their records are of, and share it here.
#pragma once

#include "carry/carry_list.cuh"
#include "common/atoms/warp.cuh"

namespace carry_select {

namespace carry = fsla::carry::detail;
namespace ops = ::denseref::ops;

//: THE TOKEN AT A PLACE of the listing: lane `p` of round `r` takes place `WORDS r + p`; -1 past the total -- this experiment's own: the library's supply lays its tokens (`carry_supply`)
template <int WORDS>
__device__ __forceinline__ int listed_token(const carry::Listing& s, int round) {
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

}  // namespace carry_select
