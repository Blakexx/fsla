// csrc/rola/src/owner/owner_drain.cuh -- THE DRAIN: a pass of the window's read-live tokens' panel rows reduced into the
// output planes -- a token's channel group of `num` as one warp's reduction, its `den` by the first channel group.
// see docs/internals/owner/owner_drain.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "owner/owner.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: THE PANEL: a read-live token's 32 channels and its den, fp32, one row a read-pool slot.
struct Panel {
  float num[kReadPool][kChannels];
  float den[kReadPool];
};

//: A DRAIN PASS: slots `[first, first + n)`, a token a warp instruction (lane = channel); `den` a token a lane, by the
//: first channel group alone. -- see docs/internals/owner/owner_drain.md#pass
template <class C>
__device__ void drain(const Panel& panel, const ListTenants<C>& sm, float* num, float* den, int bh,
                      int cg, int L, int win, int first, int n, int tid) {
  const int warp = tid >> 5, lane = tid & 31;
  float* const num_bh =
      num + static_cast<int64_t>(bh) * L * (kChannelGroups * kChannels) + cg * kChannels + lane;
#pragma unroll 1
  for (int r = warp; r < n; r += kWarps) {
    const int tg = win * kWindow + sm.token_of[1][first + r];
    ops::red_global_add_f32(num_bh + static_cast<int64_t>(tg) * (kChannelGroups * kChannels),
                            panel.num[r][lane]);
  }
  if (cg == 0) {
#pragma unroll 1
    for (int r = tid; r < n; r += kThreads)
      ops::red_global_add_f32(
          den + static_cast<int64_t>(bh) * L + win * kWindow + sm.token_of[1][first + r],
          panel.den[r]);
  }
}

}  // namespace rola::owner::detail
