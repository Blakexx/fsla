// measure/lab/experiments/reg_five.cu -- QUESTION: `reg_trade` traded registers between two warpgroups; can a CTA of
// FIVE warpgroups, 640 threads, trade them among its roles, and what does the compiler give such a kernel at entry?
// METHOD: a CTA an SM of five warpgroups, launch bounds one CTA an SM. Two warpgroups raise themselves to 160
// registers a thread, one lowers to 80 and two to 40: 480 over the five, the 96 a thread the kernel enters with.
// Each role's whole body is in its own branch, its `setmaxnreg` first. A thread holds a count of words its role's
// registers hold -- 128, 32 or 12 -- through a loop that rewrites every one each trip, and stores their sum: a role
// short of its registers spills (the lab's pack refuses local memory) or never returns from its raise. Every sum is
// checked exactly, and the kernel's span is clocked.
// A SECOND KERNEL asks the trade BY WARP: 16 warps enter at 128 registers a thread; one warp of the last warpgroup
// raises itself to 176 and its neighbour lowers itself to 80, the other fourteen untouched -- a warp's count its
// own, not its warpgroup's.
// TIMELINE: no load and no shared store. A thread's trip is one multiply-add a held word; nothing is staged.
// see docs/internals/tools/lab.md#reg-five
#include "lab.cuh"

namespace reg_five {

constexpr int kGroup = 128, kGroups = 5, kThreads = kGroups * kGroup;
//: THE REGISTERS A THREAD: what a kernel of 640 threads enters with, and each role's count; a raise takes only
//: what the other warpgroups released from the entry count, so the roles' counts sum to it
constexpr int kEntry = 96, kWide = 160, kMid = 80, kSmall = 40;
static_assert(2 * kWide + kMid + 2 * kSmall == kGroups * kEntry);
//: the words a role's thread holds
constexpr int kWideWords = 128, kMidWords = 32, kSmallWords = 12;

constexpr uint32_t seed_of(uint32_t block, uint32_t tid, uint32_t i) { return block * 7919u + tid * 131u + i; }

//: A ROLE: `N` words held in registers, each rewritten every trip, their sum stored
template <int N>
__device__ __forceinline__ void hold_role(long long* times, uint32_t* out, int trips) {
  uint32_t v[N];
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = lab::held(seed_of(blockIdx.x, threadIdx.x, static_cast<uint32_t>(i)));
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < trips; ++it) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = v[i] * 3u + static_cast<uint32_t>(it);
  }
  clock.stop(times);
  uint32_t sum = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) sum += v[i];
  out[blockIdx.x * kThreads + threadIdx.x] = sum;
}

__global__ void __launch_bounds__(kThreads, 1) five(long long* times, uint32_t* out, int trips) {
  const int group = threadIdx.x / kGroup;
  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH (docs/lab/reg_trade.md): the two lowered roles and the middle one
  //: release what the two wide ones acquire
  if (group < 2) {
    lab::regs_up<kWide>();
    hold_role<kWideWords>(times, out, trips);
  } else if (group == 2) {
    lab::regs_down<kMid>();
    hold_role<kMidWords>(times, out, trips);
  } else {
    lab::regs_down<kSmall>();
    hold_role<kSmallWords>(times, out, trips);
  }
}

//: THE TRADE BY WARP: 512 threads enter at 128; warp 12 raises to `kUp`, warp 13 lowers to `kDown`, what one
//: releases the other acquires; every other warp, the two's own warpgroup's among them, stays at the entry count
constexpr int kWarpThreads = 512, kUp = 176, kDown = 80, kUpWords = 144, kDownWords = 32, kStayWords = 96;
static_assert(kUp + kDown == 2 * 128);
constexpr int warp_words(int warp) { return warp == 12 ? kUpWords : warp == 13 ? kDownWords : kStayWords; }

__global__ void __launch_bounds__(kWarpThreads, 1) by_warp(long long* times, uint32_t* out, int trips) {
  const int warp = threadIdx.x >> 5;
  if (warp == 12) {
    lab::regs_up<kUp>();
    hold_role<kUpWords>(times, out, trips);
  } else if (warp == 13) {
    lab::regs_down<kDown>();
    hold_role<kDownWords>(times, out, trips);
  } else {
    hold_role<kStayWords>(times, out, trips);
  }
}

}  // namespace reg_five

int main(int argc, char** argv) {
  using namespace reg_five;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  constexpr int kCheckTrips = 16, kTrips = 4096;
  const std::string name = "five warpgroups: two at 160, one at 80, two at 40";
  uint32_t* out = nullptr;
  const size_t words = static_cast<size_t>(b.sms) * kThreads;
  LAB_CHECK(cudaMalloc(&out, words * 4));
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  five<<<b.sms, kThreads>>>(b.times, out, kCheckTrips);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = 0; tid < kThreads; ++tid) {
      const int group = tid / kGroup, n = group < 2 ? kWideWords : group == 2 ? kMidWords : kSmallWords;
      uint32_t want = 0;
      for (int i = 0; i < n; ++i) {
        uint32_t v = seed_of(static_cast<uint32_t>(blk), static_cast<uint32_t>(tid), static_cast<uint32_t>(i));
        for (int it = 0; it < kCheckTrips; ++it) v = v * 3u + static_cast<uint32_t>(it);
        want += v;
      }
      wrong += got[static_cast<size_t>(blk) * kThreads + tid] != want;
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(words) + " sums differ");
  const lab::Samples cycles = b.variant([&] { five<<<b.sms, kThreads>>>(b.times, out, kTrips); },
                                        [&] { return b.span(kThreads >> 5) / kTrips; });
  b.record(name, "cycles a trip a CTA, its 20 warps", cycles);
  {
    //: the trade by warp: `hold_role` stores at `block * kThreads + thread`, so a block's 512 sums lie in its 640
    const std::string by = "sixteen warps at 128: one warp of a warpgroup raised to 176, its neighbour lowered to 80";
    LAB_CHECK(cudaMemset(out, 0, words * 4));
    by_warp<<<b.sms, kWarpThreads>>>(b.times, out, kCheckTrips);
    LAB_CHECK(cudaDeviceSynchronize());
    LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
    long long off = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kWarpThreads; ++tid) {
        uint32_t want = 0;
        for (int i = 0; i < warp_words(tid >> 5); ++i) {
          uint32_t v = seed_of(static_cast<uint32_t>(blk), static_cast<uint32_t>(tid), static_cast<uint32_t>(i));
          for (int it = 0; it < kCheckTrips; ++it) v = v * 3u + static_cast<uint32_t>(it);
          want += v;
        }
        off += got[static_cast<size_t>(blk) * kThreads + tid] != want;
      }
    b.check(by, off == 0, std::to_string(off) + " of " + std::to_string(static_cast<size_t>(b.sms) * kWarpThreads) + " sums differ");
    const lab::Samples trip = b.variant([&] { by_warp<<<b.sms, kWarpThreads>>>(b.times, out, kTrips); },
                                        [&] { return b.span(kWarpThreads >> 5) / kTrips; });
    b.record(by, "cycles a trip a CTA, its 16 warps", trip);
  }
  return 0;
}
