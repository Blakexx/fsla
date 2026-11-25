// THE CARRY, its device code -- docs/internals/carry/carry_kernel.md
#pragma once

#include <cstdint>

#include "common/atoms/address.cuh"
#include "common/atoms/barrier.cuh"
#include "common/atoms/tensor.cuh"
#include "common/smem_layout.cuh"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: a warpgroup is four warps
constexpr int kGroup = 128;
//: FIXED by the roles: two fold warpgroups, then the readout's, then the drain's warps -- see docs/internals/carry/carry_kernel.md#roles
constexpr int kFolds = 2;
constexpr int kReadAt = kFolds * kGroup, kDrainAt = kReadAt + kGroup, kThreads = kDrainAt + kGroup;
constexpr int kGroupWarps = kGroup / 32;
//: FIXED by the readout's pass: a window's image is four tile groups, read two a pass onto two panels
constexpr int kChunks = 4;
constexpr int kPanels = kChunks / 2;
//: FIXED by the instruction: named barrier 0 is the CTA's; a fold warpgroup takes the next, then the readout
constexpr int kFoldId = 1;
constexpr int kReadId = kFoldId + kFolds;
//: FIXED by R15: one CTA a multiprocessor
constexpr int kCtasPerSm = 1;

//: A ROLE'S REGISTERS A THREAD -- see docs/internals/carry/carry_kernel.md#registers
template <class Bt>
struct Registers {
  //: FIXED by the lab's split, the climb's to change: a fold thread's accumulators and entries
  static constexpr int kFold = 176;
  //: FIXED by the lab's split, the climb's to change: a readout thread's sums and rows on their way
  static constexpr int kRead = 104;
  static constexpr int kDrain =
      (Bt::traits.caps.regs_per_sm - kFolds * kGroup * kFold - kGroup * kRead) / kGroup;
  static_assert(kDrain >= 24 && kDrain % 8 == 0, "the drain's registers are what the others leave");
};

//: THE ROLES' BARRIERS, a shared word each -- see docs/internals/carry/carry_kernel.md#barriers
struct Bars {
  std::uint64_t tiles_ready[kChunks];
  std::uint64_t tiles_read[kChunks];
  std::uint64_t panel_ready[kPanels];
  std::uint64_t panel_freed[kPanels];
};

struct BarsTile;

//: THE CTA'S SHARED MEMORY -- see docs/internals/carry/carry_kernel.md#smem
template <class Bt>
using CarrySmem = SmemLayout<Bt, kCtasPerSm, One<BarsTile, Bars>>;

struct Args {
  int windows;
};

__device__ __forceinline__ void arrive(const std::uint64_t* bar) {
  if ((threadIdx.x & 31) == 0) ops::mbar_arrive(ops::smem_addr(bar));
}

__device__ __forceinline__ void await(const std::uint64_t* bar, int parity) {
  ops::mbar_wait_suspended(ops::smem_addr(bar), static_cast<std::uint32_t>(parity));
}

//: A FOLD WARPGROUP -- see docs/internals/carry/carry_kernel.md#fold
template <class Bt>
__device__ __forceinline__ void fold_role(const Args& a, Bars* bars) {
  const int box = static_cast<int>(threadIdx.x) / kGroup;
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (w > 0) await(&bars->tiles_read[k], (w - 1) & 1);
      //: NOT BUILT: tile group `k` of the state's image, published in the readout's orientation
      ops::async_fence();
      __syncwarp();
      arrive(&bars->tiles_ready[k]);
    }
    //: NOT BUILT: the next window's lists; this window's value lines landed; a thread's entries expanded into its tile
    ops::async_fence();
    ops::rendezvous_group(kFoldId + box, kGroup);
    //: NOT BUILT: the groups' steps, behind their fence, committed and waited for; then the next window's copies issued
    ops::rendezvous_group(kFoldId + box, kGroup);
  }
}

//: THE READOUT'S WARPGROUP -- see docs/internals/carry/carry_kernel.md#readout
template <class Bt>
__device__ __forceinline__ void read_role(const Args& a, Bars* bars) {
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
#pragma unroll
    for (int first = 0; first < kChunks; first += kPanels) {
#pragma unroll
      for (int g = 0; g < kPanels; ++g) await(&bars->tiles_ready[first + g], w & 1);
      //: NOT BUILT: the pass's readers' coefficients expanded into its tiles
      ops::async_fence();
      ops::rendezvous_group(kReadId, kGroup);
      //: NOT BUILT: the pass's steps, behind their fence, committed and waited for
#pragma unroll
      for (int g = 0; g < kPanels; ++g) arrive(&bars->tiles_read[first + g]);

#pragma unroll
      for (int g = 0; g < kPanels; ++g) {
        const int chunk = kChunks * w + first + g, turn = chunk / kPanels;
        if (turn > 0) await(&bars->panel_freed[g], (turn - 1) & 1);
        //: NOT BUILT: the pass's sums stored once into panel `g`, in the list's order
        __syncwarp();
        arrive(&bars->panel_ready[g]);
      }
    }
  }
}

//: A DRAIN WARP -- see docs/internals/carry/carry_kernel.md#drain
template <class Bt>
__device__ __forceinline__ void drain_role(const Args& a, Bars* bars) {
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      const int panel = k % kPanels, turn = (kChunks * w + k) / kPanels;
      await(&bars->panel_ready[panel], turn & 1);
      //: NOT BUILT: the panel's lines added at their tokens' output lines
      __syncwarp();
      arrive(&bars->panel_freed[panel]);
    }
  }
}

//: A CTA: its barriers, then each role's whole body in its own branch, its registers set at the branch's head -- see docs/internals/carry/carry_kernel.md#the-cta
template <class Bt>
__device__ __forceinline__ void carry_cta(const Args& a) {
  Bars* const bars = CarrySmem<Bt>::template get<BarsTile>();
  const int thread = static_cast<int>(threadIdx.x);
  if (thread == 0) {
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      ops::mbar_init(ops::smem_addr(&bars->tiles_ready[k]), kFolds * kGroupWarps);
      ops::mbar_init(ops::smem_addr(&bars->tiles_read[k]), kGroupWarps);
    }
#pragma unroll
    for (int p = 0; p < kPanels; ++p) {
      ops::mbar_init(ops::smem_addr(&bars->panel_ready[p]), kGroupWarps);
      ops::mbar_init(ops::smem_addr(&bars->panel_freed[p]), kGroupWarps);
    }
  }
  ops::async_fence();
  __syncthreads();

  if (thread < kReadAt) {
    ops::registers_up<Registers<Bt>::kFold>();
    fold_role<Bt>(a, bars);
  } else if (thread < kDrainAt) {
    ops::registers_down<Registers<Bt>::kRead>();
    read_role<Bt>(a, bars);
  } else {
    ops::registers_down<Registers<Bt>::kDrain>();
    drain_role<Bt>(a, bars);
  }
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
