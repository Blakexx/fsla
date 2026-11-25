// THE CARRY, its device code -- docs/internals/carry/carry_kernel.md
#pragma once

#include <cstdint>

#include "common/atoms/barrier.cuh"
#include "common/atoms/tensor.cuh"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: a warpgroup is four warps
constexpr int kGroup = 128;
//: FIXED by the roles: two fold warpgroups, then the readout's, then the drain's warps -- see docs/internals/carry/carry_kernel.md#roles
constexpr int kFolds = 2;
constexpr int kReadAt = kFolds * kGroup, kDrainAt = kReadAt + kGroup, kThreads = kDrainAt + kGroup;
//: FIXED by the readout's pass: a window's image is four tile groups, read two a pass onto two panels
constexpr int kChunks = 4;
constexpr int kPanels = kChunks / 2;
//: FIXED by the instruction: named barrier 0 is the CTA's; a fold warpgroup takes the next, then the readout
constexpr int kFoldId = 1;
constexpr int kReadId = kFoldId + kFolds;
//: THE HAND-OFFS' BARRIERS, a tile group's two and a panel's two -- see docs/internals/carry/carry_kernel.md#handoff
constexpr int kTilesReadyId = kReadId + 1, kTilesReadId = kTilesReadyId + kChunks;
constexpr int kPanelReadyId = kTilesReadId + kChunks, kPanelFreedId = kPanelReadyId + kPanels;
static_assert(kPanelFreedId + kPanels <= 16, "a CTA has sixteen named barriers");
constexpr int kTilesThreads = (kFolds + 1) * kGroup, kPanelThreads = 2 * kGroup;

//: A ROLE'S REGISTERS A THREAD: `FOLD` and `READ` are the arm's, the drain's what the multiprocessor's file leaves -- see docs/internals/carry/carry_kernel.md#registers
template <class Bt, int FOLD, int READ>
struct Registers {
  static constexpr int kDrain =
      (Bt::traits.caps.regs_per_sm - kFolds * kGroup * FOLD - kGroup * READ) / kGroup;
  static_assert(kDrain >= 24 && kDrain % 8 == 0, "the drain's registers are what the others leave");
};

struct Args {
  int windows;
};

//: THE PLACES A ROLE STAMPS beside its barriers -- see docs/internals/carry/carry_kernel.md#trace
enum Site { kSiteEntry, kSiteExit, kSiteWindow };

//: THE LIBRARY'S TRACER: nothing -- see docs/internals/carry/carry_kernel.md#trace
struct NoTrace {
  __device__ __forceinline__ void wait_begin(int) {}

  __device__ __forceinline__ void wait_end(int) {}

  __device__ __forceinline__ void arrive(int) {}

  __device__ __forceinline__ void phase(int) {}
};

//: A HAND-OFF BETWEEN ROLES: the producer arrives, the waiter syncs -- see docs/internals/carry/carry_kernel.md#handoff
template <class Tr>
__device__ __forceinline__ void hand(Tr& tr, int id, int threads) {
  tr.arrive(id);
  ops::rendezvous_arrive(id, threads);
}

template <class Tr>
__device__ __forceinline__ void take(Tr& tr, int id, int threads) {
  tr.wait_begin(id);
  ops::rendezvous_group(id, threads);
  tr.wait_end(id);
}

//: A FOLD WARPGROUP -- see docs/internals/carry/carry_kernel.md#fold
template <class Tr>
__device__ __forceinline__ void fold_role(const Args& a, Tr& tr) {
  const int box = static_cast<int>(threadIdx.x) / kGroup;
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
    tr.phase(kSiteWindow);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (w > 0) take(tr, kTilesReadId + k, kTilesThreads);
      //: NOT BUILT: tile group `k` of the state's image, published in the readout's orientation and fenced
      hand(tr, kTilesReadyId + k, kTilesThreads);
    }
    //: NOT BUILT: the next window's lists; this window's value lines landed; a thread's entries expanded into its tile
    take(tr, kFoldId + box, kGroup);
    //: NOT BUILT: the groups' steps, behind their fence, committed and waited for; then the next window's copies issued
    take(tr, kFoldId + box, kGroup);
  }
}

//: THE READOUT'S WARPGROUP -- see docs/internals/carry/carry_kernel.md#readout
template <class Tr>
__device__ __forceinline__ void read_role(const Args& a, Tr& tr) {
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
    tr.phase(kSiteWindow);
#pragma unroll
    for (int first = 0; first < kChunks; first += kPanels) {
#pragma unroll
      for (int g = 0; g < kPanels; ++g) take(tr, kTilesReadyId + first + g, kTilesThreads);
      //: NOT BUILT: the pass's readers' coefficients expanded into its tiles and fenced
      take(tr, kReadId, kGroup);
      //: NOT BUILT: the pass's steps, behind their fence, committed and waited for
#pragma unroll
      for (int g = 0; g < kPanels; ++g) hand(tr, kTilesReadId + first + g, kTilesThreads);

#pragma unroll
      for (int g = 0; g < kPanels; ++g) {
        const int turn = (kChunks * w + first + g) / kPanels;
        if (turn > 0) take(tr, kPanelFreedId + g, kPanelThreads);
        //: NOT BUILT: the pass's sums stored once into panel `g`, in the list's order
        hand(tr, kPanelReadyId + g, kPanelThreads);
      }
    }
  }
}

//: A DRAIN WARP -- see docs/internals/carry/carry_kernel.md#drain
template <class Tr>
__device__ __forceinline__ void drain_role(const Args& a, Tr& tr) {
#pragma unroll 1
  for (int w = 0; w < a.windows; ++w) {
    tr.phase(kSiteWindow);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      const int panel = k % kPanels;
      take(tr, kPanelReadyId + panel, kPanelThreads);
      //: NOT BUILT: the panel's lines added at their tokens' output lines
      hand(tr, kPanelFreedId + panel, kPanelThreads);
    }
  }
}

//: A CTA: each role's whole body in its own branch, its registers set at the branch's head -- see docs/internals/carry/carry_kernel.md#the-cta
template <class Bt, int FOLD, int READ, class Tr>
__device__ __forceinline__ void carry_cta(const Args& a, Tr& tr) {
  const int thread = static_cast<int>(threadIdx.x);
  tr.phase(kSiteEntry);
  if (thread < kReadAt) {
    ops::registers_up<FOLD>();
    fold_role(a, tr);
  } else if (thread < kDrainAt) {
    ops::registers_down<READ>();
    read_role(a, tr);
  } else {
    ops::registers_down<Registers<Bt, FOLD, READ>::kDrain>();
    drain_role(a, tr);
  }
  tr.phase(kSiteExit);
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
