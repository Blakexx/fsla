// THE CARRY, its device code -- docs/internals/carry/carry_kernel.md
#pragma once

#include <cstdint>

#include "carry/carry_groups.cuh"
#include "carry/carry_list.cuh"
#include "carry/carry_values.cuh"
#include "common/atoms/address.cuh"
#include "common/atoms/barrier.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/tensor.cuh"
#include "common/atoms/warp.cuh"
#include "common/smem_layout.cuh"
#include "plan/levels.h"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: a warpgroup is four warps
constexpr int kGroup = 128;
constexpr int kGroupWarps = kGroup / ops::kBallotLanes;
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

//: FIXED by the tensor step's M: a CTA's value channels, a slice of a token's line -- see docs/internals/carry/carry_kernel.md#channels
constexpr int kChannels = 64;
//: FIXED by KERNEL_STANDARDS section 4: a fold warp holds two rounds of its rows in flight
constexpr int kHeld = 2;
constexpr int kWarpRows = kHeld * kRoundBytes<kChannels>;
//: FIXED by the step's operand of rows: a tile of rows starts on a 1,024-byte boundary
constexpr int kRowsAlign = 1024;
constexpr int kRowsBytes = kFolds * kGroupWarps * kWarpRows;

//: THE CTA'S SHARED MEMORY: the fold warps' rows -- see docs/internals/carry/carry_kernel.md#smem
struct FoldRows;

template <class Bt>
using Smem = SmemLayout<Bt, 1, Aligned<FoldRows, std::uint8_t, kRowsBytes, kRowsAlign>>;

//: A FOLD WARP'S ROWS
template <class Bt>
__device__ __forceinline__ std::uint8_t* fold_rows(int fold, int warp) {
  return Smem<Bt>::template get<FoldRows>() + (fold * kGroupWarps + warp) * kWarpRows;
}

//: A ROLE'S REGISTERS A THREAD: `FOLD` and `READ` are the arm's, the drain's what the multiprocessor's file leaves -- see docs/internals/carry/carry_kernel.md#registers
template <class Bt, int FOLD, int READ>
struct Registers {
  static constexpr int kDrain =
      (Bt::traits.caps.regs_per_sm - kFolds * kGroup * FOLD - kGroup * READ) / kGroup;
  static_assert(kDrain >= 24 && kDrain % 8 == 0, "the drain's registers are what the others leave");
};

//: THE PLACES A ROLE STAMPS beside its barriers -- see docs/internals/carry/carry_kernel.md#trace
enum Site {
  kSiteEntry,
  kSiteExit,
  kSiteWindow,
  kSiteSet,
  kSiteListed,
  kSiteWord,
  kSiteFetch,
  kSiteIssued,
  kSiteLanded
};

//: THE LIBRARY'S TRACER: nothing -- see docs/internals/carry/carry_kernel.md#trace
struct NoTrace {
  __device__ __forceinline__ void wait_begin(int) {}

  __device__ __forceinline__ void wait_end(int) {}

  __device__ __forceinline__ void arrive(int) {}

  __device__ __forceinline__ void phase(int) {}

  __device__ __forceinline__ void round(int, int, int, int, int, ops::SmemAddr) {}
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
template <class Bt, class Tr>
__device__ __forceinline__ void fold_role(const Operands& o, const plan::PrefillCarve& c,
                                          const Place& p, Tr& tr) {
  constexpr int kWords = plan::window_of_cc(Bt::build.cc) / 32;
  static_assert(kWords == ops::kBallotLanes, "a lane a word of a window");
  const int thread = static_cast<int>(threadIdx.x), fold = thread / kGroup;
  const int warp = thread % kGroup / ops::kBallotLanes, lane = thread % ops::kBallotLanes;
  //: THE WARP'S SHARE of its fold's sets and of their rounds -- see docs/internals/carry/carry_kernel.md#supply
  const int set_bits = 31 - __clz(c.fold_sets);
  const int shared_bits = set_bits < 2 ? set_bits : 2;
  const int first_set = warp & ((1 << shared_bits) - 1);
  const int stride = kGroupWarps >> shared_bits, first_round = warp >> shared_bits;
  //: the warp's sets a window, a value every lane holds: the loops below are bounded by it
  const int mine = ops::lane_broadcast(fold < c.folds ? 1 << (set_bits - shared_bits) : 0, 0);
  //: the levels the write side lists, a value every lane holds: a set's word branches on it
  const int listed = ops::lane_broadcast(listed_levels(c, false), 0);
  const int raised = ops::lane_broadcast(raised_levels(c, false), 0);

  const std::uint8_t* const lines = o.values + p.slice * kRowBytes<kChannels>;
  const ops::SmemAddr rows = ops::smem_addr(fold_rows<Bt>(fold, warp));
  //: A SET'S WORD, its parts formed outside the loop over sets: a level's three once a fold, a level's row a step on a window -- see docs/internals/carry/carry_kernel.md#supply
  int base[kLevels], shift[kLevels], mask[kLevels];
  fold_levels(c, p, fold, base, shift, mask);
  const std::uint32_t* row[kLevels];
  std::int64_t step[kLevels];
  std::uint32_t ours = window_word<kWords>(o, 0, lane);
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const int s = side_at(false) + l;
    row[l] = level_row<kWords>(o, s, c.write_size_bits[l], p.head, 0, lane);
    step[l] = level_step<kWords>(o, s, c.write_size_bits[l]);
  }
  //: THE WORDS A SET IS LISTED FROM ARE LOADED A SET AHEAD AND ANDed AT THEIR USE: ANDed where they are loaded, the warp waits there for them
  std::uint32_t got[kLevels];
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    got[l] = ~0u;
    if ((listed >> l & 1) != 0 && mine > 0) {
      const int group = base[l] + ((first_set >> shift[l]) & mask[l]);
      got[l] = group_words(row[l], group, raised >> (4 * l) & 15);
    }
  }
  std::uint32_t whose = mine > 0 ? ours : 0u;
  int issued = 0;

#pragma unroll 1
  for (int w = 0; w < o.windows; ++w) {
    tr.phase(kSiteWindow);
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (w > 0) take(tr, kTilesReadId + k, kTilesThreads);
      //: NOT BUILT: tile group `k` of the state's image, published in the readout's orientation and fenced
      hand(tr, kTilesReadyId + k, kTilesThreads);
    }

    const std::uint8_t* const from =
        lines + (p.head * o.tokens + std::int64_t{w} * kWords * 32) * o.line_bytes;
#pragma unroll 1
    for (int i = 0; i < mine; ++i) {
      tr.phase(kSiteSet);
      std::uint32_t word = whose;
#pragma unroll
      for (int l = 0; l < kLevels; ++l) word &= got[l];
      const Listing s = listing_of<kWords>(word);
      tr.phase(kSiteListed);
      //: the set after's words: the window's last set loads the next window's first, a level's row a step on; the call's last its own again
      const bool turn = i + 1 == mine, on = turn && w + 1 < o.windows;
      if (on) {
        const std::int64_t at = (std::int64_t{w} + 1) * kWords + lane;
        ours = window_word<kWords>(o, w + 1, lane);
        //: a lane past the mask's last word keeps its row: the call's own tokens count its word for nothing
#pragma unroll
        for (int l = 0; l < kLevels; ++l) row[l] += at < o.words ? step[l] : 0;
      }
      const int set_after = first_set + ((turn ? 0 : i + 1) << shared_bits);
      whose = ours;
#pragma unroll
      for (int l = 0; l < kLevels; ++l) {
        if ((listed >> l & 1) != 0) {
          const int group = base[l] + ((set_after >> shift[l]) & mask[l]);
          got[l] = group_words(row[l], group, raised >> (4 * l) & 15);
        }
      }
      tr.phase(kSiteWord);
      const int set = first_set + (i << shared_bits);
      int round = first_round;
      if (ops::lane_ballot(kWords * round < s.total) != 0u) {
#pragma unroll 1
        do {
          tr.phase(kSiteFetch);
          const int token = listed_token<kWords>(s, round);
          const int slot = issued & (kHeld - 1);
          const ops::SmemAddr to = rows + static_cast<ops::SmemAddr>(slot * kRoundBytes<kChannels>);
          fetch_round<kChannels>(to, from, o.line_bytes, token);
          tr.phase(kSiteIssued);
          ops::stage_wait<kHeld - 1>();
          tr.phase(kSiteLanded);
          tr.round(fold, set, w, round, token, to);
          round += stride;
          ++issued;
        } while (ops::lane_ballot(kWords * round < s.total) != 0u);
      }
      //: NOT BUILT: the round's entries expanded into its coefficient tile, and its steps
    }
    take(tr, kFoldId + fold, kGroup);
    //: NOT BUILT: the groups' steps, behind their fence, committed and waited for
    take(tr, kFoldId + fold, kGroup);
  }
  ops::stage_wait<0>();
}

//: THE READOUT'S WARPGROUP -- see docs/internals/carry/carry_kernel.md#readout
template <class Tr>
__device__ __forceinline__ void read_role(const Operands& o, Tr& tr) {
#pragma unroll 1
  for (int w = 0; w < o.windows; ++w) {
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
__device__ __forceinline__ void drain_role(const Operands& o, Tr& tr) {
#pragma unroll 1
  for (int w = 0; w < o.windows; ++w) {
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
__device__ __forceinline__ void carry_cta(const Operands& o, Tr& tr) {
  const int thread = static_cast<int>(threadIdx.x);
  tr.phase(kSiteEntry);
  if (thread < kReadAt) {
    ops::registers_up<FOLD>();
    const plan::PrefillCarve c = carve_of(o);
    const int slices = o.line_bytes / kRowBytes<kChannels>;
    const Place p = place_of(o, c, static_cast<std::int64_t>(blockIdx.x), slices);
    fold_role<Bt>(o, c, p, tr);
  } else if (thread < kDrainAt) {
    ops::registers_down<READ>();
    read_role(o, tr);
  } else {
    ops::registers_down<Registers<Bt, FOLD, READ>::kDrain>();
    drain_role(o, tr);
  }
  tr.phase(kSiteExit);
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
