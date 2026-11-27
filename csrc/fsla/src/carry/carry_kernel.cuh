// THE CARRY, its device code -- docs/internals/carry/carry_kernel.md
#pragma once

#include <cstdint>

#include "carry/carry_coefs.cuh"
#include "carry/carry_groups.cuh"
#include "carry/carry_list.cuh"
#include "carry/carry_values.cuh"
#include "common/atoms/address.cuh"
#include "common/atoms/barrier.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/memory.cuh"
#include "common/atoms/tensor.cuh"
#include "common/atoms/warp.cuh"
#include "common/smem_layout.cuh"
#include "common/structure_switch.cuh"
#include "plan/levels.h"

namespace fsla {
namespace carry {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the tensor step: a warpgroup is four warps
constexpr int kGroup = 128;
constexpr int kGroupWarps = kGroup / ops::kBallotLanes;
//: FIXED by the roles: a warpgroup is four warps, as log2
constexpr int kGroupWarpBits = 2;
static_assert(kGroupWarps == 1 << kGroupWarpBits, "a warpgroup is four warps");
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

//: A FOLD WARP'S COEFFICIENT TILE: a round's two steps at the widest step's 64 columns -- see docs/internals/carry/carry_kernel.md#smem
constexpr int kWarpCoefs = kCoefRoundBytes<StepShape<kStepColumnBits, 0, 0, 0>>;
//: FIXED by the tile's stores: a lane's 16 bytes at a time, and a step's bricks whole lines of the banks
constexpr int kCoefsAlign = 128;

//: THE CTA'S SHARED MEMORY: the fold warps' rows, their lists of a set's tokens and their coefficient tiles -- see docs/internals/carry/carry_kernel.md#smem
struct FoldRows;
struct FoldTokens;
struct FoldCoefs;
struct FoldStaged;

//: the window's mask words on the build's card
template <class Bt>
constexpr int kWindowWords = plan::window_of_cc(Bt::build.cc) / kWordTokens;

template <class Bt>
using Smem = SmemLayout<
    Bt, 1, Aligned<FoldRows, std::uint8_t, kRowsBytes, kRowsAlign>,
    Fixed<FoldTokens, std::uint16_t, kFolds * kGroupWarps * kListPlaces<kWindowWords<Bt>>>,
    Aligned<FoldCoefs, std::uint8_t, kFolds * kGroupWarps * kWarpCoefs, kCoefsAlign>,
    Aligned<FoldStaged, std::uint8_t, kFolds * kGroupWarps * kHeld * kStageMost, kCoefsAlign>>;

//: A FOLD WARP'S LIST of the tokens of the set it is on
template <class Bt>
__device__ __forceinline__ std::uint16_t* fold_tokens(int fold, int warp) {
  return Smem<Bt>::template get<FoldTokens>()
         + (fold * kGroupWarps + warp) * kListPlaces<kWindowWords<Bt>>;
}

//: A FOLD WARP'S ROWS
template <class Bt>
__device__ __forceinline__ std::uint8_t* fold_rows(int fold, int warp) {
  return Smem<Bt>::template get<FoldRows>() + (fold * kGroupWarps + warp) * kWarpRows;
}

//: A FOLD WARP'S COEFFICIENT TILE
template <class Bt>
__device__ __forceinline__ std::uint8_t* fold_coefs(int fold, int warp) {
  return Smem<Bt>::template get<FoldCoefs>() + (fold * kGroupWarps + warp) * kWarpCoefs;
}

//: A FOLD WARP'S STAGED BYTES: two stages' in flight, a stage the widest's -- see docs/internals/carry/carry_kernel.md#smem
template <class Bt>
__device__ __forceinline__ std::uint8_t* fold_staged(int fold, int warp) {
  return Smem<Bt>::template get<FoldStaged>() + (fold * kGroupWarps + warp) * (kHeld * kStageMost);
}

//: FIXED by the fold's box and the step: a fold thread's accumulators, 256 leaves by 64 channels over a warpgroup's 128 threads -- see docs/internals/carry/carry_kernel.md#registers
constexpr int kStateRegisters = 256 * kChannels / kGroup;

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
  kSiteLanded,
  kSiteCoefs
};

//: THE LIBRARY'S TRACER: nothing -- see docs/internals/carry/carry_kernel.md#trace
struct NoTrace {
  __device__ __forceinline__ void wait_begin(int) {}

  __device__ __forceinline__ void wait_end(int) {}

  __device__ __forceinline__ void arrive(int) {}

  __device__ __forceinline__ void phase(int) {}

  __device__ __forceinline__ void round(int, int, int, int, const std::uint16_t*, ops::SmemAddr) {}

  __device__ __forceinline__ void coefs(int, int, int, int, int, const std::uint16_t*,
                                        ops::SmemAddr, int) {}
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

//: A FOLD WARP'S WINDOWS, compiled for its fold's class (`CLASS`): the call's levels and a step's shape. A set's tokens are one line of the write side's leaf groups, the fold's sets the leaf groups from `first_leaf` on -- see docs/internals/carry/carry_kernel.md#supply
template <class Bt, class Tr, int CLASS>
__device__ __forceinline__ void fold_windows(const Operands& o, const plan::PrefillCarve& c,
                                             const Place& p, const FoldPositions& positions,
                                             int leaf_bits, int first_leaf, Tr& tr) {
  constexpr int kWords = plan::window_of_cc(Bt::build.cc) / 32;
  constexpr FoldClass kClass = kFoldClasses[CLASS];
  constexpr int kDepth = kClass.levels;
  constexpr int kB0 = kClass.bits[0], kB1 = kClass.bits[1], kB2 = kClass.bits[2];
  constexpr int kB3 = kClass.bits[3], kChunkBits = kClass.chunk_bits;
  constexpr int kSetChunks = 1 << kChunkBits;
  //: A STAGE IS AS MANY OF A SET'S CHUNKS AS FIT ITS SLOT: their bytes are staged together, a round's value lines with its first stage's
  constexpr int kStageBits = stage_chunk_bits(kDepth, kB0, kB1, kB2, kB3, kChunkBits);
  constexpr int kStageChunks = 1 << kStageBits, kStages = kSetChunks / kStageChunks;
  constexpr int kS0 = kB0 + kStageBits;
  static_assert(StageShape<kDepth, kS0, kB1, kB2, kB3>::kBytes <= kStageMost,
                "a stage fits its tile");
  using Shape = StepShape<kB0, kB1, kB2, kB3>;
  static_assert(kWords == ops::kBallotLanes, "a lane a word of a window");
  const int thread = static_cast<int>(threadIdx.x), fold = thread / kGroup;
  const int warp = thread % kGroup / ops::kBallotLanes, lane = thread % ops::kBallotLanes;
  //: THE WARP'S SHARE of its fold's sets and of their rounds -- see docs/internals/carry/carry_kernel.md#supply
  const int set_bits = 31 - __clz(c.fold_sets);
  const int shared_bits = set_bits < 2 ? set_bits : 2;
  const int first_set = warp & ((1 << shared_bits) - 1);
  //: warps that share a set split a word's tokens among them by bit: each lists its part and runs its own rounds
  const std::uint32_t part = shared_part(kGroupWarpBits - shared_bits, warp >> shared_bits);
  std::uint16_t* const list = fold_tokens<Bt>(fold, warp);
  //: the warp's sets a window, a value every lane holds: the loops below are bounded by it
  const int mine = ops::lane_broadcast(fold < c.folds ? 1 << (set_bits - shared_bits) : 0, 0);

  const std::uint8_t* const lines = o.values + p.slice * kRowBytes<kChannels>;
  const ops::SmemAddr rows = ops::smem_addr(fold_rows<Bt>(fold, warp));
  const ops::SmemAddr coefs = ops::smem_addr(fold_coefs<Bt>(fold, warp));
  const ops::SmemAddr staged = ops::smem_addr(fold_staged<Bt>(fold, warp));
  //: the fold's positions, their rows the call's: a window's place in them is `tiles_at`, a window a whole number of the rows' tiles
  const FoldPositions& at = positions;
  //: the window's place in its head's value lines and in the rows' tiles: offsets, the bases the call's
  std::uint32_t line_at = 0;
  int tiles_at = 0;
  constexpr int kWindowTiles = (kWords * 32) >> kRowTileBits;
  const std::int64_t window_bytes = std::int64_t{kWords} * 32 * o.line_bytes;
  const std::uint8_t* const from = lines + p.head * o.tokens * o.line_bytes;
  //: THE LANE'S WORD OF THE FOLD'S FIRST SET'S LINE, the window's; a window on is `step` words on. A line holds the call's own tokens alone, so nothing is masked here
  const std::uint32_t* line = leaf_lines<kWords>(o, kWriteSide, leaf_bits, p.head)
                              + std::int64_t{first_leaf} * kWords + lane;
  const std::int64_t step = leaf_step<kWords>(leaf_bits);
  //: THE WORD A SET IS LISTED FROM IS LOADED A SET AHEAD: loaded where it is used, the warp waits there for it
  std::uint32_t got = ops::load_global_u32_if(line + first_set * kWords, mine > 0, 0u);

#pragma unroll 1
  for (int w = 0; w < o.windows; ++w) {
    //: THE WINDOW'S HAND-OFFS STAND BEHIND ITS FIRST SET'S LISTING AND FIRST STAGE'S ISSUE: the stage lands under them -- see docs/internals/carry/carry_kernel.md#fold
    const auto hand_offs = [&]() {
      tr.phase(kSiteWindow);
#pragma unroll
      for (int k = 0; k < kChunks; ++k) {
        if (w > 0) take(tr, kTilesReadId + k, kTilesThreads);
        //: NOT BUILT: tile group `k` of the state's image, published in the readout's orientation and fenced
        hand(tr, kTilesReadyId + k, kTilesThreads);
      }
    };
    if (mine == 0) hand_offs();

#pragma unroll 1
    for (int i = 0; i < mine; ++i) {
      tr.phase(kSiteSet);
      const Listing s = listing_of<kWords>(got & part);
      lay_tokens<kWords>(s, list);
      tr.phase(kSiteListed);
      //: the set after's word: the window's last set loads the next window's first, a window's lines a step on; the call's last its own again
      const bool turn = i + 1 == mine, on = turn && w + 1 < o.windows;
      line += on ? step : 0;
      const int set_after = first_set + ((turn ? 0 : i + 1) << shared_bits);
      got = ops::load_global_u32_if(line + set_after * kWords, true, 0u);
      tr.phase(kSiteWord);
      const int set = first_set + (i << shared_bits);
      int round = 0;
      const bool any = ops::lane_ballot(kWords * round < s.total) != 0u;
      if (any) {
        //: A STAGE IS A CHUNK OF A ROUND, its copies one committed group: the chunk's bytes, and with a round's first chunk the round's value lines. The set's first stage is issued here, and each stage issues the one after it -- see docs/internals/carry/carry_kernel.md#supply
        //: A STAGE'S ORDER: its tokens' loads and its addresses, then the round's value lines' copies, then the bytes' last, under the wait -- see docs/internals/carry/carry_kernel.md#supply
        const StagePlan first = stage_plan<kDepth, kS0, kB1, kB2, kB3>(
            coef_source<kDepth, kB0, kB1, kB2, kB3, kChunkBits>(at, tiles_at, set, 0), list,
            staged);
        fetch_round<kChannels>(rows, from, line_at, o.line_bytes, list);
        stage_bytes<kDepth, kS0, kB1, kB2, kB3>(first);
        ops::stage_commit();
      }
      if (i == 0) hand_offs();
      if (any) {
#pragma unroll 1
        do {
          const std::uint16_t* const tokens = list + kWords * round;
          const ops::SmemAddr to =
              rows + static_cast<ops::SmemAddr>((round & (kHeld - 1)) * kRoundBytes<kChannels>);
          //: whether the set has a round after this one, a value every lane holds: a set's last round issues nothing
          const bool more = ops::lane_ballot(kWords * (round + 1) < s.total) != 0u;
#pragma unroll
          for (int stage = 0; stage < kStages; ++stage) {
            tr.phase(kSiteFetch);
            //: the stage's slot and the next's: a stage's turn where a round has several, a round's where it has one
            const int turn = (kStages > 1 ? stage : round) & (kHeld - 1);
            const ops::SmemAddr here = staged + static_cast<ops::SmemAddr>(turn * kStageMost);
            const ops::SmemAddr there =
                staged + static_cast<ops::SmemAddr>((turn ^ 1) * kStageMost);
            if (stage + 1 < kStages) {
              stage_bytes<kDepth, kS0, kB1, kB2, kB3>(stage_plan<kDepth, kS0, kB1, kB2, kB3>(
                  coef_source<kDepth, kB0, kB1, kB2, kB3, kChunkBits>(at, tiles_at, set,
                                                                      (stage + 1) * kStageChunks),
                  tokens, there));
              ops::stage_commit();
              tr.phase(kSiteIssued);
              ops::stage_wait<kHeld - 1>();
            } else if (more) {
              const StagePlan next = stage_plan<kDepth, kS0, kB1, kB2, kB3>(
                  coef_source<kDepth, kB0, kB1, kB2, kB3, kChunkBits>(at, tiles_at, set, 0),
                  tokens + kWords, there);
              fetch_round<kChannels>(rows
                                         + static_cast<ops::SmemAddr>(((round + 1) & (kHeld - 1))
                                                                      * kRoundBytes<kChannels>),
                                     from, line_at, o.line_bytes, tokens + kWords);
              stage_bytes<kDepth, kS0, kB1, kB2, kB3>(next);
              ops::stage_commit();
              tr.phase(kSiteIssued);
              ops::stage_wait<kHeld - 1>();
            } else {
              //: a set's last round issues nothing: its own stage is the only one in flight
              tr.phase(kSiteIssued);
              ops::stage_wait<0>();
            }
            tr.phase(kSiteCoefs);
#pragma unroll
            for (int held = 0; held < kStageChunks; ++held) {
              const int chunk = stage * kStageChunks + held;
              expand_round<kDepth, kS0, kB0, kB1, kB2, kB3>(
                  coef_source<kDepth, kB0, kB1, kB2, kB3, kChunkBits>(at, tiles_at, set, chunk),
                  tokens, here, (held << kB0) >> kRowBlockBits, coefs);
              tr.coefs(fold, set, w, round, chunk, tokens, coefs, kCoefRoundBytes<Shape>);
              //: NOT BUILT: the fold's meet and the chunk's steps
            }
          }
          tr.phase(kSiteLanded);
          tr.round(fold, set, w, round, tokens, to);
          ++round;
        } while (ops::lane_ballot(kWords * round < s.total) != 0u);
      }
    }
    take(tr, kFoldId + fold, kGroup);
    //: NOT BUILT: the groups' steps, behind their fence, committed and waited for
    take(tr, kFoldId + fold, kGroup);
    line_at += static_cast<std::uint32_t>(window_bytes);
    tiles_at += kWindowTiles;
  }
  ops::stage_wait<0>();
}

//: A FOLD WARPGROUP: ONE SWITCH A CALL on its fold's class, to the windows compiled for the call's levels and its step's shape -- see docs/internals/carry/carry_kernel.md#fold
template <class Bt, class Tr>
__device__ __forceinline__ void fold_role(const Operands& o, const plan::PrefillCarve& c,
                                          const Place& p, Tr& tr) {
  const int fold = static_cast<int>(threadIdx.x) / kGroup;
  //: A SET'S WORD AND ITS DIGITS, their parts formed outside the loop over sets: a level's three once a fold -- see docs/internals/carry/carry_kernel.md#supply
  int base[kLevels], shift[kLevels], mask[kLevels];
  fold_levels(c, p, fold, base, shift, mask);
  const FoldPositions positions = fold_positions(o, c, p, fold, shift, mask);
  //: THE FOLD'S SETS ARE CONSECUTIVE LEAF GROUPS of the write side, from its levels' first groups' on -- see docs/internals/common/plan/leaf_groups.md#order
  const plan::LeafGroups leaves = plan::leaf_groups(c, false, 0);
  const int first_leaf = plan::leaf_group_of(leaves, base);
  //: a head's value lines are addressed by an unsigned 32-bit offset from the call's base, a level's rows by a signed one
  std::int64_t rows_most = 0;
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const std::int64_t rows = l < c.levels ? o.tokens << o.width_bits[l] : 0;
    rows_most = rows > rows_most ? rows : rows_most;
  }
  if (o.tokens * o.line_bytes >= (std::int64_t{1} << 32) || rows_most >= (std::int64_t{1} << 31)) {
    __trap();
  }
  uniform_switch<kFoldClassCount>(positions.fold_class, [&](auto member) {
    fold_windows<Bt, Tr, decltype(member)::value>(o, c, p, positions, leaves.bits, first_leaf, tr);
  });
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
    //: THE ACCUMULATORS' REGISTERS ARE NOT YET SET ASIDE: the fold's code does not fit what they leave -- see docs/internals/carry/carry_kernel.md#registers
    static_assert(FOLD - kStateRegisters >= 24, "a fold thread's count leaves its code registers");
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
