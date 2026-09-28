// csrc/rola/src/prepass/plan.cuh -- THE TALL PLAN: the compile-time shape of the int8 carry's skeleton
// (the pre-pass is that skeleton at two columns a leaf), and its shared-memory ledger.
// See docs/internals/prepass/prepass.md#plan
#pragma once

#include "common/arch_caps.cuh"
#include "common/constants.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"
#include "common/ops.cuh"
#include "prepass/params.cuh"

namespace rola::prepass {

constexpr int round_up16(int bytes) { return (bytes + 15) / 16 * 16; }

constexpr int round_up128(int bytes) { return (bytes + 127) / 128 * 128; }

//: THE TALL PLAN for depth `D_`. -- see docs/internals/prepass/prepass.md#plan
template <int D_>
struct TallPlan {
  static constexpr int D = D_;
  static constexpr int kWarps = carry::warps_per_sm;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kLeaves = carry::tall_leaves_per_sm();
  static constexpr int kLeavesPerWarp = carry::tall_leaves_per_warp();
  static constexpr int kCols = 2;  //: mass and bound
  static constexpr int kWindow = carry::kWindow;
  static constexpr int kRounds = kWindow / 32;
  static constexpr int kTile = 32;
  static constexpr int kTiles = kWindow / kTile;
  static constexpr int kBlockLeaves = 1 << carry::kBlockLeavesBits;
  static constexpr int kBlocksPerWarp = kLeavesPerWarp / kBlockLeaves;

  //: the staged words -- plan.md#ledger
  static constexpr int kWordRows =
      (1 << (carry::geom_ilog2(kLeaves) - carry::kTallDenseSpanBitsMax))
      + (1 << carry::kTallDenseSpanBitsMax);
  static constexpr int kWordsBytes = round_up16(kWordRows * kRounds * 4);

  //: the pool -- plan.md#pool
  static constexpr int kPoolTok = 64;
  static constexpr int kPoolSlots = 2;
  static constexpr int kPoolSparseBytes = 1 << carry::kTallDenseSpanBitsMax;
  static constexpr int kPoolDenseBytes = 16;
  static constexpr int kPoolScalarBytes = 16;
  static constexpr int kPoolRowBytes = kPoolSparseBytes + kPoolDenseBytes + kPoolScalarBytes;
  static constexpr int kPoolSlotBytes = round_up128(kPoolTok * kPoolRowBytes);
  static constexpr int kPoolBytes = kPoolSlots * kPoolSlotBytes;
  static constexpr int kChunks = kWindow / kPoolTok;

  //: the read ring -- plan.md#ring
  static constexpr int kRingSlots = 2;
  static constexpr int kRingRowBytes = kBlockLeaves + 16;
  static constexpr int kRingSlotBytes = round_up16(kTile * kRingRowBytes);
  static constexpr int kRingBytes = kWarps * kRingSlots * kRingSlotBytes;

  //: the transfer -- plan.md#transfer
  static constexpr int kXferValueBytes = kLeaves * kCols * 2;
  static constexpr int kXferScaleBytes = kLeaves * kCols * 4;
  static constexpr int kXferBytes = round_up128(kXferValueBytes + kXferScaleBytes);

  //: the panel, a row a warp -- plan.md#panel
  static constexpr int kRange = 128;
  static constexpr int kRanges = kWindow / kRange;
  static constexpr int kPanelRowBytes = kRange * kCols * 4;
  static constexpr int kPanelBytes = round_up16(kWarps * kPanelRowBytes);

  //: the barriers and the ledger -- plan.md#ledger
  static constexpr int kPoolBars = 2 * kPoolSlots;
  static constexpr int kRingBars = kWarps * kRingSlots;
  static constexpr int kFrameBars = 2 * 3;
  static constexpr int kBarBytes = round_up16((kPoolBars + kRingBars + kFrameBars) * 8);
  static constexpr int kLedgerBytes = kWarps * kPpPhases * 8;
  static constexpr int kCountBytes = round_up16(kWarps * (kLeavesPerWarp / 8 + kBlocksPerWarp) * 4);

  static constexpr int kWordsOffset = 0;
  static constexpr int kPoolOffset = round_up128(kWordsOffset + 2 * kWordsBytes);
  static constexpr int kRingOffset = kPoolOffset + kPoolBytes;
  static constexpr int kXferOffset = round_up128(kRingOffset + kRingBytes);
  static constexpr int kPanelOffset = kXferOffset + kXferBytes;
  static constexpr int kBarOffset = kPanelOffset + kPanelBytes;
  static constexpr int kLedgerOffset = kBarOffset + kBarBytes;
  static constexpr int kCountOffset = kLedgerOffset + kLedgerBytes;
  static constexpr int kSmemBytes = round_up128(kCountOffset + 2 * kCountBytes);

  static_assert(kWarps == 8 && kThreads == 256, "the tall plan is eight warps");
  static_assert(kLeaves == 1024 && kLeavesPerWarp == 128,
                "the tall owner set: 1024 leaves, 128 a warp");
  static_assert(kWindow % kPoolTok == 0 && kPoolTok % kTile == 0,
                "a chunk is whole tiles of a window");
  static_assert(kWindow % kRange == 0, "a window is whole panel ranges");
  static_assert(kBlocksPerWarp == 4, "a read warp holds four K blocks");
  static_assert(denseref::design::smem_fits((size_t)kSmemBytes),
                "the tall plan's shared-memory ledger exceeds this architecture's per-CTA maximum");
};

}  // namespace rola::prepass
