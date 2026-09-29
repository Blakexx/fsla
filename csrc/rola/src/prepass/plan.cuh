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
  //: DIGIT-MAJOR: a digit's 32 token bytes are one row (a word a token quad), the write-sparse digits at a
  //: 36-byte stride (the two lists of an m-tile half, eight digits apart, land eight banks apart) plus a
  //: zero row a padding list reads, the write-dense digits at a 48-byte stride (eight consecutive digits
  //: cover the banks); then the factors [column][token], formed by the fill from the token's gain, value
  //: scale and write maxima.
  static constexpr int kPoolTok = 32;
  static constexpr int kPoolSlots = 2;
  static constexpr int kPoolSpanBytes = 1 << carry::kTallDenseSpanBitsMax;
  static constexpr int kPoolSparseStride = kPoolTok + 4;
  static constexpr int kPoolDenseStride = kPoolTok + 16;
  static constexpr int kPoolZeroRow = kPoolSpanBytes;
  static constexpr int kPoolDenseOff = round_up16((kPoolSpanBytes + 1) * kPoolSparseStride);
  static constexpr int kPoolFactorOff = kPoolDenseOff + kPoolSpanBytes * kPoolDenseStride;
  static constexpr int kPoolSlotBytes = round_up128(kPoolFactorOff + kCols * kPoolTok * 4);
  static constexpr int kPoolBytes = kPoolSlots * kPoolSlotBytes;
  static constexpr int kChunks = kWindow / kPoolTok;

  //: the fold: sixteen tiles a warp (a list x an n-tile of 8 write-dense digits) -- plan.md#fold
  static constexpr int kTilesPerWarp = kLeavesPerWarp / 8;

  //: the fp32 state, a warp its 128 leaves x 2 columns -- plan.md#state
  static constexpr int kStateBytes = kLeavesPerWarp * kCols * 4;

  //: the read ring -- plan.md#ring
  static constexpr int kRingSlots = 2;
  static constexpr int kRingSparseBytes =
      16;  //: the region's read-sparse row, 8 or 16 digits at depth two
  static constexpr int kRingRowBytes = kBlockLeaves + kRingSparseBytes + 8;
  static constexpr int kRingSparseOff = kBlockLeaves;
  static constexpr int kRingScaleOff = kBlockLeaves + kRingSparseBytes;
  static constexpr int kRingSlotBytes = round_up16(kTile * kRingRowBytes);
  static constexpr int kRingBytes = kWarps * kRingSlots * kRingSlotBytes;

  //: the transfer: s16 digits and a power-of-two scale's s8 exponent a leaf a column -- plan.md#transfer
  static constexpr int kXferValueBytes = kLeaves * kCols * 2;
  static constexpr int kXferScaleBytes = kLeaves * kCols * 1;
  static constexpr int kXferBytes = round_up128(kXferValueBytes + kXferScaleBytes);

  //: the panel, a row a warp -- plan.md#panel
  static constexpr int kRange = 64;
  static constexpr int kRanges = kWindow / kRange;
  static constexpr int kPanelRowBytes = kRange * kCols * 4;
  static constexpr int kPanelBufs = 2;
  static constexpr int kPanelBytes = round_up16(kPanelBufs * kWarps * kPanelRowBytes);

  //: the barriers and the ledger -- plan.md#ledger
  static constexpr int kPoolBars = 2 * kPoolSlots;
  static constexpr int kRingBars = kWarps * kRingSlots;
  static constexpr int kFrameBars = 2 * 4;
  static constexpr int kBarBytes = round_up16((kPoolBars + kRingBars + kFrameBars) * 8);
  static constexpr int kLedgerBytes = kWarps * kPpPhases * 8;
  static constexpr int kCountBytes = round_up16(kWarps * (kLeavesPerWarp / 8 + kBlocksPerWarp) * 4);

  static constexpr int kWordsOffset = 0;
  static constexpr int kPoolOffset = round_up128(kWordsOffset + 2 * kWordsBytes);
  static constexpr int kRingOffset = kPoolOffset + kPoolBytes;
  static constexpr int kXferOffset = round_up128(kRingOffset + kRingBytes);
  static constexpr int kPanelOffset = kXferOffset + kXferBytes;
  static constexpr int kStateOffset = round_up128(kPanelOffset + kPanelBytes);
  static constexpr int kBarOffset = kStateOffset + kWarps * kStateBytes;
  static constexpr int kLedgerOffset = kBarOffset + kBarBytes;
  static constexpr int kCountOffset = kLedgerOffset + kLedgerBytes;
  static constexpr int kSmemBytes = round_up128(kCountOffset + 2 * kCountBytes);

  static_assert(kWarps == 8 && kThreads == 256, "the tall plan is eight warps");
  static_assert(kLeaves == 1024 && kLeavesPerWarp == 128,
                "the tall owner set: 1024 leaves, 128 a warp");
  static_assert(kPoolTok == 32 && kChunks == kRounds, "a chunk is one liveness word a list");
  static_assert(kWindow % kPoolTok == 0, "a chunk divides a window");
  static_assert(kWindow % kRange == 0, "a window is whole panel ranges");
  static_assert(kBlocksPerWarp == 4, "a read warp holds four K blocks");
  static_assert(denseref::design::smem_fits((size_t)kSmemBytes),
                "the tall plan's shared-memory ledger exceeds this architecture's per-CTA maximum");
};

}  // namespace rola::prepass
