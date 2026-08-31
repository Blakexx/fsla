// csrc/rola/src/carry/box.cuh -- the CTA box's compile-time plan: the register shape the
// arm fixes, the accumulator's fragment geometry, and the shared-memory ledger's offsets.
// See docs/internals/carry/carry_kernel.md#box-plan and docs/internals/carry/smem_ledger.md
#pragma once

#include "common/arch_caps.cuh"
#include "common/constants.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/state_page.cuh"

namespace rola::carry {

//: THE FILL's unit, in tokens, and its production points -- one cursor each.
//: -- see docs/internals/carry/carry_kernel.md#fill
constexpr int kUnitTokens = 4;
enum class Fill : int {
  kCoverFactors = 0,
  kPrologueValues = 1,
  kSegment = 2,
  kReadCover = 3,
  kReadSegment = 4
};
constexpr int kFillPoints = 5;

//: THE BOX'S ROUTING FOOTPRINT, in amplitude columns, and one level's base in it.
//: -- see docs/internals/carry/carry_kernel.md#factor-rows
constexpr int span_total(int D, int bc) {
  int total = 0;
  for (int l = 0; l < D; ++l) total += owner_span(D, bc, l);
  return total;
}

constexpr int span_base(int D, int bc, int l) {
  int base = 0;
  for (int i = 0; i < l; ++i) base += owner_span(D, bc, i);
  return base;
}

//: THE SLOT STRIDES, each an access pattern in one congruence.
//: -- see docs/internals/carry/smem_ledger.md#factor-rows
constexpr int raw_stride(int span_tot) { return span_tot + ((4 - span_tot % 32) + 32) % 32; }

constexpr int tok_stride(int seg) { return seg + ((8 - seg % 16) + 16) % 16; }

//: THE J-SLICE OF A LEVEL: which bits of a warp's tile index its digit depends on.
//: -- see docs/internals/carry/carry_kernel.md#expansion
constexpr int slice_shift(int shift) { return shift > 3 ? shift - 3 : 0; }

constexpr int slice_bits(int shift, int bits) {
  const int hi = shift + bits - 3;
  return hi <= 0 ? 0 : hi - slice_shift(shift);
}

//: THE ASYNCHRONOUS QUANTUM a run is staged in: the widest that divides run, base, stride.
constexpr int run_quantum(int span_h, int base_h, int stride_h) {
  const int b = span_h * 2, o = base_h * 2, r = stride_h * 2;
  if (b % 16 == 0 && o % 16 == 0 && r % 16 == 0) return 16;
  if (b % 8 == 0 && o % 8 == 0 && r % 8 == 0) return 8;
  return 4;
}

//: THE BOX, THE ACCUMULATOR AND THE LEDGER for one arm.
template <int D_, int DV_, int WARPS_>
struct BoxPlan {
  static constexpr int D = D_;
  static constexpr int kDv = DV_;
  static constexpr int kWarps = WARPS_;
  static constexpr int kThreads = kWarps * 32;

  //: the degenerate upper levels: a cluster is one CTA and a CTA is one SM.
  static constexpr int kClusterCtas = rola::carry::kClusterCtas;

  //: THE OWNERSHIP UNIT: the CTA's box, carved per side among its warps, one each.
  static constexpr int kLeavesPerWarp = leaves_per_warp(kDv);
  static constexpr int kBC = kLeavesPerWarp * kWarps;
  static constexpr int kStreams = kWarps;
  static constexpr int kAtoms = kBC / kAtomLeaves;

  //: THE ACCUMULATOR, C[m = channel][n = leaf]. -- kernel.md#accumulator
  static constexpr int kCols = kDv + 1;
  static constexpr int kMCh = (kCols + denseref::ops::kMmaM - 1) / denseref::ops::kMmaM;
  static constexpr int kNL = kLeavesPerWarp / denseref::ops::kMmaN;
  static constexpr int kMassTile = kDv / denseref::ops::kMmaM;
  static constexpr int kMassRow = kDv - kMassTile * denseref::ops::kMmaM;
  //: -- see docs/internals/carry/carry_kernel.md#readout
  static constexpr int kDw = kLeavesPerWarp / denseref::ops::kMmaK;
  static constexpr int kTokTiles = kSegmentTokens / denseref::ops::kMmaN;
  static constexpr int kAccRegs = kMCh * kNL * 4;
  static constexpr int kStateElems = kLeavesPerWarp * kCols;

  static constexpr int kSeg = kSegmentTokens;
  static constexpr int kSegsPerWindow = kWindow / kSeg;
  static constexpr int kSegmentSlots = segment_slots(denseref::arch::kCaps);

  static constexpr int kFactorSlots = kSegmentSlots + 1;
  static constexpr int kSpanTotal = span_total(D, kBC);
  static constexpr int kRawStride = raw_stride(kSpanTotal);
  static constexpr int kTokStride = tok_stride(kSeg);
  static constexpr int kRawSlotBytes = kSeg * kRawStride * 2;
  static constexpr int kGainSlotBytes = kSeg * 2;
  static constexpr int kColSlots = 2;
  static constexpr int kColBytes = kColSlots * kSpanTotal * kTokStride * 2;
  //: -- see docs/internals/carry/smem_ledger.md#read-ring
  static constexpr int kReadSlots = 2;
  static constexpr int kReadRawBytes = kReadSlots * kRawSlotBytes;

  //: the shared tenants' shapes. -- see docs/internals/carry/smem_ledger.md
  static constexpr int kLdO = kCols + (4 - kCols % 32 + 32) % 32;
  static constexpr int kPageImageBytes = kAtomLeaves * kLdO * (int)sizeof(float);
  static constexpr int kVTileBytes = kSeg * kDv * 2;

  //: THE TRANSIT REGION, and the phase-local tenants that alias into it. -- smem_ledger.md#region
  static constexpr int kLdX =
      (kCols + denseref::ops::kMmaN - 1) / denseref::ops::kMmaN * denseref::ops::kMmaN;
  static constexpr int kRegionBytes = kBC * kLdX * 2;
  static constexpr int kStatePages =
      kAtoms < kRegionBytes / kPageImageBytes ? kAtoms : kRegionBytes / kPageImageBytes;
  static constexpr int kStateRounds = (kAtoms + kStatePages - 1) / kStatePages;
  static constexpr int kVRingBytes = kSegmentSlots * kVTileBytes;
  //: -- see docs/internals/carry/smem_ledger.md#out-block
  static constexpr int kOutSlots = 2;
  static constexpr int kOutSlotBytes = kSeg * kLdO * (int)sizeof(float);
  static constexpr int kOutBytes = kOutSlots * kOutSlotBytes;

  static_assert(kStatePages >= 1, "the region holds at least one page image");
  static_assert(kStatePages * kPageImageBytes <= kRegionBytes, "the page images fit the region");
  static_assert(kOutBytes <= kRegionBytes,
                "PHASE 1's output blocks fit the region they alias into");
  static_assert(kDw >= 1 && kLeavesPerWarp % denseref::ops::kMmaK == 0,
                "a warp's leaves are a whole number of readout k-tiles, so the fold's n-tiles "
                "pair into them");
  static_assert(span_base(D, kBC, D - 1) % 2 == 0,
                "the readout reads its innermost level's leaf PAIR as one 32-bit access, so "
                "that level's run begins on a 32-bit boundary of the staged row");
  static_assert(kMassTile == kMCh - 1,
                "the mass opens the last channel m-tile, so the tiles below it are whole "
                "value tiles and the transit's ldmatrix reaches every one of them");
  static_assert(kSpanTotal % 2 == 0 && owner_span(D, kBC, 0) % 2 == 0,
                "the transpose moves the factor row in column PAIRS, and the gain is folded "
                "into the outermost level's own columns");
  static_assert(kSeg % kUnitTokens == 0, "a segment is a whole number of fill units");

  //: the tenants beside the region. -- see docs/internals/carry/smem_ledger.md#region
  static constexpr int kPageSlotBytes = kAtoms * (int)sizeof(int);

  static constexpr int kRegionOffset = 0;
  static constexpr int kVTileOffset = kRegionOffset + kRegionBytes;
  static constexpr int kColOffset = kVTileOffset + kVRingBytes;
  static constexpr int kPageSlotOffset = kColOffset + kColBytes;
  static constexpr int kClaimOffset = kPageSlotOffset + kPageSlotBytes;
  static constexpr int kClaimBytes = kFillPoints * 4;
  static constexpr int kGainOffset = (kClaimOffset + kClaimBytes + 15) / 16 * 16;
  static constexpr int kRawOffset = kGainOffset + kFactorSlots * kGainSlotBytes;
  static constexpr int kReadRawOffset = kRawOffset + kFactorSlots * kRawSlotBytes;
  static constexpr int kSmemBytes = kReadRawOffset + kReadRawBytes;

  static_assert(kVTileOffset % 16 == 0 && kColOffset % 16 == 0,
                "the value ring and the column block are addressed in 16-byte quanta");
  static_assert(kRawOffset % 16 == 0 && kReadRawOffset % 16 == 0 && kRawSlotBytes % 8 == 0,
                "a raw factor slot's base carries the widest quantum its runs are staged in");
  static_assert(kLdX % denseref::ops::kMmaN == 0 && (kLdX * 2) % 16 == 0,
                "a transit row's base is an ldmatrix address and its channel groups are "
                "sixteen-byte quanta");

  static_assert(denseref::design::smem_fits((size_t)kSmemBytes),
                "the carry box's shared-memory ledger exceeds this architecture's per-CTA "
                "maximum");
  static_assert(kBC % kAtomLeaves == 0, "a CTA box is a whole number of pages");
  static_assert(kLeavesPerWarp % denseref::ops::kMmaN == 0,
                "a warp's leaves are a whole number of accumulator n-tiles");
  static_assert(kWindow % kSeg == 0, "the window is a whole number of segments");
  static_assert(kMassRow == 0, "the mass channel opens the accumulator's last m-tile");

  using Page = rola::state_page::Blocks<DV_>;
};

//: THE STREAM'S OWN EDGES, which are not the frame's: NAMED BARRIER ids.
//: -- see docs/internals/carry/carry_kernel.md#stream-edges
enum class StreamEdge : int { kFilled = 1, kEmpty = 2, kReadBatch = 3 };

//: THE FRAME'S EDGES, named -- barrier ids of the same kind, disjoint from the streams'.
//: -- see docs/internals/carry/carry_kernel.md#edges
enum class Edge : int { kWindowLine = 4, kOutLeg = 5, kReturnLeg = 6 };

}  // namespace rola::carry
