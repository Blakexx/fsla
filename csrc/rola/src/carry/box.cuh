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
enum class Fill : int { kCoverFactors = 0, kPrologueValues = 1, kSegment = 2 };
constexpr int kFillPoints = 3;

//: THE BOX'S ROUTING FOOTPRINT, in amplitude columns, and the catenated run's base for
//: level `l`. -- see docs/internals/carry/carry_kernel.md#factor-rows
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

//: THE SLOT STRIDES, each an access pattern in one congruence: the transpose's over the
//: raw rows, the expansion's over the column block.
//: -- see docs/internals/carry/smem_ledger.md#factor-rows
constexpr int raw_stride(int span_tot) { return span_tot + ((4 - span_tot % 32) + 32) % 32; }

constexpr int tok_stride(int seg) { return seg + ((8 - seg % 16) + 16) % 16; }

//: THE ASYNCHRONOUS QUANTUM a level's run is staged in: the widest that divides the run,
//: its base and the stride.
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
  static constexpr int kColBytes = kSpanTotal * kTokStride * 2;

  //: the shared page image, the value tile, and the ring slots the fill stages into.
  //: -- see docs/internals/carry/smem_ledger.md
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
  static constexpr int kColOffsetInRegion = kVRingBytes;

  static_assert(kStatePages >= 1, "the region holds at least one page image");
  static_assert(kStatePages * kPageImageBytes <= kRegionBytes, "the page images fit the region");
  static_assert(kVRingBytes + kColBytes <= kRegionBytes,
                "PHASE 2's value ring and column block fit the region they alias into");
  static_assert(kSpanTotal % 2 == 0 && owner_span(D, kBC, 0) % 2 == 0,
                "the transpose moves the factor row in column PAIRS, and the gain is folded "
                "into the outermost level's own columns");
  static_assert(kSeg % kUnitTokens == 0, "a segment is a whole number of fill units");

  //: the persistent tenants beside the region. -- see docs/internals/carry/smem_ledger.md
  static constexpr int kEdges = 3;
  static constexpr int kEdgeBytes = kEdges * (int)sizeof(denseref::ops::EdgeState);
  static constexpr int kPageSlotBytes = kAtoms * (int)sizeof(int);
  static constexpr int kStreamEdges = 2;
  static constexpr int kStreamEdgeBytes = kStreamEdges * (int)sizeof(denseref::ops::EdgeState);

  static constexpr int kRegionOffset = 0;
  static constexpr int kEdgeOffset = kRegionOffset + kRegionBytes;
  static constexpr int kPageSlotOffset = kEdgeOffset + kEdgeBytes;
  static constexpr int kStreamEdgeOffset = kPageSlotOffset + kPageSlotBytes;
  static constexpr int kClaimOffset = kStreamEdgeOffset + kStreamEdgeBytes;
  static constexpr int kClaimBytes = kFillPoints * 4;
  static constexpr int kGainOffset = (kClaimOffset + kClaimBytes + 15) / 16 * 16;
  static constexpr int kRawOffset = kGainOffset + kFactorSlots * kGainSlotBytes;
  static constexpr int kSmemBytes = kRawOffset + kFactorSlots * kRawSlotBytes;

  static_assert(kEdgeOffset % 8 == 0 && kStreamEdgeOffset % 8 == 0,
                "an mbarrier is eight-byte aligned");
  static_assert(kRawOffset % 16 == 0 && kRawSlotBytes % 8 == 0,
                "a raw factor slot's base carries the widest quantum its runs are staged in");

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

//: THE FRAME'S EDGES, named. -- see docs/internals/carry/carry_kernel.md#edges
enum class Edge : int { kWindowLine = 0, kOutLeg = 1, kReturnLeg = 2 };

//: THE STREAM'S OWN EDGES, which are not the frame's.
//: -- see docs/internals/carry/carry_kernel.md#stream-edges
enum class StreamEdge : int { kFilled = 0, kEmpty = 1 };

}  // namespace rola::carry
