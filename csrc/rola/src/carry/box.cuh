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

  //: the state page's shared image and the V segment's tile. -- smem_ledger.md#page-image
  static constexpr int kLdO = kCols;
  static constexpr int kPageImageBytes = kAtomLeaves * kLdO * (int)sizeof(float);
  static constexpr int kVTileBytes = kSeg * kDv * 2;

  //: THE TRANSIT REGION, and the phase-local tenants that alias into it. -- smem_ledger.md#region
  static constexpr int kLdX =
      (kCols + denseref::ops::kMmaN - 1) / denseref::ops::kMmaN * denseref::ops::kMmaN;
  static constexpr int kRegionBytes = kBC * kLdX * 2;
  static constexpr int kStatePages =
      kAtoms < kRegionBytes / kPageImageBytes ? kAtoms : kRegionBytes / kPageImageBytes;
  static constexpr int kStateRounds = (kAtoms + kStatePages - 1) / kStatePages;
  static constexpr int kVRingBytes = kWarps * kSegmentSlots * kVTileBytes;

  static_assert(kStatePages >= 1, "the region holds at least one page image");
  static_assert(kStatePages * kPageImageBytes <= kRegionBytes, "the page images fit the region");
  static_assert(kVRingBytes <= kRegionBytes, "the V ring fits the region");

  //: the persistent tenants beside the region: the frame's edges and the box's page slots.
  static constexpr int kEdges = 3;
  static constexpr int kEdgeBytes = kEdges * (int)sizeof(denseref::ops::EdgeState);
  static constexpr int kPageSlotBytes = kAtoms * (int)sizeof(int);

  static constexpr int kRegionOffset = 0;
  static constexpr int kEdgeOffset = kRegionOffset + kRegionBytes;
  static constexpr int kPageSlotOffset = kEdgeOffset + kEdgeBytes;
  static constexpr int kSmemBytes = kPageSlotOffset + kPageSlotBytes;

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

}  // namespace rola::carry
