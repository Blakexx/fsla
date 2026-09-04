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

//: THE FILL's unit, in tokens. -- see docs/internals/carry/carry_kernel.md#fill
constexpr int kUnitTokens = 4;

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

//: THE FAN-IN PANEL'S TOKEN GRAIN, in MMA n-tiles: the widest that still fits one slice
//: per warp inside the region. -- see docs/internals/carry/smem_ledger.md#panel
constexpr int panel_tiles(int warps, int tiles, int ldo, int region) {
  int t = tiles;
  while (t > 1 && warps * t * denseref::ops::kMmaN * ldo * (int)sizeof(float) > region) t >>= 1;
  return t;
}

//: THE SLOT STRIDES, each an access pattern in one congruence. -- see docs/internals/carry/smem_ledger.md#factor-rows
constexpr int raw_stride(int span_tot) { return span_tot + ((4 - span_tot % 32) + 32) % 32; }

//: THE POOL'S AMPLITUDE ROW STRIDE, in halfwords: an ODD number of sixteen-byte quanta.
//: -- see docs/internals/carry/smem_ledger.md#pool
constexpr int amp_stride(int span_tot) {
  int q = (span_tot * 2 + 15) / 16;
  if (q % 2 == 0) ++q;
  return q * 8;
}

//: THE POOL'S CAPACITY: the largest power of two <= the window whose rows and lists fit. -- see docs/internals/carry/smem_ledger.md#pool
constexpr int pool_positions(int budget, int row_bytes, int list_bytes, int window) {
  int c = window;
  while (c > 32 && (c + 1) * row_bytes + list_bytes > budget) c >>= 1;
  return c;
}

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

  static constexpr int kSpanTotal = span_total(D, kBC);
  static constexpr int kRawStride = raw_stride(kSpanTotal);
  static constexpr int kRawSlotBytes = kSeg * kRawStride * 2;
  //: -- see docs/internals/carry/smem_ledger.md#read-ring
  static constexpr int kReadSlots = 2;
  static constexpr int kReadRawBytes = kReadSlots * kRawSlotBytes;

  //: THE POOL's rows and the per-warp lists. -- see docs/internals/carry/smem_ledger.md#pool
  static constexpr int kVRowBytes = kDv * 2;
  static constexpr int kAmpStride = amp_stride(kSpanTotal);
  static constexpr int kAmpRowBytes = kAmpStride * 2;

  //: the shared tenants' shapes. -- see docs/internals/carry/smem_ledger.md
  static constexpr int kLdO = kCols + (4 - kCols % 32 + 32) % 32;
  static constexpr int kPageImageBytes = kAtomLeaves * kLdO * (int)sizeof(float);

  //: THE TRANSIT REGION, and the phase-local tenants that alias into it. -- smem_ledger.md#region
  static constexpr int kLdX =
      (kCols + denseref::ops::kMmaN - 1) / denseref::ops::kMmaN * denseref::ops::kMmaN;
  static constexpr int kRegionBytes = kBC * kLdX * 2;
  static constexpr int kStatePages =
      kAtoms < kRegionBytes / kPageImageBytes ? kAtoms : kRegionBytes / kPageImageBytes;
  static constexpr int kStateRounds = (kAtoms + kStatePages - 1) / kStatePages;
  //: -- see docs/internals/carry/smem_ledger.md#panel
  static constexpr int kPanelTiles =
      panel_tiles(kWarps, kSeg / denseref::ops::kMmaN, kLdO, kRegionBytes);
  static constexpr int kPanelTokens = kPanelTiles * denseref::ops::kMmaN;
  static constexpr int kPanelRounds = kSeg / kPanelTokens;
  static constexpr int kPanelSliceBytes = kPanelTokens * kLdO * (int)sizeof(float);
  static constexpr int kPanelBytes = kWarps * kPanelSliceBytes;

  static_assert(kStatePages >= 1, "the region holds at least one page image");
  static_assert(kStatePages * kPageImageBytes <= kRegionBytes, "the page images fit the region");
  static_assert(kPanelBytes <= kRegionBytes,
                "the fan-in's per-warp panel slices fit the region they alias into");
  static_assert(kDw >= 1 && kLeavesPerWarp % denseref::ops::kMmaK == 0,
                "a warp's leaves are a whole number of readout k-tiles, so the fold's n-tiles "
                "pair into them");
  static_assert(span_base(D, kBC, D - 1) % 2 == 0,
                "the readout reads its innermost level's leaf PAIR as one 32-bit access, so "
                "that level's run begins on a 32-bit boundary of the staged row");
  static_assert(kMassTile == kMCh - 1,
                "the mass opens the last channel m-tile, so the tiles below it are whole "
                "value tiles and the transit's ldmatrix reaches every one of them");
  static_assert(owner_span(D, kBC, 0) % 2 == 0,
                "the outermost run is whole 32-bit words: the fill scales it by the gain a "
                "word at a time");
  static_assert(kSeg % kUnitTokens == 0, "a segment is a whole number of fill units");

  //: a side's streams at its finest grain: the catalog per warp, one list each.
  //: -- see docs/internals/carry/smem_ledger.md#streams
  static constexpr int kStreamGrainMin = kAtomLeaves;
  static constexpr int kStreamCatalog = kLeavesPerWarp / kStreamGrainMin;
  static constexpr int kSideStreams = kWarps * kStreamCatalog;
  //: THE GRAIN SWITCH: a side's grain is a launch parameter selecting one of these bodies.
  static constexpr int kGrainBitsMin = kAtomLeavesBits;
  static constexpr int kGrainCases = kStreamCatalog == 1 ? 1 : (kStreamCatalog == 2 ? 2 : 3);
  static_assert(kStreamCatalog <= 4, "the grain switch covers a catalog of up to four streams");
  static constexpr int kRounds = kWindow / 32;

  static constexpr int kPageSlotBytes = kAtoms * (int)sizeof(int);

  //: THE LIVE WORDS per side: a stream's word per round and its exclusive prefix.
  static constexpr int kStreamWordBytes = kSideStreams * kRounds * 4;
  static constexpr int kStreamPrefixBytes = kSideStreams * kRounds * 2;
  static constexpr int kLiveBytes = kStreamWordBytes + kStreamPrefixBytes;
  static constexpr int kFillScratchBytes = 32 * 2;

  static constexpr int kReadBlockBytes = kReadRawBytes;
  static constexpr int kClaimBytes = 4;
  //: the pool aliases the region and may outgrow it. -- see docs/internals/carry/smem_ledger.md#region
  static constexpr int kPoolBudget =
      denseref::arch::kCaps.smem_per_cta_max
      - (kReadBlockBytes + kPageSlotBytes + kClaimBytes + 2 * kLiveBytes + 64);
  static constexpr int kPoolPos =
      pool_positions(kPoolBudget, kVRowBytes + kAmpRowBytes,
                     kSideStreams* kWindow * 2 + kWarps * kFillScratchBytes, kWindow);
  static constexpr int kListEntries = kPoolPos;
  static constexpr int kListBytes = kListEntries * 2;
  static constexpr int kRegionOffset = 0;
  static constexpr int kPoolVOffset = kRegionOffset;
  static constexpr int kPoolAmpOffset = kPoolVOffset + (kPoolPos + 1) * kVRowBytes;
  static constexpr int kListOffset = kPoolAmpOffset + (kPoolPos + 1) * kAmpRowBytes;
  static constexpr int kFillScratchOffset = kListOffset + kSideStreams * kListBytes;
  static constexpr int kPoolBytes = kFillScratchOffset + kWarps * kFillScratchBytes - kRegionOffset;
  static constexpr int kReadAmpOffset = kRegionOffset;
  static constexpr int kReadListOffset = kReadAmpOffset + (kPoolPos + 1) * kAmpRowBytes;
  static constexpr int kReadScratchOffset = kReadListOffset + kSideStreams * kListBytes;
  static constexpr int kPoolTokenOffset = kReadScratchOffset + kWarps * kFillScratchBytes;
  //: THE STAGING TILE, one per warp: `[token][channel + mass]` fp32 at a `DV + 4` stride
  //: (a fragment's 64-bit pair stores two-way at most, a token's row contiguous), and a
  //: seventeenth row holding the tile's sixteen token ids. -- smem_ledger.md#pool
  static constexpr int kStageStride = kDv + 4;
  static constexpr int kStageBytes = ((kSeg + 1) * kStageStride * 4 + 15) / 16 * 16;
  static constexpr int kStageOffset = (kPoolTokenOffset + (kPoolPos + 1) * 2 + 15) / 16 * 16;
  static constexpr int kReadPoolBytes = kStageOffset + kWarps * kStageBytes - kRegionOffset;
  static_assert(kStageStride % 16 == 4 && kDv % 32 == 0, "the staging rows are DV + 4 floats");
  static constexpr int kRegionTenantBytes =
      ((kPoolBytes > kRegionBytes ? (kPoolBytes > kReadPoolBytes ? kPoolBytes : kReadPoolBytes)
                                  : (kRegionBytes > kReadPoolBytes ? kRegionBytes : kReadPoolBytes))
       + 15)
      / 16 * 16;
  static constexpr int kReadBlockOffset = kRegionOffset + kRegionTenantBytes;
  static constexpr int kReadRawOffset = kReadBlockOffset;
  static constexpr int kPageSlotOffset = kReadBlockOffset + kReadBlockBytes;
  static constexpr int kClaimOffset = kPageSlotOffset + kPageSlotBytes;
  static constexpr int kLiveOffset = kClaimOffset + kClaimBytes;
  static constexpr int kSmemBytes = (kLiveOffset + 2 * kLiveBytes + 15) / 16 * 16;

  static_assert(kPoolPos % 32 == 0 && kWindow % kPoolPos == 0,
                "the pool holds whole rounds of thirty-two positions and the window is a whole "
                "number of pools");
  static_assert(kPoolAmpOffset % 16 == 0 && kListOffset % 16 == 0 && kAmpRowBytes % 16 == 0,
                "the pool's rows are ldmatrix addresses and sixteen-byte cp.async targets");
  static_assert(kRawSlotBytes % 16 == 0,
                "a raw factor slot's bytes keep the next slot's base aligned");
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
enum class StreamEdge : int { kReadBatch = 3 };

//: THE FRAME'S EDGES, named -- barrier ids of the same kind, disjoint from the streams'.
//: -- see docs/internals/carry/carry_kernel.md#edges
enum class Edge : int { kWindowLine = 4, kOutLeg = 5, kReturnLeg = 6, kFoldRound = 7 };

}  // namespace rola::carry
