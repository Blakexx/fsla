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

//: THE POOL'S AMPLITUDE ROW STRIDE, in halfwords. -- see docs/internals/carry/smem_ledger.md#pool
constexpr int amp_stride(int span_tot) {
  int q = (span_tot * 2 + 15) / 16;
  if (q % 2 == 0) ++q;
  return q * 8;
}

//: THE POOL'S CAPACITY: the largest power of two <= the window whose rows and lists fit. -- see docs/internals/carry/smem_ledger.md#pool
constexpr int pool_positions(int budget, int row_bytes, int list_bytes_per_pos, int scratch_bytes,
                             int window) {
  int c = window;
  while (c > 32 && (c + 1) * row_bytes + c * list_bytes_per_pos + scratch_bytes > budget) c >>= 1;
  return c;
}

//: THE J-SLICE OF A LEVEL. -- see docs/internals/carry/carry_kernel.md#expansion
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
//: `Streams`: the read form's ledger. -- see docs/internals/carry/smem_ledger.md#defer
template <int D_, int DV_, int WARPS_, bool Streams = false>
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
  static constexpr int kAccRegs = kMCh * kNL * 4;
  static constexpr int kStateElems = kLeavesPerWarp * kCols;

  static constexpr int kSeg = kSegmentTokens;
  static constexpr int kSegsPerWindow = kWindow / kSeg;

  //: THE POOL's rows and the per-warp lists. -- see docs/internals/carry/smem_ledger.md#pool
  static constexpr int kVRowBytes = kDv * 2;
  static constexpr int kAmpStride = amp_stride(span_total(D, kBC));
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

  static_assert(kStatePages >= 1, "the region holds at least one page image");
  static_assert(kStatePages * kPageImageBytes <= kRegionBytes, "the page images fit the region");
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

  //: a side's streams at its finest grain. -- see docs/internals/carry/smem_ledger.md#streams
  static constexpr int kStreamGrainMin = kAtomLeaves;
  static constexpr int kStreamCatalog = kLeavesPerWarp / kStreamGrainMin;
  static constexpr int kSideStreams = kWarps * kStreamCatalog;
  //: THE GRAIN CASES: a side's grain selects a kernel symbol. -- carry_kernel.md#forms
  static constexpr int kGrainBitsMin = kAtomLeavesBits;
  static constexpr int kGrainCases = kStreamCatalog == 1 ? 1 : (kStreamCatalog == 2 ? 2 : 3);
  static_assert(kStreamCatalog <= 4, "the grain switch covers a catalog of up to four streams");
  static constexpr int kRounds = kWindow / 32;

  static constexpr int kPageSlotBytes = kAtoms * (int)sizeof(int);

  //: THE LIVE WORDS per side: a stream's word per round and its exclusive prefix.
  static constexpr int kStreamWordBytes = kSideStreams * kRounds * 4;
  static constexpr int kStreamPrefixBytes = kSideStreams * kRounds * 2;
  static constexpr int kLiveBytes = kStreamWordBytes + kStreamPrefixBytes;
  static constexpr int kLiveBlocks = 2;
  static constexpr int kKstepWordBytes = kSideStreams * kRounds * 4;
  static constexpr int kBoxRows = span_total(D, kBC);
  static constexpr int kBoxWordsBytes = kBoxRows * kRounds * 4;
  static constexpr int kFillScratchBytes = 32 * 2 * 2;

  //: THE PERSISTENT TENANTS' BYTES, which the pools' budget is the file less.
  static constexpr int kKb = kBC / denseref::ops::kMmaK;
  static constexpr int kMemoColStride = 2 + (D - 1);
  static constexpr int kMemoBytes = kKb * 32 * 4 + kKb * 4 * 4 * 4 + kKb * kMemoColStride * 2;
  static constexpr int kUniformWords = 2 * kRounds + 2 * kSideStreams;
  static constexpr int kPersistentBytes = kPageSlotBytes + kLiveBlocks * kLiveBytes
                                          + kKstepWordBytes + kBoxWordsBytes + kMemoBytes
                                          + kWindow * 2 + 2 * kUniformWords * 4 + 7 * 16;
  //: the pool aliases the region and may outgrow it. -- see docs/internals/carry/smem_ledger.md#region
  static constexpr int kPoolBudget = denseref::arch::kCaps.smem_per_cta_max - kPersistentBytes;
  //: the fold pool's cap under the stream form. -- see smem_ledger.md#defer
  static constexpr int kPoolCap = Streams ? kRegionBytes + kAtoms * 256 : kPoolBudget;
  static constexpr int kPoolPos =
      pool_positions(kPoolBudget < kPoolCap ? kPoolBudget : kPoolCap, kVRowBytes + kAmpRowBytes,
                     kSideStreams * 2, kWarps* kFillScratchBytes, kWindow);
  static constexpr int kListEntries = kPoolPos;
  static constexpr int kListBytes = kListEntries * 2;
  static constexpr int kRegionOffset = 0;
  static constexpr int kPoolVOffset = kRegionOffset;
  static constexpr int kPoolAmpOffset = kPoolVOffset + (kPoolPos + 1) * kVRowBytes;
  static constexpr int kListOffset = kPoolAmpOffset + (kPoolPos + 1) * kAmpRowBytes;
  static constexpr int kFillScratchOffset = kListOffset + kSideStreams * kListBytes;
  //: THE POOL'S GAIN, one bf16 a position, from the window's gain table (below).
  static constexpr int kPoolGainOffset = kFillScratchOffset + kWarps * kFillScratchBytes;
  static constexpr int kPoolBytes = kPoolGainOffset + (kPoolPos + 1) * 2 - kRegionOffset;

  //: THE READ POOL AND THE STAGING TILES, behind the region. -- see smem_ledger.md#pool
  //: THE STAGING TILE: one segment of one slice at a time. -- smem_ledger.md#pool
  static constexpr int kStageStride = 32 + 8;
  static constexpr int kStageBytes = ((kSeg + 1) * kStageStride * 4 + 15) / 16 * 16;

  //: THE SLICES a warp owns of a batch: two where the region, the read pool and the staging
  //: tiles fit the budget at two, else one. -- see docs/internals/carry/carry_kernel.md#readout
  static constexpr int read_pool_bytes(int slices) {
    const int pos = slices * kWarps * kSeg;
    const int tok = kRegionBytes + (pos + 1) * kAmpRowBytes + kWarps * kFillScratchBytes;
    return (tok + (pos + 1) * 6 + 3 + 15) / 16 * 16 + kSideStreams * pos * 2 + kWarps * kStageBytes;
  }

  static constexpr int kSlices = read_pool_bytes(2) <= kPoolBudget ? 2 : 1;
  static constexpr int kReadPos = kSlices * kWarps * kSeg;
  static constexpr int kReadAmpOffset = kRegionOffset + kRegionBytes;
  static constexpr int kReadScratchOffset = kReadAmpOffset + (kReadPos + 1) * kAmpRowBytes;
  static constexpr int kPoolTokenOffset = kReadScratchOffset + kWarps * kFillScratchBytes;
  static constexpr int kPoolGroupsOffset = (kPoolTokenOffset + (kReadPos + 1) * 2 + 3) / 4 * 4;
  static constexpr int kReadListOffset = (kPoolGroupsOffset + (kReadPos + 1) * 4 + 15) / 16 * 16;
  static constexpr int kReadListBytes = kReadPos * 2;
  static constexpr int kStageOffset = kReadListOffset + kSideStreams * kReadListBytes;
  static constexpr int kReadPoolBytes = kStageOffset + kWarps * kStageBytes - kRegionOffset;
  //: THE DEFERRED TILES, one a warp. -- see docs/internals/carry/smem_ledger.md#defer
  static constexpr int kDeferStride = kDv + 8;
  static constexpr int kDeferTileBytes = ((kSeg * kDeferStride * 4 + kSeg * 2 + 15) / 16) * 16;
  static constexpr int kDeferOffset =
      ((kPoolBytes > kRegionBytes ? kPoolBytes : kRegionBytes) + 15) / 16 * 16;
  static_assert(!Streams
                    || kDeferOffset + kWarps * kDeferTileBytes
                           <= kStageOffset + kWarps * kStageBytes,
                "the deferred tiles fit behind the fold pool inside the readout's tenants");
  static_assert(kReadPoolBytes == read_pool_bytes(kSlices), "the ledger and its estimate agree");
  static_assert(kStageStride % 32 == 8 && kSlices * 2 * kSeg * 2 <= kStageStride * 4
                    && 32 + kSlices <= kStageStride,
                "a staging row is one segment, a mass column a slice and the pad that makes the "
                "fragment pair's stores bank free; the id sets fit a row");
  static_assert(kReadPos % 32 == 0 && kWindow % kReadPos == 0,
                "the read batch holds whole rounds of thirty-two positions and the window is a "
                "whole number of batches");
  static_assert(kReadPoolBytes <= kPoolBudget,
                "the region, the read pool and the staging tiles fit the pool budget");
  static constexpr int kRegionTenantBytes =
      ((kPoolBytes > kRegionBytes ? (kPoolBytes > kReadPoolBytes ? kPoolBytes : kReadPoolBytes)
                                  : (kRegionBytes > kReadPoolBytes ? kRegionBytes : kReadPoolBytes))
       + 15)
      / 16 * 16;
  static constexpr int kPageSlotOffset = kRegionOffset + kRegionTenantBytes;
  static constexpr int kLiveOffset = kPageSlotOffset + kPageSlotBytes;
  //: THE READ MEMO, by k-step of the box. -- see docs/internals/carry/smem_ledger.md#memo
  static constexpr int kKstepWordOffset = (kLiveOffset + kLiveBlocks * kLiveBytes + 15) / 16 * 16;
  static constexpr int kBoxWordsOffset = (kKstepWordOffset + kKstepWordBytes + 15) / 16 * 16;
  static constexpr int kMemoRowsOffset = (kBoxWordsOffset + kBoxWordsBytes + 15) / 16 * 16;
  static constexpr int kMemoMassOffset = kMemoRowsOffset + kKb * 32 * 4;
  static constexpr int kMemoColOffset = kMemoMassOffset + kKb * 4 * 4 * 4;
  static_assert(kMemoColOffset + kKb * kMemoColStride * 2 - kMemoRowsOffset == kMemoBytes,
                "the memo's tables are its bytes");
  //: THE WINDOW'S GAIN TABLE, persistent. -- see docs/internals/carry/smem_ledger.md#pool
  static constexpr int kGainTableOffset = (kMemoRowsOffset + kMemoBytes + 15) / 16 * 16;
  //: THE UNIFORM TABLES, per side. -- see docs/internals/carry/carry_kernel.md#uniform
  static constexpr int kUniformOffset = (kGainTableOffset + kWindow * 2 + 15) / 16 * 16;
  static constexpr int kSmemBytes = (kUniformOffset + 2 * kUniformWords * 4 + 15) / 16 * 16;

  static_assert(kPoolPos % 32 == 0 && kWindow % kPoolPos == 0,
                "the pool holds whole rounds of thirty-two positions and the window is a whole "
                "number of pools");
  static_assert(kPoolAmpOffset % 16 == 0 && kListOffset % 16 == 0 && kAmpRowBytes % 16 == 0,
                "the pool's rows are ldmatrix addresses and sixteen-byte cp.async targets");
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

//: A WARP'S OWN EDGE, a named barrier. -- see docs/internals/carry/carry_kernel.md#edges
constexpr int kWarpEdge = 8;

__device__ __forceinline__ void warp_edge(int warp) {
  denseref::ops::rendezvous_group(kWarpEdge + warp, 32);
}

}  // namespace rola::carry
