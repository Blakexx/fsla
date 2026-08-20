#pragma once

// THE T=1 FACTOR TABLES -- the per-level objects the step's lattice walk is enumerated
// from, and nothing else.
//
// Three facts a reader must not miss:
//   1. NOTHING HERE IS SIZED IN `N`. One bit per DIGIT (`sum_l B_l` bits per side) and
//      one entry per live OWNER COORDINATE (`sum_l g_l` at most) is the whole apparatus;
//      the leaf space is never represented.
//   2. Membership is the EXACT-NONZERO test on the amplitude, never a threshold -- the
//      same defining expression the schedule's support bitmap uses.
//   3. The owner list is a UNION list carrying both sides' flags, because the walk prunes
//      on `read-alive OR write-alive` and stops at the first level where neither holds.
//
// The lattice's formulas, the capacities and the cost accounting:
// docs/internals/decode/decode_lattice.md

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {

using LatticeScan = cub::BlockScan<int, kDecodeThreads>;

//: An owner-list entry: the coordinate in the low 16 bits, its two side flags above.
constexpr int kOwnerFlagShift = 16;
constexpr int kOwnerCoordMask = 0xffff;
constexpr int kSideRead = 1;
constexpr int kSideWrite = 2;

//: One side's DIGIT MASK, one bit per digit of every level, packed level by level. One
//: thread owns one destination word and reads the amplitudes it covers, so the pass is
//: race-free without atomics and costs `sum_l ceil(B_l / 32)` words.
template <int D>
__device__ __forceinline__ void build_digit_mask(const DecodeParams& p, const float* amp,
                                                 uint32_t* mask) {
  for (int idx = threadIdx.x; idx < p.mask_total; idx += kDecodeThreads) {
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (idx >= p.mask_off[q]) l = q;
    }
    const int base = (idx - p.mask_off[l]) << 5;
    const int width = p.level_width[l];
    const float* row = amp + p.level_amp_offset[l];
    uint32_t v = 0u;
    const int take = (width - base) < 32 ? (width - base) : 32;
    for (int b = 0; b < take; ++b) {
      if (row[base + b] != 0.0f) v |= 1u << b;
    }
    mask[idx] = v;
  }
}

//: Level `l`'s digits `[lo, lo + count)` as one word. `count <= kDecodeMaxSpan` is the
//: host's declared capacity, which is what makes an owner span and a sub-box span each a
//: single extract.
__device__ __forceinline__ uint32_t level_slice(const DecodeParams& p, const uint32_t* mask, int l,
                                                int lo, int count) {
  return extract_bits(mask + p.mask_off[l], p.mask_words[l], lo, count);
}

//: THE UNION OWNER LISTS: per level, the ascending coordinates at which either side has
//: amplitude, each carrying its two flags. `n_o[l]` is the level's count and the product
//: of those counts is the walk's OWNER RANK SPACE -- the object the grid partitions.
//: One block scan per level; `lat_g[l] <= kDecodeThreads` follows from the level-width
//: capacity. The caller's barrier before this must have published both masks.
template <int D>
__device__ __forceinline__ void build_owner_lists(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, int32_t* olist,
                                                  int32_t* n_o,
                                                  LatticeScan::TempStorage& scan_temp) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int span = p.lat_s[l];
    int flag = 0;
    if (tid < p.lat_g[l]) {
      const int lo = tid * span;
      flag = (level_slice(p, mask_r, l, lo, span) != 0u ? kSideRead : 0)
             | (level_slice(p, mask_w, l, lo, span) != 0u ? kSideWrite : 0);
    }
    int excl = 0;
    int total = 0;
    LatticeScan(scan_temp).ExclusiveSum(flag != 0 ? 1 : 0, excl, total);
    if (flag != 0) olist[p.olist_off[l] + excl] = (flag << kOwnerFlagShift) | tid;
    if (tid == 0) n_o[l] = total;
    __syncthreads();
  }
}

//: ONE UNIT of the walk: the (owner, sub-box) pair a rank names, with both sides'
//: intra-sub-box digit slices and the staging offset each level's digits are read at.
//: Every array is indexed by a COMPILE-TIME level, which is what keeps the whole object
//: in registers. docs/internals/decode/decode_lattice.md#the-unit
template <int D>
struct SubBox {
  int owner;  // O, the owner's mixed-radix index
  int cap;    // U, the sub-box's index inside the owner
  bool live_r;
  bool live_w;
  int dbase[D];
  uint32_t cr[D];
  uint32_t cw[D];
};

//: THE UNIT INDEX, decomposed. The high bits are the OWNER RANK, which decomposes over
//: the union lists; the low `log2(m)` are the capacity radix. A level whose slice is
//: empty on a side kills that side for the whole sub-box, which is the pruning the
//: product structure buys: docs/internals/decode/decode_lattice.md#the-unit
template <int D>
__device__ __forceinline__ SubBox<D> resolve_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, const int32_t* olist,
                                                  const int* n_o, const int* nsuf, int unit) {
  SubBox<D> sb;
  sb.owner = 0;
  sb.cap = unit & (p.lat_m - 1);
  sb.live_r = true;
  sb.live_w = true;
  const int rank = unit >> p.lat_msb;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int o = olist[p.olist_off[l] + (rank / nsuf[l]) % n_o[l]] & kOwnerCoordMask;
    sb.owner += o * p.lat_gsuf[l];
    const int lo = o * p.lat_s[l] + ((sb.cap / p.lat_msuf[l]) % p.lat_m_l[l]) * p.lat_k;
    sb.dbase[l] = p.level_amp_offset[l] + lo;
    sb.cr[l] = level_slice(p, mask_r, l, lo, p.lat_k);
    sb.cw[l] = level_slice(p, mask_w, l, lo, p.lat_k);
    sb.live_r = sb.live_r && sb.cr[l] != 0u;
    sb.live_w = sb.live_w && sb.cw[l] != 0u;
  }
  return sb;
}

}  // namespace decode
}  // namespace rola
