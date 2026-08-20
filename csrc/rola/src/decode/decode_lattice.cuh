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
//   3. THE RANK SPACE IS A BIT FIELD, not a mixed radix over the realized counts: each
//      level's live count is padded up to a power of two so a unit decomposes by shifts.
//      A device integer division is tens of cycles and the enumeration runs once per
//      candidate atom per CTA, which is where it would be paid.
//
// TWO LISTS: the walk enumerates the UNION (a leaf either side reaches), the candidate
// atoms enumerate the WRITE side alone (an atom the step deposits into). One list would
// make the second enumeration the first one's size, and a page-table probe per entry is
// what that costs.
//
// The lattice's formulas, the capacities and the cost accounting:
// docs/internals/decode/decode_lattice.md

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {

using LatticeScan = cub::BlockScan<int, kDecodeThreads>;

//: The two enumerations, as the index of one owner-list plane.
constexpr int kListUnion = 0;
constexpr int kListWrite = 1;
constexpr int kLists = 2;

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

__device__ __forceinline__ int ceil_log2_dev(int n) {
  int bits = 0;
  while ((1 << bits) < n) ++bits;
  return bits;
}

//: The write side's own unit: no read slices and no staging offsets, because the
//: candidate pass asks one question -- which atoms does this step deposit into?
template <int D>
struct SubBoxWrite {
  int owner;
  int cap;
  bool valid;
  bool live;
  uint32_t cw[D];
};

//: THE OWNER LISTS: per list and level, the ascending coordinates at which that list's
//: side has amplitude. `n_o` is the realized count, `bits` its power-of-two padding and
//: `shift` the padded field's position in the rank, so a rank decomposes by shifts and a
//: padded-past-the-end index is skipped rather than divided around. `units` is the whole
//: enumeration -- the rank space times the capacity radix.
//: One block scan per (list, level); `lat_g[l] <= kDecodeThreads` follows from the
//: level-width capacity. The caller's barrier before this must have published both masks.
template <int D>
__device__ __forceinline__ void build_owner_lists(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, int32_t* olist,
                                                  int32_t* n_o, int32_t* bits, int32_t* shift,
                                                  int32_t* units, int32_t* wulist, int32_t* nu,
                                                  int32_t* nubits, int32_t* nushift,
                                                  LatticeScan::TempStorage& scan_temp) {
  const int tid = threadIdx.x;
  const uint32_t kmask = (p.lat_k >= 32) ? 0xffffffffu : ((1u << p.lat_k) - 1u);
#pragma unroll
  for (int list = 0; list < kLists; ++list) {
#pragma unroll
    for (int l = 0; l < D; ++l) {
      const int span = p.lat_s[l];
      bool live = false;
      if (tid < p.lat_g[l]) {
        const int lo = tid * span;
        const bool w = level_slice(p, mask_w, l, lo, span) != 0u;
        live = (list == kListWrite) ? w : (w || level_slice(p, mask_r, l, lo, span) != 0u);
      }
      int excl = 0;
      int total = 0;
      LatticeScan(scan_temp).ExclusiveSum(live ? 1 : 0, excl, total);
      if (live) olist[list * p.olist_total + p.olist_off[l] + excl] = tid;
      if (tid == 0) n_o[list * kMaxLevels + l] = total;
      //: The write list's owners carry their own live sub-boxes, compacted the same way:
      //: a dead sub-box is a page-table probe the candidate pass must not spend.
      if (list == kListWrite) {
        if (tid == 0) nubits[l] = 0;
        __syncthreads();
        if (live) {
          const uint32_t ws = level_slice(p, mask_w, l, tid * span, span);
          int32_t* row = wulist + p.wulist_off[l] + excl * p.lat_m_l[l];
          int count = 0;
          for (int u = 0; u < p.lat_m_l[l]; ++u) {
            if (((ws >> (u * p.lat_k)) & kmask) != 0u) row[count++] = u;
          }
          nu[p.olist_off[l] + excl] = count;
          atomicMax(&nubits[l], count);
        }
      }
      __syncthreads();
      if (list == kListWrite && tid == 0) nubits[l] = ceil_log2_dev(nubits[l]);
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0;
#pragma unroll
      for (int l = D - 1; l >= 0; --l) {
        shift[list * kMaxLevels + l] = acc;
        acc += ceil_log2_dev(n_o[list * kMaxLevels + l]);
        bits[list * kMaxLevels + l] = acc - shift[list * kMaxLevels + l];
      }
      bool empty = false;
#pragma unroll
      for (int l = 0; l < D; ++l) empty = empty || n_o[list * kMaxLevels + l] == 0;
      if (list == kListWrite) {
        int nacc = 0;
#pragma unroll
        for (int l = D - 1; l >= 0; --l) {
          nushift[l] = nacc;
          nacc += nubits[l];
        }
        units[list] = empty ? 0 : (1 << (acc + nacc));
      } else {
        units[list] = empty ? 0 : (1 << (acc + p.lat_msb));
      }
    }
    __syncthreads();
  }
}

//: ONE CANDIDATE ATOM'S SUB-BOX, resolved from the write side's two compacted tiers.
//: `valid` is false where either padded field overshoots its realized count -- the price
//: of decomposing by shifts, and cheaper than the division it replaces.
template <int D>
__device__ __forceinline__ SubBoxWrite<D> resolve_write_unit(
    const DecodeParams& p, const uint32_t* mask_w, const int32_t* wlist, const int32_t* wulist,
    const int32_t* nu, const int32_t* n_o, const int32_t* bits, const int32_t* shift,
    const int32_t* nubits, const int32_t* nushift, int rank, int capidx) {
  SubBoxWrite<D> sb;
  sb.owner = 0;
  sb.cap = 0;
  sb.valid = true;
  sb.live = true;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int i = (rank >> shift[l]) & ((1 << bits[l]) - 1);
    if (i >= n_o[l]) {
      sb.valid = false;
      return sb;
    }
    const int j = (capidx >> nushift[l]) & ((1 << nubits[l]) - 1);
    if (j >= nu[p.olist_off[l] + i]) {
      sb.valid = false;
      return sb;
    }
    const int o = wlist[p.olist_off[l] + i];
    const int u = wulist[p.wulist_off[l] + i * p.lat_m_l[l] + j];
    sb.owner += o * p.lat_gsuf[l];
    sb.cap += u << p.lat_ushift[l];
    const int lo = o * p.lat_s[l] + (u << p.lat_kshift);
    sb.cw[l] = level_slice(p, mask_w, l, lo, p.lat_k);
    sb.live = sb.live && sb.cw[l] != 0u;
  }
  return sb;
}

//: ONE UNIT of the walk: the (owner, sub-box) pair a rank names, with both sides'
//: intra-sub-box digit slices and the staging offset each level's digits are read at.
//: Every array is indexed by a COMPILE-TIME level, which is what keeps the whole object
//: in registers. `valid` is false where the padded rank space overshoots a level's
//: realized count. docs/internals/decode/decode_lattice.md#the-unit
template <int D>
struct SubBox {
  int owner;  // O, the owner's mixed-radix index
  int cap;    // U, the sub-box's index inside the owner
  bool valid;
  bool live_r;
  bool live_w;
  int dbase[D];
  uint32_t cr[D];
  uint32_t cw[D];
};

//: THE UNIT INDEX, decomposed by shifts alone. The high bits are the OWNER RANK, one
//: padded field per level; the low `log2(m)` are the capacity radix. A level whose slice
//: is empty on a side kills that side for the whole sub-box, which is the pruning the
//: product structure buys: docs/internals/decode/decode_lattice.md#the-unit
template <int D>
__device__ __forceinline__ SubBox<D> resolve_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, const int32_t* list,
                                                  const int32_t* n_o, const int32_t* bits,
                                                  const int32_t* shift, int unit) {
  SubBox<D> sb;
  sb.owner = 0;
  sb.cap = unit & (p.lat_m - 1);
  sb.valid = true;
  sb.live_r = true;
  sb.live_w = true;
  const int rank = unit >> p.lat_msb;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    int i = (rank >> shift[l]) & ((1 << bits[l]) - 1);
    if (i >= n_o[l]) {
      sb.valid = false;
      i = 0;
    }
    const int o = list[p.olist_off[l] + i];
    sb.owner += o * p.lat_gsuf[l];
    const int u = (sb.cap >> p.lat_ushift[l]) & (p.lat_m_l[l] - 1);
    const int lo = o * p.lat_s[l] + (u << p.lat_kshift);
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
