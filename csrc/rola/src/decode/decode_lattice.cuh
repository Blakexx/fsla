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
//: host's declared capacity, which is what makes an owner's whole span a single extract.
__device__ __forceinline__ uint32_t level_slice(const DecodeParams& p, const uint32_t* mask, int l,
                                                int lo, int count) {
  return extract_bits(mask + p.mask_off[l], p.mask_words[l], lo, count);
}

__device__ __forceinline__ int ceil_log2_dev(int n) {
  int bits = 0;
  while ((1 << bits) < n) ++bits;
  return bits;
}

//: THE OWNER LISTS: per list and level, the ascending coordinates at which that list's
//: side has amplitude. `n_o` is the realized count, `bits` its power-of-two padding and
//: `shift` the padded field's position in the rank, so a rank decomposes by shifts and a
//: padded-past-the-end index is skipped rather than divided around. `units` is the whole
//: enumeration -- the rank space times the unit radix inside an owner.
//: One block scan per (list, level); `lat_g[l] <= kDecodeThreads` follows from the
//: level-width capacity. The caller's barrier before this must have published both masks.
template <int D>
__device__ __forceinline__ void build_owner_lists(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, int32_t* olist,
                                                  int32_t* n_o, int32_t* bits, int32_t* shift,
                                                  int32_t* units,
                                                  LatticeScan::TempStorage& scan_temp) {
  const int tid = threadIdx.x;
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
      units[list] = empty ? 0 : (1 << (acc + p.lat_outer_bits));
    }
    __syncthreads();
  }
}

//: ONE UNIT of the walk: the `(owner, r_0 .. r_{lat_p - 1})` pair a rank names -- an owner
//: together with its OUTER run offsets. Under the mixed-radix intra-box order the
//: remaining levels' runs are the CONTIGUOUS tail of the owner-local index, so a unit's
//: leaves are `1 << lat_unit_bits` consecutive lattice indices at an aligned base: at the
//: flagship spans `(8, 16)` that is exactly ONE 16-leaf atom, at full occupancy, with the
//: innermost level's 16 digits as the unit's slots.
//:
//: What the unit carries is therefore ASYMMETRIC BY CONSTRUCTION, and that is the whole
//: saving over a walk whose unit cut every level: an OUTER level contributes ONE FIXED
//: DIGIT (`dbase[l]`, a single amplitude and a single mask bit, hoisted clean out of the
//: slot loop), a MID level one digit per mid index, and only the INNERMOST level a SLICE
//: (`cr`/`cw`, one 32-bit extract) that the slot loop iterates by `__ffs`.
//: `valid` is false where the padded rank space overshoots a level's realized count.
//: docs/internals/decode/decode_lattice.md#the-unit
template <int D>
struct Unit {
  int owner;  // O, the owner's mixed-radix index over lat_g
  int outer;  // the unit's index inside the owner: r_0 .. r_{lat_p - 1}, by shifts
  bool valid;
  bool live_r;  // every OUTER level carries read amplitude at its fixed digit
  bool live_w;
  int dbase[D];  // level l's staging offset: its fixed digit (l < lat_p) or its run base
  uint32_t cr;   // level D-1's whole run, as one slice, per side
  uint32_t cw;
};

//: THE UNIT INDEX, decomposed by shifts alone. The high bits are the OWNER RANK, one
//: padded field per level; the low `lat_outer_bits` are the outer run offsets in the SAME
//: mixed radix the owner-local index uses, so `outer << lat_unit_bits` is the unit's base
//: inside its owner without a multiply. A level whose digit is dead on a side kills that
//: side for the whole unit, which is the pruning the product structure buys:
//: docs/internals/decode/decode_lattice.md#the-unit
template <int D>
__device__ __forceinline__ Unit<D> resolve_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                const uint32_t* mask_w, const int32_t* list,
                                                const int32_t* n_o, const int32_t* bits,
                                                const int32_t* shift, int unit) {
  Unit<D> u;
  u.owner = 0;
  u.outer = unit & ((1 << p.lat_outer_bits) - 1);
  u.valid = true;
  u.live_r = true;
  u.live_w = true;
  u.cr = 0u;
  u.cw = 0u;
  const int rank = unit >> p.lat_outer_bits;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    int i = (rank >> shift[l]) & ((1 << bits[l]) - 1);
    if (i >= n_o[l]) {
      u.valid = false;
      i = 0;
    }
    const int o = list[p.olist_off[l] + i];
    u.owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    if (l < p.lat_p) {
      const int digit = run_base + ((u.outer >> p.lat_oshift[l]) & (p.lat_s[l] - 1));
      u.dbase[l] = p.level_amp_offset[l] + digit;
      u.live_r = u.live_r && level_slice(p, mask_r, l, digit, 1) != 0u;
      u.live_w = u.live_w && level_slice(p, mask_w, l, digit, 1) != 0u;
    } else {
      u.dbase[l] = p.level_amp_offset[l] + run_base;
      if (l == D - 1) {
        u.cr = level_slice(p, mask_r, l, run_base, p.lat_s[l]);
        u.cw = level_slice(p, mask_w, l, run_base, p.lat_s[l]);
      }
    }
  }
  return u;
}

}  // namespace decode
}  // namespace rola
