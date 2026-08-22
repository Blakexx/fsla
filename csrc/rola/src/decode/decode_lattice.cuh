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
// ONE SIDE AT A TIME: every object here takes ONE side's digit mask and produces that
// side's owner list, and a unit resolved against it carries that side's amplitude alone.
// A union over the sides would enumerate `prod_l (R_l u W_l)` -- the DENSE product
// wherever every level has a dense side, which is what the split-level declaration makes
// true at every level.
//
// The lattice's formulas, the capacities and the cost accounting:
// docs/internals/decode/decode_lattice.md

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {

using LatticeScan = cub::BlockScan<int, kDecodeThreads>;

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

//: ONE SIDE'S OWNER LIST: per level, the ascending coordinates at which that side has
//: amplitude. `n_o` is the realized count, `bits` its power-of-two padding and `shift` the
//: padded field's position in the rank, so a rank decomposes by shifts and a
//: padded-past-the-end index is skipped rather than divided around. `units` is the whole
//: enumeration -- the rank space times the unit radix inside an owner -- and it is the
//: SIDE'S OWN PRODUCT `prod_l |live_l|`, never a union's.
//: One block scan per level; `lat_g[l] <= kDecodeThreads` follows from the level-width
//: capacity. The caller's barrier before this must have published the mask.
template <int D>
__device__ __forceinline__ void build_owner_list(const DecodeParams& p, const uint32_t* mask,
                                                 int32_t* olist, int32_t* n_o, int32_t* bits,
                                                 int32_t* shift, int32_t* units,
                                                 LatticeScan::TempStorage& scan_temp) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int span = p.lat_s[l];
    const bool live = tid < p.lat_g[l] && level_slice(p, mask, l, tid * span, span) != 0u;
    int excl = 0;
    int total = 0;
    LatticeScan(scan_temp).ExclusiveSum(live ? 1 : 0, excl, total);
    if (live) olist[p.olist_off[l] + excl] = tid;
    if (tid == 0) n_o[l] = total;
    __syncthreads();
  }
  if (tid == 0) {
    int acc = 0;
#pragma unroll
    for (int l = D - 1; l >= 0; --l) {
      shift[l] = acc;
      acc += ceil_log2_dev(n_o[l]);
      bits[l] = acc - shift[l];
    }
    bool empty = false;
#pragma unroll
    for (int l = 0; l < D; ++l) empty = empty || n_o[l] == 0;
    *units = empty ? 0 : (1 << (acc + p.lat_outer_bits));
  }
  __syncthreads();
}

//: ONE DIGIT'S MEMBERSHIP, as ONE load and a shift. `level_slice` reads two words and
//: funnel-shifts them because an owner's SPAN can straddle a word boundary; a single digit
//: never can, and the unit asks this question `2 * (D - 1)` times.
__device__ __forceinline__ bool mask_bit(const DecodeParams& p, const uint32_t* mask, int l,
                                         int bit) {
  return ((mask[p.mask_off[l] + (bit >> 5)] >> (static_cast<uint32_t>(bit) & 31u)) & 1u) != 0u;
}

//: ONE UNIT of the walk: the `(owner, r_0 .. r_{D-2})` pair a rank names -- an owner
//: together with EVERY run offset but the innermost. Under the mixed-radix intra-box order
//: the innermost level's run is the CONTIGUOUS tail of the owner-local index, so a unit's
//: leaves are `s_{D-1}` consecutive lattice indices at an aligned base: at the flagship
//: spans `(8, 16)` that is EXACTLY ONE 16-leaf atom, at full occupancy, with the innermost
//: level's 16 digits as the unit's slots.
//:
//: What the unit carries is therefore ASYMMETRIC BY CONSTRUCTION, and that asymmetry IS
//: the unit's point: every level but the innermost contributes ONE FIXED DIGIT, so its
//: amplitude and its membership are resolved ONCE HERE and the slot loop below costs one
//: innermost amplitude per leaf rather than `D` of them. Only the innermost level
//: contributes a SLICE (`c`, one 32-bit extract) that the slot loop iterates by `__ffs`.
//: `valid` is false where the padded rank space overshoots a level's realized count.
//: docs/internals/decode/decode_lattice.md#the-unit
template <int D>
struct Unit {
  int base;  // the unit's first leaf, as a LATTICE index
  bool valid;
  bool live;  // every level but the innermost carries this side's amplitude at its digit
  int dlast;  // level D-1's staging offset: its run base
  float a;    // the fixed levels' amplitude product, and the decay rate's
  float rate;
  uint32_t c;  // level D-1's whole run, as one slice
};

//: THE UNIT INDEX, decomposed by shifts alone. The high bits are the OWNER RANK, one
//: padded field per level; the low `lat_outer_bits` are the fixed run offsets in the SAME
//: mixed radix the owner-local index uses, so `outer << lat_inner_bits` is the unit's base
//: inside its owner without a multiply. A level whose digit is dead kills the whole unit,
//: which is the pruning the product structure buys.
//: THE MASK, THE AMPLITUDES AND THE LIST ARE ONE SIDE'S, always the same side: the rank
//: space this index runs over is the one `build_owner_list` built from that mask.
//: `DECAY` is a template arm so the dial product is not merely multiplied by one.
//: docs/internals/decode/decode_lattice.md#the-unit
template <int D, bool DECAY>
__device__ __forceinline__ Unit<D> resolve_unit(const DecodeParams& p, const uint32_t* mask,
                                                const float* sa, const float* sd,
                                                const int32_t* list, const int32_t* n_o,
                                                const int32_t* bits, const int32_t* shift,
                                                int unit) {
  Unit<D> u;
  int owner = 0;
  u.valid = true;
  u.live = true;
  u.a = 1.0f;
  u.rate = 1.0f;
  const int outer = unit & ((1 << p.lat_outer_bits) - 1);
  const int rank = unit >> p.lat_outer_bits;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    int i = (rank >> shift[l]) & ((1 << bits[l]) - 1);
    if (i >= n_o[l]) {
      u.valid = false;
      i = 0;
    }
    const int o = list[p.olist_off[l] + i];
    owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    if (l == D - 1) {
      u.dlast = p.level_amp_offset[l] + run_base;
      u.c = level_slice(p, mask, l, run_base, p.lat_s[l]);
    } else {
      const int digit = run_base + ((outer >> p.lat_oshift[l]) & (p.lat_s[l] - 1));
      const int off = p.level_amp_offset[l] + digit;
      u.live = u.live && mask_bit(p, mask, l, digit);
      u.a *= sa[off];
      if (DECAY) u.rate *= sd[off];
    }
  }
  u.base = (owner << p.lat_local_bits) + (outer << p.lat_inner_bits);
  return u;
}

}  // namespace decode
}  // namespace rola
