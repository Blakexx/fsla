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
// THE ENUMERATION IS THE UNION OF THE TWO PRODUCTS, NOT THE PRODUCT OF THE TWO UNIONS,
// and that is this header's whole subject. `prod_l (R_l u W_l)` is the DENSE product
// wherever every level has a dense side -- which the split-level declaration makes true
// at EVERY level -- while `(prod_l R_l) u (prod_l W_l)` is the set the step actually
// touches. The two are separated by a factor of up to `2^D` at the flagship.
//
// The union of two products is not a product, but it IS a DISJOINT UNION of `D + 1` of
// them, which is what makes it walkable by the same three loops:
//
//     (prod_l W_l)  u  [ (prod_l R_l) \ (prod_l W_l) ]
//        = TERM 0   ]+[ (+)_{l=0}^{D-1} TERM (l + 1)
//     TERM 0     = prod_l W_l
//     TERM l + 1 = [prod_{l' < l} (R_l' & W_l')] x (R_l & ~W_l) x [prod_{l' > l} R_l']
//
// EXHAUSTIVE: a leaf of `prod R` outside `prod W` has a SMALLEST level `l` whose digit is
// not in `W_l`, and that `l` names its term. DISJOINT: term 0's level-`l` digit is in
// `W_l` and term `l + 1`'s is not; terms `i < j` disagree at level `i - 1`. So every leaf
// either side reaches is enumerated EXACTLY ONCE -- which is what keeps UPDATE-THEN-READ
// a single visit and the deposit a partition over write owners.
//
// FOUR KINDS, ONE PER PER-LEVEL DIGIT SET the terms ask for (`W`, `R`, `R & W`,
// `R & ~W`), each with its own owner list; the term table says which kind each level of
// each term takes. The two DIGIT MASKS stay two -- a kind's bits are a boolean function
// of the read and write bits, formed where they are read.
//
// The lattice's formulas, the capacities and the cost accounting:
// docs/internals/decode/decode_lattice.md

#include <cub/block/block_scan.cuh>

#include "decode.cuh"

namespace rola {
namespace decode {

using LatticeScan = cub::BlockScan<int, kDecodeThreads>;

//: THE FOUR PER-LEVEL DIGIT SETS the terms are built from, as the index of one owner-list
//: plane each. `kKindW` is also TERM 0's kind at every level, so the write product -- the
//: set the candidate atoms and the pool claim are enumerated from -- is a term of the walk
//: and needs no second list plane.
//: THERE IS NO FIFTH, UNION KIND, and that is a MEASURED refusal rather than an omission:
//: a `R_l | W_l` list plus a fallback term that walks it whenever it enumerates fewer
//: units than the decomposition was built, gated and benchmarked, and LOST on every cell
//: (its fifth list costs a block scan per level in a prologue-bound step, and the unit
//: count it selects on does not predict the wall -- the scattered cell at `BH = 1` was
//: 21 % SLOWER on the enumeration with 1.5x FEWER units).
//: docs/internals/decode/decode_lattice.md#the-refused-fallback
constexpr int kKindW = 0;
constexpr int kKindR = 1;
constexpr int kKindRW = 2;   // R_l & W_l
constexpr int kKindRmW = 3;  // R_l & ~W_l
constexpr int kKinds = 4;

//: `D + 1` terms: the write product, then one per level.
constexpr int kMaxTerms = kMaxLevels + 1;

//: THE TERM TABLE, as arithmetic rather than as a table: term `0` is the write product,
//: and term `j >= 1` splits at level `j - 1`.
__device__ __forceinline__ int term_kind(int term, int l) {
  if (term == 0) return kKindW;
  const int split = term - 1;
  return l < split ? kKindRW : (l == split ? kKindRmW : kKindR);
}

//: A KIND'S BITS, from the two sides' bits. One expression, used on single digits and on
//: whole owner spans alike, which is what keeps the digit masks at two planes.
__device__ __forceinline__ uint32_t kind_bits(uint32_t r, uint32_t w, int kind) {
  switch (kind) {
    case kKindW:
      return w;
    case kKindR:
      return r;
    case kKindRW:
      return r & w;
    default:
      return r & ~w;
  }
}

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

//: THE OWNER LISTS: per KIND and level, the ascending coordinates at which that kind's
//: digit set has any digit. `n_o` is the realized count. Each thread reads ITS OWNER'S
//: SPAN ONCE, on both sides, and all four kinds fall out of that one pair of slices;
//: `lat_g[l] <= kDecodeThreads` follows from the level-width capacity. The caller's
//: barrier before this must have published both masks.
//:
//: TWO SCANS PER LEVEL FOR FOUR LISTS, and the packing is why the prologue did not grow
//: with the kind count: a level's live-owner count is at most `kDecodeThreads`, so two
//: counts ride in the two halves of one `int` and cannot carry into one another. THE
//: PROLOGUE IS THE STEP'S DOMINANT COST WHEREVER `BH` IS SMALL (`n_split` is largest
//: there, and every CTA pays it), so a scan avoided is worth more than the arithmetic
//: it saves: docs/internals/decode/decode.md#per-cta-prologue
template <int D>
__device__ __forceinline__ void build_owner_lists(const DecodeParams& p, const uint32_t* mask_r,
                                                  const uint32_t* mask_w, int32_t* olist,
                                                  int32_t* n_o,
                                                  LatticeScan::TempStorage& scan_temp) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int span = p.lat_s[l];
    uint32_t r = 0u;
    uint32_t w = 0u;
    if (tid < p.lat_g[l]) {
      const int lo = tid * span;
      r = level_slice(p, mask_r, l, lo, span);
      w = level_slice(p, mask_w, l, lo, span);
    }
#pragma unroll
    for (int pair = 0; pair < kKinds / 2; ++pair) {
      const int k_lo = 2 * pair;
      const int k_hi = k_lo + 1;
      const bool live_lo = tid < p.lat_g[l] && kind_bits(r, w, k_lo) != 0u;
      const bool live_hi = tid < p.lat_g[l] && kind_bits(r, w, k_hi) != 0u;
      int excl = 0;
      int total = 0;
      LatticeScan(scan_temp).ExclusiveSum((live_lo ? 1 : 0) | (live_hi ? 1 << 16 : 0), excl, total);
      if (live_lo) olist[k_lo * p.olist_total + p.olist_off[l] + (excl & 0xffff)] = tid;
      if (live_hi) olist[k_hi * p.olist_total + p.olist_off[l] + (excl >> 16)] = tid;
      if (tid == 0) {
        n_o[k_lo * kMaxLevels + l] = total & 0xffff;
        n_o[k_hi * kMaxLevels + l] = total >> 16;
      }
      __syncthreads();
    }
  }
}

//: THE TERM LAYOUT: each term's own rank space. `bits` is a level's power-of-two padding
//: and `shift` the padded field's position in that term's rank, so a rank decomposes by
//: shifts and a padded-past-the-end index is skipped rather than divided around; a term
//: whose kind is empty at any level contributes NO units at all, which is how
//: `R_l <= W_l` at every level -- the dense cell -- collapses the whole read remainder to
//: nothing and leaves the walk exactly the write product.
//: ONE THREAD PER TERM, because the terms are independent: `D + 1` threads do at most
//: `D` `ceil_log2`s each instead of one thread doing twenty in series while 255 wait.
template <int D>
__device__ __forceinline__ void build_term_layout(const DecodeParams& p, const int32_t* n_o,
                                                  int32_t* bits, int32_t* shift, int32_t* units) {
  const int t = threadIdx.x;
  if (t <= D) {
    int acc = 0;
    bool empty = false;
#pragma unroll
    for (int l = D - 1; l >= 0; --l) {
      const int count = n_o[term_kind(t, l) * kMaxLevels + l];
      shift[t * kMaxLevels + l] = acc;
      acc += ceil_log2_dev(count);
      bits[t * kMaxLevels + l] = acc - shift[t * kMaxLevels + l];
      empty = empty || count == 0;
    }
    units[t] = empty ? 0 : (1 << (acc + p.lat_outer_bits));
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
//: contributes a SLICE (`cr`/`cw`, one 32-bit extract) that the slot loop iterates by
//: `__ffs`. `valid` is false where the padded rank space overshoots a level's realized
//: count. docs/internals/decode/decode_lattice.md#the-unit
template <int D>
struct Unit {
  int owner;  // O, the owner's mixed-radix index over lat_g
  int base;   // the unit's first leaf, as a LATTICE index
  bool valid;
  bool live_r;  // every level but the innermost carries read amplitude at its fixed digit
  bool live_w;
  int dlast;  // level D-1's staging offset: its run base
  float ar;   // the fixed levels' amplitude product, per side, and the decay rate's
  float aw;
  float rate;
  uint32_t cr;  // level D-1's whole run, as one slice, per side
  uint32_t cw;
};

//: THE UNIT INDEX, decomposed by shifts alone within its TERM. THE TERM IS THE WALK'S
//: OUTER LOOP AND THAT IS A LATENCY DECISION, not a bookkeeping one: it makes the kind of
//: every level -- and therefore the list base, the count, the field widths and every
//: predicate below -- LOOP-INVARIANT, so the per-unit dependent chain is the one the union
//: walk had. Laying the terms end to end in one index space and naming the term per unit
//: was measured at `+2.3 ns` of dependent latency per unit, which on the scattered cell
//: cost more than the units the decomposition saves:
//: docs/internals/decode/decode.md#the-term-loop
//: The grid partitions EACH TERM'S index space by the modular map it always used, so the
//: deposit -- which lives in term 0 alone -- is still one owning CTA, warp and lane per row.
//: Within a term the high bits are the OWNER RANK, one padded field per level, and the low
//: `lat_outer_bits` are the fixed run offsets in the SAME mixed radix the owner-local index
//: uses, so `outer << lat_inner_bits` is the unit's base inside its owner without a
//: multiply. A level whose digit is dead in the TERM'S OWN KIND kills the unit, which is
//: the pruning the product structure buys.
//:
//: WHAT THE UNIT HANDS BACK IS ALREADY IN THE WALK'S VOCABULARY, and that is deliberate:
//: the slot loop iterates `live_w ? cw : cr` and tests `in_r`/`in_w` exactly as a union
//: walk does, because the two slices are CONDITIONED here so the term never reaches it.
//:   * TERM 0 is the write product: `cw` IS the enumerated run, every one of its leaves is
//:     in `supp(W)`, and `cr`/`live_r` answer the read question as they always did. A dead
//:     write digit above the innermost level zeroes both slices, which is the unit dying.
//:   * TERM `l + 1` is disjoint from `supp(W)` by construction and inside `supp(R)`, so it
//:     reports `live_w = false` (no leaf of it is written), `live_r = true`, and its
//:     enumerated run in `cr` (where the union walk's read slice would have been).
//: The walk therefore has NO TERM ARM, NO EXTRA SLICE REGISTER and one expression per question -- the
//: arithmetic the tight loop's register budget is measured on is unchanged.
//: docs/internals/decode/decode_lattice.md#the-unit
//: `DECAY` is a template arm so the dial product is not merely multiplied by one.
template <int D, bool DECAY>
__device__ __forceinline__ Unit<D> resolve_unit(const DecodeParams& p, const uint32_t* mask_r,
                                                const uint32_t* mask_w, const float* sa_r,
                                                const float* sa_w, const float* sd,
                                                const int32_t* olist, const int32_t* n_o,
                                                const int32_t* bits, const int32_t* shift, int term,
                                                int local) {
  Unit<D> u;
  u.owner = 0;
  u.valid = true;
  u.live_r = true;
  u.live_w = true;
  u.ar = 1.0f;
  u.aw = 1.0f;
  u.rate = 1.0f;
  bool live_kind = true;
  uint32_t c_kind = 0u;
  const int outer = local & ((1 << p.lat_outer_bits) - 1);
  const int rank = local >> p.lat_outer_bits;
#pragma unroll
  for (int l = 0; l < D; ++l) {
    const int kind = term_kind(term, l);
    int i = (rank >> shift[term * kMaxLevels + l]) & ((1 << bits[term * kMaxLevels + l]) - 1);
    if (i >= n_o[kind * kMaxLevels + l]) {
      u.valid = false;
      i = 0;
    }
    const int o = olist[kind * p.olist_total + p.olist_off[l] + i];
    u.owner += o * p.lat_gsuf[l];
    const int run_base = o * p.lat_s[l];
    if (l == D - 1) {
      u.dlast = p.level_amp_offset[l] + run_base;
      u.cr = level_slice(p, mask_r, l, run_base, p.lat_s[l]);
      u.cw = level_slice(p, mask_w, l, run_base, p.lat_s[l]);
      c_kind = kind_bits(u.cr, u.cw, kind);
    } else {
      const int digit = run_base + ((outer >> p.lat_oshift[l]) & (p.lat_s[l] - 1));
      const int off = p.level_amp_offset[l] + digit;
      const uint32_t rb = mask_bit(p, mask_r, l, digit) ? 1u : 0u;
      const uint32_t wb = mask_bit(p, mask_w, l, digit) ? 1u : 0u;
      u.live_r = u.live_r && rb != 0u;
      u.live_w = u.live_w && wb != 0u;
      live_kind = live_kind && kind_bits(rb, wb, kind) != 0u;
      u.ar *= sa_r[off];
      u.aw *= sa_w[off];
      if (DECAY) u.rate *= sd[off];
    }
  }
  //: THE CONDITIONING the walk's one expression rests on, and the whole of the term's
  //: presence in this file's contract.
  const bool alive = u.valid && live_kind;
  if (term == 0) {
    if (!alive) {
      u.cr = 0u;
      u.cw = 0u;
    }
  } else {
    u.cr = alive ? c_kind : 0u;
    u.cw = 0u;
    u.live_w = false;
    u.live_r = true;
  }
  u.base = (u.owner << p.lat_local_bits) + (outer << p.lat_inner_bits);
  return u;
}

}  // namespace decode
}  // namespace rola
