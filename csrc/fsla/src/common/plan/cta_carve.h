// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREFILL'S CARVE (docs/internals/common/plan/cta_carve.md): from a call's level widths and each side's densities alone, each level's size from its density and the groups raised from the sizes, how the levels share the leaves one CTA owns,
// the folds' split and the write and read sides' sets. Pure, constexpr, total, integers and fixed arrays alone: nothing in it is beyond a kernel's reach.
#pragma once

#include <cstdint>

//: what a kernel calls of the carve is the device's too; a host tool compiles the same text
//: a loop over the levels is unrolled in a kernel, so that no array of the carve is indexed at run time there; a loop over the doublings stays a loop (`FSLA_CARVE_TRIPS`), so that a kernel holds one trip's values and not nine -- see docs/internals/common/plan/cta_carve.md#register-shaped
#if defined(__CUDACC__)
#define FSLA_CARVE_FN __host__ __device__
#define FSLA_CARVE_UNROLL _Pragma("unroll")
#define FSLA_CARVE_TRIPS _Pragma("unroll 1")
#else
#define FSLA_CARVE_FN
#define FSLA_CARVE_UNROLL
#define FSLA_CARVE_TRIPS
#endif

namespace fsla::plan {

inline constexpr int kCarveMaxLevels = 4;

//: a level is at most 256 digits wide: 8 bits
inline constexpr int kCarveMaxWidthBits = 8;

//: a CTA owns at most 512 leaves: 9 doublings from one digit a level
inline constexpr int kCtaLeafBits = 9;
inline constexpr int kCtaLeaves = 1 << kCtaLeafBits;

//: a side's density, in 256ths of a level's digits, at which it is live on every digit of the level, and the density at least
inline constexpr int kWholeDensity = 256;

//: A LEVEL-SIDE'S DENSITY AS THE CARVE TAKES IT, in 256ths: its byte `b` is `b + 1`, and a level-side with no mask (`masked`, a bit a level) is whole whatever its byte -- the mask's absence is the dense flag
FSLA_CARVE_FN constexpr int carve_density(int masked, int level, int byte) {
  return (masked >> level & 1) != 0 ? byte + 1 : kWholeDensity;
}

inline constexpr int kLeastDensity = 1;

//: the allowed group sizes are every power of two of digits from 1 to the widest level's 256, a size its log2 (those no larger than a level's width are allowed on it, so a level's own size always is)
inline constexpr int kGroupSizes = kCarveMaxWidthBits + 1;

//: `value` held within `lo` .. `hi`
FSLA_CARVE_FN constexpr int carve_clamp(int value, int lo, int hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

//: THE GROUP OF A DENSITY (docs/internals/common/plan/prefill_launch.md#groups), one rule for both sides: the log2 of the allowed group size nearest to `density * width / 256` digits,
//: a tie to the smaller size, compared exactly: the distance of a size `a` is `|a * 256 - density * width|`
FSLA_CARVE_FN constexpr int prefill_group_bits(int density, int width_bits) {
  //: the live digits in 256ths, at most 2^16; a size `2^b` is nearer than `2^(b + 1)` up to their midpoint, `3 * 2^(b + 7)` 256ths, the tie the smaller's: the group is the midpoints passed
  const int live = density << width_bits;
  int bits = 0;
  FSLA_CARVE_UNROLL
  for (int b = 0; b < kCarveMaxWidthBits; ++b) {
    bits += b < width_bits && live > (3 << (b + 7)) ? 1 : 0;
  }
  return bits;
}

//: A SIDE'S LOOP ORDER over `levels` levels (docs/internals/common/plan/cta_carve.md#loop-order): `order[0]` the outermost, the side's sparsest level, to the densest innermost; equal densities keep the lower level outside
FSLA_CARVE_FN constexpr void carve_loop_order(const int* density, int levels, int* order) {
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) order[l] = l;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    //: a level's place: the levels sparser than it, and the lower levels as sparse
    int place = 0;
    FSLA_CARVE_UNROLL
    for (int m = 0; m < kCarveMaxLevels; ++m) {
      const bool sparser = density[m] < density[l] || (density[m] == density[l] && m < l);
      place += m < levels && sparser ? 1 : 0;
    }
    FSLA_CARVE_UNROLL
    for (int p = 0; p < kCarveMaxLevels; ++p) {
      if (l < levels && p == place) order[p] = l;
    }
  }
}

//: FIXED by the owner's ruling ("a minimum leaf group size, such as 16"): the least leaves of a set, as log2; 16 is the narrowest tensor step the tree's kernels run, on any card -- see docs/internals/common/plan/prefill_launch.md#least
inline constexpr int kLeastSetBits = 4;

//: A SIDE'S GROUPS RAISED until a set holds 16 leaves, where its box holds as many: the group of the side's densest level that has room in the box is doubled, and again -- see docs/internals/common/plan/prefill_launch.md#least
FSLA_CARVE_FN constexpr void carve_least_sets(const int* order, int levels, const int* box_bits,
                                              int* group_bits) {
  FSLA_CARVE_TRIPS
  for (int step = 0; step < kLeastSetBits; ++step) {
    int set = 0, grow = -1;
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      set += group_bits[l] < box_bits[l] ? group_bits[l] : box_bits[l];
    }
    //: the densest level with room is the last such in the side's order
    FSLA_CARVE_UNROLL
    for (int i = 0; i < kCarveMaxLevels; ++i) {
      FSLA_CARVE_UNROLL
      for (int l = 0; l < kCarveMaxLevels; ++l) {
        const int at = order[i], group = group_bits[l], box = box_bits[l];
        grow = i < levels && at == l && group < box ? l : grow;
      }
    }
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      group_bits[l] += set < kLeastSetBits && l == grow ? 1 : 0;
    }
  }
}

struct CtaCarve {
  int levels;
  int carve_bits[kCarveMaxLevels];
};

//: THE CTA'S CARVE of `levels` levels (docs/internals/common/plan/cta_carve.md): the carve bits of each, from the widths' bits and each side's sizes a level (the group its density gives, as log2).
//: EACH DOUBLING GOES TO THE LEVEL THAT SAVES THE MOST FETCHES: a side's LEFT is the product of size over share on the levels where its size is still the larger, and a level's saving half of each side's left that still needs it; then the smaller share, a level the read side needs, the lower level
FSLA_CARVE_FN constexpr CtaCarve cta_carve(const int* width_bits, const int* write_bits,
                                           const int* read_bits, int levels) {
  CtaCarve a{levels, {}};
  //: a side's shortfall, the log2 of its left: the doublings its sizes are above the shares, kept as the shares grow
  int write_short = 0, read_short = 0;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    write_short += l < levels ? write_bits[l] : 0;
    read_short += l < levels ? read_bits[l] : 0;
  }
  FSLA_CARVE_TRIPS
  for (int step = 0; step < kCtaLeafBits; ++step) {
    const bool tied = write_short == read_short, write_more = write_short > read_short;
    int best = -1, best_key = -1, best_writes = 0, best_reads = 0;
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      const int share = a.carve_bits[l];
      const bool open = l < levels && share + 1 <= width_bits[l];
      const int writes = write_bits[l] > share ? 1 : 0, reads = read_bits[l] > share ? 1 : 0;
      //: the savings compared without their products: the side with more left counts first and the other second; sides with as much left count alike
      const int more = write_more ? writes : reads, less = write_more ? reads : writes;
      const int first = tied ? writes + reads : more;
      const int second = tied ? 0 : less;
      const int spare = kCarveMaxWidthBits - share;
      const int key = open ? (first << 6) | (second << 5) | (spare << 1) | reads : -1;
      const bool takes = key > best_key;
      best = takes ? l : best;
      best_key = takes ? key : best_key;
      best_writes = takes ? writes : best_writes;
      best_reads = takes ? reads : best_reads;
    }
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) a.carve_bits[l] += l == best ? 1 : 0;
    write_short -= best_writes;
    read_short -= best_reads;
  }
  return a;
}

//: every level index is 0-based; -1 is none. `write_density` and `read_density` are each side's densities as clamped, `write_group_bits` and `read_group_bits` the groups a kernel reads (the sizes, raised where a set would hold fewer than 16 leaves);
//: `write_order` and `read_order` are each side's loop order over the levels, outermost first (`carve_loop_order`); `carve_bits` are the CTA's carve of each level; `split_level` is the level the two fold warpgroups split along
//: and `folds` their count; `fold_bits` a fold's bits of each level; `fold_set_leaves` and `fold_sets` the leaves a token is expanded over at once on the write side and the
//: separate sets a fold holds; `read_set_leaves` and `read_sets` the same for the read side over the whole carve; `unit_bits` the bits of a level's CTA index its write group spans, and `unit_ctas` the CTAs inside one combination of write groups (they see one token schedule and are taken together)
struct PrefillCarve {
  int levels;
  int width_bits[kCarveMaxLevels];
  int write_density[kCarveMaxLevels];
  int read_density[kCarveMaxLevels];
  int write_group_bits[kCarveMaxLevels];
  int read_group_bits[kCarveMaxLevels];
  int write_order[kCarveMaxLevels];
  int read_order[kCarveMaxLevels];
  int carve_bits[kCarveMaxLevels];
  int cta_leaves;
  int split_level;
  int folds;
  int fold_bits[kCarveMaxLevels];
  int fold_leaves;
  int fold_set_leaves;
  int fold_sets;
  int read_set_leaves;
  int read_sets;
  int unit_bits[kCarveMaxLevels];
  std::int64_t unit_ctas;
};

//: THE CARVE of a call, from `levels` (0 to 4) levels' width bits (0 to 8) and each side's densities, `width_bits[0 .. levels)`, `write_density[0 .. levels)` and `read_density[0 .. levels)`;
//: TOTAL: a density outside 1..256 (256ths of a level's digits) is CLAMPED into 1..256, `levels` into 0..4 and a width into 0..8; it has no failure path
FSLA_CARVE_FN constexpr PrefillCarve prefill_carve(int levels, const int* width_bits,
                                                   const int* write_density,
                                                   const int* read_density) {
  PrefillCarve c{};
  levels = carve_clamp(levels, 0, kCarveMaxLevels);
  c.levels = levels;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    if (l < levels) {
      c.width_bits[l] = carve_clamp(width_bits[l], 0, kCarveMaxWidthBits);
      c.write_density[l] = carve_clamp(write_density[l], kLeastDensity, kWholeDensity);
      c.read_density[l] = carve_clamp(read_density[l], kLeastDensity, kWholeDensity);
      //: a level's size from its density, which a set's least may raise below
      c.write_group_bits[l] = prefill_group_bits(c.write_density[l], c.width_bits[l]);
      c.read_group_bits[l] = prefill_group_bits(c.read_density[l], c.width_bits[l]);
    }
  }
  carve_loop_order(c.write_density, levels, c.write_order);
  carve_loop_order(c.read_density, levels, c.read_order);
  //: the shares are picked from the sizes; a set's least raises a group after, and never past a share
  const CtaCarve a = cta_carve(c.width_bits, c.write_group_bits, c.read_group_bits, levels);
  c.cta_leaves = 1;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    c.carve_bits[l] = a.carve_bits[l];
    c.cta_leaves <<= a.carve_bits[l];
  }

  //: the fold warpgroups split along the first level the write side does not walk whole that the carve holds 2 digits of, else the first level the carve holds 2 digits of
  c.split_level = -1;
  FSLA_CARVE_UNROLL
  for (int l = kCarveMaxLevels - 1; l >= 0; --l) {
    if (c.carve_bits[l] >= 1) c.split_level = l;
  }
  FSLA_CARVE_UNROLL
  for (int l = kCarveMaxLevels - 1; l >= 0; --l) {
    if (c.carve_bits[l] >= 1 && c.write_group_bits[l] < c.width_bits[l]) {
      c.split_level = l;
    }
  }
  c.folds = c.split_level >= 0 ? 2 : 1;
  c.fold_leaves = c.cta_leaves / c.folds;

  c.fold_set_leaves = 1;
  c.read_set_leaves = 1;
  c.unit_ctas = 1;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    c.fold_bits[l] = c.carve_bits[l] - (l == c.split_level ? 1 : 0);
  }
  carve_least_sets(c.write_order, levels, c.fold_bits, c.write_group_bits);
  carve_least_sets(c.read_order, levels, c.carve_bits, c.read_group_bits);
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    const int write_set =
        c.write_group_bits[l] < c.fold_bits[l] ? c.write_group_bits[l] : c.fold_bits[l];
    const int read_set =
        c.read_group_bits[l] < c.carve_bits[l] ? c.read_group_bits[l] : c.carve_bits[l];
    c.fold_set_leaves <<= write_set;
    c.read_set_leaves <<= read_set;
    const int over = c.write_group_bits[l] - c.carve_bits[l];
    c.unit_bits[l] = over > 0 ? over : 0;
    c.unit_ctas <<= c.unit_bits[l];
  }
  c.fold_sets = c.fold_leaves / c.fold_set_leaves;
  c.read_sets = c.cta_leaves / c.read_set_leaves;
  return c;
}

//: THE CTA AT A PLACE OF THE SCHEDULE (docs/internals/common/plan/cta_carve.md#schedule): `cta[l]` its index along level `l`, of the head's `ctas_per_head` CTAs in the order they are taken --
//: the write side's groups in its loop order, the densest level's varying fastest, and inside a group its CTAs in the read side's, the read side's sparsest level varying fastest
FSLA_CARVE_FN constexpr void prefill_cta_at(const PrefillCarve& c, std::int64_t place, int* cta) {
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) cta[l] = 0;
  //: inside a group: the bits of a level its write group spans CTAs along (`unit_bits`)
  FSLA_CARVE_UNROLL
  for (int i = 0; i < kCarveMaxLevels; ++i) {
    int taken = 0;
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      const bool here = i < c.levels && c.read_order[i] == l;
      const std::int64_t low = place & ((std::int64_t{1} << c.unit_bits[l]) - 1);
      cta[l] = here ? static_cast<int>(low) : cta[l];
      taken += here ? c.unit_bits[l] : 0;
    }
    place >>= taken;
  }
  //: the groups: the bits of a level above those
  FSLA_CARVE_UNROLL
  for (int i = kCarveMaxLevels - 1; i >= 0; --i) {
    int taken = 0;
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      const bool here = i < c.levels && c.write_order[i] == l;
      const int bits = c.width_bits[l] - c.carve_bits[l] - c.unit_bits[l];
      const std::int64_t low = place & ((std::int64_t{1} << bits) - 1);
      cta[l] |= here ? static_cast<int>(low) << c.unit_bits[l] : 0;
      taken += here ? bits : 0;
    }
    place >>= taken;
  }
}

namespace carve_proof {

//: the measured cell: two levels of 256 digits, level 0 sparse for the write side and whole for the read side, level 1 the reverse
constexpr PrefillCarve cell() {
  const int width_bits[2] = {8, 8};
  const int write_density[2] = {4, 256};
  const int read_density[2] = {256, 4};
  return prefill_carve(2, width_bits, write_density, read_density);
}

constexpr int cta_along(std::int64_t place, int level) {
  int cta[kCarveMaxLevels] = {};
  prefill_cta_at(cell(), place, cta);
  return cta[level];
}

static_assert(cell().carve_bits[0] == 5 && cell().carve_bits[1] == 4,
              "the cell's carve is 32 x 16");
static_assert(cell().write_order[0] == 0 && cell().read_order[0] == 1,
              "each side's sparsest level is outermost");
static_assert(cell().unit_ctas == 16, "a write group of the cell spans 16 CTAs");
static_assert(cta_along(0, 1) == 0 && cta_along(15, 1) == 15 && cta_along(15, 0) == 0,
              "a group's 16 CTAs are taken together, along the level the read side lists");
static_assert(cta_along(16, 0) == 1 && cta_along(16, 1) == 0 && cta_along(127, 0) == 7,
              "then the next group");

}  // namespace carve_proof

}  // namespace fsla::plan
