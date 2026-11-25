// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREFILL'S CARVE (docs/internals/common/plan/cta_carve.md): from a call's level widths and each side's densities alone, the groups of the densities, how the levels share the leaves one CTA owns,
// the folds' split and the write and read sides' sets. Pure, constexpr, total, integers and fixed arrays alone: nothing in it is beyond a kernel's reach.
#pragma once

#include <cstdint>

//: what a kernel calls of the carve is the device's too; a host tool compiles the same text
#if defined(__CUDACC__)
#define FSLA_CARVE_FN __host__ __device__
#else
#define FSLA_CARVE_FN
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
inline constexpr int kLeastDensity = 1;

//: the allowed group sizes in digits, as log2: 1, 4, 16, 32, 64, 128, 256 (those no larger than a level's width are allowed on it)
inline constexpr int kGroupSizeBits[] = {0, 2, 4, 5, 6, 7, 8};
inline constexpr int kGroupSizes = 7;

//: `kGroupSizeBits[i]` as arithmetic: a kernel evaluates it where it cannot read the table
FSLA_CARVE_FN constexpr int group_size_bits(int i) { return i < 3 ? 2 * i : i + 2; }

constexpr bool group_size_bits_is_the_table() {
  for (int i = 0; i < kGroupSizes; ++i) {
    if (group_size_bits(i) != kGroupSizeBits[i]) return false;
  }
  return sizeof(kGroupSizeBits) == kGroupSizes * sizeof(int);
}

static_assert(group_size_bits_is_the_table(), "group_size_bits is kGroupSizeBits");

//: `value` held within `lo` .. `hi`
FSLA_CARVE_FN constexpr int carve_clamp(int value, int lo, int hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

//: THE GROUP OF A DENSITY (docs/internals/common/plan/prefill_launch.md#groups), one rule for both sides: the log2 of the allowed group size nearest to `density * width / 256` digits,
//: a tie to the smaller size, compared exactly: the distance of a size `a` is `|a * 256 - density * width|`
FSLA_CARVE_FN constexpr int prefill_group_bits(int density, int width_bits) {
  const std::int64_t live = std::int64_t{density} << width_bits;
  int best = 0;
  std::int64_t best_distance = -1;
  for (int i = 0; i < kGroupSizes; ++i) {
    const int bits = group_size_bits(i);
    if (bits > width_bits) break;
    const std::int64_t size = (std::int64_t{1} << bits) << 8;
    const std::int64_t distance = size > live ? size - live : live - size;
    if (best_distance < 0 || distance < best_distance) {
      best = bits;
      best_distance = distance;
    }
  }
  return best;
}

//: A SIDE'S LOOP ORDER over `levels` levels (docs/internals/common/plan/cta_carve.md#loop-order): `order[0]` the outermost, the side's sparsest level, to the densest innermost; equal densities keep the lower level outside
constexpr void carve_loop_order(const int* density, int levels, int* order) {
  for (int l = 0; l < levels; ++l) order[l] = l;
  for (int i = 1; i < levels; ++i) {
    for (int j = i; j > 0 && density[order[j]] < density[order[j - 1]]; --j) {
      const int held = order[j];
      order[j] = order[j - 1];
      order[j - 1] = held;
    }
  }
}

struct CtaCarve {
  int levels;
  int carve_bits[kCarveMaxLevels];
};

//: whether level `l` takes the doubling before level `best` (`best < l`): higher priority (a level's WEIGHT, the sum of its two densities, over its holdings plus one), then larger read density, then the lower index
constexpr bool carve_takes_doubling(const CtaCarve& a, const int* write_density,
                                    const int* read_density, int best, int l) {
  const int lhs = (write_density[l] + read_density[l]) * (a.carve_bits[best] + 1);
  const int rhs = (write_density[best] + read_density[best]) * (a.carve_bits[l] + 1);
  if (lhs != rhs) return lhs > rhs;
  return read_density[l] > read_density[best];
}

//: THE CTA'S CARVE of `levels` levels (docs/internals/common/plan/cta_carve.md): the carve bits of each, from the widths' bits and each side's densities (in 1..256)
constexpr CtaCarve cta_carve(const int* width_bits, const int* write_density,
                             const int* read_density, int levels) {
  CtaCarve a{levels, {}};
  for (int step = 0; step < kCtaLeafBits; ++step) {
    int best = -1;
    for (int l = 0; l < levels; ++l) {
      if (a.carve_bits[l] + 1 > width_bits[l]) continue;
      if (best < 0 || carve_takes_doubling(a, write_density, read_density, best, l)) best = l;
    }
    if (best < 0) break;
    ++a.carve_bits[best];
  }
  return a;
}

//: every level index is 0-based; -1 is none. `write_density` and `read_density` are each side's densities as clamped, `write_group_bits` and `read_group_bits` the groups derived from them;
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
constexpr PrefillCarve prefill_carve(int levels, const int* width_bits, const int* write_density,
                                     const int* read_density) {
  PrefillCarve c{};
  levels = carve_clamp(levels, 0, kCarveMaxLevels);
  c.levels = levels;
  for (int l = 0; l < levels; ++l) {
    c.width_bits[l] = carve_clamp(width_bits[l], 0, kCarveMaxWidthBits);
    c.write_density[l] = carve_clamp(write_density[l], kLeastDensity, kWholeDensity);
    c.read_density[l] = carve_clamp(read_density[l], kLeastDensity, kWholeDensity);
    c.write_group_bits[l] = prefill_group_bits(c.write_density[l], c.width_bits[l]);
    c.read_group_bits[l] = prefill_group_bits(c.read_density[l], c.width_bits[l]);
  }
  carve_loop_order(c.write_density, levels, c.write_order);
  carve_loop_order(c.read_density, levels, c.read_order);
  const CtaCarve a = cta_carve(c.width_bits, c.write_density, c.read_density, levels);
  c.cta_leaves = 1;
  for (int l = 0; l < levels; ++l) {
    c.carve_bits[l] = a.carve_bits[l];
    c.cta_leaves <<= a.carve_bits[l];
  }

  //: the fold warpgroups split along the first level the write side does not walk whole that the carve holds 2 digits of, else the first level the carve holds 2 digits of
  c.split_level = -1;
  for (int l = levels - 1; l >= 0; --l) {
    if (c.carve_bits[l] >= 1) c.split_level = l;
  }
  for (int l = levels - 1; l >= 0; --l) {
    if (c.carve_bits[l] >= 1 && c.write_group_bits[l] < c.width_bits[l]) {
      c.split_level = l;
    }
  }
  c.folds = c.split_level >= 0 ? 2 : 1;
  c.fold_leaves = c.cta_leaves / c.folds;

  c.fold_set_leaves = 1;
  c.read_set_leaves = 1;
  c.unit_ctas = 1;
  for (int l = 0; l < levels; ++l) {
    c.fold_bits[l] = c.carve_bits[l] - (l == c.split_level ? 1 : 0);
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
constexpr void prefill_cta_at(const PrefillCarve& c, std::int64_t place, int* cta) {
  //: inside a group: the bits of a level its write group spans CTAs along (`unit_bits`)
  for (int i = 0; i < c.levels; ++i) {
    const int l = c.read_order[i];
    cta[l] = static_cast<int>(place & ((std::int64_t{1} << c.unit_bits[l]) - 1));
    place >>= c.unit_bits[l];
  }
  //: the groups: the bits of a level above those
  for (int i = c.levels - 1; i >= 0; --i) {
    const int l = c.write_order[i];
    const int bits = c.width_bits[l] - c.carve_bits[l] - c.unit_bits[l];
    cta[l] |= static_cast<int>(place & ((std::int64_t{1} << bits) - 1)) << c.unit_bits[l];
    place >>= bits;
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
