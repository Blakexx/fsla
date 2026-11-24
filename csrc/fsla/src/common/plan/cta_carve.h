// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREFILL'S CARVE (docs/internals/common/plan/cta_carve.md): from a call's level widths and each side's weights alone, the groups of the weights, how the levels share the leaves one CTA owns,
// the folds' split and the write and read sides' sets. Pure, constexpr, total, integers and fixed arrays alone: nothing in it is beyond a kernel's reach.
#pragma once

#include <cstdint>

namespace fsla::plan {

inline constexpr int kCarveMaxLevels = 4;

//: a level is at most 256 digits wide: 8 bits
inline constexpr int kCarveMaxWidthBits = 8;

//: a CTA owns at most 512 leaves: 9 doublings from one digit a level
inline constexpr int kCtaLeafBits = 9;
inline constexpr int kCtaLeaves = 1 << kCtaLeafBits;

//: a side's weight at which it is live on every digit of a level (256ths of the level), and the weight at least
inline constexpr int kWholeWeight = 256;
inline constexpr int kLeastWeight = 1;

//: the allowed group sizes in digits, as log2: 1, 4, 16, 32, 64, 128, 256 (those no larger than a level's width are allowed on it)
inline constexpr int kGroupSizeBits[] = {0, 2, 4, 5, 6, 7, 8};

//: `value` held within `lo` .. `hi`
constexpr int carve_clamp(int value, int lo, int hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

//: THE GROUP OF A WEIGHT (docs/internals/common/plan/prefill_launch.md#groups), one rule for both sides: the log2 of the allowed group size nearest to `weight * width / 256` digits,
//: a tie to the smaller size, compared exactly: the distance of a size `a` is `|a * 256 - weight * width|`
constexpr int prefill_group_bits(int weight, int width_bits) {
  const std::int64_t live = std::int64_t{weight} << width_bits;
  int best = 0;
  std::int64_t best_distance = -1;
  for (const int bits : kGroupSizeBits) {
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

struct CtaCarve {
  int levels;
  int carve_bits[kCarveMaxLevels];
};

//: whether level `l` takes the doubling before level `best` (`best < l`): higher priority, then larger read weight, then the lower index
constexpr bool carve_takes_doubling(const CtaCarve& a, const int* write_weight,
                                    const int* read_weight, int best, int l) {
  const int lhs = (write_weight[l] + read_weight[l]) * (a.carve_bits[best] + 1);
  const int rhs = (write_weight[best] + read_weight[best]) * (a.carve_bits[l] + 1);
  if (lhs != rhs) return lhs > rhs;
  return read_weight[l] > read_weight[best];
}

//: THE CTA'S CARVE of `levels` levels (docs/internals/common/plan/cta_carve.md): the carve bits of each, from the widths' bits and each side's weights (in 1..256)
constexpr CtaCarve cta_carve(const int* width_bits, const int* write_weight, const int* read_weight,
                             int levels) {
  CtaCarve a{levels, {}};
  for (int step = 0; step < kCtaLeafBits; ++step) {
    int best = -1;
    for (int l = 0; l < levels; ++l) {
      if (a.carve_bits[l] + 1 > width_bits[l]) continue;
      if (best < 0 || carve_takes_doubling(a, write_weight, read_weight, best, l)) best = l;
    }
    if (best < 0) break;
    ++a.carve_bits[best];
  }
  return a;
}

//: every level index is 0-based; -1 is none. `write_weight` and `read_weight` are each side's weights as clamped, `write_group_bits` and `read_group_bits` the groups derived from them;
//: `carve_bits` are the CTA's carve of each level; `split_level` is the level the two fold warpgroups split along
//: and `folds` their count; `fold_bits` a fold's bits of each level; `fold_set_leaves` and `fold_sets` the leaves a token is expanded over at once on the write side and the
//: separate sets a fold holds; `read_set_leaves` and `read_sets` the same for the read side over the whole carve; `unit_ctas` the CTAs inside one combination of write groups (they read one value stream)
struct PrefillCarve {
  int levels;
  int width_bits[kCarveMaxLevels];
  int write_weight[kCarveMaxLevels];
  int read_weight[kCarveMaxLevels];
  int write_group_bits[kCarveMaxLevels];
  int read_group_bits[kCarveMaxLevels];
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
  std::int64_t unit_ctas;
};

//: THE CARVE of a call, from `levels` (0 to 4) levels' width bits (0 to 8) and each side's weights, `width_bits[0 .. levels)`, `write_weight[0 .. levels)` and `read_weight[0 .. levels)`;
//: TOTAL: a weight outside 1..256 is CLAMPED into 1..256 (the mask format's range), `levels` into 0..4 and a width into 0..8; it has no failure path
constexpr PrefillCarve prefill_carve(int levels, const int* width_bits, const int* write_weight,
                                     const int* read_weight) {
  PrefillCarve c{};
  levels = carve_clamp(levels, 0, kCarveMaxLevels);
  c.levels = levels;
  for (int l = 0; l < levels; ++l) {
    c.width_bits[l] = carve_clamp(width_bits[l], 0, kCarveMaxWidthBits);
    c.write_weight[l] = carve_clamp(write_weight[l], kLeastWeight, kWholeWeight);
    c.read_weight[l] = carve_clamp(read_weight[l], kLeastWeight, kWholeWeight);
    c.write_group_bits[l] = prefill_group_bits(c.write_weight[l], c.width_bits[l]);
    c.read_group_bits[l] = prefill_group_bits(c.read_weight[l], c.width_bits[l]);
  }
  const CtaCarve a = cta_carve(c.width_bits, c.write_weight, c.read_weight, levels);
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
    c.unit_ctas <<=
        c.write_group_bits[l] > c.carve_bits[l] ? c.write_group_bits[l] - c.carve_bits[l] : 0;
  }
  c.fold_sets = c.fold_leaves / c.fold_set_leaves;
  c.read_sets = c.cta_leaves / c.read_set_leaves;
  return c;
}

}  // namespace fsla::plan
