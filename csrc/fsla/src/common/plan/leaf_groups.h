// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// A SIDE'S LEAF GROUPS (docs/internals/common/plan/leaf_groups.md): how many a call's carve gives a side, the order they are laid in, and the most a call's shape allows whatever its densities.
// Pure, constexpr, total, integers and fixed arrays alone: nothing in it is beyond a kernel's reach.
#pragma once

#include <cstdint>

#include "cta_carve.h"

namespace fsla::plan {

//: A SIDE'S LEAF GROUPS: a leaf group is one group on each of the side's levels, `bits` their count as log2. A level's group index is two fields of a leaf group's index:
//: its `inner[l]` low bits, those that vary inside a box (a fold on the write side, the CTA on the read side), at `inner_at[l]`, and its `outer[l]` bits above them, at `outer_at[l]`.
//: THE INNER FIELDS ARE THE INDEX'S LOW `inner_bits`, the side's densest level's lowest: a box's sets are consecutive leaf groups, its first `outer << inner_bits`
struct LeafGroups {
  int bits;
  int inner_bits;
  int inner[kCarveMaxLevels];
  int inner_at[kCarveMaxLevels];
  int outer[kCarveMaxLevels];
  int outer_at[kCarveMaxLevels];
};

//: THE LEAF GROUPS of a side of a carve -- see docs/internals/common/plan/leaf_groups.md#order
FSLA_CARVE_FN constexpr LeafGroups leaf_groups(const PrefillCarve& c, bool read) {
  LeafGroups g{};
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    const int group = read ? c.read_group_bits[l] : c.write_group_bits[l];
    const int box = read ? c.carve_bits[l] : c.fold_bits[l];
    const int along = l < c.levels ? c.width_bits[l] - group : 0;
    const int inside = l < c.levels && box > group ? box - group : 0;
    g.inner[l] = inside;
    g.outer[l] = along - inside;
    g.bits += along;
  }
  //: the inner fields, the side's densest level's lowest; each of the carve's arrays is read before the level is compared with anything
  int taken = 0;
  FSLA_CARVE_UNROLL
  for (int i = kCarveMaxLevels - 1; i >= 0; --i) {
    FSLA_CARVE_UNROLL
    for (int l = 0; l < kCarveMaxLevels; ++l) {
      const int order = read ? c.read_order[i] : c.write_order[i], inside = g.inner[l];
      const bool here = i < c.levels && order == l;
      g.inner_at[l] = here ? taken : g.inner_at[l];
      taken += here ? inside : 0;
    }
  }
  g.inner_bits = taken;
  //: the outer fields above them, the lowest level's lowest
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    g.outer_at[l] = taken;
    taken += g.outer[l];
  }
  return g;
}

//: THE LEAF GROUP of one group a level, `group[l]` its index along level `l`
FSLA_CARVE_FN constexpr int leaf_group_of(const LeafGroups& g, const int* group) {
  int leaf = 0;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    const int inside = group[l] & ((1 << g.inner[l]) - 1), above = group[l] >> g.inner[l];
    leaf |= (inside << g.inner_at[l]) | (above << g.outer_at[l]);
  }
  return leaf;
}

//: A LEAF GROUP'S GROUP ON EACH LEVEL, `group[l]` its index along level `l`: `leaf_group_of` undone
FSLA_CARVE_FN constexpr void leaf_group_levels(const LeafGroups& g, int leaf, int* group) {
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    const int inside = (leaf >> g.inner_at[l]) & ((1 << g.inner[l]) - 1);
    const int above = (leaf >> g.outer_at[l]) & ((1 << g.outer[l]) - 1);
    group[l] = (above << g.inner[l]) | inside;
  }
}

//: THE MOST LEAF GROUPS A SIDE CAN HAVE, as log2, whatever its densities (docs/internals/common/plan/leaf_groups.md#most): `width_bits` of the call's `levels` levels and `masked`, a bit a level the side has a mask on.
//: A level with no mask is walked whole, one group; and a leaf group is 16 leaves at least wherever a fold holds 16 (a call of 32 leaves or more), since a set is raised to 16 and a leaf group holds its set
FSLA_CARVE_FN constexpr int leaf_group_bits_most(int levels, const int* width_bits, int masked) {
  levels = carve_clamp(levels, 0, kCarveMaxLevels);
  int leaves = 0, listed = 0;
  FSLA_CARVE_UNROLL
  for (int l = 0; l < kCarveMaxLevels; ++l) {
    const int width = l < levels ? carve_clamp(width_bits[l], 0, kCarveMaxWidthBits) : 0;
    leaves += width;
    listed += (masked >> l & 1) != 0 ? width : 0;
  }
  const int least = leaves > kLeastSetBits ? leaves - kLeastSetBits : leaves;
  return listed < least ? listed : least;
}

namespace leaf_proof {

constexpr PrefillCarve two_levels(int write0, int write1, int read0, int read1) {
  const int width_bits[2] = {8, 8};
  const int write_density[2] = {write0, write1};
  const int read_density[2] = {read0, read1};
  return prefill_carve(2, width_bits, write_density, read_density);
}

//: the measured cell: level 0 sparse for the write side and whole for the read side, level 1 the reverse
constexpr PrefillCarve cell() { return two_levels(4, 256, 256, 4); }

//: a leaf group and back, over every leaf group of a side
constexpr bool round_trips(const PrefillCarve& c, bool read) {
  const LeafGroups g = leaf_groups(c, read);
  for (int leaf = 0; leaf < (1 << g.bits); ++leaf) {
    int group[kCarveMaxLevels] = {};
    leaf_group_levels(g, leaf, group);
    if (leaf_group_of(g, group) != leaf) return false;
  }
  return true;
}

static_assert(leaf_groups(cell(), false).bits == 6 && leaf_groups(cell(), true).bits == 6,
              "the cell has 64 leaf groups a side");
static_assert(leaf_groups(two_levels(4, 4, 4, 4), false).bits == 12,
              "two levels at 4 live digits of 256 are 4,096 leaf groups of 16 leaves");
static_assert(round_trips(cell(), false) && round_trips(cell(), true)
                  && round_trips(two_levels(4, 4, 4, 4), false)
                  && round_trips(two_levels(4, 4, 4, 4), true)
                  && round_trips(two_levels(1, 64, 16, 2), false)
                  && round_trips(two_levels(1, 64, 16, 2), true),
              "a leaf group's index is its levels' groups, one to one");

constexpr int most_of_two(int masked) {
  const int width_bits[2] = {8, 8};
  return leaf_group_bits_most(2, width_bits, masked);
}

static_assert(most_of_two(3) == 12 && most_of_two(1) == 8 && most_of_two(0) == 0,
              "65,536 leaves: 4,096 leaf groups at most, 256 with one level whole, one with both");

}  // namespace leaf_proof

}  // namespace fsla::plan
