// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREFILL LAUNCH FUNCTION'S CARVE (docs/internals/common/plan/prefill_launch.md): how a call's levels fall into 256-leaf boxes, each side's
// list and column levels, the CTA's pair of boxes, the folds' split, the counts of a batch-head and the window, from the call's exact facts alone.
#pragma once

#include <cstdint>
#include <cstdlib>

#include "plan/prefill/facts.h"
#include "plan/levels.h"
#include "plan/rules.h"

namespace fsla::plan {

using PrefillDescriptor = prefill::Facts;

inline constexpr int kPrefillMaxLevels = 4;

//: a box is 256 leaves: 8 address bits; a side takes 4 of them as list bits and 4 as column bits
inline constexpr int kBoxBits = 8;
inline constexpr int kBoxHalfBits = 4;

//: the window of the hopper build (cc 900), provisional until the Hopper carry's design picks it
inline constexpr int kHopperWindow = 1024;

//: THE WINDOW the device of a descriptor runs, in tokens: the one statement of it, a function of the device's facts as the arm is (docs/internals/common/plan/prefill_launch.md#window);
//: compute capability 900 gives 1024, any other aborts
constexpr int prefill_window(const PrefillDescriptor& d) {
  if (!prefill::Reader::cc(d).point() || prefill::Reader::cc(d).lo != kHopperCc) std::abort();
  return kHopperWindow;
}

//: every level index is 0-based; -1 is none. `box_bits`, `read_list_bits` and `write_list_bits` are a level's bits in a box and each side's
//: list bits of them (the column bits are `box_bits - list_bits`); `read_list_level` and `write_list_level` are each side's list level
//: (a one-level call: level 0); `pair_level` is the level a CTA's two boxes neighbour along; `split_level` the level its two folds split along;
//: a fold's extents are along the write side's list level and column level, and `fold_outer` along a pair level outside the box; the counts are a batch-head's; `window` and `windows` are the
//: window's tokens and the call's count of them
struct PrefillCarve {
  int levels;
  int width_bits[kPrefillMaxLevels];
  int box_bits[kPrefillMaxLevels];
  int read_list_bits[kPrefillMaxLevels];
  int write_list_bits[kPrefillMaxLevels];
  int read_list_level;
  int write_list_level;
  int pair_level;
  int split_level;
  int boxes_per_cta;
  int fold_lists;
  int fold_columns;
  int fold_outer;
  std::int64_t boxes_per_head;
  std::int64_t ctas_per_head;
  std::int64_t units_per_head;
  std::int64_t unit_ctas;
  int window;
  std::int64_t windows;
};

//: the exact facts the carve reads, each a point and in its range; a range aborts
constexpr void prefill_exact(const PrefillDescriptor& d) {
  bool exact = prefill::Reader::cc(d).point() && prefill::Reader::tokens(d).point()
               && prefill::Reader::levels(d).point();
  exact = exact && prefill::Reader::levels(d).lo >= 1
          && prefill::Reader::levels(d).lo <= kPrefillMaxLevels;
  exact = exact && prefill::Reader::tokens(d).lo >= 1 && prefill::Reader::cc(d).lo == kHopperCc;
  for (int l = 0; l < kPrefillMaxLevels; ++l) {
    exact = exact && level_width_bits(d, l).point() && level_read_listed(d, l).point()
            && level_write_listed(d, l).point();
    exact = exact && level_read_listed(d, l).lo >= 0 && level_read_listed(d, l).lo <= 1
            && level_write_listed(d, l).lo >= 0 && level_write_listed(d, l).lo <= 1;
  }
  if (!exact) std::abort();
}

//: the two levels a box's halves of 4 bits are drawn from, `lo <= hi`: the write side's first listed and first whole level, else the first
//: two levels; a call of one level draws both halves from it (`lo == hi`), its high 4 bits the lists and its low 4 the columns
struct PrefillBoxLevels {
  int lo;
  int hi;
};

constexpr PrefillBoxLevels prefill_box_levels(const PrefillDescriptor& d) {
  const int levels = static_cast<int>(prefill::Reader::levels(d).lo);
  const int last = levels - 1;
  int listed = -1, whole = -1;
  for (int l = levels - 1; l >= 0; --l) {
    if (level_write_listed(d, l).lo == 1) {
      listed = l;
    } else {
      whole = l;
    }
  }
  if (listed < 0 || whole < 0) return {0, last < 1 ? last : 1};
  return listed < whole ? PrefillBoxLevels{listed, whole} : PrefillBoxLevels{whole, listed};
}

//: level `l`'s bits in a box: a half for each of the box's two halves drawn from it
constexpr int prefill_box_bits(const PrefillBoxLevels& box, int l) {
  return (l == box.lo ? kBoxHalfBits : 0) + (l == box.hi ? kBoxHalfBits : 0);
}

//: whether every level is wide enough to fill its bits in a box
constexpr bool prefill_carves(const PrefillDescriptor& d) {
  prefill_exact(d);
  const PrefillBoxLevels box = prefill_box_levels(d);
  bool wide = true;
  for (int l = 0; l < kPrefillMaxLevels; ++l) {
    wide = wide && level_width_bits(d, l).lo >= prefill_box_bits(box, l);
  }
  return wide;
}

//: a side's list level among the box levels `lo < hi` (the one listed where exactly one is, else `lo`)
constexpr int prefill_list_level(Interval (*listed)(const PrefillDescriptor&, int),
                                 const PrefillDescriptor& d, const PrefillBoxLevels& box) {
  const bool lo_listed = listed(d, box.lo).lo == 1;
  const bool hi_listed = listed(d, box.hi).lo == 1;
  return hi_listed && !lo_listed ? box.hi : box.lo;
}

//: the carve of a call from its exact facts; a range, or a call that is not carved, aborts
constexpr PrefillCarve prefill_carve(const PrefillDescriptor& d) {
  if (!prefill_carves(d)) std::abort();
  PrefillCarve c{};
  c.levels = static_cast<int>(prefill::Reader::levels(d).lo);
  const PrefillBoxLevels box = prefill_box_levels(d);
  for (int l = 0; l < c.levels; ++l) {
    c.width_bits[l] = static_cast<int>(level_width_bits(d, l).lo);
    c.box_bits[l] = prefill_box_bits(box, l);
  }
  const auto boxes_along = [&](int l) {
    return std::int64_t{1} << (c.width_bits[l] - c.box_bits[l]);
  };

  c.read_list_level = prefill_list_level(level_read_listed<PrefillDescriptor>, d, box);
  c.write_list_level = prefill_list_level(level_write_listed<PrefillDescriptor>, d, box);
  c.read_list_bits[c.read_list_level] = kBoxHalfBits;
  c.write_list_bits[c.write_list_level] = kBoxHalfBits;
  const int read_column = c.read_list_level == box.lo ? box.hi : box.lo;
  const int write_column = c.write_list_level == box.lo ? box.hi : box.lo;

  //: the pair, the first that holds: the read side's column level where it walks it whole with 2 or more boxes; the lowest other level it walks whole with
  //: 2 or more; else, where no whole level has 2, its column level, its list level, then the lowest other level with 2 or more
  const auto read_whole = [&](int l) { return level_read_listed(d, l).lo == 0; };
  const auto lowest_other = [&](bool whole_only, int skip) {
    for (int l = 0; l < c.levels; ++l) {
      const bool usable = l != skip && l != read_column && boxes_along(l) >= 2;
      if (usable && (!whole_only || read_whole(l))) return l;
    }
    return -1;
  };
  c.pair_level = -1;
  if (read_whole(read_column) && boxes_along(read_column) >= 2) {
    c.pair_level = read_column;
  } else {
    c.pair_level = lowest_other(true, -1);
  }
  if (c.pair_level < 0) {
    if (boxes_along(read_column) >= 2) {
      c.pair_level = read_column;
    } else if (boxes_along(c.read_list_level) >= 2) {
      c.pair_level = c.read_list_level;
    } else {
      c.pair_level = lowest_other(false, c.read_list_level);
    }
  }
  c.boxes_per_cta = c.pair_level >= 0 ? 2 : 1;

  const bool lists_walked = level_write_listed(d, c.write_list_level).lo == 1;
  c.split_level = lists_walked || c.pair_level < 0 ? c.write_list_level : c.pair_level;
  //: where the write side's lists and columns are one level's bits, the pair and the split take its lists (the high bits)
  const bool pair_lists = c.pair_level == c.write_list_level;
  const bool pair_columns = !pair_lists && c.pair_level == write_column;
  const bool split_lists = c.split_level == c.write_list_level;
  const bool split_columns = !split_lists && c.split_level == write_column;
  c.fold_lists = (pair_lists ? 32 : 16) / (split_lists ? 2 : 1);
  c.fold_columns = (pair_columns ? 32 : 16) / (split_columns ? 2 : 1);
  const bool pair_outside = c.pair_level >= 0 && c.box_bits[c.pair_level] == 0;
  c.fold_outer = pair_outside && c.split_level != c.pair_level ? 2 : 1;

  c.boxes_per_head = 1;
  c.units_per_head = 1;
  for (int l = 0; l < c.levels; ++l) {
    c.boxes_per_head *= boxes_along(l);
    if (level_write_listed(d, l).lo != 1) continue;
    c.units_per_head *= l == c.pair_level ? boxes_along(l) / 2 : boxes_along(l);
  }
  c.ctas_per_head = c.boxes_per_head / c.boxes_per_cta;
  c.unit_ctas = c.ctas_per_head / c.units_per_head;

  c.window = prefill_window(d);
  c.windows = (prefill::Reader::tokens(d).lo + c.window - 1) / c.window;
  return c;
}

}  // namespace fsla::plan
