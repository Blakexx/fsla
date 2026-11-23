// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PER-LEVEL FACTS (docs/internals/common/plan/levels.md): the one lookup of a level's width and of each side's walk of it, for the launch functions.
#pragma once

#include "plan/facts.h"
#include "plan/rules.h"

namespace fsla::plan {

//: the call's width fact of level `l` (0-based): the log2 of that level's width
constexpr Interval level_width_bits(const Facts& d, int l) {
  switch (l) {
    case 0:
      return Reader::width_bits_1(d);
    case 1:
      return Reader::width_bits_2(d);
    case 2:
      return Reader::width_bits_3(d);
    default:
      return Reader::width_bits_4(d);
  }
}

//: the call's read-side flag of level `l` (0-based): 1 when the read side walks it by its lists
constexpr Interval level_read_listed(const Facts& d, int l) {
  switch (l) {
    case 0:
      return Reader::read_listed_1(d);
    case 1:
      return Reader::read_listed_2(d);
    case 2:
      return Reader::read_listed_3(d);
    default:
      return Reader::read_listed_4(d);
  }
}

//: the call's write-side flag of level `l` (0-based): 1 when the write side walks it by its lists
constexpr Interval level_write_listed(const Facts& d, int l) {
  switch (l) {
    case 0:
      return Reader::write_listed_1(d);
    case 1:
      return Reader::write_listed_2(d);
    case 2:
      return Reader::write_listed_3(d);
    default:
      return Reader::write_listed_4(d);
  }
}

}  // namespace fsla::plan
