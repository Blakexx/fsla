// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PER-LEVEL FACTS (docs/internals/common/plan/levels.md): the one lookup of a level's width and of each side's walk of it, for the launch functions of
// every kernel, over the kernel's `Facts` (its `Read` is its reader; the read-side walk is a fact of the prefill's alone, so its lookup is instantiated there).
#pragma once

#include "plan/rules.h"

namespace fsla::plan {

//: the compute capability of the hopper device (`sm_90a`), the one both launch functions' matchers name
inline constexpr int kHopperCc = 900;

//: the call's width fact of level `l` (0-based): the log2 of that level's width
template <typename F>
constexpr Interval level_width_bits(const F& d, int l) {
  using R = typename F::Read;
  switch (l) {
    case 0:
      return R::width_bits_1(d);
    case 1:
      return R::width_bits_2(d);
    case 2:
      return R::width_bits_3(d);
    default:
      return R::width_bits_4(d);
  }
}

//: the call's read-side flag of level `l` (0-based): 1 when the read side walks it by its lists
template <typename F>
constexpr Interval level_read_listed(const F& d, int l) {
  using R = typename F::Read;
  switch (l) {
    case 0:
      return R::read_listed_1(d);
    case 1:
      return R::read_listed_2(d);
    case 2:
      return R::read_listed_3(d);
    default:
      return R::read_listed_4(d);
  }
}

//: the call's write-side flag of level `l` (0-based): 1 when the write side walks it by its lists
template <typename F>
constexpr Interval level_write_listed(const F& d, int l) {
  using R = typename F::Read;
  switch (l) {
    case 0:
      return R::write_listed_1(d);
    case 1:
      return R::write_listed_2(d);
    case 2:
      return R::write_listed_3(d);
    default:
      return R::write_listed_4(d);
  }
}

}  // namespace fsla::plan
