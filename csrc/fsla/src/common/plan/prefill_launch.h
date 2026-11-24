// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PREFILL LAUNCH FUNCTION'S HOST HALF (docs/internals/common/plan/prefill_launch.md): what the host knows of a call from its descriptor; the densities are the carve's (`plan/cta_carve.h`), never the host's.
#pragma once

#include <cstdint>
#include <cstdlib>

#include "plan/cta_carve.h"
#include "plan/prefill/facts.h"
#include "plan/levels.h"
#include "plan/rules.h"

namespace fsla::plan {

using PrefillDescriptor = prefill::Facts;

//: THE WINDOW the device of a descriptor runs, in tokens (`window_of_cc`, docs/internals/common/plan/prefill_launch.md#window); a card with none aborts
constexpr int prefill_window(const PrefillDescriptor& d) {
  const int window = prefill::Reader::cc(d).point() ? window_of_cc(prefill::Reader::cc(d).lo) : 0;
  if (window == 0) std::abort();
  return window;
}

//: `leaves` is the call's leaves a batch-head (the product of the levels' widths); `cta_leaves` a CTA's, the smaller of 512 and `leaves`; `ctas_per_head = leaves / cta_leaves` (it does not depend on how
//: the carve falls across the levels); `window` and `windows` are the window's tokens and the call's count of them
struct PrefillLaunch {
  int levels;
  int width_bits[kCarveMaxLevels];
  std::int64_t leaves;
  int cta_leaves;
  std::int64_t ctas_per_head;
  int window;
  std::int64_t windows;
};

//: the launch's counts of `levels` levels from their width bits (`window` and `windows` left 0)
constexpr PrefillLaunch prefill_launch_of(const int* width_bits, int levels) {
  PrefillLaunch p{};
  p.levels = levels;
  p.leaves = 1;
  for (int l = 0; l < levels; ++l) {
    p.width_bits[l] = width_bits[l];
    p.leaves <<= width_bits[l];
  }
  p.cta_leaves = p.leaves < kCtaLeaves ? static_cast<int>(p.leaves) : kCtaLeaves;
  p.ctas_per_head = p.leaves / p.cta_leaves;
  return p;
}

constexpr void prefill_exact(const PrefillDescriptor& d) {
  bool exact = prefill::Reader::cc(d).point() && prefill::Reader::tokens(d).point()
               && prefill::Reader::levels(d).point();
  exact = exact && prefill::Reader::levels(d).lo >= 1
          && prefill::Reader::levels(d).lo <= kCarveMaxLevels;
  exact = exact && prefill::Reader::tokens(d).lo >= 1 && prefill::Reader::cc(d).lo == kHopperCc;
  for (int l = 0; l < kCarveMaxLevels; ++l) exact = exact && level_width_bits(d, l).point();
  if (!exact) std::abort();
}

//: the launch of a call from its exact facts; a range aborts
constexpr PrefillLaunch prefill_launch(const PrefillDescriptor& d) {
  prefill_exact(d);
  int width_bits[kCarveMaxLevels] = {};
  const int levels = static_cast<int>(prefill::Reader::levels(d).lo);
  for (int l = 0; l < levels; ++l) width_bits[l] = static_cast<int>(level_width_bits(d, l).lo);
  PrefillLaunch p = prefill_launch_of(width_bits, levels);
  p.window = prefill_window(d);
  p.windows = (prefill::Reader::tokens(d).lo + p.window - 1) / p.window;
  return p;
}

}  // namespace fsla::plan
