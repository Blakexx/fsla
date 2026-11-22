// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// DECODE'S LAUNCH FUNCTION (docs/internals/common/plan/decode_launch.md): the arm to launch and its
// arguments, from a descriptor; the arm half is evaluable over ranges, the args half is not.
#pragma once

#include <cstdlib>
#include <string>

#include "plan/facts.h"
#include "plan/rules.h"

namespace fsla::plan {

using Descriptor = Facts;

//: one decode arm, the key of its kernel `decode_step_kernel<Build, d_v, levels, decay>`
struct DecodeArm {
  int d_v;
  int levels;
  int decay;

  constexpr bool operator==(const DecodeArm&) const = default;

  constexpr bool operator<(const DecodeArm& o) const {
    if (d_v != o.d_v) return d_v < o.d_v;
    return levels != o.levels ? levels < o.levels : decay < o.decay;
  }
};

struct DecodeArgs {
  int threads;
  int smem_budget;
  int pad;
};

struct DecodeLaunch {
  DecodeArm arm;
  DecodeArgs args;
};

inline constexpr std::size_t kMaxDecodeArms = 64;

using DecodeArmSet = ArmSet<DecodeArm, kMaxDecodeArms>;

inline constexpr int kDecodeRegsPerThread = 128;

constexpr DecodeArmSet decode_arms(const Descriptor& d) {
  ArmSet<DecodeArm, kMaxDecodeArms> out;
  constexpr int kWidths[] = {32, 64};
  for (int d_v : kWidths)
    for (int levels = 1; levels <= 4; ++levels)
      for (int decay = 0; decay <= 1; ++decay) {
        Truth reached = all(eq(Reader::cc(d), 900), eq(Reader::d_v(d), d_v),
                            eq(Reader::levels(d), levels), eq(Reader::decay(d), decay));
        if (reached != Truth::No) out.add({d_v, levels, decay});
      }
  return out;
}

//: exact values only: a range aborts
constexpr DecodeArgs decode_args(const Descriptor& d, const DecodeArm& arm) {
  bool exact = Reader::regs_per_sm(d).point() && Reader::smem_per_cta_optin(d).point();
  if (!exact || !Reader::d_v(d).point()) std::abort();
  return {static_cast<int>(Reader::regs_per_sm(d).lo / kDecodeRegsPerThread),
          static_cast<int>(Reader::smem_per_cta_optin(d).lo),
          arm.d_v - static_cast<int>(Reader::d_v(d).lo)};
}

constexpr DecodeLaunch decode_launch(const Descriptor& d) {
  DecodeArmSet set = decode_arms(d);
  if (set.size() != 1) std::abort();
  return {set.arms[0], decode_args(d, set.arms[0])};
}

inline std::string decode_facts_read(const Descriptor& d) {
  auto at = [](Interval v) {
    return v.point() ? std::to_string(v.lo) : std::to_string(v.lo) + ".." + std::to_string(v.hi);
  };
  std::string out = "cc=" + at(Reader::cc(d)) + " d_v=" + at(Reader::d_v(d));
  out += " levels=" + at(Reader::levels(d)) + " decay=" + at(Reader::decay(d));
  return out;
}

}  // namespace fsla::plan
