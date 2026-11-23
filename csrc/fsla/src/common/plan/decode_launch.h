// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// DECODE'S LAUNCH FUNCTION (docs/internals/common/plan/decode_launch.md): the arm to launch and its
// arguments, from a descriptor; the arm half is evaluable over ranges, the args half is not.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <string>

#include "plan/decode/facts.h"
#include "plan/levels.h"
#include "plan/rules.h"

namespace fsla::plan {

using DecodeDescriptor = decode::Facts;

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

inline constexpr int kDecodeMaxLevels = 4;

//: what the launch takes besides the arm: the compute capability of the build to launch, the CTA's threads, the shared
//: memory a CTA may opt in to, the zero columns the call's value width is padded by, the device's SMs, each level's width
//: (0 beyond the call's `levels`) and its offset into the flattened per-level rows (the prefix sums of the widths), and
//: whether the step deposits whole atoms (0 or 1): THE LAUNCH FUNCTION'S OWN DERIVATION, 1 when the innermost level's write side is
//: not listed (`write_listed` of the call's last level is 0)
struct DecodeArgs {
  int cc;
  int threads;
  int smem_budget;
  int pad;
  int sms;
  int widths[kDecodeMaxLevels];
  int row_offsets[kDecodeMaxLevels];
  int write_dense;
};

struct DecodeLaunch {
  DecodeArm arm;
  DecodeArgs args;
};

inline constexpr std::size_t kMaxDecodeArms = 64;

//: THE WALK'S REGISTERS A THREAD: a warp holds a quarter of an atom's leaves a lane group in flight.
//: -- see docs/internals/decode/decode.md#decode-threads
inline constexpr int kDecodeRegsPerThread = 128;

//: THE SMALLEST UNIT SHARE worth a CTA's prologue (its fold and factor tables): the grid's supply bound.
//: -- see docs/internals/decode/decode.md#the-split-policy
inline constexpr int kDecodeUnitsPerCta = 16;

//: the CTAs an SM keeps resident: the step's declared residency (docs/internals/decode/decode.md#declared-residency)
inline constexpr int kDecodeBlocksPerSm = 1;

//: the value widths the decode arms are built at; a call's `d_v` is served by the narrowest arm that holds it
inline constexpr int kDecodeArmWidths[] = {32, 64};

inline constexpr int kDecodeLevelsUpto[] = {1, 2, 3, 4};
inline constexpr int kDecodeDecayUpto[] = {0, 1};

using DecodeReach = Reach<DecodeArm, kMaxDecodeArms>;

constexpr void decode_decays(DecodeReach& r, const DecodeDescriptor& d, int d_v, int levels) {
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t k) { r.arm({d_v, levels, static_cast<int>(k)}); };
  ladder(decode::Reader::decay(d), 0, kDecodeDecayUpto, 2, hole, leaf);
}

constexpr void decode_levels(DecodeReach& r, const DecodeDescriptor& d, int d_v) {
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t l) { decode_decays(r, d, d_v, static_cast<int>(l) + 1); };
  ladder(decode::Reader::levels(d), 1, kDecodeLevelsUpto, 4, hole, leaf);
}

//: THE ARM HALF, a decision tree on Truth, exact over ranges (docs/internals/common/plan/rules.md#exact)
constexpr DecodeReach decode_reach(const DecodeDescriptor& d) {
  DecodeReach r;
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t w) { decode_levels(r, d, kDecodeArmWidths[w]); };
  const auto widths = [&] {
    ladder(decode::Reader::d_v(d), 1, kDecodeArmWidths, std::size(kDecodeArmWidths), hole, leaf);
  };
  fork(eq(decode::Reader::cc(d), kHopperCc), widths, hole);
  return r;
}

//: the arm of a point: the same walk, which over a point visits one leaf; a range, or a point no arm serves, aborts
constexpr DecodeArm decode_arm(const DecodeDescriptor& d) { return decode_reach(d).one(); }

//: exact values only: a range aborts
constexpr DecodeArgs decode_args(const DecodeDescriptor& d, const DecodeArm& arm) {
  bool exact = decode::Reader::cc(d).point() && decode::Reader::regs_per_sm(d).point()
               && decode::Reader::smem_per_cta_optin(d).point()
               && decode::Reader::sm_count(d).point() && decode::Reader::d_v(d).point()
               && decode::Reader::levels(d).point();
  for (int l = 0; l < kDecodeMaxLevels; ++l) {
    exact = exact && level_width_bits(d, l).point() && level_write_listed(d, l).point();
  }
  if (!exact) std::abort();
  const bool innermost_dense =
      level_write_listed(d, static_cast<int>(decode::Reader::levels(d).lo) - 1).lo == 0;
  DecodeArgs args{static_cast<int>(decode::Reader::cc(d).lo),
                  static_cast<int>(decode::Reader::regs_per_sm(d).lo / kDecodeRegsPerThread),
                  static_cast<int>(decode::Reader::smem_per_cta_optin(d).lo),
                  arm.d_v - static_cast<int>(decode::Reader::d_v(d).lo),
                  static_cast<int>(decode::Reader::sm_count(d).lo),
                  {},
                  {},
                  static_cast<int>(innermost_dense)};
  int offset = 0;
  for (int l = 0; l < arm.levels; ++l) {
    const auto bits = level_width_bits(d, l).lo;
    args.widths[l] = bits > 0 ? 1 << bits : 0;
    args.row_offsets[l] = offset;
    offset += args.widths[l];
  }
  return args;
}

//: THE STEP'S GRID, from exact values right before the launch: one wave of the declared residency, at most `bh` heads' unit supply of
//: `unit_ceiling` units each; `forced` (a gate's count, at least 1) replaces it, -1 leaves it
constexpr int decode_grid(const DecodeArgs& args, std::int64_t unit_ceiling, std::int64_t bh,
                          std::int64_t forced) {
  if (forced >= 1) return static_cast<int>(forced);
  const std::int64_t supply = unit_ceiling > kDecodeUnitsPerCta
                                  ? (unit_ceiling + kDecodeUnitsPerCta - 1) / kDecodeUnitsPerCta
                                  : 1;
  const std::int64_t resident = std::int64_t{args.sms} * kDecodeBlocksPerSm;
  const std::int64_t grid = resident < bh * supply ? resident : bh * supply;
  return static_cast<int>(grid > 1 ? grid : 1);
}

constexpr DecodeLaunch decode_launch(const DecodeDescriptor& d) {
  const DecodeArm arm = decode_arm(d);
  return {arm, decode_args(d, arm)};
}

//: whether a launch's arguments are the descriptor's own: its device facts, the arm's padding and each level's width
constexpr bool decode_args_reflect(const DecodeDescriptor& d, const DecodeLaunch& launch) {
  const DecodeArgs& a = launch.args;
  bool ok = a.cc == decode::Reader::cc(d).lo && a.sms == decode::Reader::sm_count(d).lo
            && a.smem_budget == decode::Reader::smem_per_cta_optin(d).lo
            && a.threads == decode::Reader::regs_per_sm(d).lo / kDecodeRegsPerThread
            && a.pad == launch.arm.d_v - decode::Reader::d_v(d).lo && a.pad >= 0;
  for (int l = 0; l < launch.arm.levels; ++l) {
    const std::int64_t bits = level_width_bits(d, l).lo;
    ok = ok && a.widths[l] == (bits > 0 ? std::int64_t{1} << bits : 0);
  }
  return ok;
}

inline std::string decode_facts_read(const DecodeDescriptor& d) {
  auto at = [](Interval v) {
    return v.point() ? std::to_string(v.lo) : std::to_string(v.lo) + ".." + std::to_string(v.hi);
  };
  std::string out = "cc=" + at(decode::Reader::cc(d)) + " d_v=" + at(decode::Reader::d_v(d));
  out += " levels=" + at(decode::Reader::levels(d)) + " decay=" + at(decode::Reader::decay(d));
  return out;
}

}  // namespace fsla::plan
