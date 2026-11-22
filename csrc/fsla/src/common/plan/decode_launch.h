// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// DECODE'S LAUNCH FUNCTION (docs/internals/common/plan/decode_launch.md): the arm to launch and its
// arguments, from a descriptor; the arm half is evaluable over ranges, the args half is not.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <iterator>
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

//: what the launch takes besides the arm: the compute capability of the build to launch, the CTA's threads, the shared
//: memory a CTA may opt in to, the zero columns the call's value width is padded by, and the device's SMs
struct DecodeArgs {
  int cc;
  int threads;
  int smem_budget;
  int pad;
  int sms;
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

constexpr void decode_decays(DecodeReach& r, const Descriptor& d, int d_v, int levels) {
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t k) { r.arm({d_v, levels, static_cast<int>(k)}); };
  ladder(Reader::decay(d), 0, kDecodeDecayUpto, 2, hole, leaf);
}

constexpr void decode_levels(DecodeReach& r, const Descriptor& d, int d_v) {
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t l) { decode_decays(r, d, d_v, static_cast<int>(l) + 1); };
  ladder(Reader::levels(d), 1, kDecodeLevelsUpto, 4, hole, leaf);
}

//: THE ARM HALF, a decision tree on Truth, exact over ranges (docs/internals/common/plan/rules.md#exact)
constexpr DecodeReach decode_reach(const Descriptor& d) {
  DecodeReach r;
  const auto hole = [&] { r.hole(); };
  const auto leaf = [&](std::size_t w) { decode_levels(r, d, kDecodeArmWidths[w]); };
  const auto widths = [&] {
    ladder(Reader::d_v(d), 1, kDecodeArmWidths, std::size(kDecodeArmWidths), hole, leaf);
  };
  fork(eq(Reader::cc(d), 900), widths, hole);
  return r;
}

//: the arm of a point: the same walk, which over a point visits one leaf; a range, or a point no arm serves, aborts
constexpr DecodeArm decode_arm(const Descriptor& d) { return decode_reach(d).one(); }

//: exact values only: a range aborts
constexpr DecodeArgs decode_args(const Descriptor& d, const DecodeArm& arm) {
  bool exact = Reader::cc(d).point() && Reader::regs_per_sm(d).point()
               && Reader::smem_per_cta_optin(d).point() && Reader::sm_count(d).point()
               && Reader::d_v(d).point();
  if (!exact) std::abort();
  return {static_cast<int>(Reader::cc(d).lo),
          static_cast<int>(Reader::regs_per_sm(d).lo / kDecodeRegsPerThread),
          static_cast<int>(Reader::smem_per_cta_optin(d).lo),
          arm.d_v - static_cast<int>(Reader::d_v(d).lo), static_cast<int>(Reader::sm_count(d).lo)};
}

//: the scratch a call needs beside its operands: its bytes and the alignment of its address (decode has no per-call scratch: its buffers belong to
//: the sequence)
struct WorkspaceNeed {
  std::int64_t bytes;
  std::int64_t alignment;
};

constexpr WorkspaceNeed decode_workspace(const Descriptor&) { return {0, 256}; }

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

constexpr DecodeLaunch decode_launch(const Descriptor& d) {
  const DecodeArm arm = decode_arm(d);
  return {arm, decode_args(d, arm)};
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
