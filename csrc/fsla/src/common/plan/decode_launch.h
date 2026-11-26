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
#include "plan/lattice.h"
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

//: what the launch takes besides the arm: the compute capability of the build to launch, the CTA's threads, the zero columns the call's value width is padded by, the device's SMs, each level's width
//: (0 beyond the call's `levels`) and its offset into the flattened per-level rows (the prefix sums of the widths), and
//: whether the step deposits whole atoms (0 or 1): THE LAUNCH FUNCTION'S OWN DERIVATION, 1 when the innermost level's write side is
//: not listed (`write_listed` of the call's last level is 0)
struct DecodeArgs {
  int cc;
  int threads;
  int pad;
  int sms;
  int widths[kDecodeMaxLevels];
  int row_offsets[kDecodeMaxLevels];
  int write_dense;
};

//: THE LAUNCH'S LAYOUT, a function of the call's facts and the device's: the lattice box `(lat_k, lat_m)` of the level widths, every
//: field the step's parameters carry of it (named as `DecodeParams` names them), the tables' word layout, `write_atoms`,
//: the `leaves`, `total_rows`, `unit_ceiling` (the units a head supplies the grid), and `step_words`, what the call needs of the step's
//: run-time tile; every array is 0 beyond the call's levels -- see docs/internals/common/plan/decode_launch.md
struct DecodeLayout {
  int lat_k;
  int lat_m;
  int lat_s[kDecodeMaxLevels];
  int lat_g[kDecodeMaxLevels];
  int lat_gsuf[kDecodeMaxLevels];
  int lat_sbits[kDecodeMaxLevels];
  int lat_run_shift[kDecodeMaxLevels];
  int lat_local_bits;
  int lat_inner_bits;
  int lat_outer_bits;
  int lat_oshift[kDecodeMaxLevels];
  int mask_off[kDecodeMaxLevels];
  int mask_words[kDecodeMaxLevels];
  int mask_total;
  int olist_off[kDecodeMaxLevels];
  int olist_total;
  int dlist_off[kDecodeMaxLevels];
  int dlist_total;
  int atoms_per_unit_shift;
  int write_atoms;
  std::int64_t leaves;
  int total_rows;
  std::int64_t unit_ceiling;
  std::int64_t step_words;
};

struct DecodeLaunch {
  DecodeArm arm;
  DecodeArgs args;
  DecodeLayout layout;
};

inline constexpr std::size_t kMaxDecodeArms = 64;

//: THE WALK'S REGISTERS A THREAD: a warp holds a quarter of an atom's leaves a lane group in flight.
//: -- see docs/internals/decode/decode.md#decode-threads
inline constexpr int kDecodeRegsPerThread = 128;

//: THE CTA'S THREADS from a register file of `regs_per_sm`: the one function of the device's fact (`decode_args`) and of a build's
//: (`DecodePlan<B>::kThreads`)
constexpr int decode_threads(std::int64_t regs_per_sm) {
  return static_cast<int>(regs_per_sm / kDecodeRegsPerThread);
}

//: the digit sets the terms are built from (W, R, R&W, R&~W): each takes a level's owner grid of list words in the step's shared memory
inline constexpr int kDecodeListKinds = 4;

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
               && decode::Reader::sm_count(d).point() && decode::Reader::d_v(d).point()
               && decode::Reader::levels(d).point();
  for (int l = 0; l < kDecodeMaxLevels; ++l) {
    exact = exact && level_width_bits(d, l).point() && level_write_listed(d, l).point();
  }
  if (!exact) std::abort();
  const bool innermost_dense =
      level_write_listed(d, static_cast<int>(decode::Reader::levels(d).lo) - 1).lo == 0;
  DecodeArgs args{static_cast<int>(decode::Reader::cc(d).lo),
                  decode_threads(decode::Reader::regs_per_sm(d).lo),
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

//: WHAT A CALL NEEDS OF THE STEP'S RUN-TIME TILE, in 4-byte words: both sides' staged amplitudes (and the decay's) and the tables; the kernel's layout holds the rest
constexpr std::int64_t decode_step_words(int total_rows, bool decay, int mask_total,
                                         int olist_total, int dlist_total) {
  const std::int64_t amps = std::int64_t{total_rows} * (decay ? 3 : 2);
  const std::int64_t tables = 2 * std::int64_t{mask_total}
                              + kDecodeListKinds * std::int64_t{olist_total}
                              + std::int64_t{dlist_total};
  return amps + tables;
}

//: THE LAUNCH'S LAYOUT from the arm and its arguments: the lattice's box of the level widths (`lattice.h`) laid out as the step's parameters
//: carry it
constexpr DecodeLayout decode_layout(const DecodeArm& arm, const DecodeArgs& args) {
  const int levels = arm.levels;
  const auto box = lattice_solve<kDecodeMaxLevels>(args.widths, levels);
  DecodeLayout out{};
  out.lat_k = box.k;
  out.lat_m = static_cast<int>(box.m);

  int run_shift = 0;
  int gsuf = 1;
  int rows = 0;
  std::int64_t leaves = 1;
  for (int l = 0; l < levels; ++l) {
    out.lat_s[l] = static_cast<int>(box.s[l]);
    out.lat_g[l] = static_cast<int>(box.g[l]);
    out.lat_sbits[l] = lattice_log2(box.s[l]);
    out.lat_local_bits += out.lat_sbits[l];
    leaves *= args.widths[l];
    rows += args.widths[l];
  }

  for (int l = levels - 1; l >= 0; --l) {
    out.lat_gsuf[l] = gsuf;
    out.lat_run_shift[l] = run_shift;
    gsuf *= out.lat_g[l];
    run_shift += out.lat_sbits[l];
  }

  out.lat_inner_bits = out.lat_sbits[levels - 1];
  out.lat_outer_bits = out.lat_local_bits - out.lat_inner_bits;
  for (int l = 0; l < levels; ++l) {
    out.lat_oshift[l] = l < levels - 1 ? out.lat_run_shift[l] - out.lat_inner_bits : 0;
  }

  out.atoms_per_unit_shift =
      out.lat_inner_bits > kLatticeAtomShift ? out.lat_inner_bits - kLatticeAtomShift : 0;

  for (int l = 0; l < levels; ++l) {
    out.mask_off[l] = out.mask_total;
    out.mask_words[l] = (args.widths[l] + 31) / 32;
    out.mask_total += out.mask_words[l];
    out.olist_off[l] = out.olist_total;
    out.olist_total += out.lat_g[l];
    out.dlist_off[l] = out.dlist_total;
    if (l < levels - 1) out.dlist_total += args.widths[l];
  }

  out.write_atoms = (args.write_dense != 0 && out.lat_inner_bits >= kLatticeAtomShift) ? 1 : 0;
  out.leaves = leaves;
  out.total_rows = rows;
  out.unit_ceiling = leaves / out.lat_s[levels - 1];
  out.step_words =
      decode_step_words(rows, arm.decay != 0, out.mask_total, out.olist_total, out.dlist_total);
  return out;
}

//: the refusal of a step that needs more of the run-time tile than the kernel's layout holds on the card (`capacity`, in words)
inline std::string decode_smem_refusal(const DecodeLaunch& launch, std::int64_t capacity) {
  std::string out = "the decode step's staged operands need ";
  out += std::to_string(launch.layout.step_words);
  out += " words of shared memory against the ";
  out += std::to_string(capacity);
  out += " its layout holds on this card; sum_l width_l = ";
  out += std::to_string(launch.layout.total_rows);
  out += ". This is a declared capacity, refused loudly rather than taken as a slower path.";
  return out;
}

constexpr DecodeLaunch decode_launch(const DecodeDescriptor& d) {
  const DecodeArm arm = decode_arm(d);
  const DecodeArgs args = decode_args(d, arm);
  return {arm, args, decode_layout(arm, args)};
}

//: whether a launch's arguments are the descriptor's own: its device facts, the arm's padding and each level's width
constexpr bool decode_args_reflect(const DecodeDescriptor& d, const DecodeLaunch& launch) {
  const DecodeArgs& a = launch.args;
  bool ok = a.cc == decode::Reader::cc(d).lo && a.sms == decode::Reader::sm_count(d).lo
            && a.threads == decode_threads(decode::Reader::regs_per_sm(d).lo)
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
