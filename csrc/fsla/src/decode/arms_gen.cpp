// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// DECODE'S ARM GENERATOR (docs/internals/decode/arms_gen.md): a host tool the graph builds and
// runs. It evaluates decode's launch function over every context of the contract and writes the arm list
// the graph instantiates, and the selection record. A context the matcher
// maps to no arm (some point of its ranges reaches none), fails the
// run naming the context.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "plan/decode_launch.h"
#include "plan/facts.h"
#include "plan/rules.h"

namespace {

using fsla::plan::DecodeArm;
using Arm = DecodeArm;
using fsla::plan::EnumSet;
using fsla::plan::Facts;
using fsla::plan::Interval;

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream in(s);
  for (std::string cell; std::getline(in, cell, sep);) out.push_back(cell);
  return out;
}

struct Context {
  std::string id;
  Facts facts{};
  std::vector<std::pair<std::string, Interval>> ranges;  // the int facts that are not points
};

[[noreturn]] void refuse(const std::string& why) {
  std::fprintf(stderr, "arms_gen: %s\n", why.c_str());
  std::exit(1);
}

EnumSet parse_set(const Context& c, const std::string& name, const std::string& value) {
  EnumSet set{0};
  for (const std::string& m : split(value.substr(1, value.size() - 2), ',')) {
    int index = fsla::plan::member_of(name, m);
    if (index < 0) refuse("context '" + c.id + "': " + name + " has no member '" + m + "'");
    set.bits |= std::uint64_t{1} << index;
  }
  return set;
}

Interval parse_range(const Context& c, const std::string& name, const std::string& value) {
  std::size_t colon = value.find(':');
  if (colon == std::string::npos) {
    refuse("context '" + c.id + "': " + name + "='" + value + "' is not lo:hi");
  }
  return {std::stoll(value.substr(0, colon)), std::stoll(value.substr(colon + 1))};
}

Context parse(const std::string& line) {
  std::vector<std::string> cells = split(line, '\t');
  Context c;
  c.id = cells.at(0);
  std::set<std::string> seen;
  for (std::size_t i = 1; i < cells.size(); ++i) {
    std::size_t eq = cells[i].find('=');
    if (eq == std::string::npos) {
      refuse("context '" + c.id + "': '" + cells[i] + "' is no fact=value");
    }
    std::string name = cells[i].substr(0, eq), value = cells[i].substr(eq + 1);
    bool ok;
    if (fsla::plan::is_set(name)) {
      ok = c.facts.assign(name, parse_set(c, name, value));
    } else {
      Interval range = parse_range(c, name, value);
      if (!range.point()) c.ranges.emplace_back(name, range);
      ok = c.facts.assign(name, range);
    }
    if (!ok) refuse("context '" + c.id + "': the schema names no fact '" + name + "'");
    seen.insert(name);
  }
  for (std::string_view name : fsla::plan::kFactNames) {
    if (!seen.count(std::string(name))) {
      std::string missing(name);
      refuse("context '" + c.id + "' states no value for the schema's fact '" + missing + "'");
    }
  }
  return c;
}

std::string spell(const Arm& a) {
  std::ostringstream out;
  out << "(d_v=" << a.d_v << ", levels=" << a.levels << ", decay=" << a.decay << ")";
  return out.str();
}

std::string key_of(const Arm& a) {
  std::ostringstream out;
  out << "[" << a.d_v << ", " << a.levels << ", " << a.decay << "]";
  return out.str();
}

//: THE COVERAGE CHECK, exact and without sampling: `decode_reach` walks the arm half's decision tree over the context's ranges, following both
//: branches of a Maybe, so it reaches every arm some point of the ranges reaches and the NONE leaf exactly when some point reaches no arm
//: (docs/internals/common/plan/rules.md#exact). The count of ranged facts does not matter.
bool reaches_a_gap(const Facts& f) {
  fsla::plan::DecodeReach r = fsla::plan::decode_reach(f);
  return r.gap || r.arms.none();
}

//: a point of the context that reaches no arm, found by halving each ranged fact toward the half where the walk still reaches the NONE leaf
//: (one such half always exists: the walk is exact)
Facts gap_witness(const Context& c) {
  Facts f = c.facts;
  for (const auto& [name, range] : c.ranges) {
    std::int64_t lo = range.lo, hi = range.hi;
    while (lo < hi) {
      std::int64_t mid = lo + (hi - lo) / 2;
      Facts left = f;
      left.assign(name, Interval{lo, mid});
      if (reaches_a_gap(left)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    f.assign(name, Interval{lo, lo});
  }
  return f;
}

//: THE SANITY CHECK of the argument (`--sample N --seed S`, tests/unit/test_decode_arms_gen.py): N seeded random points inside each context, each
//: of which must reach exactly one arm (`decode_arm`, the same walk over a point) among the context's reach
bool sampled_points_agree(const Context& c, const fsla::plan::DecodeReach& reached,
                          std::mt19937_64& rng, std::size_t samples) {
  for (std::size_t i = 0; i < samples; ++i) {
    Facts f = c.facts;
    for (const auto& [name, range] : c.ranges) {
      std::int64_t v = std::uniform_int_distribution<std::int64_t>(range.lo, range.hi)(rng);
      f.assign(name, Interval{v, v});
    }
    Arm arm = fsla::plan::decode_arm(f);
    bool among = false;
    for (std::size_t k = 0; k < reached.arms.size(); ++k) among |= reached.arms.arms[k] == arm;
    if (!among) {
      std::fprintf(stderr,
                   "arms_gen: context '%s': the point %s reaches %s, which the context's reach "
                   "does not hold\n",
                   c.id.c_str(), fsla::plan::decode_facts_read(f).c_str(), spell(arm).c_str());
      return false;
    }
  }
  return true;
}

void write_rows(const std::string& path, const std::vector<Arm>& list) {
  std::ofstream rows(path);
  rows << "{\"family\": \"decode\", \"key\": [\"d_v\", \"levels\", \"decay\"], \"arms\": [";
  for (std::size_t i = 0; i < list.size(); ++i) rows << (i ? ", " : "") << key_of(list[i]);
  rows << "]}\n";
}

//: the launch function of an exact descriptor, proved at compile time: one arm and its arguments
constexpr fsla::plan::Descriptor exact() {
  fsla::plan::Descriptor d;
  d.assign("cc", Interval{900, 900});
  d.assign("regs_per_sm", Interval{65536, 65536});
  d.assign("smem_per_cta_optin", Interval{232448, 232448});
  d.assign("sm_count", Interval{132, 132});
  d.assign("d_v", Interval{48, 48});
  d.assign("levels", Interval{2, 2});
  d.assign("decay", Interval{1, 1});
  d.assign("write_dense", Interval{1, 1});
  d.assign("width_bits_1", Interval{4, 4});
  d.assign("width_bits_2", Interval{5, 5});
  return d;
}

constexpr fsla::plan::DecodeLaunch kLaunch = fsla::plan::decode_launch(exact());
static_assert(kLaunch.arm == Arm{64, 2, 1});
static_assert(kLaunch.args.threads == 512 && kLaunch.args.smem_budget == 232448);
static_assert(kLaunch.args.pad == 16 && kLaunch.args.sms == 132 && kLaunch.args.cc == 900);
static_assert(kLaunch.args.widths[0] == 16 && kLaunch.args.widths[1] == 32
              && kLaunch.args.widths[2] == 0);
static_assert(kLaunch.args.row_offsets[0] == 0 && kLaunch.args.row_offsets[1] == 16
              && kLaunch.args.write_dense == 1);

}  // namespace

int main(int argc, char** argv) {
  std::string contexts_path, out_dir;
  std::size_t samples = 0;
  std::uint64_t seed = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--contexts") {
      contexts_path = argv[i + 1];
    } else if (flag == "--out-dir") {
      out_dir = argv[i + 1];
    } else if (flag == "--sample") {
      samples = std::stoull(argv[i + 1]);
    } else if (flag == "--seed") {
      seed = std::stoull(argv[i + 1]);
    } else {
      refuse("unknown argument " + flag);
    }
  }
  if (argc % 2 == 0 || contexts_path.empty() || out_dir.empty()) {
    refuse("usage: arms_gen --contexts <file> --out-dir <dir> [--sample <points> --seed <n>]");
  }

  std::ifstream in(contexts_path);
  if (!in) refuse("cannot read " + contexts_path);
  std::vector<Context> contexts;
  std::string contract = "full";  // the first line: `# contract: <name>`
  for (std::string line; std::getline(in, line);) {
    constexpr std::string_view kNamed = "# contract: ";
    if (line.rfind(kNamed, 0) == 0) {
      contract = line.substr(kNamed.size());
    } else if (!line.empty()) {
      contexts.push_back(parse(line));
    }
  }
  if (contexts.empty()) refuse(contexts_path + " holds no context: nothing is supported");

  std::set<Arm> built;
  std::vector<std::pair<std::string, std::vector<Arm>>> reach;
  std::mt19937_64 rng(seed);
  int failures = 0;
  for (const Context& c : contexts) {
    fsla::plan::DecodeReach set = fsla::plan::decode_reach(c.facts);
    if (set.gap || set.arms.none()) {
      std::string maps = "it";
      if (!c.ranges.empty()) {
        maps = "the point " + fsla::plan::decode_facts_read(gap_witness(c)) + " of it";
      }
      std::fprintf(stderr,
                   "arms_gen: the contract holds context '%s' (%s) and decode's launch function "
                   "(csrc/fsla/src/common/plan/decode_launch.h) maps %s to NONE: add the arm "
                   "or narrow the contract (hardware/parts) in the same commit\n",
                   c.id.c_str(), fsla::plan::decode_facts_read(c.facts).c_str(), maps.c_str());
      ++failures;
    } else if (samples && !sampled_points_agree(c, set, rng, samples)) {
      ++failures;
    } else {
      std::vector<Arm> arms(set.arms.arms, set.arms.arms + set.arms.size());
      built.insert(arms.begin(), arms.end());
      reach.emplace_back(c.id, arms);
    }
  }
  if (failures) {
    std::fprintf(stderr, "arms_gen: %d of %zu contexts refused\n", failures, contexts.size());
    return 1;
  }

  std::vector<Arm> list(built.begin(), built.end());  // std::set order is the canonical order
  std::map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < list.size(); ++i) index[spell(list[i])] = i;
  write_rows(out_dir + "/decode_arm_rows.json", list);
  std::ofstream record(out_dir + "/decode_selection_record.txt");
  record << "# GENERATED by csrc/fsla/src/decode/arms_gen.cpp from the " << contract
         << " contract: the contract's context, then the arm index and key it reaches\n";
  for (const auto& [id, arms] : reach) {
    record << id << "\t->";
    for (const Arm& a : arms) record << " " << index[spell(a)] << spell(a);
    record << "\n";
  }
  std::printf("arms_gen: the %s contract's %zu contexts reach %zu decode arms\n", contract.c_str(),
              contexts.size(), list.size());
  return 0;
}
