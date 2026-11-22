// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// DECODE'S ARM GENERATOR (docs/internals/decode/arms_gen.md): a host tool the graph builds and
// runs. It evaluates decode's launch function over every context of the contract and writes the arm list
// the graph instantiates, and the selection record. A context the matcher
// maps to no arm, or a point that reaches several arms (or one a range does not reach), fails the
// run naming the context.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
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

//: THE POINT PROPERTY: a launch function over a point returns exactly one arm, and the arms of the points of a range are among the range's.
//: The points tried: every combination of the ranged facts' ends (each at its lo or its hi), and each ranged fact swept through every value
//: of its range (up to kSweep) with the others at their lo. A context with more ranged facts than kMaxRanged is refused: its corners are not tried.
constexpr std::size_t kMaxRanged = 8;
constexpr std::int64_t kSweep = 4096;

bool corners_agree(const Context& c, const fsla::plan::DecodeArmSet& reached) {
  std::size_t n = c.ranges.size();
  if (n > kMaxRanged) {
    std::fprintf(
        stderr,
        "arms_gen: context '%s' has %zu ranged facts; the point property is tried at most over "
        "%zu: state the facts as points or alternatives, or narrow the context\n",
        c.id.c_str(), n, kMaxRanged);
    return false;
  }

  std::vector<Facts> points;
  for (std::size_t mask = 0; mask < (std::size_t{1} << n); ++mask) {
    Facts f = c.facts;
    for (std::size_t i = 0; i < n; ++i) {
      std::int64_t v = (mask >> i) & 1 ? c.ranges[i].second.hi : c.ranges[i].second.lo;
      f.assign(c.ranges[i].first, Interval{v, v});
    }
    points.push_back(f);
  }

  for (std::size_t axis = 0; axis < n; ++axis) {
    Facts base = c.facts;
    for (std::size_t i = 0; i < n; ++i) {
      base.assign(c.ranges[i].first, Interval{c.ranges[i].second.lo, c.ranges[i].second.lo});
    }
    const Interval range = c.ranges[axis].second;
    for (std::int64_t v = range.lo; v <= range.hi && v < range.lo + kSweep; ++v) {
      Facts f = base;
      f.assign(c.ranges[axis].first, Interval{v, v});
      points.push_back(f);
    }
  }

  for (const Facts& f : points) {
    fsla::plan::DecodeArmSet point = fsla::plan::decode_arms(f);
    if (point.size() != 1) {
      std::fprintf(stderr, "arms_gen: context '%s': the point %s maps to %zu arms, not one\n",
                   c.id.c_str(), fsla::plan::decode_facts_read(f).c_str(), point.size());
      return false;
    }
    bool among = false;
    for (std::size_t i = 0; i < reached.size(); ++i) among |= reached.arms[i] == point.arms[0];
    if (!among) {
      std::fprintf(stderr,
                   "arms_gen: context '%s': the point %s reaches %s, which the context's range "
                   "does not\n",
                   c.id.c_str(), fsla::plan::decode_facts_read(f).c_str(),
                   spell(point.arms[0]).c_str());
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
  return d;
}

constexpr fsla::plan::DecodeLaunch kLaunch = fsla::plan::decode_launch(exact());
static_assert(kLaunch.arm == Arm{64, 2, 1});
static_assert(kLaunch.args.threads == 512 && kLaunch.args.smem_budget == 232448);
static_assert(kLaunch.args.pad == 16 && kLaunch.args.sms == 132 && kLaunch.args.cc == 900);

}  // namespace

int main(int argc, char** argv) {
  std::string contexts_path, out_dir;
  if (argc != 5) refuse("usage: arms_gen --contexts <file> --out-dir <dir>");
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--contexts") {
      contexts_path = argv[i + 1];
    } else if (flag == "--out-dir") {
      out_dir = argv[i + 1];
    } else {
      refuse("unknown argument " + flag);
    }
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
  int failures = 0;
  for (const Context& c : contexts) {
    fsla::plan::DecodeArmSet set = fsla::plan::decode_arms(c.facts);
    if (set.none()) {
      std::fprintf(stderr,
                   "arms_gen: the contract holds context '%s' (%s) and decode's launch function "
                   "(csrc/fsla/src/common/plan/decode_launch.h) maps it to NONE: add the arm "
                   "or narrow the contract (hardware/parts) in the same commit\n",
                   c.id.c_str(), fsla::plan::decode_facts_read(c.facts).c_str());
      ++failures;
    } else if (!corners_agree(c, set)) {
      ++failures;
    } else {
      std::vector<Arm> arms(set.arms, set.arms + set.size());
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
