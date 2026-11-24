// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE ARM GENERATOR (docs/internals/common/plan/arms_gen.md): a host tool the graph builds and
// runs, once for each arm family. It evaluates the family's matcher over every context of the family's
// contract and writes the arm list the graph instantiates, and the selection record. A context the matcher
// maps to no arm (some point of its ranges reaches none), fails the run naming the context.
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

#include "plan/arm_family.h"
#include "plan/rules.h"

namespace {

using fsla::plan::EnumSet;
using fsla::plan::Interval;

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream in(s);
  for (std::string cell; std::getline(in, cell, sep);) out.push_back(cell);
  return out;
}

[[noreturn]] void refuse(const std::string& why) {
  std::fprintf(stderr, "arms_gen: %s\n", why.c_str());
  std::exit(1);
}

struct Options {
  std::string contexts_path;
  std::string out_dir;
  std::size_t samples = 0;
  std::uint64_t seed = 0;
};

//: ONE FAMILY `K`'s run: its contexts parsed, its matcher evaluated over each, its arms and record written
template <typename K>
class Generator {
 public:
  using Family = fsla::plan::ArmFamily<K>;
  using Facts = typename Family::Facts;
  using Arm = typename Family::Arm;

  struct Context {
    std::string id;
    Facts facts{};
    std::vector<std::pair<std::string, Interval>> ranges;  // the int facts that are not points
  };

  static EnumSet parse_set(const Context& c, const std::string& name, const std::string& value) {
    EnumSet set{0};
    for (const std::string& m : split(value.substr(1, value.size() - 2), ',')) {
      int index = K::member_of(name, m);
      if (index < 0) refuse("context '" + c.id + "': " + name + " has no member '" + m + "'");
      set.bits |= std::uint64_t{1} << index;
    }
    return set;
  }

  static Interval parse_range(const Context& c, const std::string& name, const std::string& value) {
    std::size_t colon = value.find(':');
    if (colon == std::string::npos) {
      refuse("context '" + c.id + "': " + name + "='" + value + "' is not lo:hi");
    }
    return {std::stoll(value.substr(0, colon)), std::stoll(value.substr(colon + 1))};
  }

  static Context parse(const std::string& line) {
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
      if (K::is_set(name)) {
        ok = c.facts.assign(name, parse_set(c, name, value));
      } else {
        Interval range = parse_range(c, name, value);
        if (!range.point()) c.ranges.emplace_back(name, range);
        ok = c.facts.assign(name, range);
      }
      if (!ok) refuse("context '" + c.id + "': the schema names no fact '" + name + "'");
      seen.insert(name);
    }
    for (std::string_view name : K::fact_names) {
      if (!seen.count(std::string(name))) {
        std::string missing(name);
        refuse("context '" + c.id + "' states no value for the schema's fact '" + missing + "'");
      }
    }
    return c;
  }

  static std::string spell(const Arm& a) {
    const auto key = Family::key(a);
    std::string out = "(";
    for (std::size_t i = 0; i < key.size(); ++i) {
      out += (i ? ", " : "") + std::string(Family::fields[i]) + "=" + std::to_string(key[i]);
    }
    return out + ")";
  }

  static std::string key_of(const Arm& a) {
    const auto key = Family::key(a);
    std::string out = "[";
    for (std::size_t i = 0; i < key.size(); ++i) out += (i ? ", " : "") + std::to_string(key[i]);
    return out + "]";
  }

  //: THE COVERAGE CHECK, exact and without sampling: the matcher's reach walks the arm half's decision tree over the context's ranges, following both
  //: branches of a Maybe, so it reaches every arm some point of the ranges reaches and the NONE leaf exactly when some point reaches no arm
  //: (docs/internals/common/plan/rules.md#exact). The count of ranged facts does not matter.
  static bool reaches_a_gap(const Facts& f) {
    typename Family::Reach r = Family::reach(f);
    return r.gap || r.arms.none();
  }

  //: a point of the context that reaches no arm, found by halving each ranged fact toward the half where the walk still reaches the NONE leaf
  //: (one such half always exists: the walk is exact)
  static Facts gap_witness(const Context& c) {
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
  //: of which must reach exactly one arm (the matcher's arm over a point) among the context's reach
  static bool sampled_points_agree(const Context& c, const typename Family::Reach& reached,
                                   std::mt19937_64& rng, std::size_t samples) {
    for (std::size_t i = 0; i < samples; ++i) {
      Facts f = c.facts;
      for (const auto& [name, range] : c.ranges) {
        std::int64_t v = std::uniform_int_distribution<std::int64_t>(range.lo, range.hi)(rng);
        f.assign(name, Interval{v, v});
      }
      Arm arm = Family::arm(f);
      bool among = false;
      for (std::size_t k = 0; k < reached.arms.size(); ++k) among |= reached.arms.arms[k] == arm;
      if (!among) {
        std::fprintf(stderr,
                     "arms_gen: context '%s': the point %s reaches %s, which the context's reach "
                     "does not hold\n",
                     c.id.c_str(), Family::read(f).c_str(), spell(arm).c_str());
        return false;
      }
    }
    return true;
  }

  static void write_rows(const std::string& path, const std::vector<Arm>& list) {
    std::ofstream rows(path);
    rows << "{\"family\": \"" << K::name << "\", \"key\": [";
    for (std::size_t i = 0; i < Family::fields.size(); ++i) {
      rows << (i ? ", " : "") << "\"" << Family::fields[i] << "\"";
    }
    rows << "], \"arms\": [";
    for (std::size_t i = 0; i < list.size(); ++i) rows << (i ? ", " : "") << key_of(list[i]);
    rows << "]}\n";
  }

  //: THE SAMPLE DEVICE AND CALL of a context: each of its facts a point, the context's own where it holds a point and the upper end where it holds a range
  static Facts sample_of(const Context& c) {
    Facts f = c.facts;
    for (const auto& [name, range] : c.ranges) f.assign(name, Interval{range.hi, range.hi});
    return f;
  }

  //: THE LAUNCH FUNCTION ON THE SAMPLE: its arguments are the sample's own facts
  static bool launch_reflects_the_sample(const Context& c) {
    const Facts f = sample_of(c);
    if (Family::launch_reflects(f)) return true;
    std::fprintf(stderr,
                 "arms_gen: context '%s': the launch function's arguments for the point %s are not "
                 "that point's facts\n",
                 c.id.c_str(), Family::read(f).c_str());
    return false;
  }

  static int run(const Options& o, const std::vector<std::string>& lines,
                 const std::string& contract) {
    std::vector<Context> contexts;
    for (const std::string& line : lines) contexts.push_back(parse(line));
    if (contexts.empty()) refuse(o.contexts_path + " holds no context: nothing is supported");

    const std::string name(K::name);
    std::set<Arm> built;
    std::vector<std::pair<std::string, std::vector<Arm>>> reach;
    std::mt19937_64 rng(o.seed);
    int failures = 0;
    for (const Context& c : contexts) {
      typename Family::Reach set = Family::reach(c.facts);
      if (set.gap || set.arms.none()) {
        std::string maps = "it";
        if (!c.ranges.empty()) {
          maps = "the point " + Family::read(gap_witness(c)) + " of it";
        }
        std::fprintf(stderr,
                     "arms_gen: the contract holds context '%s' (%s) and %s's launch function "
                     "(%s) maps %s to NONE: add the arm "
                     "or narrow the contract (hardware/kernels/%s) in the same commit\n",
                     c.id.c_str(), Family::read(c.facts).c_str(), name.c_str(), Family::matcher,
                     maps.c_str(), name.c_str());
        ++failures;
      } else if (o.samples && !sampled_points_agree(c, set, rng, o.samples)) {
        ++failures;
      } else if (!launch_reflects_the_sample(c)) {
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
    write_rows(o.out_dir + "/" + name + "_arm_rows.json", list);
    std::ofstream record(o.out_dir + "/" + name + "_selection_record.txt");
    record << "# GENERATED by csrc/fsla/src/common/plan/arms_gen.cpp from the " << contract
           << " contract: the contract's context, then the arm index and key it reaches\n";
    for (const auto& [id, arms] : reach) {
      record << id << "\t->";
      for (const Arm& a : arms) record << " " << index[spell(a)] << spell(a);
      record << "\n";
    }
    std::printf("arms_gen: the %s contract's %zu contexts reach %zu %s arms\n", contract.c_str(),
                contexts.size(), list.size(), name.c_str());
    return 0;
  }
};

//: THE FAMILIES the generator serves: `run` is called with the definition of the kernel the contract file names
template <typename Run>
int with_family(const std::string& kernel, Run run) {
  using namespace fsla::plan;
  if (kernel == decode::Kernel::name) return run(decode::Kernel{});
  if (kernel == factor_forward::Kernel::name) return run(factor_forward::Kernel{});
  if (kernel == factor_backward::Kernel::name) return run(factor_backward::Kernel{});
  if (kernel == softmax_forward::Kernel::name) return run(softmax_forward::Kernel{});
  if (kernel == softmax_backward::Kernel::name) return run(softmax_backward::Kernel{});
  refuse("no family '" + kernel + "': the families are decode, "
         "factor_forward, factor_backward, softmax_forward, softmax_backward");
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--contexts") {
      o.contexts_path = argv[i + 1];
    } else if (flag == "--out-dir") {
      o.out_dir = argv[i + 1];
    } else if (flag == "--sample") {
      o.samples = std::stoull(argv[i + 1]);
    } else if (flag == "--seed") {
      o.seed = std::stoull(argv[i + 1]);
    } else {
      refuse("unknown argument " + flag);
    }
  }
  if (argc % 2 == 0 || o.contexts_path.empty() || o.out_dir.empty()) {
    refuse("usage: arms_gen --contexts <file> --out-dir <dir> [--sample <points> --seed <n>]");
  }

  std::ifstream in(o.contexts_path);
  if (!in) refuse("cannot read " + o.contexts_path);
  std::vector<std::string> lines;
  std::string contract = "full";  // the first lines: `# contract: <name>`, `# kernel: <name>`
  std::string kernel;
  for (std::string line; std::getline(in, line);) {
    constexpr std::string_view kNamed = "# contract: ";
    constexpr std::string_view kKernel = "# kernel: ";
    if (line.rfind(kNamed, 0) == 0) {
      contract = line.substr(kNamed.size());
    } else if (line.rfind(kKernel, 0) == 0) {
      kernel = line.substr(kKernel.size());
    } else if (!line.empty()) {
      lines.push_back(line);
    }
  }
  if (kernel.empty()) refuse(o.contexts_path + " does not name its kernel (`# kernel: <name>`)");
  return with_family(kernel,
                     [&](auto k) { return Generator<decltype(k)>::run(o, lines, contract); });
}
