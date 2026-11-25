// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PLANNER ON A SYNTHETIC ENVIRONMENT (docs/internals/common/plan/plan_tool.md)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "plan/arm_family.h"
#include "plan/cta_carve.h"
#include "plan/decode/contract.h"
#include "plan/decode_launch.h"
#include "plan/lattice.h"
#include "plan/mask_prepare/contract.h"
#include "plan/planner.h"
#include "plan/prefill/contract.h"
#include "plan/prefill_launch.h"

namespace {

using fsla::plan::Named;
using Decode = fsla::plan::decode::Kernel;
using Prefill = fsla::plan::prefill::Kernel;

[[noreturn]] void refuse(const std::string& why) {
  std::fprintf(stderr, "plan_tool: %s\n", why.c_str());
  std::exit(1);
}

Named parse(const std::string& text, std::vector<std::string>& keep) {
  Named out;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find(',', at);
    if (end == std::string::npos) end = text.size();
    std::string cell = text.substr(at, end - at);
    std::size_t eq = cell.find('=');
    if (eq == std::string::npos) refuse("'" + cell + "' is no name=value");
    keep.push_back(cell.substr(0, eq));
    out.emplace_back(std::string_view(keep.back()), std::stoll(cell.substr(eq + 1)));
    at = end + 1;
  }
  return out;
}

//: the numbers `values[0 .. levels)` as a comma list
template <typename T>
std::string list_of(const T* values, int levels) {
  std::string out;
  for (int l = 0; l < levels; ++l) out += (l ? "," : "") + std::to_string(values[l]);
  return out;
}

//: `values[0 .. levels)` as a JSON list
template <typename T>
std::string json_list(const T* values, int levels) {
  return "[" + list_of(values, levels) + "]";
}

//: the launch's host half of a prefill call as JSON members, `separator` between them (`window` and `windows` follow, the descriptor's)
void print_launch_fields(const fsla::plan::PrefillLaunch& p, const char* separator) {
  const auto n = [](std::int64_t v) { return std::to_string(v); };
  std::printf("\"levels\": %s%s", n(p.levels).c_str(), separator);
  std::printf("\"width_bits\": %s%s", json_list(p.width_bits, p.levels).c_str(), separator);
  std::printf("\"leaves\": %s%s", n(p.leaves).c_str(), separator);
  std::printf("\"cta_leaves\": %s%s", n(p.cta_leaves).c_str(), separator);
  std::printf("\"ctas_per_head\": %s", n(p.ctas_per_head).c_str());
}

//: the carve's fields from `levels` to `unit_ctas` as JSON members, `separator` between them
void print_carve_fields(const fsla::plan::PrefillCarve& c, const char* separator) {
  const auto list = [&](const int* v) { return json_list(v, c.levels); };
  const auto n = [](std::int64_t v) { return std::to_string(v); };
  bool first = true;
  const auto line = [&](const std::string& key, const std::string& value) {
    std::printf("%s\"%s\": %s", first ? "" : separator, key.c_str(), value.c_str());
    first = false;
  };
  line("levels", n(c.levels));
  line("width_bits", list(c.width_bits));
  line("write_density", list(c.write_density));
  line("read_density", list(c.read_density));
  line("write_group_bits", list(c.write_group_bits));
  line("read_group_bits", list(c.read_group_bits));
  line("write_order", list(c.write_order));
  line("read_order", list(c.read_order));
  line("carve_bits", list(c.carve_bits));
  line("cta_leaves", n(c.cta_leaves));
  line("split_level", n(c.split_level));
  line("folds", n(c.folds));
  line("fold_bits", list(c.fold_bits));
  line("fold_leaves", n(c.fold_leaves));
  line("fold_set_leaves", n(c.fold_set_leaves));
  line("fold_sets", n(c.fold_sets));
  line("read_set_leaves", n(c.read_set_leaves));
  line("read_sets", n(c.read_sets));
  line("unit_bits", list(c.unit_bits));
  line("unit_ctas", n(c.unit_ctas));
}

//: the launch of a prefill call as JSON, from the facts given: the planner's verdict on the call (held,
//: owed with its kernels, or outside) is printed beside it, never a gate on it
void print_launch(const fsla::plan::PrefillDescriptor& device, const Named& call,
                  const std::string& model) {
  using namespace fsla::plan;
  PrefillDescriptor d = device;
  for (const auto& [name, value] : call) {
    if (!Prefill::is_call(name) || Prefill::is_set(name)
        || !d.assign(name, Interval{value, value})) {
      refuse("the call names '" + std::string(name) + "', which is no fact a call supplies");
    }
  }
  for (std::string_view name : Prefill::fact_names) {
    bool supplied = false;
    for (const auto& cell : call) supplied |= cell.first == name;
    if (Prefill::is_call(name) && !supplied) {
      refuse("the call states no value for '" + std::string(name) + "'");
    }
  }
  const Planned<Prefill> verdict = describe<Prefill>(device, call, model);
  std::string kernels;
  for (const std::string& k : verdict.owed) {
    kernels += (kernels.empty() ? "\"" : ", \"") + k + "\"";
  }
  std::string held = "owed";
  if (verdict.ok()) held = "built";
  if (!verdict.ok() && verdict.owed.empty()) held = "outside the contract";
  const PrefillLaunch p = prefill_launch(d);
  std::printf("{\"kernels\": \"%s\",\n \"owed\": [%s],\n ", held.c_str(), kernels.c_str());
  print_launch_fields(p, ",\n ");
  std::printf(",\n \"window\": %d,\n \"windows\": %lld}\n", p.window,
              static_cast<long long>(p.windows));
}

void print_box(int k, long long m, long long bc, long long owners, const std::string& spans,
               const std::string& grids) {
  std::printf("lattice: k=%d m=%lld bc=%lld owners=%lld spans=%s grids=%s\n", k, m, bc, owners,
              spans.c_str(), grids.c_str());
}

//: THE LATTICE of level widths `text` (a comma list): the box the search picks, and the law's constants
void print_lattice(const std::string& text) {
  using namespace fsla::plan;
  int widths[kDecodeMaxLevels] = {};
  int count = 0;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find(',', at);
    if (end == std::string::npos) end = text.size();
    if (count == kDecodeMaxLevels) {
      refuse("--lattice takes at most " + std::to_string(kDecodeMaxLevels) + " widths");
    }
    widths[count++] = std::stoi(text.substr(at, end - at));
    at = end + 1;
  }
  if (count == 0) refuse("--lattice takes the level widths, a comma list");
  const auto box = lattice_solve<kDecodeMaxLevels>(widths, count);
  print_box(box.k, box.m, box.bc, box.owners, list_of(box.s, count), list_of(box.g, count));
  std::printf("constants: atom_leaves=%d cap_bc=%d max_span=%d max_span_k=%d\n", kLatticeAtomLeaves,
              kLatticeCapBc, kLatticeMaxSpan, kLatticeMaxSpanK);
}

//: THE LAUNCH'S LAYOUT: the lattice's box, the unit's bits, the tables' sizes and the step's derived sizes
void print_layout(const fsla::plan::DecodeLaunch& launch) {
  const auto& layout = launch.layout;
  const int levels = launch.arm.levels;
  print_box(layout.lat_k, layout.lat_m, 1LL << layout.lat_local_bits,
            layout.leaves >> layout.lat_local_bits, list_of(layout.lat_s, levels),
            list_of(layout.lat_g, levels));
  std::printf("unit: local_bits=%d inner_bits=%d outer_bits=%d\n", layout.lat_local_bits,
              layout.lat_inner_bits, layout.lat_outer_bits);
  std::printf("tables: mask_total=%d olist_total=%d dlist_total=%d write_atoms=%d\n",
              layout.mask_total, layout.olist_total, layout.dlist_total, layout.write_atoms);
  std::printf("step: leaves=%lld total_rows=%d unit_ceiling=%lld step_words=%lld\n",
              static_cast<long long>(layout.leaves), layout.total_rows,
              static_cast<long long>(layout.unit_ceiling),
              static_cast<long long>(layout.step_words));
}

//: THE DECODE KERNEL ON A DEVICE AND A CALL: the descriptor, the arm and the launch's arguments, and the grid when `--grid` names its inputs
void print_decode(const Named& device_facts, const Named& call_facts, const std::string& model,
                  const std::string& grid_text, std::vector<std::string>& keep) {
  const auto device = fsla::plan::device_descriptor<Decode>(device_facts, model);
  if (!device.ok()) refuse(device.refusal);
  const auto call = fsla::plan::describe<Decode>(device.descriptor, call_facts, model);
  if (!call.ok()) refuse(call.refusal);
  const auto launch = fsla::plan::decode_launch(call.descriptor);
  const auto& arm = launch.arm;
  const auto& args = launch.args;

  std::printf("descriptor: %s\n", fsla::plan::spell<Decode>(call.descriptor).c_str());
  std::printf("arm: d_v=%d levels=%d decay=%d\n", arm.d_v, arm.levels, arm.decay);
  std::printf("args: cc=%d threads=%d pad=%d sms=%d\n", args.cc, args.threads, args.pad, args.sms);

  std::string widths, offsets;
  for (int l = 0; l < arm.levels; ++l) {
    widths += (l ? "," : "") + std::to_string(args.widths[l]);
    offsets += (l ? "," : "") + std::to_string(args.row_offsets[l]);
  }
  std::printf("levels: widths=%s row_offsets=%s write_dense=%d\n", widths.c_str(), offsets.c_str(),
              args.write_dense);
  print_layout(launch);

  if (!grid_text.empty()) {
    const Named grid = parse(grid_text, keep);
    if (grid.size() != 3) refuse("--grid is unit_ceiling=n,bh=n,forced=n");
    const int ctas = fsla::plan::decode_grid(args, grid[0].second, grid[1].second, grid[2].second);
    std::printf("grid: %d\n", ctas);
  }
}

//: THE WIDTHS OF A LAUNCH'S LEVELS, a comma list
std::vector<std::int64_t> parse_widths(const std::string& text) {
  std::vector<std::int64_t> out;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find(',', at);
    if (end == std::string::npos) end = text.size();
    out.push_back(std::stoll(text.substr(at, end - at)));
    at = end + 1;
  }
  return out;
}

//: THE CARVE of one call: the levels' width bits and each side's densities (comma lists of one length), as one JSON object of the carve's fields and the launch's counts of its widths
void print_cta_carve(const std::string& widths_text, const std::string& write_text,
                     const std::string& read_text) {
  using namespace fsla::plan;
  const auto bits = [&](const std::string& text, const char* flag) {
    const std::vector<std::int64_t> v = parse_widths(text);
    if (v.empty() || v.size() > kCarveMaxLevels) {
      refuse(std::string(flag) + " takes 1 to " + std::to_string(kCarveMaxLevels)
             + " numbers, a comma list");
    }
    return v;
  };
  const auto width = bits(widths_text, "--cta-carve");
  const auto write = bits(write_text, "--write-density");
  const auto read = bits(read_text, "--read-density");
  if (write.size() != width.size() || read.size() != width.size()) {
    refuse("--cta-carve, --write-density and --read-density take one number a level");
  }

  const int levels = static_cast<int>(width.size());
  int w[kCarveMaxLevels] = {};
  int density_write[kCarveMaxLevels] = {};
  int density_read[kCarveMaxLevels] = {};
  for (int l = 0; l < levels; ++l) {
    w[l] = static_cast<int>(width[l]);
    density_write[l] = static_cast<int>(write[l]);
    density_read[l] = static_cast<int>(read[l]);
  }
  //: the carve is total (a density is clamped into 1..256); the CTAs a head are the launch's, `units_per_head` their quotient by the carve's unit CTAs
  const PrefillCarve c = prefill_carve(levels, w, density_write, density_read);
  const PrefillLaunch p = prefill_launch_of(c.width_bits, c.levels);
  std::printf("{");
  print_carve_fields(c, ", ");
  std::printf(", \"ctas_per_head\": %lld, \"units_per_head\": %lld}\n",
              static_cast<long long>(p.ctas_per_head),
              static_cast<long long>(p.ctas_per_head / c.unit_ctas));
}

//: ONE ENTMAX KERNEL ON A DEVICE AND A CALL: the descriptor, the arm and the launch; `widths` (the levels' widths of the launch) are held to the launch's one width class
template <typename K>
void print_entmax(const Named& device_facts, const Named& call_facts, const std::string& model,
                  const std::string& widths_text) {
  using Family = fsla::plan::ArmFamily<K>;
  const auto device = fsla::plan::device_descriptor<K>(device_facts, model);
  if (!device.ok()) refuse(device.refusal);
  const auto call = fsla::plan::describe<K>(device.descriptor, call_facts, model);
  if (!call.ok()) refuse(call.refusal);
  const auto launch = fsla::plan::entmax_launch<K>(call.descriptor);
  const auto key = Family::key(launch.arm);

  std::printf("descriptor: %s\n", fsla::plan::spell<K>(call.descriptor).c_str());
  std::string arm;
  for (std::size_t i = 0; i < key.size(); ++i) {
    arm += (i ? " " : "") + std::string(Family::fields[i]) + "=" + std::to_string(key[i]);
  }
  std::printf("arm: %s\n", arm.c_str());
  std::printf("launch: klass=%d levels=%d\n", launch.klass, launch.levels);
  if (!widths_text.empty()) {
    const std::string refusal =
        fsla::plan::entmax_class_refusal(parse_widths(widths_text), launch.klass);
    if (!refusal.empty()) refuse(refusal);
    std::printf("class: one\n");
  }
}

//: THE FOUR ENTMAX KERNELS: `run` is called with the definition of the one named `kernel`, if it is one of them
template <typename Run>
bool with_entmax_kernel(const std::string& kernel, Run run) {
  using namespace fsla::plan;
  const auto each = [&](auto k) {
    if (kernel != decltype(k)::name) return false;
    run(k);
    return true;
  };
  return each(factor_forward::Kernel{}) || each(factor_backward::Kernel{})
         || each(softmax_forward::Kernel{}) || each(softmax_backward::Kernel{});
}

bool kernel_is_entmax(const std::string& kernel) {
  return with_entmax_kernel(kernel, [](auto) {});
}

void print_entmax_kernel(const std::string& kernel, const Named& device_facts,
                         const Named& call_facts, const std::string& model,
                         const std::string& widths_text) {
  with_entmax_kernel(kernel, [&](auto k) {
    print_entmax<decltype(k)>(device_facts, call_facts, model, widths_text);
  });
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel, device_text, call_text, model, grid_text, launch_text, lattice_text,
      widths_text, cta_carve_text, write_density_text, read_density_text;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--lattice") {
      lattice_text = argv[i + 1];
    } else if (flag == "--cta-carve") {
      cta_carve_text = argv[i + 1];
    } else if (flag == "--write-density") {
      write_density_text = argv[i + 1];
    } else if (flag == "--read-density") {
      read_density_text = argv[i + 1];
    } else if (flag == "--kernel") {
      kernel = argv[i + 1];
    } else if (flag == "--device") {
      device_text = argv[i + 1];
    } else if (flag == "--call") {
      call_text = argv[i + 1];
    } else if (flag == "--model") {
      model = argv[i + 1];
    } else if (flag == "--launch") {
      launch_text = argv[i + 1];
    } else if (flag == "--grid") {
      grid_text = argv[i + 1];
    } else if (flag == "--widths") {
      widths_text = argv[i + 1];
    } else {
      refuse("unknown argument " + flag);
    }
  }
  const bool launch_given = !kernel.empty() && !device_text.empty() && !call_text.empty();
  if (argc % 2 == 0 || (lattice_text.empty() && cta_carve_text.empty() && !launch_given)) {
    refuse(
        "usage: plan_tool --kernel decode|prefill|mask_prepare|<entmax kernel> --device name=v,... --call name=v,... [--model text] "
        "[--grid ...] (decode) | [--launch json] (prefill: the descriptor, or the launch's host half) | [--widths w,...] (an entmax kernel: the launch's "
        "levels' widths; --call takes several calls, `;` between them) | plan_tool --lattice w1,w2,... (the lattice of level widths, no device) | "
        "plan_tool --cta-carve width_bits,... --write-density w,... --read-density w,... (the carve of a call's widths and densities, no device)");
  }
  if (!cta_carve_text.empty()) {
    if (!lattice_text.empty() || !kernel.empty() || !device_text.empty() || !call_text.empty()) {
      refuse("--cta-carve stands alone");
    }
    //: calls, `;` between them: one line of JSON each
    const auto calls = [](const std::string& text) {
      std::vector<std::string> out;
      std::size_t at = 0;
      while (at <= text.size()) {
        std::size_t end = text.find(';', at);
        if (end == std::string::npos) end = text.size();
        out.push_back(text.substr(at, end - at));
        at = end + 1;
      }
      return out;
    };
    const auto widths = calls(cta_carve_text);
    const auto writes = calls(write_density_text);
    const auto reads = calls(read_density_text);
    if (writes.size() != widths.size() || reads.size() != widths.size()) {
      refuse("--cta-carve, --write-density and --read-density take the same number of calls");
    }
    for (std::size_t i = 0; i < widths.size(); ++i) print_cta_carve(widths[i], writes[i], reads[i]);
    return 0;
  }
  if (!lattice_text.empty()) {
    if (!kernel.empty() || !device_text.empty() || !call_text.empty()) {
      refuse("--lattice stands alone");
    }
    print_lattice(lattice_text);
    return 0;
  }

  std::vector<std::string> keep;
  keep.reserve(64);
  const Named device_facts = parse(device_text, keep);
  const std::size_t device_names = keep.size();
  if (kernel_is_entmax(kernel)) {
    //: an entmax kernel's calls, `;` between them: each printed in turn, the first refused ends the run
    std::size_t at = 0;
    while (at <= call_text.size()) {
      std::size_t end = call_text.find(';', at);
      if (end == std::string::npos) end = call_text.size();
      keep.resize(device_names);
      print_entmax_kernel(kernel, device_facts, parse(call_text.substr(at, end - at), keep), model,
                          widths_text);
      at = end + 1;
    }
    return 0;
  }
  const Named call_facts = parse(call_text, keep);
  if (kernel == "decode") {
    if (!launch_text.empty()) refuse("--launch is the prefill's");
    print_decode(device_facts, call_facts, model, grid_text, keep);
  } else if (kernel == "prefill") {
    const auto device = fsla::plan::device_descriptor<Prefill>(device_facts, model);
    if (!device.ok()) refuse(device.refusal);
    if (!grid_text.empty()) refuse("--grid is decode's");
    if (!launch_text.empty()) {
      if (launch_text != "json") refuse("--launch takes json");
      print_launch(device.descriptor, call_facts, model);
      return 0;
    }
    const auto call = fsla::plan::describe<Prefill>(device.descriptor, call_facts, model);
    if (!call.ok()) refuse(call.refusal);
    std::printf("descriptor: %s\n", fsla::plan::spell<Prefill>(call.descriptor).c_str());
  } else if (kernel == fsla::plan::mask_prepare::Kernel::name) {
    using MaskPrepare = fsla::plan::mask_prepare::Kernel;
    if (!grid_text.empty() || !launch_text.empty()) {
      refuse("--grid is decode's and --launch the prefill's");
    }
    const auto device = fsla::plan::device_descriptor<MaskPrepare>(device_facts, model);
    if (!device.ok()) refuse(device.refusal);
    const auto call = fsla::plan::describe<MaskPrepare>(device.descriptor, call_facts, model);
    if (!call.ok()) refuse(call.refusal);
    std::printf("descriptor: %s\n", fsla::plan::spell<MaskPrepare>(call.descriptor).c_str());
  } else {
    refuse("no kernel '" + kernel
           + "': the kernels are decode, prefill, mask_prepare and the entmax kernels");
  }
  return 0;
}
