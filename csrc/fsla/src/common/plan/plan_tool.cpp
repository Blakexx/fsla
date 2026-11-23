// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PLANNER ON A SYNTHETIC ENVIRONMENT (docs/internals/common/plan/plan_tool.md)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "plan/decode_launch.h"
#include "plan/planner.h"
#include "plan/prefill_launch.h"

namespace {

using fsla::plan::Named;

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

//: the carve of a prefill call as JSON, from the facts given: the planner's verdict on the call (held,
//: owed with its kernels, or outside) is printed beside it, never a gate on it
void print_carve(const fsla::plan::Descriptor& device, const Named& call,
                 const std::string& model) {
  using namespace fsla::plan;
  Descriptor d = device;
  for (const auto& [name, value] : call) {
    if (!is_call(name) || is_set(name) || !d.assign(name, Interval{value, value})) {
      refuse("the call names '" + std::string(name) + "', which is no fact a call supplies");
    }
  }
  for (std::string_view name : kFactNames) {
    bool supplied = false;
    for (const auto& cell : call) supplied |= cell.first == name;
    if (is_call(name) && !supplied) {
      refuse("the call states no value for '" + std::string(name) + "'");
    }
  }
  if (!prefill_carves(d)) {
    std::printf("{\"carved\": false}\n");
    return;
  }
  const Planned verdict = describe(device, call, model);
  std::string kernels;
  for (const std::string& k : verdict.owed) {
    kernels += (kernels.empty() ? "\"" : ", \"") + k + "\"";
  }
  std::string held = "owed";
  if (verdict.ok()) held = "built";
  if (!verdict.ok() && verdict.owed.empty()) held = "outside the contract";
  const PrefillCarve c = prefill_carve(d);
  const auto list = [&](const int* v) {
    std::string out = "[";
    for (int i = 0; i < c.levels; ++i) out += (i ? ", " : "") + std::to_string(v[i]);
    return out + "]";
  };
  const auto line = [](const std::string& key, const std::string& value, bool last = false) {
    std::printf(" \"%s\": %s%s\n", key.c_str(), value.c_str(), last ? "}" : ",");
  };
  const auto n = [](std::int64_t v) { return std::to_string(v); };
  std::printf("{\"carved\": true,\n");
  line("kernels", "\"" + held + "\"");
  line("owed", "[" + kernels + "]");
  line("levels", n(c.levels));
  line("box_bits", list(c.box_bits));
  line("read_list_bits", list(c.read_list_bits));
  line("write_list_bits", list(c.write_list_bits));
  line("read_list_level", n(c.read_list_level));
  line("write_list_level", n(c.write_list_level));
  line("pair_level", n(c.pair_level));
  line("split_level", n(c.split_level));
  line("boxes_per_cta", n(c.boxes_per_cta));
  line("fold_lists", n(c.fold_lists));
  line("fold_columns", n(c.fold_columns));
  line("fold_outer", n(c.fold_outer));
  line("boxes_per_head", n(c.boxes_per_head));
  line("ctas_per_head", n(c.ctas_per_head));
  line("units_per_head", n(c.units_per_head));
  line("unit_ctas", n(c.unit_ctas));
  line("window", n(c.window));
  line("windows", n(c.windows), true);
}

}  // namespace

int main(int argc, char** argv) {
  std::string device_text, call_text, model, grid_text, carve_text;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--device") {
      device_text = argv[i + 1];
    } else if (flag == "--call") {
      call_text = argv[i + 1];
    } else if (flag == "--model") {
      model = argv[i + 1];
    } else if (flag == "--carve") {
      carve_text = argv[i + 1];
    } else if (flag == "--grid") {
      grid_text = argv[i + 1];
    } else {
      refuse("unknown argument " + flag);
    }
  }
  if (argc % 2 == 0 || device_text.empty() || call_text.empty()) {
    refuse(
        "usage: plan_tool --device name=v,... --call name=v,... [--model text] [--grid ...] "
        "[--carve json]");
  }

  std::vector<std::string> keep;
  keep.reserve(64);
  const auto device = fsla::plan::device_descriptor(parse(device_text, keep), model);
  if (!device.ok()) refuse(device.refusal);
  const bool carve = carve_text == "json";
  if (!carve_text.empty() && !carve) refuse("--carve takes json");
  const Named call_facts = parse(call_text, keep);
  if (carve) {
    print_carve(device.descriptor, call_facts, model);
    return 0;
  }
  const auto call = fsla::plan::describe(device.descriptor, call_facts, model);
  if (!call.ok()) refuse(call.refusal);
  const auto launch = fsla::plan::decode_launch(call.descriptor);
  const auto& arm = launch.arm;
  const auto& args = launch.args;

  std::printf("descriptor: %s\n", fsla::plan::spell(call.descriptor).c_str());
  std::printf("arm: d_v=%d levels=%d decay=%d\n", arm.d_v, arm.levels, arm.decay);
  std::printf("args: cc=%d threads=%d smem_budget=%d pad=%d sms=%d\n", args.cc, args.threads,
              args.smem_budget, args.pad, args.sms);

  std::string widths, offsets;
  for (int l = 0; l < arm.levels; ++l) {
    widths += (l ? "," : "") + std::to_string(args.widths[l]);
    offsets += (l ? "," : "") + std::to_string(args.row_offsets[l]);
  }
  std::printf("levels: widths=%s row_offsets=%s write_dense=%d\n", widths.c_str(), offsets.c_str(),
              args.write_dense);

  if (!grid_text.empty()) {
    const Named grid = parse(grid_text, keep);
    if (grid.size() != 3) refuse("--grid is unit_ceiling=n,bh=n,forced=n");
    const int ctas = fsla::plan::decode_grid(args, grid[0].second, grid[1].second, grid[2].second);
    std::printf("grid: %d\n", ctas);
  }
  return 0;
}
