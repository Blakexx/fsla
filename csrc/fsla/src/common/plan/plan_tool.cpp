// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PLANNER ON A SYNTHETIC ENVIRONMENT (docs/internals/common/plan/plan_tool.md)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "plan/decode_launch.h"
#include "plan/planner.h"

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

}  // namespace

int main(int argc, char** argv) {
  std::string device_text, call_text, model, grid_text;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--device") {
      device_text = argv[i + 1];
    } else if (flag == "--call") {
      call_text = argv[i + 1];
    } else if (flag == "--model") {
      model = argv[i + 1];
    } else if (flag == "--grid") {
      grid_text = argv[i + 1];
    } else {
      refuse("unknown argument " + flag);
    }
  }
  if (argc % 2 == 0 || device_text.empty() || call_text.empty()) {
    refuse("usage: plan_tool --device name=v,... --call name=v,... [--model text] [--grid ...]");
  }

  std::vector<std::string> keep;
  keep.reserve(64);
  const auto device = fsla::plan::device_descriptor(parse(device_text, keep), model);
  if (!device.ok()) refuse(device.refusal);
  const auto call = fsla::plan::describe(device.descriptor, parse(call_text, keep), model);
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
