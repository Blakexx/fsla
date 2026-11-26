#pragma once

// THE PLANNER'S HOST-FACING DECLARATIONS, as `csrc/fsla/fsla_api.cpp` binds them;
// see docs/internals/common/plan/environment.md#entries

#include <cstdint>
#include <string>
#include <vector>

namespace fsla {
namespace plan {

std::string fsla_call_facts(std::string kernel);

std::vector<int64_t> fsla_descriptor(std::string kernel, int64_t device, std::vector<int64_t> call);

std::string fsla_owed_kernels(std::string kernel, int64_t device, std::vector<int64_t> call);

std::vector<int64_t> fsla_segment_plan(int64_t length, int64_t maximum, int64_t device);

std::vector<int64_t> fsla_step_bytes(std::vector<int64_t> widths, std::vector<int64_t> masked,
                                     std::vector<int64_t> heads, int64_t value_width,
                                     int64_t tokens, int64_t device);

int64_t fsla_step_tokens(std::vector<int64_t> widths, std::vector<int64_t> masked,
                         std::vector<int64_t> heads, int64_t value_width, int64_t limit,
                         int64_t device);

}  // namespace plan
}  // namespace fsla
