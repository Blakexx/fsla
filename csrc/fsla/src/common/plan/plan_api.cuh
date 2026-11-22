#pragma once

// THE PLANNER'S HOST-FACING DECLARATIONS, as `csrc/fsla/fsla_api.cpp` binds them;
// see docs/internals/common/plan/environment.md#entries

#include <cstdint>
#include <string>
#include <vector>

namespace fsla {
namespace plan {

std::string fsla_call_facts();

std::vector<int64_t> fsla_descriptor(int64_t device, std::vector<int64_t> call);

std::string fsla_owed_kernels(int64_t device, std::vector<int64_t> call);

}  // namespace plan
}  // namespace fsla
