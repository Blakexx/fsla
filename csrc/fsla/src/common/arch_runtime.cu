// the runtime half of the closed-world arch rule (R4): docs/internals/common/arch_runtime.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include "common/arch_caps.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"

namespace fsla {

//: the current device's card admitted (`common/select.cu`) -- see docs/internals/common/arch_runtime.md#check-arch-table
void check_arch_table() { (void)current_card(); }

}  // namespace fsla
