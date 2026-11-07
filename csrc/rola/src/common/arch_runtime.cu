// csrc/rola/src/common/arch_runtime.cu -- the runtime half of the closed-world arch rule (setup.py's R4), the one
// definition every launching entry calls first. See docs/internals/common/arch_runtime.md

#include "common/torch_seam.cuh"
#include <cuda_runtime.h>

#include <cstring>
#include <map>
#include <mutex>

#include "common/arch_caps.cuh"
#include "common/arch_runtime.cuh"
#include "common/select.cuh"

namespace rola {

//: the current device's card admitted (`common/select.cu`) -- see docs/internals/common/arch_runtime.md#check-arch-table
void check_arch_table() { (void)current_card(); }

//: -- see docs/internals/common/arch_runtime.md#mma-quanta
const long long* mma_quanta(int device) {
  static std::mutex mu;
  static std::map<int, const long long*> resolved;
  const std::lock_guard<std::mutex> hold(mu);
  const auto at = resolved.find(device);
  if (at != resolved.end()) return at->second;
  cudaDeviceProp prop{};
  ROLA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int cc = prop.major * 100 + prop.minor * 10;
  const long long* out = nullptr;
  for (const auto& row : denseref::arch::kMmaRows) {
    if (row.cc != cc) continue;
    if (row.device == nullptr && out == nullptr) out = row.mma;
    if (row.device != nullptr && std::strcmp(row.device, prop.name) == 0) {
      out = row.mma;
      break;
    }
  }
  resolved[device] = out;
  return out;
}

}  // namespace rola
