// csrc/rola/src/carry/carry_arm_abi.cuh -- ONE ARM'S ABI: pointers and scalars only, so a per-arm translation unit
// never parses the torch headers the seam needs. see docs/internals/carry/carry_arm_abi.md
#pragma once

#include <cuda_runtime.h>

#include "carry/carry.cuh"

namespace rola::carry {

//: ONE ARM'S CENSUS, as the device build reports it. -- see docs/internals/carry/carry_arm_abi.md#census
struct ArmCensus {
  int regs, local_bytes, smem_bytes, max_threads, binary_version;
};

#define ROLA_CARRY_ARM_DECLARE(I_, FR_, FF_, RR_, RF_, CH_, PR_)                                       \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream); \
  ArmCensus carry_arm_census_##I_();

}  // namespace rola::carry
