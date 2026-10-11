// csrc/rola/src/carry/carry_arm.cuh -- THE LAUNCH of one arm: the grid the carve implies, the shared memory its plan
// sizes, and the census the device build reports. Included only by the generated per-arm translation units.
// see docs/internals/carry/carry_arm.md
#pragma once

#include "carry/carry_arm_abi.cuh"
#include "carry/carry_kernel.cuh"
#include "common/arm_switch.cuh"

namespace rola::carry {

//: THE GRID: one CTA a (box, batch-head), the CTA the SM. -- see docs/internals/carry/carry_arm.md#launch
template <class Arm>
inline cudaError_t carry_launch(const CarryParams& p, int boxes, int bh, cudaStream_t stream) {
  using P = Plan<Arm>;
  const auto fn = carry_kernel<Arm>;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, P::kSmemBytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  fn<<<dim3((unsigned)boxes, (unsigned)bh), (unsigned)kThreads, P::kSmemBytes, stream>>>(p);
  return cudaGetLastError();
}

template <class Arm>
inline ArmCensus carry_census() {
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, carry_kernel<Arm>);
  return ArmCensus{(int)a.numRegs, (int)a.localSizeBytes, Plan<Arm>::kSmemBytes,
                   (int)a.maxThreadsPerBlock, (int)a.binaryVersion};
}

#define ROLA_CARRY_ARM_DEFINE(I_, FR_, FF_, RR_, RF_, CH_, PR_)                                      \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream) { \
    return carry_launch<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, PR_>>(p, boxes, bh, stream);        \
  }                                                                                                  \
  ArmCensus carry_arm_census_##I_() { return carry_census<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, PR_>>(); }

}  // namespace rola::carry
