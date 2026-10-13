// csrc/rola/src/carry/carry_arm.cuh -- THE LAUNCH of one arm: the grid the carve implies, the shared memory its plan
// sizes, and the census the device build reports. Included only by the generated per-arm translation units.
// see docs/internals/carry/carry_arm.md
#pragma once

#include "carry/carry_arm_abi.cuh"
#include "carry/carry_kernel.cuh"
#include "common/arm.cuh"

namespace rola::carry {

//: THE PLAN'S SHARED BYTES on the current device's card: the plan the device code was compiled with is the card's,
//: and the kernel traps at entry on any other launched size. -- see docs/internals/carry/carry_arm.md#plan
template <class Arm>
inline int carry_smem_bytes() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  switch (major * 100 + minor * 10) {
    case 800:
      return Plan<Arm, 800>::kSmemBytes;
    case 860:
      return Plan<Arm, 860>::kSmemBytes;
    case 870:
      return Plan<Arm, 870>::kSmemBytes;
    case 890:
      return Plan<Arm, 890>::kSmemBytes;
    default:
      return 0;
  }
}

//: THE GRID: one CTA a (box, batch-head), the CTA the SM. -- see docs/internals/carry/carry_arm.md#launch
template <class Arm>
inline cudaError_t carry_launch(const CarryParams& p, int boxes, int bh, cudaStream_t stream) {
  const auto fn = carry_kernel<Arm>;
  const int bytes = carry_smem_bytes<Arm>();
  if (bytes == 0) return cudaErrorInvalidDevice;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  fn<<<dim3((unsigned)boxes, (unsigned)bh), (unsigned)kThreads, bytes, stream>>>(p);
  return cudaGetLastError();
}

template <class Arm>
inline ArmCensus carry_census() {
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, carry_kernel<Arm>);
  return ArmCensus{(int)a.numRegs, (int)a.localSizeBytes, carry_smem_bytes<Arm>(),
                   (int)a.maxThreadsPerBlock, (int)a.binaryVersion};
}

#define ROLA_CARRY_ARM_DEFINE(I_, FR_, FF_, RR_, RF_, CH_, PR_)                                      \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream) { \
    return carry_launch<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, PR_>>(p, boxes, bh, stream);        \
  }                                                                                                  \
  ArmCensus carry_arm_census_##I_() { return carry_census<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, PR_>>(); }

}  // namespace rola::carry
