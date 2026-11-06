// csrc/rola/src/carry/carry_arm.cuh -- THE LAUNCH of one arm: the grid the carve implies, the shared memory its plan
// sizes, and the census the device build reports. Included only by the generated per-arm translation units.
// see docs/internals/carry/carry_arm.md
#pragma once

#include "carry/carry_arm_abi.cuh"
#include "carry/carry_kernel.cuh"
#include "common/arm.cuh"

namespace rola::carry {

template <class P>
constexpr CarryForm form_of_plan() {
  return CarryForm{P::kSmemBytes, P::kArea, P::kReadRow, P::kWriteRow};
}

//: THE PLAN'S FORM on the current device's card: the plan the device code was compiled with is the card's, and the
//: kernel traps at entry on any other launched size. -- see docs/internals/carry/carry_arm.md#plan
//: THE RUNNING DEVICE'S COMPUTE CAPABILITY (x10), the key every per-card choice is read at on the host.
inline int current_cc() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major * 100 + minor * 10;
}

template <class Arm>
inline CarryForm carry_form() {
  // R9-CASES: the tabulated compute capabilities (common/hardware.cuh)
  switch (current_cc()) {
    case 800:
      return form_of_plan<Plan<Arm, 800>>();
    case 860:
      return form_of_plan<Plan<Arm, 860>>();
    case 870:
      return form_of_plan<Plan<Arm, 870>>();
    case 890:
      return form_of_plan<Plan<Arm, 890>>();
    default:
      return CarryForm{};
  }
}

template <class Arm, int Cc>
inline cudaError_t carry_launch_on(const CarryParams& p0, int boxes, int bh, cudaStream_t stream) {
  using P = Plan<Arm, Cc>;
  constexpr int bytes = P::kSmemBytes;
  //: the call's read tables on the device's card, and an area that holds the least plan
  CarryParams p = p0;
  p.tables = tables_of<P::kWindow>(p0.tables.heads, bytes);
  if (!least_plan_fits<P>(p.tables.area, p.reduce)) return cudaErrorInvalidConfiguration;
  const int classes = (p.read_lists.groups > 0 ? p.read_lists.groups : 1)
                      * (p.write_lists.groups > 0 ? p.write_lists.groups : 1);
  window_plan_kernel<Arm>
      <<<dim3((unsigned)classes, (unsigned)bh), (unsigned)kPlanThreads, 0, stream>>>(p);
  const auto fn = carry_kernel<Arm>;
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
inline cudaError_t carry_launch(const CarryParams& p, int boxes, int bh, cudaStream_t stream) {
  // R9-CASES: the tabulated compute capabilities (common/hardware.cuh)
  switch (current_cc()) {
    case 800:
      return carry_launch_on<Arm, 800>(p, boxes, bh, stream);
    case 860:
      return carry_launch_on<Arm, 860>(p, boxes, bh, stream);
    case 870:
      return carry_launch_on<Arm, 870>(p, boxes, bh, stream);
    case 890:
      return carry_launch_on<Arm, 890>(p, boxes, bh, stream);
    default:
      return cudaErrorInvalidDevice;
  }
}

template <class Arm>
inline ArmCensus carry_census() {
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, carry_kernel<Arm>);
  return ArmCensus{(int)a.numRegs, (int)a.localSizeBytes, (int)a.maxThreadsPerBlock,
                   (int)a.binaryVersion, carry_form<Arm>()};
}

#define ROLA_CARRY_ARM_DEFINE(I_, FR_, FF_, RR_, RF_, CH_, SS_)                                      \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int boxes, int bh, cudaStream_t stream) { \
    return carry_launch<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, SS_>>(p, boxes, bh, stream);        \
  }                                                                                                  \
  ArmCensus carry_arm_census_##I_() { return carry_census<::rola::Arm<I_, FR_, FF_, RR_, RF_, CH_, SS_>>(); }

}  // namespace rola::carry
