// csrc/rola/src/carry/carry_arm.cuh -- THE LAUNCH of one arm: the grid the box plan
// implies, the shared block the ledger sizes, and the census the device build reports.
// Included ONLY by the generated per-arm translation units.
// See docs/internals/carry/carry.md
#pragma once

#include "carry/box.cuh"
#include "carry/carry_arm_abi.cuh"
#include "carry/carry_kernel.cuh"

namespace rola::carry {

//: THE PARAMETER BLOCK REACHES THE KERNEL AS A GRID CONSTANT: the body reads its fields
//: through a reference, and only `__grid_constant__` lets that reference address the
//: constant bank instead of forcing a per-thread local copy of the whole block.
//: THE GRID: one CTA per (box, batch-head), and the CTA is the SM.
//: -- see docs/internals/carry/carry.md#launch
//: ONE GRAIN PAIR, ONE SYMBOL. -- see docs/internals/carry/carry_kernel.md#forms
template <int D, int DV, int W, int GR, int GW, int RF>
inline cudaError_t carry_launch_grain(const CarryParams& p, int owners, int bh,
                                      cudaStream_t stream) {
  using BP = BoxPlan<D, DV, W>;
  const auto fn = carry_kernel<D, DV, W, GR, GW, RF>;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, BP::kSmemBytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  fn<<<dim3((unsigned)owners, (unsigned)bh), (unsigned)BP::kThreads, BP::kSmemBytes, stream>>>(p);
  return cudaGetLastError();
}

template <int D, int DV, int W>
inline cudaError_t carry_launch(const CarryParams& p, int owners, int bh, cudaStream_t stream) {
  using BP = BoxPlan<D, DV, W>;
  cudaError_t e = cudaErrorInvalidValue;
  rola::static_for<BP::kGrainCases>([&](auto Rc) {
    rola::static_for<BP::kGrainCases>([&](auto Wc) {
      rola::static_for<kReadForms>([&](auto Fc) {
        constexpr int gr = BP::kGrainBitsMin + decltype(Rc)::value;
        constexpr int gw = BP::kGrainBitsMin + decltype(Wc)::value;
        constexpr int rf = decltype(Fc)::value;
        if (p.readout_grain_bits == gr && p.fold_grain_bits == gw && p.read_form == rf)
          e = carry_launch_grain<D, DV, W, gr, gw, rf>(p, owners, bh, stream);
      });
    });
  });
  return e;
}

template <int D, int DV, int W>
inline ArmCensus carry_census() {
  using BP = BoxPlan<D, DV, W>;
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a,
                        carry_kernel<D, DV, W, BP::kGrainBitsMin + BP::kGrainCases - 1,
                                     BP::kGrainBitsMin + BP::kGrainCases - 1, kReadFormStreams>);
  return ArmCensus{(int)a.numRegs,
                   0,
                   0,
                   (int)a.localSizeBytes,
                   BP::kSmemBytes,
                   (int)a.maxThreadsPerBlock,
                   (int)a.binaryVersion,
                   BP::kThreads,
                   BP::kBC,
                   BP::kAtoms,
                   BP::kAccRegs,
                   BP::kStateElems,
                   BP::kStateElems / 32,
                   sub_boxes(D, BP::kBC, BP::kStreams).count};
}

#define ROLA_CARRY_ARM_DEFINE(I_, D_, DV_, W_)                                       \
  cudaError_t carry_arm_forward_##I_(const CarryParams& p, int owners, int bh,        \
                                     cudaStream_t stream) {                           \
    return carry_launch<D_, DV_, W_>(p, owners, bh, stream);                          \
  }                                                                                   \
  ArmCensus carry_arm_census_##I_() { return carry_census<D_, DV_, W_>(); }

}  // namespace rola::carry
