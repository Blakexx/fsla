// The fixture arm family's arm header (tests/BUILD.bazel's `fixture` family): one kernel instantiation an arm, defined by
// the macro bazel/arms.bzl's generated unit calls under the arm's selection guard. See docs/build-graph.md#arms and
// tests/unit/test_arm_family.py.
#pragma once

namespace fixture {

template <int Index, int DV, int Levels, int Decay>
__global__ void arm_kernel(int* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = Index + DV + Levels + Decay;
}

}  // namespace fixture

#define ROLA_FIXTURE_ARM_DEFINE(I, DV, LV, DC) \
  template __global__ void fixture::arm_kernel<I, DV, LV, DC>(int*);
