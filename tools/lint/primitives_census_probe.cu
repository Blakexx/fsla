// tools/lint/primitives_census_probe.cu -- one probe kernel per audited primitive of tools/lint/primitives_census.py, compiled by the
// graph's CUDA rule (tools/lint/BUILD.bazel's `primitives_census_probe`). Each calls the REAL `ops.cuh` entry point, never a
// hand-rolled restatement of what it should compile to, so a regression in the shipped primitive is exactly what trips the census.
#include <cuda_bf16.h>
#include "common/ops.cuh"

extern "C" __global__ void census_mul_bf16x2(uint32_t* out, uint32_t a, uint32_t b) {
  out[0] = denseref::ops::mul_bf16x2(a, b);
}

extern "C" __global__ void census_pack_bf16x2(uint32_t* out, float a, float b) {
  out[0] = denseref::ops::pack_bf16x2(a, b);
}

extern "C" __global__ void census_hi_bf16x2(uint32_t* out, float a, float b) {
  out[0] = denseref::ops::hi_bf16x2(a, b);
}
