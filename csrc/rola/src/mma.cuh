#pragma once

// Bare tensor-core PTX primitives (m16n8k16 bf16, m16n8k8 tf32x3, and the bf16x2
// pack). No RoLA semantics -- these are the instruction wrappers `readout_mma.cuh`
// and `consumer_kernel.cuh` issue. (The `gram` intra lowering, the third historical
// caller, was deleted with the old consumer at item 6b.)
//
// PROVENANCE: this file was `p0_shared_gram_mma.cuh`, the only V2 header the V3
// consumer `#include`s. The `rola-v3` purge deleted the rest of the V2 kernel
// family; this one is load-bearing (consumer.cu fails to compile without it),
// so it was renamed into the V3 namespace rather than deleted or left carrying a
// `p0_` prefix that no longer names anything in the tree.

#include <cuda_bf16.h>

namespace rola::mma {

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
  union {
    __nv_bfloat162 h;
    unsigned u;
  } packed;
  packed.h = __floats2bfloat162_rn(lo, hi);
  return packed.u;
}

// D = A(m16xk16, bf16) * B(k16xn8, bf16) + C, fp32 accumulate.
__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1, float c0, float c1, float c2,
                                             float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ unsigned f2tf32(float value) {
  unsigned result;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float tf32_round(float value) { return __uint_as_float(f2tf32(value)); }

__device__ __forceinline__ void mma_m16n8k8_tf32(float& d0, float& d1, float& d2, float& d3,
                                                 float a0, float a1, float a2, float a3, float b0,
                                                 float b1, float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(f2tf32(a0)), "r"(f2tf32(a1)), "r"(f2tf32(a2)), "r"(f2tf32(a3)), "r"(f2tf32(b0)),
        "r"(f2tf32(b1)), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_m16n8k8_tf32x2(float& d0, float& d1, float& d2, float& d3,
                                                   float a0, float a1, float a2, float a3, float b0,
                                                   float b1, float c0, float c1, float c2,
                                                   float c3) {
  const float a0_hi = tf32_round(a0);
  const float a1_hi = tf32_round(a1);
  const float a2_hi = tf32_round(a2);
  const float a3_hi = tf32_round(a3);
  mma_m16n8k8_tf32(d0, d1, d2, d3, a0_hi, a1_hi, a2_hi, a3_hi, b0, b1, c0, c1, c2, c3);
  mma_m16n8k8_tf32(d0, d1, d2, d3, a0 - a0_hi, a1 - a1_hi, a2 - a2_hi, a3 - a3_hi, b0, b1, d0, d1,
                   d2, d3);
}

__device__ __forceinline__ void mma_m16n8k8_tf32x3(float& d0, float& d1, float& d2, float& d3,
                                                   float a0, float a1, float a2, float a3, float b0,
                                                   float b1, float c0, float c1, float c2,
                                                   float c3) {
  const float a0_hi = tf32_round(a0);
  const float a1_hi = tf32_round(a1);
  const float a2_hi = tf32_round(a2);
  const float a3_hi = tf32_round(a3);
  const float b0_hi = tf32_round(b0);
  const float b1_hi = tf32_round(b1);
  mma_m16n8k8_tf32(d0, d1, d2, d3, a0_hi, a1_hi, a2_hi, a3_hi, b0_hi, b1_hi, c0, c1, c2, c3);
  mma_m16n8k8_tf32(d0, d1, d2, d3, a0_hi, a1_hi, a2_hi, a3_hi, b0 - b0_hi, b1 - b1_hi, d0, d1, d2,
                   d3);
  mma_m16n8k8_tf32(d0, d1, d2, d3, a0 - a0_hi, a1 - a1_hi, a2 - a2_hi, a3 - a3_hi, b0_hi, b1_hi, d0,
                   d1, d2, d3);
}

}  // namespace rola::mma
