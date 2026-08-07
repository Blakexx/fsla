#pragma once

// Bare tensor-core PTX primitives (m16n8k16 bf16 and the bf16x2 pack). No RoLA
// semantics -- these are the instruction wrappers `readout_mma.cuh` and
// `consumer_kernel.cuh` issue.
//
// bf16 is the ONLY storage format (the one precision story: bf16 storage, fp32
// accumulation, fp64 reference as oracle); the tf32x3 extended-precision wrappers
// died with the PREC axis (revival = git).

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

}  // namespace rola::mma
