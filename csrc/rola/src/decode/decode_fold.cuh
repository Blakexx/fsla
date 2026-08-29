#pragma once

// THE DECODE FOLD -- every per-token operand widened, normalized and staged in SHARED
// MEMORY by the CTA that is about to consume it.
//
// Three facts a reader must not miss:
//   1. THE FOLD IS A GATHER, NOT A COPY PASS. Its inputs are whatever VIEW the producer
//      handed out -- a level is ordinarily a slice of one packed plane -- so the fold
//      indexes by stride and the `[B, 1, H, W] -> [B * H, W]` fold is the address it
//      computes, never a materialized intermediate.
//   2. THE SUM'S ORDER IS DECLARED, not incidental: lane-strided ascending partials
//      folded by a 5-step `__shfl_down_sync` tree. A reduction whose order is a
//      property of whatever the library did that release is a reduction no gate can
//      hold to bit-identity.
//   3. IT IS A HEADER, NOT A KERNEL. The step folds its own operands and the growth
//      probe folds its own write side; between them there is no launch and no plane.
//
// The declared order, the dtype arms and the cast argument:
// docs/internals/decode/decode_fold.md

#include <cuda_bf16.h>

#include "decode.cuh"

namespace rola {
namespace decode {

__device__ __forceinline__ float widen(float x) { return x; }
__device__ __forceinline__ float widen(const __nv_bfloat16& x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

//: THE DECLARED SUM. Lane `j` accumulates `row[j], row[j + 32], ...` in ascending index
//: order into an fp32 register, then the five `__shfl_down_sync` steps fold lane `j`
//: into lane `j - off`. Lanes past the row's width contribute an exact `0.0f`.
//: docs/internals/decode/decode_fold.md#declared-order
template <typename T>
__device__ __forceinline__ float row_sum(const T* row, int64_t sw, int width, int lane) {
  float acc = 0.0f;
  #pragma unroll 1
  for (int k = lane; k < width; k += 32) acc += widen(row[k * sw]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  return acc;
}

//: THE ROW SUMS, for a side that carries its own mass. Warp `l < D` owns level `l`'s sum,
//: which is why `D <= kMaxLevels <= kDecodeWarps` is the condition that makes ONE barrier
//: after this enough. `s_sum` is `[kMaxLevels]` of the caller's shared memory.
//: NO BARRIER IS TAKEN HERE -- the caller places it, because a device function that
//: synchronizes on entry and exit costs a barrier at every seam between two of them.
template <typename T>
__device__ __forceinline__ void fold_row_sums(const DecodeParams& p, int b, int h,
                                              const Span* spans, const int* normalize,
                                              float* s_sum) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp < p.D && normalize[warp] != 0) {
    const Span& s = spans[warp];
    const float total = row_sum(row_of<T>(s, b, h), s.sw, p.level_width[warp], lane);
    if (lane == 0) s_sum[warp] = total;
  }
}

//: ONE SIDE'S `D` LEVELS into the flat `[sum_l b_l]` shared staging the walk indexes by
//: digit. `normalize` is `nullptr` for a side that never carries its own mass (the write
//: side, always); otherwise `s_sum` must already hold `fold_row_sums`' answer.
template <typename T>
__device__ __forceinline__ void fold_rows(const DecodeParams& p, int b, int h, const Span* spans,
                                          const int* normalize, const float* s_sum, float* out) {
  const int tid = threadIdx.x;
  #pragma unroll 1
  for (int l = 0; l < p.D; ++l) {
    const int width = p.level_width[l];
    const T* in = row_of<T>(spans[l], b, h);
    const int64_t sw = spans[l].sw;
    float* dst = out + p.level_amp_offset[l];
    if (normalize != nullptr && normalize[l] != 0) {
      const float total = s_sum[l];
      #pragma unroll 1
      for (int g = tid; g < width; g += kDecodeThreads) dst[g] = widen(in[g * sw]) / total;
    } else {
      #pragma unroll 1
      for (int g = tid; g < width; g += kDecodeThreads) dst[g] = widen(in[g * sw]);
    }
  }
}

//: The token's value row, resident for the whole walk.
template <typename T>
__device__ __forceinline__ void fold_value(const DecodeParams& p, int b, int h, int d_v,
                                           float* out) {
  const T* in = row_of<T>(p.v, b, h);
  const int64_t sw = p.v.sw;
  #pragma unroll 1
  for (int c = threadIdx.x; c < d_v; c += kDecodeThreads) out[c] = widen(in[c * sw]);
}

//: The write gain, a per-token scalar every thread of the CTA reads.
template <typename T>
__device__ __forceinline__ float fold_gain(const DecodeParams& p, int b, int h) {
  return widen(*row_of<T>(p.g_write, b, h));
}

}  // namespace decode
}  // namespace rola
