#pragma once

// THE DECODE FOLD -- every per-token operand, read from the internal format, normalized and staged in
// SHARED MEMORY by the CTA that is about to consume it.
// see docs/internals/decode/decode_fold.md#note-l3

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "decode.cuh"

namespace rola {
namespace decode {

__device__ __forceinline__ float widen(float x) { return x; }

__device__ __forceinline__ float widen(const __nv_bfloat16& x) { return __bfloat162float(x); }

//: A stored coefficient digit's raw value: u8 as its integer, f16 widened. -- see docs/internals/decode/decode_fold.md#raw
__device__ __forceinline__ float raw(uint8_t x) { return static_cast<float>(x); }

__device__ __forceinline__ float raw(const __half& x) { return __half2float(x); }

//: The raw value of a row's max: the digit's full scale.
template <typename T>
__device__ __forceinline__ float full_scale();

template <>
__device__ __forceinline__ float full_scale<uint8_t>() {
  return 255.0f;
}

template <>
__device__ __forceinline__ float full_scale<__half>() {
  return 1.0f;
}

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

__device__ __forceinline__ float scale_of(const Span& s, int b, int h) {
  return *row_of<float>(s, b, h);
}

//: THE DECLARED SUM of a row's raw values. Lane `j` accumulates -- see docs/internals/decode/decode_fold.md#row-sum
template <typename T>
__device__ __forceinline__ float row_sum(const T* row, int64_t sw, int width, int lane) {
  float acc = 0.0f;
#pragma unroll 1
  for (int k = lane; k < width; k += 32) acc += raw(row[k * sw]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  return acc;
}

//: THE ROW SUMS, for a side that carries its own mass. Warp `l < D` owns level -- see docs/internals/decode/decode_fold.md#fold-row-sums
template <typename T>
__device__ __forceinline__ void fold_row_sums(const DecodeParams& p, int b, int h, const Span* rows,
                                              const int* normalize, float* s_sum) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp < p.D && normalize[warp] != 0) {
    const Span& s = rows[warp];
    const float total = row_sum(row_of<T>(s, b, h), s.sw, p.level_width[warp], lane);
    if (lane == 0) s_sum[warp] = total;
  }
}

//: ONE SIDE'S `D` LEVELS into the flat `[sum_l b_l]` staging the walk indexes by: `raw / sum` normalized, else `raw * scale / full scale` -- see docs/internals/decode/decode_fold.md#near-line-53
template <typename T>
__device__ __forceinline__ void fold_rows(const DecodeParams& p, int b, int h, const Span* rows,
                                          const Span* scales, const int* normalize,
                                          const float* s_sum, float* out) {
  const int tid = threadIdx.x;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) {
    const int width = p.level_width[l];
    const T* in = row_of<T>(rows[l], b, h);
    const int64_t sw = rows[l].sw;
    float* dst = out + p.level_amp_offset[l];
    if (normalize != nullptr && normalize[l] != 0) {
      const float total = s_sum[l];
#pragma unroll 1
      for (int g = tid; g < width; g += kDecodeThreads) dst[g] = raw(in[g * sw]) / total;
    } else {
      const float unit = scale_of(scales[l], b, h) / full_scale<T>();
#pragma unroll 1
      for (int g = tid; g < width; g += kDecodeThreads) dst[g] = raw(in[g * sw]) * unit;
    }
  }
}

//: The token's value row from its digits, resident for the whole walk. -- see docs/internals/decode/decode_fold.md#fold-value
__device__ __forceinline__ void fold_value(const DecodeParams& p, int b, int h, int d_v,
                                           float* out) {
  const int8_t* hi = row_of<int8_t>(p.v, b, h);
  const uint8_t* lo = p.v_lo.base != nullptr ? row_of<uint8_t>(p.v_lo, b, h) : nullptr;
  const float unit_q = scale_of(p.v_scale, b, h) / (lo != nullptr ? 32767.0f : 127.0f);
#pragma unroll 1
  for (int c = threadIdx.x; c < d_v; c += kDecodeThreads) {
    const int q = lo != nullptr ? hi[c * p.v.sw] * 256 + lo[c * p.v_lo.sw] : hi[c * p.v.sw];
    out[c] = static_cast<float>(q) * unit_q;
  }
}

//: The write gain, a per-token scalar every thread of the CTA reads.
template <typename T>
__device__ __forceinline__ float fold_gain(const DecodeParams& p, int b, int h) {
  return widen(*row_of<T>(p.g_write, b, h));
}

}  // namespace decode
}  // namespace rola
