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

//: A row's max digit: a u8 digit times its row's scale over this is the amplitude. -- see docs/internals/decode/decode_fold.md#raw
constexpr float kRowMaxDigit = 255.0f;

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

__device__ __forceinline__ float scale_of(const Span& s, int b, int h) {
  return *row_of<float>(s, b, h);
}

//: THE DIGIT PASS: every digit of both sides, a thread's slots' loads issued before any is used, staged and -- see docs/internals/decode/decode_fold.md#fold-digits
template <int D>
__device__ __forceinline__ void fold_digits(const DecodeParams& p, int b, int h, float* sa_r,
                                            float* sa_w, uint32_t* mask_r, uint32_t* mask_w) {
  //: a side is at most `kMaxLevels` levels of `kDecodeMaxLevelWidth` digits, so a thread holds at most this many slots.
  constexpr int kSlots = (2 * kMaxLevels * kDecodeMaxLevelWidth) / kDecodeThreads;
  const int lane = threadIdx.x & 31;
  const int slots = p.mask_total << 5;
  float x[kSlots];
  float scale[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = threadIdx.x + j * kDecodeThreads;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int digit = ((word - p.mask_off[l]) << 5) + lane;
    const bool valid = s < 2 * slots && digit < p.level_width[l];
    const Span& rows = write_side ? p.write[l] : p.read[l];
    x[j] = valid ? static_cast<float>(row_of<uint8_t>(rows, b, h)[digit * rows.sw]) : 0.0f;
    scale[j] = valid ? scale_of(write_side ? p.write_scale[l] : p.read_scale[l], b, h) : 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = threadIdx.x + j * kDecodeThreads;
    if (s >= 2 * slots) break;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int digit = ((word - p.mask_off[l]) << 5) + lane;
    const uint32_t live = __ballot_sync(0xffffffffu, x[j] != 0.0f);
    if (lane == 0) (write_side ? mask_w : mask_r)[word] = live;
    if (digit < p.level_width[l]) {
      //: a level the fold normalizes stays RAW until its sum is known (`normalize_rows`).
      const bool normalized = !write_side && p.normalize[l] != 0;
      const float unit = normalized ? 1.0f : scale[j] / kRowMaxDigit;
      (write_side ? sa_w : sa_r)[p.level_amp_offset[l] + digit] = x[j] * unit;
    }
  }
}

//: THE DECLARED SUM of a normalized level's raw digits, from their staging. Warp `first + l` owns level `l` -- see docs/internals/decode/decode_fold.md#row-sum
__device__ __forceinline__ void sum_staged_rows(const DecodeParams& p, const float* sa_r,
                                                float* s_sum, int first) {
  const int lane = threadIdx.x & 31;
  const int l = (threadIdx.x >> 5) - first;
  if (l >= 0 && l < p.D && p.normalize[l] != 0) {
    const float* row = sa_r + p.level_amp_offset[l];
    float acc = 0.0f;
#pragma unroll 1
    for (int k = lane; k < p.level_width[l]; k += 32) acc += row[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_sum[l] = acc;
  }
}

//: A NORMALIZED LEVEL'S AMPLITUDES, `raw / sum`, in place over its staging. -- see docs/internals/decode/decode_fold.md#normalize-rows
__device__ __forceinline__ void normalize_rows(const DecodeParams& p, float* sa_r,
                                               const float* s_sum) {
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) {
    if (p.normalize[l] == 0) continue;
    float* row = sa_r + p.level_amp_offset[l];
    const float total = s_sum[l];
#pragma unroll 1
    for (int g = threadIdx.x; g < p.level_width[l]; g += kDecodeThreads) row[g] = row[g] / total;
  }
}

//: Channel `c` of the token's value, LOADED: its f16 unit and the token's scale, the product left for after the digit pass. -- see docs/internals/decode/decode_fold.md#fold-value
struct ValueUnit {
  __half unit;
  float scale;
};

__device__ __forceinline__ ValueUnit load_value(const DecodeParams& p, int b, int h, int c) {
  return ValueUnit{row_of<__half>(p.v, b, h)[c * p.v.sw], scale_of(p.v_scale, b, h)};
}

//: The channel's value, `unit * scale`.
__device__ __forceinline__ float value_of(const ValueUnit& d) {
  return __half2float(d.unit) * d.scale;
}

//: The write gain, a per-token scalar every thread of the CTA reads.
template <typename T>
__device__ __forceinline__ float fold_gain(const DecodeParams& p, int b, int h) {
  return widen(*row_of<T>(p.g_write, b, h));
}

}  // namespace decode
}  // namespace rola
