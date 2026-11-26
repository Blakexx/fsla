#pragma once

// THE DECODE FOLD -- every per-token operand, read from the internal format as stored and staged in
// SHARED MEMORY by the CTA that is about to consume it.
// see docs/internals/decode/decode_fold.md#note-l3

#include <cuda_bf16.h>

#include <cstdint>

#include "common/ops.cuh"
#include "decode.cuh"

namespace fsla {
namespace decode {

//: THE DIGITS against the row max: a u8 digit is the row max at 255 (`fsla.routing.quant.AMP_LEVELS`).
constexpr float kRowMaxDigit = 255.0f;

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

__device__ __forceinline__ float scale_of(const Span& s, int b, int h) {
  return *row_of<float>(s, b, h);
}

//: THE DIGIT PASS: every digit of both sides, a thread's slots' loads issued before any is used, staged and -- see docs/internals/decode/decode_fold.md#fold-digits
template <class Dp, int D>
__device__ __forceinline__ void fold_digits(const DecodeParams& p, int b, int h, float* sa_r,
                                            float* sa_w, uint32_t* mask_r, uint32_t* mask_w) {
  //: a side is at most `kMaxLevels` levels of `kDecodeMaxLevelWidth` digits, so a thread holds at most this many slots.
  constexpr int kSlots = (2 * kMaxLevels * kDecodeMaxLevelWidth) / Dp::kThreads;
  //: the thread's index, opaque here: a slot's level and offsets are formed a head, not held across every head's walk
  const int me = denseref::ops::opaque(static_cast<int>(threadIdx.x));
  const int lane = me & 31;
  const int slots = p.mask_total << 5;
  float x[kSlots];
  float scale[kSlots];
  uint32_t live[kSlots];
#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = me + j * Dp::kThreads;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int level_word = word - p.mask_off[l];
    const int digit = (level_word << 5) + lane;
    const bool valid = s < 2 * slots && digit < p.level_width[l];
    const Span& rows = write_side ? p.write[l] : p.read[l];
    //: THE SUPPLIED MASK's word: digit-packed, one warp-uniform load; the op computes with the coefficient times its bit
    const int words = (p.level_width[l] + 31) >> 5;
    const int32_t* m = write_side ? p.write_mask[l] : p.read_mask[l];
    //: the word's bits past the level's width are outside the mask's shape, never live
    const int rem = p.level_width[l] - (level_word << 5);
    const uint32_t shape_bits = rem >= 32 ? 0xffffffffu : ((1u << rem) - 1u);
    //: branch-free: a slot past both sides reads its level's first word and keeps no bit of it
    const bool in_sides = s < 2 * slots;
    const int at = (b * p.H + h) * words + (in_sides ? level_word : 0);
    const uint32_t word_bits = static_cast<uint32_t>(m[at]);
    live[j] = word_bits & (in_sides ? shape_bits : 0u);
    const bool on = valid && ((live[j] >> lane) & 1u);
    //: on the byte scale: a u8 digit, at every level, zero where its bit is off
    x[j] = on ? static_cast<float>(row_of<uint8_t>(rows, b, h)[digit * rows.sw]) : 0.0f;
    scale[j] = valid ? scale_of(write_side ? p.write_scale[l] : p.read_scale[l], b, h) : 0.0f;
  }

#pragma unroll
  for (int j = 0; j < kSlots; ++j) {
    const int s = me + j * Dp::kThreads;
    if (s >= 2 * slots) break;
    const bool write_side = s >= slots;
    const int word = (write_side ? s - slots : s) >> 5;
    int l = 0;
#pragma unroll
    for (int q = 1; q < D; ++q) {
      if (word >= p.mask_off[q]) l = q;
    }
    const int digit = ((word - p.mask_off[l]) << 5) + lane;
    //: every lane stores the same word: no lane guard, so no divergence in the step's loop
    (write_side ? mask_w : mask_r)[word] = live[j];
    if (digit < p.level_width[l])
      (write_side ? sa_w : sa_r)[p.level_amp_offset[l] + digit] = x[j] * (scale[j] / kRowMaxDigit);
  }
}

//: A WRITE LEVEL'S ROW SUM, from its staging: the decay's share divides by their product. Warp `first + l` owns level
//: `l`. -- see docs/internals/decode/decode_fold.md#row-sum
__device__ __forceinline__ void sum_staged_rows(const DecodeParams& p, const float* sa_w,
                                                float* s_sum, int first) {
  const int lane = threadIdx.x & 31;
  const int l = (threadIdx.x >> 5) - first;
  if (l >= 0 && l < p.D) {
    const float* row = sa_w + p.level_amp_offset[l];
    float acc = 0.0f;
#pragma unroll 1
    for (int k = lane; k < p.level_width[l]; k += 32) acc += row[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_sum[l] = acc;
  }
}

//: THE DECAY'S SHARE SCALE: `1 / prod_l sum_l` over the write levels' row sums, 0 for a token that writes nothing (its
//: clock still). -- see docs/internals/decode/decode_fold.md#row-sum
__device__ __forceinline__ float share_scale(const DecodeParams& p, const float* s_sum) {
  float total = 1.0f;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) total *= s_sum[l];
  return total > 0.0f ? 1.0f / total : 0.0f;
}

//: Channel `c` of the token's value, LOADED: its bf16 unit and the token's scale, the product left for after the digit pass. -- see docs/internals/decode/decode_fold.md#fold-value
struct ValueUnit {
  __nv_bfloat16 unit;
  float scale;
};

__device__ __forceinline__ ValueUnit load_value(const DecodeParams& p, int b, int h, int c) {
  return ValueUnit{row_of<__nv_bfloat16>(p.v, b, h)[c * p.v.sw], scale_of(p.v_scale, b, h)};
}

//: The channel's value, `unit * scale`.
__device__ __forceinline__ float value_of(const ValueUnit& d) {
  return __bfloat162float(d.unit) * d.scale;
}

}  // namespace decode
}  // namespace fsla
