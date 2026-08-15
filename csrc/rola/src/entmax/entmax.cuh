// THE PRODUCER'S UNION ENTMAX SOLVE -- host declarations. The implementation, both
// kernels and the host dispatch, is `entmax.cu`.
//
// THE STRUCTURAL IDEA: ONE ROW PER (LOGICAL) WARP. Every collective in the solve runs
// along one token's row, so all of them are warp-scope and shuffle-based -- which is
// what makes register demand O(items-per-lane), the warp count a free occupancy knob,
// and the shared-memory need zero.
//
// THE SUPPORT WORD LAYOUT IS FROZEN (the per-leaf write bitmap has nine downstream
// consumers): one CTA owns exactly the 32 tokens of one support word.
//
// Why this is CUDA and not Triton, the measured occupancy history, and the whole
// solve: docs/internals/entmax/entmax.md
//
// The device-side block below is the warp-row vocabulary the union solve and the
// independent solves (`factor.cu`) both speak; the host declarations follow it.
#pragma once

#include <cstdint>

#include <torch/types.h>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

namespace rola {
namespace entmax {

#ifdef __CUDACC__
//: `rola_api.cpp` includes this header and is compiled by the HOST compiler, so every
//: device declaration here is behind `__CUDACC__` -- docs/internals/entmax/entmax.md#warp-row
namespace detail {

//: FROZEN (nine downstream consumers): one CTA owns exactly the 32 tokens of one
//: support word, so `1 << (token % 32)` needs no global atomic and no zero-init pass
//: -- docs/internals/entmax/entmax.md#frozen-support-word
constexpr int WORD_TOKENS = 32;
constexpr int BLOCK_WARPS = 4;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
//: With `BLOCK_THREADS`, this is `__launch_bounds__(128, 8)`: a 64-register target
//: stated to ptxas so a regression becomes a compile-time allocation decision instead
//: of a silent spill, which `tools/ratify.py` then fails the BUILD on.
constexpr int MIN_BLOCKS_PER_SM = 8;

//: `(stream, token, col, tile, lane)` -- ONE address expression serves both route-value
//: layouts because the caller zeroes the members its layout does not use, which is why
//: layout is data here -- docs/internals/entmax/entmax.md#strides
struct Strides {
  int64_t stream, token, col, tile, lane;
};

__device__ __forceinline__ int64_t addr(const Strides& s, int64_t stream, int64_t tok,
                                        int64_t col) {
  return stream * s.stream + tok * s.token + (tok >> 5) * s.tile + col * s.col
         + (tok & 31) * s.lane;
}

template <typename T>
__device__ __forceinline__ void store_value(T* p, int64_t off, float v);
template <>
__device__ __forceinline__ void store_value<float>(float* p, int64_t off, float v) {
  p[off] = v;
}
template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(__nv_bfloat16* p, int64_t off, float v) {
  p[off] = __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float load_value(const T* p, int64_t off);
template <>
__device__ __forceinline__ float load_value<float>(const float* p, int64_t off) {
  return p[off];
}
template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(const __nv_bfloat16* p, int64_t off) {
  return __bfloat162float(p[off]);
}

struct Descending {
  __device__ __forceinline__ bool operator()(float a, float b) const { return a > b; }
};

struct Pair {
  float p, q;
};
struct PairAdd {
  __device__ __forceinline__ Pair operator()(const Pair& a, const Pair& b) const {
    return {a.p + b.p, a.q + b.q};
  }
};

__device__ __forceinline__ bool bf16_nonzero(float v) {
  return __bfloat162float(__float2bfloat16(v)) != 0.0f;
}

//: HAZARD logical-warp-mask -- docs/internals/entmax/entmax.md#logical-warp-mask
//: The member mask for every hand `__shfl_sync` in this family. It names exactly ONE
//: logical warp; a full `0xffffffff` deadlocks on a ragged support word at `LW < 32`.
template <int LW>
__device__ __forceinline__ unsigned logical_warp_mask() {
  return (LW == WARP_SIZE) ? 0xffffffffu
                           : (((1u << LW) - 1u) << (((threadIdx.x % WARP_SIZE) / LW) * LW));
}

//: The 5-tuple a tensor's own RANK implies: rank-4 is the tile-major arena (no token
//: stride of its own), rank-3 the dense plane (no tile/lane stride). Deriving it keeps
//: one launch site per kernel instead of one per layout.
inline Strides five(const torch::Tensor& t) {
  if (t.dim() == 4) return {t.stride(0), 0, t.stride(2), t.stride(1), t.stride(3)};
  return {t.stride(0), t.stride(1), t.stride(2), 0, 0};
}

}  // namespace detail
#endif  // __CUDACC__

//: One launch = one union routing LEVEL = one width class = one alpha; the caller's
//: per-level loop supplies the levels -- docs/internals/entmax/entmax.md#launch-shape
//:
//: `read_values`/`write_values` accept BOTH route-value layouts (rank-3 dense,
//: rank-4 tile-major) through ONE stride 5-tuple, so layout is data here and never a
//: codegen axis -- docs/internals/entmax/entmax.md#strides
void union_forward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                   torch::Tensor& midpoint_values, torch::Tensor& support_words,
                   torch::Tensor& read_values, torch::Tensor& write_values, double alpha);

void union_backward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                    const torch::Tensor& midpoint_values, const torch::Tensor& support_words,
                    const torch::Tensor& read_values, const torch::Tensor& write_values,
                    const torch::Tensor& d_read, const torch::Tensor& d_write,
                    torch::Tensor& d_read_logits, torch::Tensor& d_write_logits, double alpha);

}  // namespace entmax
}  // namespace rola
