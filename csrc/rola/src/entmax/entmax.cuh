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
#include <vector>

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

//: `(stream, stream2, token, col, tile, lane)` -- ONE address expression serves every
//: operand layout in this family because the caller zeroes the members its layout does
//: not use, which is why layout is data here -- docs/internals/entmax/entmax.md#strides
//:
//: THE STREAM AXIS IS TWO STRIDES because a token-major logit plane needs two: `b` and
//: `h` are separate axes of `[B, T, H, C]` and the solve's stream is `b*H + h`. The
//: caller splits `blockIdx.y` ONCE per block -- docs/internals/entmax/entmax.md#stream-split
struct Strides {
  int64_t stream, stream2, token, col, tile, lane;
};

//: THE FLAT ADDRESS, unchanged: ONE stream register, and every term after it
//: rematerializes from constant memory. Keeping it that way is a REGISTER decision --
//: see `addr_split` -- docs/internals/entmax/entmax.md#stream-split
__device__ __forceinline__ int64_t addr(const Strides& s, int64_t stream, int64_t tok,
                                        int64_t col) {
  return stream * s.stream + tok * s.token + (tok >> 5) * s.tile + col * s.col
         + (tok & 31) * s.lane;
}

//: THE SPLIT ADDRESS, for the LOGIT plane alone. `b` and `h` are separate axes of a
//: token-major `[B, T, H, C]`, so its stream term is two products rather than one.
//:
//: This is a SECOND expression and not a widening of `addr` because the widening was
//: MEASURED: making every operand's stream term two products cost the family its
//: 64-register bound -- 4 spilling instantiations became 56, with 180-244 byte spills at
//: `IPT = 8`. Confining it to the one operand that needs it keeps every other address
//: rematerializable from a single register, which is what it always was.
__device__ __forceinline__ int64_t addr_split(const Strides& s, int64_t sb, int64_t sh, int64_t tok,
                                              int64_t col) {
  return sb * s.stream + sh * s.stream2 + tok * s.token + col * s.col;
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

//: THE PER-LAUNCH LEVEL TABLE. It travels in the kernel's PARAMETER space (448 B at
//: this bound, against a 4 KB floor), so batching a plan's levels into one launch costs
//: no device table, no H2D copy and no indirection. Offsets are COLUMN indices into the
//: base tensor, which is what `addr` adds them to -- docs/internals/entmax/factor.md#batched-levels
constexpr int MAX_BATCHED_LEVELS = 16;

struct LevelTable {
  int64_t logit_off[MAX_BATCHED_LEVELS];
  int64_t value_off[MAX_BATCHED_LEVELS];
  int64_t support_off[MAX_BATCHED_LEVELS];
  //: The union solve's midpoint plane is packed over the UNION levels alone, so it does
  //: not share `value_off`; every other operand does.
  int64_t midpoint_off[MAX_BATCHED_LEVELS];
  int width[MAX_BATCHED_LEVELS];
};

//: The tuple a tensor's own RANK implies: rank-4 is the tile-major arena (no token
//: stride of its own), rank-3 the dense plane (no tile/lane stride). Deriving it keeps
//: one launch site per kernel instead of one per layout.
//:
inline Strides five(const torch::Tensor& t) {
  if (t.dim() == 4) return {t.stride(0), 0, 0, t.stride(2), t.stride(1), t.stride(3)};
  return {t.stride(0), 0, t.stride(1), t.stride(2), 0, 0};
}

//: THE LOGIT PLANE, in whichever of its two layouts the caller holds it: token-major
//: `[B, T, H, C]` (the packed router GEMM's own output, so taking it here is what
//: deletes the transpose that used to follow), or the flat `[BH, T, C]` whose head
//: stride is simply 0 -- docs/internals/entmax/entmax.md#stream-split
inline Strides five_logits(const torch::Tensor& t, int heads) {
  TORCH_CHECK(heads >= 1, "the stream split needs heads >= 1; 1 is one head");
  if (t.dim() == 4) {
    TORCH_CHECK(t.size(2) == heads, "a token-major logit plane's head axis must be `heads`");
    return {t.stride(0), t.stride(2), t.stride(1), t.stride(3), 0, 0};
  }
  TORCH_CHECK(heads == 1, "a flat [BH, T, W] logit plane declares no split (heads == 1)");
  return {t.stride(0), 0, t.stride(1), t.stride(2), 0, 0};
}

inline Strides five_or_zero(const c10::optional<torch::Tensor>& t) {
  if (!t.has_value()) return {0, 0, 0, 0, 0, 0};
  return five(*t);
}

template <typename T>
T* ptr_or_null(const c10::optional<torch::Tensor>& t) {
  return t.has_value() ? (T*)t->data_ptr() : nullptr;
}

//: The padded width is what selects the `(lane width, items per thread)` pair, and every
//: level in one table must share it -- otherwise the sort and the prefix scan would
//: associate differently per level and the batch would not be the loop it replaces.
inline int padded_width(int width) {
  int w = 1;
  while (w < width) w <<= 1;
  return w < 2 ? 2 : w;
}

inline LevelTable build_table(const std::vector<int64_t>& logit_off,
                              const std::vector<int64_t>& value_off,
                              const std::vector<int64_t>& support_off,
                              const std::vector<int64_t>& widths,
                              const std::vector<int64_t>& midpoint_off = {}) {
  const int n = (int)widths.size();
  TORCH_CHECK(n > 0 && n <= MAX_BATCHED_LEVELS, "batched level count must be in [1, ",
              MAX_BATCHED_LEVELS, "], got ", n);
  TORCH_CHECK((int)logit_off.size() == n && (int)value_off.size() == n,
              "level offset tables must match the width table");
  TORCH_CHECK(support_off.empty() || (int)support_off.size() == n,
              "support offset table must match the width table");
  const int klass = padded_width((int)widths[0]);
  LevelTable table{};
  for (int i = 0; i < n; ++i) {
    TORCH_CHECK(padded_width((int)widths[i]) == klass,
                "a batched solve must be one width class; level ", i, " has width ", widths[i],
                " against class ", klass);
    table.logit_off[i] = logit_off[i];
    table.value_off[i] = value_off[i];
    table.support_off[i] = support_off.empty() ? 0 : support_off[i];
    table.midpoint_off[i] = midpoint_off.empty() ? value_off[i] : midpoint_off[i];
    table.width[i] = (int)widths[i];
  }
  return table;
}

}  // namespace detail
#endif  // __CUDACC__

//: ONE LAUNCH, MANY LEVELS: the entries take BASE tensors plus per-level COLUMN
//: offsets, so every union level of one width class and one alpha runs in one launch
//: doing exactly what the per-level loop did -- docs/internals/entmax/entmax.md#launch-shape
//:
//: `read_values`/`write_values` accept BOTH route-value layouts (rank-3 dense,
//: rank-4 tile-major) through ONE stride 5-tuple, so layout is data here and never a
//: codegen axis -- docs/internals/entmax/entmax.md#strides
void union_forward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                   torch::Tensor& midpoint_values, torch::Tensor& support_words,
                   torch::Tensor& read_values, torch::Tensor& write_values,
                   const std::vector<int64_t>& logit_offsets,
                   const std::vector<int64_t>& value_offsets,
                   const std::vector<int64_t>& midpoint_offsets,
                   const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                   int64_t heads, double alpha);

void union_backward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                    const torch::Tensor& midpoint_values, const torch::Tensor& support_words,
                    const torch::Tensor& read_values, const torch::Tensor& write_values,
                    const torch::Tensor& d_read, const torch::Tensor& d_write,
                    torch::Tensor& d_read_logits, torch::Tensor& d_write_logits,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& midpoint_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha);

}  // namespace entmax
}  // namespace rola
