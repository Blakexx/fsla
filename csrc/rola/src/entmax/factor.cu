// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- hand CUDA, CUB warp primitives.
// Design of record: docs/internals/entmax/factor.md.
// Gates: `tests/integration/test_production_levels.py` (fp64 oracle),
// `tests/unit/test_entmax_production.py`, `tests/oracle/test_entmax_cuda_gates.py`.
//
// Four kernels, two families, one launch shape: the exact alpha=1.5/2.0 threshold solve
// (forward + VJP) and the softmax pair. They share `entmax.cuh`'s warp-row vocabulary
// with the union solve next door, and the entmax forward IS the union forward's phases
// B/C/D fed one logit tensor instead of a midpoint.
//
// ONE LAUNCH ACROSS LEVELS: `blockIdx.z` selects a LevelTable entry, so a plan's levels
// that share a width class, an alpha and an output dtype run in ONE launch with
// bit-identical arithmetic -- docs/internals/entmax/factor.md#batched-levels

#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_switch.cuh"
#include "entmax.cuh"
#include "factor.cuh"

namespace rola {
namespace entmax {
namespace detail {

//: The per-launch level table travels in the kernel's PARAMETER space (448 B at this
//: bound, against a 4 KB floor), so batching costs no H2D copy and no extra
//: indirection -- docs/internals/entmax/factor.md#batched-levels
constexpr int MAX_BATCHED_LEVELS = 16;

struct LevelTable {
  int64_t logit_off[MAX_BATCHED_LEVELS];
  int64_t value_off[MAX_BATCHED_LEVELS];
  int64_t support_off[MAX_BATCHED_LEVELS];
  int width[MAX_BATCHED_LEVELS];
};

// ---------------------------------------------------------------------------
// FORWARD -- the threshold-after-sort solve
// ---------------------------------------------------------------------------
template <int LW, int IPT, bool ALPHA15, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void factor_forward_kernel(
    const float* __restrict__ logits, const bool* __restrict__ mask, OutT* __restrict__ values,
    OutT* __restrict__ values_second, int32_t* __restrict__ support_words, int tokens,
    LevelTable table, Strides sz, Strides smask, Strides sv, Strides sv2, Strides ssup) {
  constexpr int W_PAD = LW * IPT;
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;

  using WarpSort = cub::WarpMergeSort<float, IPT, LW>;
  using WarpScanP = cub::WarpScan<Pair, LW>;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedI = cub::WarpReduce<int, LW>;
  using WarpRedU = cub::WarpReduce<unsigned long long, LW>;

  //: HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias
  //: One per-physical-warp slot serves the sort scratch and then the support-word
  //: staging. `SLOT` rounds to 16 B because `WarpMergeSort::TempStorage` is
  //: `4 + 4*W_PAD` bytes, not a multiple of its own alignment.
  constexpr int SORT_BYTES = (WARP_SIZE / LW) * (int)sizeof(typename WarpSort::TempStorage);
  constexpr int STAGE_BYTES = W_PAD * (int)sizeof(uint32_t);
  constexpr int RAW_SLOT = SORT_BYTES > STAGE_BYTES ? SORT_BYTES : STAGE_BYTES;
  constexpr int SLOT = (RAW_SLOT + 15) & ~15;
  __shared__ __align__(16) char smem[BLOCK_WARPS * SLOT];
  static_assert(sizeof(typename WarpScanP::TempStorage) <= 4, "WarpScan must be shfl-only");
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedI::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedU::TempStorage) <= 4, "WarpReduce must be shfl-only");

  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int logical = threadIdx.x / LW;
  const int logical_in_warp = (threadIdx.x % WARP_SIZE) / LW;
  char* slot = smem + warp * SLOT;
  auto& sort_store = reinterpret_cast<typename WarpSort::TempStorage*>(slot)[logical_in_warp];
  typename WarpScanP::TempStorage scan_store;
  typename WarpRedF::TempStorage red_store;
  typename WarpRedI::TempStorage redi_store;
  typename WarpRedU::TempStorage redu_store;

  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t support_base = table.support_off[level];

  uint32_t part[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) part[i] = 0u;

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    // ---- PHASE A: load striped (coalesced); the candidate mask is per ELEMENT ----
    float z[IPT];
    uint32_t valid_mask = 0u;
    int local_valid = 0;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool in_width = col < width;
      const bool valid =
          in_width && (mask == nullptr || mask[addr(smask, stream, token, logit_base + col)]);
      if (valid) {
        valid_mask |= (1u << i);
        ++local_valid;
      }
      z[i] = valid ? logits[addr(sz, stream, token, logit_base + col)] : 0.0f;
    }
    //: A masked row can be EMPTY, so the member count is a per-row reduction and not
    //: `width`; `production_factor`'s empty-row autograd contract is exactly this case.
    int valid_count = width;
    if (mask != nullptr) {
      valid_count = WarpRedI(redi_store).Reduce(local_valid, cub::Sum());
      valid_count = __shfl_sync(lmask, valid_count, 0, LW);
    }
    const bool has_valid = valid_count > 0;

    // ---- PHASE B: row max/min, sentinel-pad, sort descending ----
    float local_max = -INFINITY, local_min = INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      if ((valid_mask >> i) & 1u) {
        local_max = fmaxf(local_max, z[i]);
        local_min = fminf(local_min, z[i]);
      }
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, cub::Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);
    row_max = has_valid ? row_max : 0.0f;
    float row_min = WarpRedF(red_store).Reduce(local_min, cub::Min());
    row_min = __shfl_sync(lmask, row_min, 0, LW);

    //: A finite row-relative sentinel keeps padded lanes strictly below every real
    //: value without an infinity in the scan. `row_min` is pre-shift, so shift it too.
    const float sentinel = has_valid ? (row_min - row_max) - 1.0f : -1.0f;
    float keys[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool valid = (valid_mask >> i) & 1u;
      z[i] = valid ? z[i] - row_max : 0.0f;
      keys[i] = valid ? z[i] : sentinel;
    }
    WarpSort(sort_store).Sort(keys, Descending());

    // ---- PHASE C: tau. Every full-width intermediate COLLAPSES TO A SCALAR. ----
    Pair local_sum{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      local_sum.p += keys[i];
      local_sum.q += keys[i] * keys[i];
    }
    Pair carry;
    WarpScanP(scan_store).ExclusiveScan(local_sum, carry, Pair{0.0f, 0.0f}, PairAdd());

    int best_col = -1;
    float best_tau = 0.0f;
    float run_p = carry.p, run_q = carry.q;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      run_p += keys[i];
      run_q += keys[i] * keys[i];
      const int rank = lane * IPT + i;
      const float k = (float)(rank + 1);
      float tau_c;
      bool cand;
      if (ALPHA15) {
        const float disc = run_p * run_p - k * (run_q - 4.0f);
        tau_c = (run_p - sqrtf(fmaxf(disc, 0.0f))) / k;
        cand = (rank < valid_count) && (disc >= 0.0f) && (keys[i] > tau_c);
      } else {
        tau_c = (run_p - 1.0f) / k;
        cand = (rank < valid_count) && (keys[i] > tau_c);
      }
      if (cand) {
        best_col = rank;
        best_tau = tau_c;
      }
    }
    //: ONE reduction selects the greatest valid k AND carries its tau: the column in the
    //: high 32 bits makes integer Max pick it, and `best_col = -1` maps to key 0 so an
    //: all-empty row yields tau = 0 -- docs/internals/entmax/entmax.md#forward-phases
    unsigned long long packed_key = ((unsigned long long)(uint32_t)(best_col + 1) << 32)
                                    | (unsigned long long)__float_as_uint(best_tau);
    packed_key = WarpRedU(redu_store).Reduce(packed_key, cub::Max());
    packed_key = __shfl_sync(lmask, packed_key, 0, LW);
    const int chosen = (int)(uint32_t)(packed_key >> 32) - 1;
    const float tau = chosen >= 0 ? __uint_as_float((uint32_t)(packed_key & 0xffffffffu)) : 0.0f;

    // ---- PHASE D: amplitudes, both outputs, and the support bit ----
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = ((valid_mask >> i) & 1u) && (z[i] > tau);
      float base = (ALPHA15 ? 0.5f : 1.0f) * (z[i] - tau);
      base = sup ? fmaxf(base, 0.0f) : 0.0f;
      const float p = ALPHA15 ? base * base : base;
      const int64_t off = addr(sv, stream, token, value_base + col);
      store_value(values, off, p);
      //: The second output is the SAME solve, never a second one: a `shares_solve` level
      //: hands read and write one set of amplitudes.
      if (values_second != nullptr)
        store_value(values_second, addr(sv2, stream, token, value_base + col), p);
      //: HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary
      //: Support is defined at the native bf16 storage boundary, never at the
      //: pre-quantization solve, whatever dtype this level's values are stored in.
      if (sup && bf16_nonzero(p)) part[i] |= (1u << (token & 31));
    }
  }

  // ---- EPILOGUE: pack the 32-token support word. ONE barrier. ----
  //: The butterfly runs over exactly the bits at or above LW, which is what separates
  //: lanes holding the same column: docs/internals/entmax/entmax.md#epilogue-butterfly
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    for (int d = LW; d < WARP_SIZE; d <<= 1) part[i] |= __shfl_xor_sync(0xffffffffu, part[i], d);
  }
  uint32_t* stage = reinterpret_cast<uint32_t*>(smem + warp * SLOT);
  if (logical_in_warp == 0) {
#pragma unroll
    for (int i = 0; i < IPT; ++i) stage[lane + i * LW] = part[i];
  }
  __syncthreads();
  if (warp == 0) {
    for (int col = lane; col < width; col += LW) {
      uint32_t acc = 0u;
#pragma unroll
      for (int w = 0; w < BLOCK_WARPS; ++w)
        acc |= reinterpret_cast<const uint32_t*>(smem + w * SLOT)[col];
      support_words[stream * ssup.stream + (int64_t)word * ssup.token
                    + (support_base + col) * ssup.col] = (int32_t)acc;
    }
  }
}

// ---------------------------------------------------------------------------
// BACKWARD -- the alpha-entmax rank-1 Jacobian action.
//
// No sort and no scan: ONE fused (numerator, denominator) reduction over a support that
// is an INPUT, unpacked from the same frozen word the forward emitted, so the two
// directions cannot disagree about the support set by construction. Off support every
// output term is an exact zero by multiplication.
//
// It RE-READS `values` and the cotangents in the second pass rather than staging them:
// the union backward's measured spill trade, and the same `IPT`-sized arrays would apply
// here -- docs/internals/entmax/factor.md#backward
// ---------------------------------------------------------------------------
template <int LW, int IPT, bool ALPHA15, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void factor_backward_kernel(
    const OutT* __restrict__ values, const int32_t* __restrict__ support_words,
    const bool* __restrict__ mask, const OutT* __restrict__ d_values,
    const OutT* __restrict__ d_values_second, float* __restrict__ d_logits, int tokens,
    bool accumulate, LevelTable table, Strides sv, Strides ssup, Strides smask, Strides sdv,
    Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedP = cub::WarpReduce<Pair, LW>;
  static_assert(sizeof(typename WarpRedP::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedP::TempStorage redp_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t support_base = table.support_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    uint32_t live_mask = 0u;
    Pair partial{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid =
          (col < width) && (mask == nullptr || mask[addr(smask, stream, token, logit_base + col)]);
      const uint32_t wword =
          (col < width) ? (uint32_t)support_words[stream * ssup.stream + (int64_t)word * ssup.token
                                                  + (support_base + col) * ssup.col]
                        : 0u;
      const bool live = valid && ((wword >> (token & 31)) & 1u);
      if (live) live_mask |= (1u << i);
      const float v = live ? load_value(values, addr(sv, stream, token, value_base + col)) : 0.0f;
      const float s = live ? (ALPHA15 ? sqrtf(v) : 1.0f) : 0.0f;
      float dp = live ? load_value(d_values, addr(sdv, stream, token, value_base + col)) : 0.0f;
      if (live && d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      partial.p += s * dp;
      partial.q += s;
    }
    Pair red = WarpRedP(redp_store).Reduce(partial, PairAdd());
    float num = __shfl_sync(lmask, red.p, 0, LW);
    float den = __shfl_sync(lmask, red.q, 0, LW);
    den = den > 0.0f ? den : 1.0f;
    const float shift = num / den;

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool live = (live_mask >> i) & 1u;
      float dz = 0.0f;
      if (live) {
        const float v = load_value(values, addr(sv, stream, token, value_base + col));
        const float s = ALPHA15 ? sqrtf(v) : 1.0f;
        float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
        if (d_values_second != nullptr)
          dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
        dz = s * (dp - shift);
      }
      float* out = d_logits + addr(sdz, stream, token, logit_base + col);
      if (accumulate)
        atomicAdd(out, dz);
      else
        *out = dz;
    }
  }
}

// ---------------------------------------------------------------------------
// SOFTMAX -- the full-support level (`dense_routing`). No threshold, no support word.
// ---------------------------------------------------------------------------
template <int LW, int IPT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void softmax_forward_kernel(
    const float* __restrict__ logits, OutT* __restrict__ values, OutT* __restrict__ values_second,
    int tokens, LevelTable table, Strides sz, Strides sv, Strides sv2) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    float e[IPT];
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      e[i] = (col < width) ? logits[addr(sz, stream, token, logit_base + col)] : -INFINITY;
      local_max = fmaxf(local_max, e[i]);
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, cub::Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      e[i] = (lane + i * LW < width) ? expf(e[i] - row_max) : 0.0f;
      local_sum += e[i];
    }
    float den = WarpRedF(red_store).Reduce(local_sum, cub::Sum());
    den = __shfl_sync(lmask, den, 0, LW);

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = e[i] / den;
      store_value(values, addr(sv, stream, token, value_base + col), p);
      if (values_second != nullptr)
        store_value(values_second, addr(sv2, stream, token, value_base + col), p);
    }
  }
}

template <int LW, int IPT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_BLOCKS_PER_SM) void softmax_backward_kernel(
    const OutT* __restrict__ values, const OutT* __restrict__ d_values,
    const OutT* __restrict__ d_values_second, float* __restrict__ d_logits, int tokens,
    bool accumulate, LevelTable table, Strides sv, Strides sdv, Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    float local_dot = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = load_value(values, addr(sv, stream, token, value_base + col));
      float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
      if (d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      local_dot += p * dp;
    }
    float dot = WarpRedF(red_store).Reduce(local_dot, cub::Sum());
    dot = __shfl_sync(lmask, dot, 0, LW);

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const float p = load_value(values, addr(sv, stream, token, value_base + col));
      float dp = load_value(d_values, addr(sdv, stream, token, value_base + col));
      if (d_values_second != nullptr)
        dp += load_value(d_values_second, addr(sdv2, stream, token, value_base + col));
      const float dz = p * (dp - dot);
      float* out = d_logits + addr(sdz, stream, token, logit_base + col);
      if (accumulate)
        atomicAdd(out, dz);
      else
        *out = dz;
    }
  }
}

// ---------------------------------------------------------------------------
// HOST DISPATCH -- the width CLASS is chosen here, so the kernel branches on nothing.
// The level table's entries are the caller's grouping: same class, same alpha, same
// dtype -- docs/internals/entmax/factor.md#batched-levels
// ---------------------------------------------------------------------------
inline Strides five_or_zero(const c10::optional<torch::Tensor>& t) {
  if (!t.has_value()) return {0, 0, 0, 0, 0};
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

LevelTable build_table(const std::vector<int64_t>& logit_off, const std::vector<int64_t>& value_off,
                       const std::vector<int64_t>& support_off,
                       const std::vector<int64_t>& widths) {
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
    table.width[i] = (int)widths[i];
  }
  return table;
}

template <typename OutT>
void launch_factor_forward(const torch::Tensor& logits, const c10::optional<torch::Tensor>& mask,
                           torch::Tensor& values, const c10::optional<torch::Tensor>& values_second,
                           torch::Tensor& support, const LevelTable& table, int levels, int klass,
                           double alpha) {
  const int n_streams = logits.size(0), tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = at::cuda::getCurrentCUDAStream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "independent entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(alpha == 1.5, [&](auto A15) {
          factor_forward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  logits.data_ptr<float>(), ptr_or_null<const bool>(mask), (OutT*)values.data_ptr(),
                  ptr_or_null<OutT>(values_second), support.data_ptr<int32_t>(), tokens, table,
                  five(logits), five_or_zero(mask), five(values), five_or_zero(values_second),
                  five(support));
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename OutT>
void launch_factor_backward(const torch::Tensor& values, const torch::Tensor& support,
                            const c10::optional<torch::Tensor>& mask, const torch::Tensor& d_values,
                            const c10::optional<torch::Tensor>& d_values_second,
                            torch::Tensor& d_logits, const LevelTable& table, int levels, int klass,
                            bool accumulate, double alpha) {
  const int n_streams = d_logits.size(0), tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = at::cuda::getCurrentCUDAStream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "independent entmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        bool_switch(alpha == 1.5, [&](auto A15) {
          factor_backward_kernel<kLaneWidth, kItemsPerThread, decltype(A15)::value, OutT>
              <<<grid, BLOCK_THREADS, 0, stream>>>(
                  (const OutT*)values.data_ptr(), support.data_ptr<int32_t>(),
                  ptr_or_null<const bool>(mask), (const OutT*)d_values.data_ptr(),
                  ptr_or_null<const OutT>(d_values_second), d_logits.data_ptr<float>(), tokens,
                  accumulate, table, five(values), five(support), five_or_zero(mask),
                  five(d_values), five_or_zero(d_values_second), five(d_logits));
        });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename OutT>
void launch_softmax_forward(const torch::Tensor& logits, torch::Tensor& values,
                            const c10::optional<torch::Tensor>& values_second,
                            const LevelTable& table, int levels, int klass) {
  const int n_streams = logits.size(0), tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = at::cuda::getCurrentCUDAStream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "softmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        softmax_forward_kernel<kLaneWidth, kItemsPerThread, OutT>
            <<<grid, BLOCK_THREADS, 0, stream>>>(logits.data_ptr<float>(), (OutT*)values.data_ptr(),
                                                 ptr_or_null<OutT>(values_second), tokens, table,
                                                 five(logits), five(values),
                                                 five_or_zero(values_second));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename OutT>
void launch_softmax_backward(const torch::Tensor& values, const torch::Tensor& d_values,
                             const c10::optional<torch::Tensor>& d_values_second,
                             torch::Tensor& d_logits, const LevelTable& table, int levels,
                             int klass, bool accumulate) {
  const int n_streams = d_logits.size(0), tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = at::cuda::getCurrentCUDAStream();
  int_switch<2, 4, 8, 16, 32, 64, 128, 256>(
      klass, "softmax padded width (MAX_BRANCH_WIDTH = 256)", [&](auto WPAD) {
        constexpr int kWPad = decltype(WPAD)::value;
        constexpr int kLaneWidth = kWPad < 32 ? kWPad : 32;
        constexpr int kItemsPerThread = kWPad / kLaneWidth;
        softmax_backward_kernel<kLaneWidth, kItemsPerThread, OutT>
            <<<grid, BLOCK_THREADS, 0, stream>>>(
                (const OutT*)values.data_ptr(), (const OutT*)d_values.data_ptr(),
                ptr_or_null<const OutT>(d_values_second), d_logits.data_ptr<float>(), tokens,
                accumulate, table, five(values), five(d_values), five_or_zero(d_values_second),
                five(d_logits));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_values(const torch::Tensor& values, const c10::optional<torch::Tensor>& second) {
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 || values.scalar_type() == torch::kBFloat16,
              "route values must be fp32 or bf16");
  TORCH_CHECK(!second.has_value() || second->scalar_type() == values.scalar_type(),
              "both route value outputs must share a dtype");
}

}  // namespace detail

void factor_forward(const torch::Tensor& logits, const c10::optional<torch::Tensor>& mask,
                    torch::Tensor& values, const c10::optional<torch::Tensor>& values_second,
                    torch::Tensor& support_words, const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    double alpha) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32,
              "independent entmax requires fp32 CUDA logits");
  TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "independent entmax supports alpha 1.5 / 2.0 only");
  TORCH_CHECK(support_offsets.size() == widths.size(),
              "the entmax forward writes a support word per level");
  detail::check_values(values, values_second);
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const at::cuda::CUDAGuard guard(logits.device());
  if (values.scalar_type() == torch::kFloat32)
    detail::launch_factor_forward<float>(logits, mask, values, values_second, support_words, table,
                                         (int)widths.size(), klass, alpha);
  else
    detail::launch_factor_forward<__nv_bfloat16>(logits, mask, values, values_second, support_words,
                                                 table, (int)widths.size(), klass, alpha);
}

void factor_backward(const torch::Tensor& values, const torch::Tensor& support_words,
                     const c10::optional<torch::Tensor>& mask, const torch::Tensor& d_values,
                     const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, bool accumulate, double alpha) {
  TORCH_CHECK(d_logits.is_cuda() && d_logits.scalar_type() == torch::kFloat32,
              "independent entmax backward writes fp32 CUDA logit gradients");
  TORCH_CHECK(alpha == 1.5 || alpha == 2.0, "independent entmax supports alpha 1.5 / 2.0 only");
  TORCH_CHECK(support_offsets.size() == widths.size(),
              "the entmax backward reads a support word per level");
  detail::check_values(values, d_values_second);
  TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
              "cotangents must share the route values' dtype");
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const at::cuda::CUDAGuard guard(d_logits.device());
  if (values.scalar_type() == torch::kFloat32)
    detail::launch_factor_backward<float>(values, support_words, mask, d_values, d_values_second,
                                          d_logits, table, (int)widths.size(), klass, accumulate,
                                          alpha);
  else
    detail::launch_factor_backward<__nv_bfloat16>(values, support_words, mask, d_values,
                                                  d_values_second, d_logits, table,
                                                  (int)widths.size(), klass, accumulate, alpha);
}

void softmax_forward(const torch::Tensor& logits, torch::Tensor& values,
                     const c10::optional<torch::Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& widths) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32,
              "softmax routing requires fp32 CUDA logits");
  detail::check_values(values, values_second);
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const at::cuda::CUDAGuard guard(logits.device());
  if (values.scalar_type() == torch::kFloat32)
    detail::launch_softmax_forward<float>(logits, values, values_second, table, (int)widths.size(),
                                          klass);
  else
    detail::launch_softmax_forward<__nv_bfloat16>(logits, values, values_second, table,
                                                  (int)widths.size(), klass);
}

void softmax_backward(const torch::Tensor& values, const torch::Tensor& d_values,
                      const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      bool accumulate) {
  TORCH_CHECK(d_logits.is_cuda() && d_logits.scalar_type() == torch::kFloat32,
              "softmax routing backward writes fp32 CUDA logit gradients");
  detail::check_values(values, d_values_second);
  TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
              "cotangents must share the route values' dtype");
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths);
  const int klass = detail::padded_width((int)widths[0]);
  const at::cuda::CUDAGuard guard(d_logits.device());
  if (values.scalar_type() == torch::kFloat32)
    detail::launch_softmax_backward<float>(values, d_values, d_values_second, d_logits, table,
                                           (int)widths.size(), klass, accumulate);
  else
    detail::launch_softmax_backward<__nv_bfloat16>(values, d_values, d_values_second, d_logits,
                                                   table, (int)widths.size(), klass, accumulate);
}

}  // namespace entmax
}  // namespace rola
