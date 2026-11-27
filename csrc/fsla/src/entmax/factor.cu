// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- hand CUDA, CUB warp primitives.
// Design of record: docs/internals/entmax/factor.md.
// Gates: `tests/integration/test_production_levels.py` (fp64 oracle),
// `tests/unit/test_entmax_production.py`.
//
// Four kernels, two families, one launch shape: the exact alpha=1.5/2.0 threshold solve
// (forward + VJP) and the softmax pair. They share `entmax.cuh`'s warp-row vocabulary.
//
// ONE LAUNCH ACROSS LEVELS: `blockIdx.z` selects a LevelTable entry, so a plan's levels
// that share a width class, an alpha and an output dtype run in ONE launch with
// bit-identical arithmetic -- docs/internals/entmax/factor.md#batched-levels

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/smem_layout.cuh"
#include "common/arm_switch.cuh"
#include "entmax.cuh"
#include "factor.cuh"
#include "factor_backward_selection.inc"
#include "factor_forward_selection.inc"
#include "plan/entmax_launch.h"
#include "plan/environment.h"
#include "plan/factor_backward/contract.h"
#include "plan/factor_forward/contract.h"
#include "plan/softmax_backward/contract.h"
#include "plan/softmax_forward/contract.h"
#include "softmax_backward_selection.inc"
#include "softmax_forward_selection.inc"

namespace fsla {
namespace entmax {
namespace detail {

// ---------------------------------------------------------------------------
// FORWARD -- the threshold-after-sort solve
// ---------------------------------------------------------------------------
//: THE FORWARD'S SLOT, a warp's: the sort's bytes or the stage's, whichever is more -- see docs/internals/entmax/factor.md#sort-bytes
template <int LW, int IPT>
constexpr int forward_slot_bytes() {
  const int sort = RowSort<LW, IPT>::kSlotBytes;
  const int stage = LW * IPT * static_cast<int>(sizeof(uint32_t));
  return ((sort > stage ? sort : stage) + 15) & ~15;
}

struct ForwardSlots;

//: THE FORWARD'S SHARED MEMORY: a slot a warp, on a card whose SM its `kMinBlocks` CTAs share
template <class Bt, int LW, int IPT>
using ForwardSmem =
    SmemLayout<Bt, EntmaxPlan<Bt::build>::kMinBlocks,
               Fixed<ForwardSlots, char, BLOCK_WARPS * forward_slot_bytes<LW, IPT>()>>;

template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::kMinBlocks) void
factor_forward_kernel(const LogitT* __restrict__ logits, OutT* __restrict__ values,
                      OutT* __restrict__ values_second, int32_t* __restrict__ support_words,
                      int tokens, int heads, LevelTable table, Strides sz, Strides sv, Strides sv2,
                      Strides ssup) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;

  //: HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias -- see docs/internals/entmax/factor.md#sort-bytes
  constexpr int SLOT = forward_slot_bytes<LW, IPT>();
  char* const smem = ForwardSmem<Bt, LW, IPT>::template get<ForwardSlots>();

  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int logical = threadIdx.x / LW;
  const int logical_in_warp = (threadIdx.x % WARP_SIZE) / LW;

  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
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

    // ---- PHASE A: load striped (coalesced); every column of the width is a candidate ----
    float z[IPT];
    uint32_t valid_mask = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      if (valid) valid_mask |= (1u << i);
      z[i] = valid ? load_value(logits, addr_split(sz, sb, sh, token, logit_base + col)) : 0.0f;
    }

    // ---- PHASES B, C: the threshold (`threshold`) ----
    const float tau = threshold<LW, IPT, ALPHA15>(z, valid_mask, width, lane, lmask,
                                                  smem + warp * SLOT, logical_in_warp);

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
      //: The second output is the SAME solve, never a second one: a level a read head ties -- see docs/internals/entmax/factor.md#near-line-200
      if (values_second != nullptr)
        store_value(values_second, addr(sv2, stream, token, value_base + col), p);
      //: HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary -- see docs/internals/entmax/factor.md#near-line-204
      if (sup && bf16_nonzero(p)) part[i] |= (1u << (token & 31));
    }
  }

  // ---- EPILOGUE: pack the 32-token support word. ONE barrier. ----
  //: The butterfly runs over exactly the bits at or above LW, which is what separates -- see docs/internals/entmax/factor.md#near-line-212
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
#pragma unroll 1
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

//: A LOGICAL WARP'S SUM, in every one of its lanes: a butterfly over the lane bits below `LW`, the whole warp shuffling -- see docs/internals/entmax/factor.md#backward-stages
template <int LW>
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 1; d < LW; d <<= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}

//: THE BACKWARD'S READS of one block, WALKED: a row offset a tensor, stepped a token, and a column step an item -- see docs/internals/entmax/factor.md#backward-stages
template <int LW, int IPT, typename OutT>
struct BackwardReads {
  const OutT* values;
  const OutT* d_values;
  const OutT* d_values_second;
  Strides sv, sdv, sdv2;
  int lane = 0, width = 0, tokens = 0, token = 0;
  int64_t rv = 0, rd = 0, r2 = 0;
  int64_t fv = 0, fd = 0, f2 = 0;

  //: the logical warp's first token's offsets (`r*`), and the block's first entry's (`f*`), which an item with none reads
  __device__ __forceinline__ void start(int64_t stream, int t0, int logical, int64_t base) {
    token = t0 + logical;
    rv = addr(sv, stream, t0, base + lane) + logical * (sv.token + sv.lane);
    rd = addr(sdv, stream, t0, base + lane) + logical * (sdv.token + sdv.lane);
    r2 = addr(sdv2, stream, t0, base + lane) + logical * (sdv2.token + sdv2.lane);
    fv = addr(sv, stream, t0, base);
    fd = addr(sdv, stream, t0, base);
    f2 = addr(sdv2, stream, t0, base);
  }

  //: THE NEXT TOKEN'S LOADS, issued RAW into a stage -- nothing here reads what they return -- then the step to the token after
  __device__ __forceinline__ void issue(OutT (&v)[IPT], OutT (&d)[IPT], OutT (&d2)[IPT]) {
    const bool in_call = token < tokens;
    int64_t av = rv;
    int64_t ad = rd;
    int64_t a2 = r2;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool has = in_call && lane + i * LW < width;
      v[i] = values[has ? av : fv];
      d[i] = d_values[has ? ad : fd];
      if (d_values_second != nullptr) d2[i] = d_values_second[has ? a2 : f2];
      av += LW * sv.col;
      ad += LW * sdv.col;
      a2 += LW * sdv2.col;
    }
    rv += (BLOCK_THREADS / LW) * (sv.token + sv.lane);
    rd += (BLOCK_THREADS / LW) * (sdv.token + sdv.lane);
    r2 += (BLOCK_THREADS / LW) * (sdv2.token + sdv2.lane);
    token += BLOCK_THREADS / LW;
  }
};

// ---------------------------------------------------------------------------
// BACKWARD -- the alpha-entmax rank-1 Jacobian action.
// see docs/internals/entmax/factor.md#launch-bounds
template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::backward_min_blocks(IPT)) void
factor_backward_kernel(const OutT* __restrict__ values, const int32_t* __restrict__ support_words,
                       const OutT* __restrict__ d_values, const OutT* __restrict__ d_values_second,
                       LogitT* __restrict__ d_logits, int tokens, int heads, LevelTable table,
                       Strides sv, Strides ssup, Strides sdv, Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  static_assert((LW & (LW - 1)) == 0, "a logical warp is a power of two of lanes");

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-2
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t support_base = table.support_off[level];

  //: A BLOCK OF MORE LOGICAL WARPS THAN ITS WORD HAS TOKENS idles whole warps -- see docs/internals/entmax/factor.md#backward-stages
  if (logical >= WORD_TOKENS) return;

  //: THE SUPPORT WORDS of the block's 32 tokens, a column each, read once -- see docs/internals/entmax/factor.md#backward-stages
  uint32_t wword[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + i * LW;
    const int64_t held = support_base + (col < width ? col : 0);
    const int64_t at = stream * ssup.stream + (int64_t)word * ssup.token + held * ssup.col;
    const uint32_t w = (uint32_t)support_words[at];
    wword[i] = col < width ? w : 0u;
  }

  //: A STAGE AHEAD: a trip takes its token's terms from what the last trip loaded, issues the next token's loads, then reduces and stores -- see docs/internals/entmax/factor.md#backward-stages
  const int t0 = word * WORD_TOKENS;
  BackwardReads<LW, IPT, OutT> reads{values, d_values, d_values_second, sv, sdv, sdv2};
  reads.lane = lane;
  reads.width = width;
  reads.tokens = tokens;
  reads.start(stream, t0, logical, value_base);
  OutT v[IPT] = {};
  OutT d[IPT] = {};
  OutT d2[IPT] = {};
  reads.issue(v, d, d2);

  constexpr int UNROLL = ITERS <= 2 ? ITERS : 1;
  int token = t0 + logical;
  int64_t rz = addr_split(sdz, sb, sh, token, logit_base + lane);
#pragma unroll UNROLL
  for (int it = 0; it < ITERS; ++it) {
    const bool in_call = token < tokens;
    uint32_t live_mask = 0u;
    float s[IPT];
    float dp[IPT];
    float num = 0.0f;
    float den = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool live = in_call && ((wword[i] >> (token & 31)) & 1u);
      live_mask |= (uint32_t)live << i;
      float x = load_value(d, i);
      if (d_values_second != nullptr) x += load_value(d2, i);
      const float root = ALPHA15 ? sqrtf(load_value(v, i)) : 1.0f;
      s[i] = live ? root : 0.0f;
      dp[i] = live ? x : 0.0f;
      num += s[i] * dp[i];
      den += s[i];
    }
    if (it + 1 < ITERS) reads.issue(v, d, d2);
    num = warp_sum<LW>(num);
    den = warp_sum<LW>(den);
    den = den > 0.0f ? den : 1.0f;
    const float shift = num / den;
    int64_t az = rz;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const float dz = ((live_mask >> i) & 1u) ? s[i] * (dp[i] - shift) : 0.0f;
      if (in_call && lane + i * LW < width) store_value(d_logits, az, dz);
      az += LW * sdz.col;
    }
    rz += LOGICAL_PER_BLOCK * sdz.token;
    token += LOGICAL_PER_BLOCK;
  }
}

// ---------------------------------------------------------------------------
// SOFTMAX -- the full-support level (`softmax()`). No threshold, no support word.
// ---------------------------------------------------------------------------
template <class Bt, int LW, int IPT, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::kMinBlocks) void
softmax_forward_kernel(const LogitT* __restrict__ logits, OutT* __restrict__ values,
                       OutT* __restrict__ values_second, int tokens, int heads, LevelTable table,
                       Strides sz, Strides sv, Strides sv2) {
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
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-3
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
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
      e[i] = (col < width) ? load_value(logits, addr_split(sz, sb, sh, token, logit_base + col))
                           : -INFINITY;
      local_max = fmaxf(local_max, e[i]);
    }
    float row_max = WarpRedF(red_store).Reduce(local_max, Max());
    row_max = __shfl_sync(lmask, row_max, 0, LW);

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      e[i] = (lane + i * LW < width) ? expf(e[i] - row_max) : 0.0f;
      local_sum += e[i];
    }
    float den = WarpRedF(red_store).Reduce(local_sum, Sum());
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

template <class Bt, int LW, int IPT, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::backward_min_blocks(IPT)) void
softmax_backward_kernel(const OutT* __restrict__ values, const OutT* __restrict__ d_values,
                        const OutT* __restrict__ d_values_second, LogitT* __restrict__ d_logits,
                        int tokens, int heads, LevelTable table, Strides sv, Strides sdv,
                        Strides sdv2, Strides sdz) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  static_assert((LW & (LW - 1)) == 0, "a logical warp is a power of two of lanes");

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/factor.md#stream32-4
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];

  //: A BLOCK OF MORE LOGICAL WARPS THAN ITS WORD HAS TOKENS idles whole warps -- see docs/internals/entmax/factor.md#backward-stages
  if (logical >= WORD_TOKENS) return;

  //: A STAGE AHEAD: a trip takes its token's terms from what the last trip loaded, issues the next token's loads, then reduces and stores -- see docs/internals/entmax/factor.md#backward-stages
  const int t0 = word * WORD_TOKENS;
  BackwardReads<LW, IPT, OutT> reads{values, d_values, d_values_second, sv, sdv, sdv2};
  reads.lane = lane;
  reads.width = width;
  reads.tokens = tokens;
  reads.start(stream, t0, logical, value_base);
  OutT v[IPT] = {};
  OutT d[IPT] = {};
  OutT d2[IPT] = {};
  reads.issue(v, d, d2);

  constexpr int UNROLL = ITERS <= 2 ? ITERS : 1;
  int token = t0 + logical;
  int64_t rz = addr_split(sdz, sb, sh, token, logit_base + lane);
#pragma unroll UNROLL
  for (int it = 0; it < ITERS; ++it) {
    const bool in_call = token < tokens;
    float p[IPT];
    float dp[IPT];
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool own = in_call && (lane + i * LW < width);
      float x = load_value(d, i);
      if (d_values_second != nullptr) x += load_value(d2, i);
      p[i] = own ? load_value(v, i) : 0.0f;
      dp[i] = own ? x : 0.0f;
      dot += p[i] * dp[i];
    }
    if (it + 1 < ITERS) reads.issue(v, d, d2);
    dot = warp_sum<LW>(dot);
    int64_t az = rz;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const float dz = p[i] * (dp[i] - dot);
      if (in_call && lane + i * LW < width) store_value(d_logits, az, dz);
      az += LW * sdz.col;
    }
    rz += LOGICAL_PER_BLOCK * sdz.token;
    token += LOGICAL_PER_BLOCK;
  }
}

// ---------------------------------------------------------------------------
// HOST DISPATCH -- the matcher (`common/plan/entmax_launch.h`) chooses the arm from the call's
// facts, so the kernel branches on nothing. The level table's entries are the caller's grouping: same
// class, same alpha, same dtype -- docs/internals/entmax/factor.md#batched-levels
// ---------------------------------------------------------------------------
using FactorForward = plan::factor_forward::Kernel;
using FactorBackward = plan::factor_backward::Kernel;
using SoftmaxForward = plan::softmax_forward::Kernel;
using SoftmaxBackward = plan::softmax_backward::Kernel;

//: THE ARM SETS, defined over the generated selection headers: the key `(lw, ipt, alpha15, logit, value)` of the entmax kernels, `(lw, ipt, logit, value)` of the softmax ones
FSLA_DEFINE_ARM_SET(FactorForwardArmSet, FACTOR_FORWARD);
FSLA_DEFINE_ARM_SET(FactorBackwardArmSet, FACTOR_BACKWARD);
FSLA_DEFINE_ARM_SET(SoftmaxForwardArmSet, SOFTMAX_FORWARD);
FSLA_DEFINE_ARM_SET(SoftmaxBackwardArmSet, SOFTMAX_BACKWARD);

template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
void launch_factor_forward_at(const Tensor& logits, Tensor& values,
                              const std::optional<Tensor>& values_second, Tensor& support,
                              const LevelTable& table, int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams
  const int n_streams = logits.size(0) * heads, tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  const auto kernel = factor_forward_kernel<Bt, LW, IPT, ALPHA15, LogitT, OutT>;
  kernel<<<grid, BLOCK_THREADS, ForwardSmem<Bt, LW, IPT>::bytes, stream>>>(
      (const LogitT*)logits.data_ptr(), (OutT*)values.data_ptr(), ptr_or_null<OutT>(values_second),
      support.mutable_data_ptr<int32_t>(), tokens, heads, table, five_logits(logits, heads),
      five(values), five_or_zero(values_second), five(support));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_factor_forward_at` at the arm the matcher chose, at `logits`'s card's build: the one branch to the arm, and inside it the one branch to the build, outside every launch.
void launch_factor_forward(const Tensor& logits, Tensor& values,
                           const std::optional<Tensor>& values_second, Tensor& support,
                           const LevelTable& table, const plan::EntmaxLaunch<FactorForward>& launch,
                           int heads) {
  const plan::EntmaxAlphaArm& arm = launch.arm;
  arm_switch<FactorForwardArmSet>(
      {{arm.lw, arm.ipt, arm.alpha15, arm.logit, arm.value}}, [&](auto built) {
        constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
        constexpr bool kAlpha15 = decltype(built)::values[2] != 0;
        using LogitT = DtypeOfCode<decltype(built)::values[3]>;
        using OutT = DtypeOfCode<decltype(built)::values[4]>;
        build_switch(card_of(logits.get_device_index()), [&](auto b) {
          constexpr ::denseref::arch::Build kB = decltype(b)::build;
          if constexpr (in_image(kB)) {
            using Bt = ::denseref::arch::BuildTag<kB>;
            launch_factor_forward_at<Bt, kLane, kItems, kAlpha15, LogitT, OutT>(
                logits, values, values_second, support, table, launch.levels, heads);
          }
        });
      });
}

template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
void launch_factor_backward_at(const Tensor& values, const Tensor& support, const Tensor& d_values,
                               const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                               const LevelTable& table, int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-2
  const int n_streams = d_logits.size(0) * heads, tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  factor_backward_kernel<Bt, LW, IPT, ALPHA15, LogitT, OutT><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const OutT*)values.data_ptr(), support.mutable_data_ptr<int32_t>(),
      (const OutT*)d_values.data_ptr(), ptr_or_null<const OutT>(d_values_second),
      (LogitT*)d_logits.data_ptr(), tokens, heads, table, five(values), five(support),
      five(d_values), five_or_zero(d_values_second), five_logits(d_logits, heads));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_factor_backward_at` at the arm the matcher chose, at `values`'s card's build.
void launch_factor_backward(const Tensor& values, const Tensor& support, const Tensor& d_values,
                            const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                            const LevelTable& table,
                            const plan::EntmaxLaunch<FactorBackward>& launch, int heads) {
  const plan::EntmaxAlphaArm& arm = launch.arm;
  arm_switch<FactorBackwardArmSet>(
      {{arm.lw, arm.ipt, arm.alpha15, arm.logit, arm.value}}, [&](auto built) {
        constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
        constexpr bool kAlpha15 = decltype(built)::values[2] != 0;
        using LogitT = DtypeOfCode<decltype(built)::values[3]>;
        using OutT = DtypeOfCode<decltype(built)::values[4]>;
        build_switch(card_of(values.get_device_index()), [&](auto b) {
          constexpr ::denseref::arch::Build kB = decltype(b)::build;
          if constexpr (in_image(kB)) {
            using Bt = ::denseref::arch::BuildTag<kB>;
            launch_factor_backward_at<Bt, kLane, kItems, kAlpha15, LogitT, OutT>(
                values, support, d_values, d_values_second, d_logits, table, launch.levels, heads);
          }
        });
      });
}

template <class Bt, int LW, int IPT, typename LogitT, typename OutT>
void launch_softmax_forward_at(const Tensor& logits, Tensor& values,
                               const std::optional<Tensor>& values_second, const LevelTable& table,
                               int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-3
  const int n_streams = logits.size(0) * heads, tokens = logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  softmax_forward_kernel<Bt, LW, IPT, LogitT, OutT><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const LogitT*)logits.data_ptr(), (OutT*)values.data_ptr(), ptr_or_null<OutT>(values_second),
      tokens, heads, table, five_logits(logits, heads), five(values), five_or_zero(values_second));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_softmax_forward_at` at the arm the matcher chose, at `logits`'s card's build.
void launch_softmax_forward(const Tensor& logits, Tensor& values,
                            const std::optional<Tensor>& values_second, const LevelTable& table,
                            const plan::EntmaxLaunch<SoftmaxForward>& launch, int heads) {
  const plan::EntmaxPlainArm& arm = launch.arm;
  arm_switch<SoftmaxForwardArmSet>({{arm.lw, arm.ipt, arm.logit, arm.value}}, [&](auto built) {
    constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
    using LogitT = DtypeOfCode<decltype(built)::values[2]>;
    using OutT = DtypeOfCode<decltype(built)::values[3]>;
    build_switch(card_of(logits.get_device_index()), [&](auto b) {
      constexpr ::denseref::arch::Build kB = decltype(b)::build;
      if constexpr (in_image(kB)) {
        using Bt = ::denseref::arch::BuildTag<kB>;
        launch_softmax_forward_at<Bt, kLane, kItems, LogitT, OutT>(logits, values, values_second,
                                                                   table, launch.levels, heads);
      }
    });
  });
}

template <class Bt, int LW, int IPT, typename LogitT, typename OutT>
void launch_softmax_backward_at(const Tensor& values, const Tensor& d_values,
                                const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                                const LevelTable& table, int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/factor.md#nstreams-4
  const int n_streams = d_logits.size(0) * heads, tokens = d_logits.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  softmax_backward_kernel<Bt, LW, IPT, LogitT, OutT><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const OutT*)values.data_ptr(), (const OutT*)d_values.data_ptr(),
      ptr_or_null<const OutT>(d_values_second), (LogitT*)d_logits.data_ptr(), tokens, heads, table,
      five(values), five(d_values), five_or_zero(d_values_second), five_logits(d_logits, heads));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_softmax_backward_at` at the arm the matcher chose, at `values`'s card's build.
void launch_softmax_backward(const Tensor& values, const Tensor& d_values,
                             const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                             const LevelTable& table,
                             const plan::EntmaxLaunch<SoftmaxBackward>& launch, int heads) {
  const plan::EntmaxPlainArm& arm = launch.arm;
  arm_switch<SoftmaxBackwardArmSet>({{arm.lw, arm.ipt, arm.logit, arm.value}}, [&](auto built) {
    constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
    using LogitT = DtypeOfCode<decltype(built)::values[2]>;
    using OutT = DtypeOfCode<decltype(built)::values[3]>;
    build_switch(card_of(values.get_device_index()), [&](auto b) {
      constexpr ::denseref::arch::Build kB = decltype(b)::build;
      if constexpr (in_image(kB)) {
        using Bt = ::denseref::arch::BuildTag<kB>;
        launch_softmax_backward_at<Bt, kLane, kItems, LogitT, OutT>(
            values, d_values, d_values_second, d_logits, table, launch.levels, heads);
      }
    });
  });
}

void check_values(const Tensor& values, const std::optional<Tensor>& second) {
  STD_TORCH_CHECK(!second.has_value() || second->scalar_type() == values.scalar_type(),
                  "both route value outputs must share a dtype");
}

}  // namespace detail

//: THE ARMS THIS BINARY CARRIES, as `[lw, ipt, alpha15, logit, value]` rows (`[lw, ipt, logit, value]` for the softmax kernels)
std::vector<std::vector<int64_t>> factor_forward_arms() {
  return detail::FactorForwardArmSet::rows();
}

std::vector<std::vector<int64_t>> factor_backward_arms() {
  return detail::FactorBackwardArmSet::rows();
}

std::vector<std::vector<int64_t>> softmax_forward_arms() {
  return detail::SoftmaxForwardArmSet::rows();
}

std::vector<std::vector<int64_t>> softmax_backward_arms() {
  return detail::SoftmaxBackwardArmSet::rows();
}

void factor_forward(const Tensor& logits, Tensor& values,
                    const std::optional<Tensor>& values_second, Tensor& support_words,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(logits);
  STD_TORCH_CHECK(support_offsets.size() == widths.size(),
                  "the entmax forward writes a support word per level");
  detail::check_values(values, values_second);
  using K = detail::FactorForward;
  const int device = logits.get_device_index();
  const int logit = detail::dtype_code(logits.scalar_type());
  const int value = detail::dtype_code(values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_alpha_call<K>(width, levels, alpha, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table =
      detail::build_table(logit_offsets, value_offsets, support_offsets, widths, launch.klass);
  const DeviceGuard guard(device);
  detail::launch_factor_forward(logits, values, values_second, support_words, table, launch,
                                (int)heads);
}

void factor_backward(const Tensor& values, const Tensor& support_words, const Tensor& d_values,
                     const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(d_logits);
  STD_TORCH_CHECK(support_offsets.size() == widths.size(),
                  "the entmax backward reads a support word per level");
  detail::check_values(values, d_values_second);
  STD_TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
                  "cotangents must share the route values' dtype");
  using K = detail::FactorBackward;
  const int device = d_logits.get_device_index();
  const int logit = detail::dtype_code(d_logits.scalar_type());
  const int value = detail::dtype_code(values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_alpha_call<K>(width, levels, alpha, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table =
      detail::build_table(logit_offsets, value_offsets, support_offsets, widths, launch.klass);
  const DeviceGuard guard(device);
  detail::launch_factor_backward(values, support_words, d_values, d_values_second, d_logits, table,
                                 launch, (int)heads);
}

void softmax_forward(const Tensor& logits, Tensor& values,
                     const std::optional<Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                     int64_t heads) {
  check_arch_table();
  detail::check_logits(logits);
  detail::check_values(values, values_second);
  using K = detail::SoftmaxForward;
  const int device = logits.get_device_index();
  const int logit = detail::dtype_code(logits.scalar_type());
  const int value = detail::dtype_code(values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_call<K>(width, levels, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths, launch.klass);
  const DeviceGuard guard(device);
  detail::launch_softmax_forward(logits, values, values_second, table, launch, (int)heads);
}

void softmax_backward(const Tensor& values, const Tensor& d_values,
                      const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      int64_t heads) {
  check_arch_table();
  detail::check_logits(d_logits);
  detail::check_values(values, d_values_second);
  STD_TORCH_CHECK(d_values.scalar_type() == values.scalar_type(),
                  "cotangents must share the route values' dtype");
  using K = detail::SoftmaxBackward;
  const int device = d_logits.get_device_index();
  const int logit = detail::dtype_code(d_logits.scalar_type());
  const int value = detail::dtype_code(values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_call<K>(width, levels, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table = detail::build_table(logit_offsets, value_offsets, {}, widths, launch.klass);
  const DeviceGuard guard(device);
  detail::launch_softmax_backward(values, d_values, d_values_second, d_logits, table, launch,
                                  (int)heads);
}

}  // namespace entmax
}  // namespace fsla
