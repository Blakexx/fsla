// THE PRODUCER'S UNION ENTMAX SOLVE -- hand CUDA, CUB warp primitives.
// Design of record: docs/internals/entmax/entmax.md.
// Gates: `tests/unit/test_entmax_cuda_gates.py` (fp64 oracle only; the Triton kernel is
// a debugging diagnostic and is never asserted against).
//
// ONE ROW PER LOGICAL WARP: every collective runs along one token's row, so all of them
// are warp-scope and shuffle-based. Three facts follow, and each is load-bearing rather
// than incidental -- register demand is O(items-per-lane) and NOT O(CTA tile); the solve
// uses ZERO shared memory beyond the sort scratch, because every `cub::WarpScan` /
// `WarpReduce` TempStorage here resolves to a measured 1-byte pure-shfl specialization;
// and there is exactly ONE `__syncthreads` in each kernel, in the epilogue.
//
// The Triton history that forced hand CUDA, the phase-by-phase walk, the backward's
// Jacobian derivation and its measured spill trades: docs/internals/entmax/entmax.md

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/arm_switch.cuh"
#include "entmax.cuh"
#include "plan/entmax_launch.h"
#include "plan/environment.h"
#include "plan/union_backward/contract.h"
#include "plan/union_forward/contract.h"
#include "union_backward_selection.inc"
#include "union_forward_selection.inc"

namespace fsla {
namespace entmax {
namespace detail {

// ---------------------------------------------------------------------------
// FORWARD
// ---------------------------------------------------------------------------
template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::kMinBlocks) void
union_forward_kernel(const LogitT* __restrict__ read_logits,
                     const LogitT* __restrict__ write_logits, float* __restrict__ midpoint_values,
                     int32_t* __restrict__ support_words, OutT* __restrict__ read_values,
                     OutT* __restrict__ write_values, int tokens, int heads, LevelTable table,
                     Strides sr, Strides sw, Strides smid, Strides sread, Strides swrite,
                     Strides ssup) {
  constexpr int W_PAD = LW * IPT;
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;

  //: HAZARD smem-slot-alias -- docs/internals/entmax/entmax.md#smem-slot-alias -- see docs/internals/entmax/entmax.md#sort-bytes
  constexpr int SORT_BYTES = RowSort<LW, IPT>::kSlotBytes;
  constexpr int STAGE_BYTES = W_PAD * (int)sizeof(uint32_t);
  constexpr int RAW_SLOT = SORT_BYTES > STAGE_BYTES ? SORT_BYTES : STAGE_BYTES;
  constexpr int SLOT = (RAW_SLOT + 15) & ~15;
  __shared__ __align__(16) char smem[BLOCK_WARPS * SLOT];

  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int logical = threadIdx.x / LW;  // logical warp id within the block
  const int logical_in_warp = (threadIdx.x % WARP_SIZE) / LW;

  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/entmax.md#stream32
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t midpoint_base = table.midpoint_off[level];
  const int64_t support_base = table.support_off[level];

  uint32_t part[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) part[i] = 0u;

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    //: A ROW PAST THE END has no token: it is skipped -- see docs/internals/entmax/entmax.md#continue
    if (local >= WORD_TOKENS || token >= tokens) continue;

    // ---- PHASE A: load striped (coalesced), fold, RETIRE both logit tiles ----
    float midpoint[IPT], delta[IPT];
    uint32_t support_mask = 0u, saturated_mask = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const float r =
          valid ? load_value(read_logits, addr_split(sr, sb, sh, token, logit_base + col)) : 0.0f;
      const float w =
          valid ? load_value(write_logits, addr_split(sw, sb, sh, token, logit_base + col)) : 0.0f;
      midpoint[i] = 0.5f * r + 0.5f * w;
      bool sat = false;
      delta[i] = overflow_free_delta(r, w, &sat);
      if (sat) saturated_mask |= (1u << i);
    }

    // ---- PHASES B, C: the threshold of the midpoint row (`threshold`) ----
    uint32_t members = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) members |= (lane + i * LW < width ? 1u : 0u) << i;
    const float tau = threshold<LW, IPT, ALPHA15>(midpoint, members, width, lane, lmask,
                                                  smem + warp * SLOT, logical_in_warp);

    // ---- PHASE D: values in place over `midpoint`; `read` emitted HERE, while `delta`
    //      is still the logit difference and before phase E overwrites it. ----
    float values[IPT];
    uint32_t read_live_mask = 0u;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool sup = (col < width) && (midpoint[i] > tau);
      if (sup) support_mask |= (1u << i);
      float base = (ALPHA15 ? 0.5f : 1.0f) * (midpoint[i] - tau);
      base = sup ? fmaxf(base, 0.0f) : 0.0f;
      values[i] = ALPHA15 ? base * base : base;
      if (col < width) {
        midpoint_values[addr(smid, stream, token, midpoint_base + col)] = values[i];
        const float read = values[i] * (1.0f / (1.0f + expf(-delta[i])));
        store_value(read_values, addr(sread, stream, token, value_base + col), read);
        if (bf16_nonzero(read)) read_live_mask |= (1u << i);
      }
    }

    // ---- PHASE E: the write weights IN PLACE over `delta` (`union_write`) ----
    union_write<LW, IPT>(values, delta, support_mask, lmask);
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = (support_mask >> i) & 1u;
      store_value(write_values, addr(swrite, stream, token, value_base + col), delta[i]);
      //: HAZARD bf16-support-boundary -- docs/internals/entmax/entmax.md#bf16-support-boundary -- see docs/internals/entmax/entmax.md#near-line-267
      if (sup && (((read_live_mask >> i) & 1u) || bf16_nonzero(delta[i])))
        part[i] |= (1u << (token & 31));
    }
  }

  // ---- EPILOGUE: pack the 32-token support word. ONE barrier. ----
  //: The butterfly runs over exactly the bits at or above LW, which is what separates -- see docs/internals/entmax/entmax.md#near-line-275
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

// ---------------------------------------------------------------------------
// BACKWARD -- the alpha-entmax Jacobian action through the union split.
// see docs/internals/entmax/entmax.md#launch-bounds
template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
__global__ __launch_bounds__(BLOCK_THREADS, EntmaxPlan<Bt::build>::backward_min_blocks(IPT)) void
union_backward_kernel(const LogitT* __restrict__ read_logits,
                      const LogitT* __restrict__ write_logits,
                      const float* __restrict__ midpoint_values,
                      const int32_t* __restrict__ support_words,
                      const OutT* __restrict__ read_values, const OutT* __restrict__ write_values,
                      const OutT* __restrict__ d_read, const OutT* __restrict__ d_write,
                      LogitT* __restrict__ d_read_logits, LogitT* __restrict__ d_write_logits,
                      int tokens, int heads, LevelTable table, Strides sr, Strides sw, Strides smid,
                      Strides srv, Strides swv, Strides sdr, Strides sdw, Strides sdrl,
                      Strides sdwl, Strides ssup) {
  constexpr int LOGICAL_PER_BLOCK = BLOCK_THREADS / LW;
  constexpr int ITERS = (WORD_TOKENS + LOGICAL_PER_BLOCK - 1) / LOGICAL_PER_BLOCK;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedP = cub::WarpReduce<Pair, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedP::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;
  typename WarpRedP::TempStorage redp_store;

  const int lane = threadIdx.x % LW;
  const int logical = threadIdx.x / LW;
  const unsigned lmask = logical_warp_mask<LW>();
  const int64_t stream = blockIdx.y;
  const int word = blockIdx.x;
  //: ONE 32-BIT division per BLOCK on a uniform value: `blockIdx.y` is the -- see docs/internals/entmax/entmax.md#stream32-2
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int level = blockIdx.z;
  const int width = table.width[level];
  const int64_t logit_base = table.logit_off[level];
  const int64_t value_base = table.value_off[level];
  const int64_t midpoint_base = table.midpoint_off[level];
  const int64_t support_base = table.support_off[level];

  for (int it = 0; it < ITERS; ++it) {
    const int local = it * LOGICAL_PER_BLOCK + logical;
    const int token = word * WORD_TOKENS + local;
    if (local >= WORD_TOKENS || token >= tokens) continue;

    uint32_t support_mask = 0u, saturated_mask = 0u;
    // ---- the write-side rank-1: sum(d_write * write_values) ----
    //: This pass carries NOTHING into the next one but two scalars (`local_dot` and the -- see docs/internals/entmax/entmax.md#localdot
    float local_dot = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const uint32_t wword =
          valid ? (uint32_t)support_words[stream * ssup.stream + (int64_t)word * ssup.token
                                          + (support_base + col) * ssup.col]
                : 0u;
      if (valid && ((wword >> (token & 31)) & 1u)) support_mask |= (1u << i);
      const float wv =
          valid ? load_value(write_values, addr(swv, stream, token, value_base + col)) : 0.0f;
      const float dw =
          valid ? load_value(d_write, addr(sdw, stream, token, value_base + col)) : 0.0f;
      local_dot += dw * wv;
    }
    float dot = WarpRedF(red_store).Reduce(local_dot, Sum());
    dot = __shfl_sync(lmask, dot, 0, LW);

    //: `s` is NOT kept: it is recomputed from a re-read `midpoint_values` in the final -- see docs/internals/entmax/entmax.md#near-line-368
    float d_delta[IPT], d_mid[IPT];
    Pair partial{0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      const bool valid = col < width;
      const bool sup = (support_mask >> i) & 1u;
      const float wv =
          valid ? load_value(write_values, addr(swv, stream, token, value_base + col)) : 0.0f;
      const float dw =
          valid ? load_value(d_write, addr(sdw, stream, token, value_base + col)) : 0.0f;
      const float dlw = wv * (dw - dot);
      const float r =
          valid ? load_value(read_logits, addr_split(sr, sb, sh, token, logit_base + col)) : 0.0f;
      const float w =
          valid ? load_value(write_logits, addr_split(sw, sb, sh, token, logit_base + col)) : 0.0f;
      bool sat = false;
      const float delta = overflow_free_delta(r, w, &sat);
      if (sat) saturated_mask |= (1u << i);
      const float read_gate = 1.0f / (1.0f + expf(-delta));
      const float write_gate = 1.0f / (1.0f + expf(delta));
      const float mid =
          valid ? midpoint_values[addr(smid, stream, token, midpoint_base + col)] : 0.0f;
      const float dr =
          valid ? load_value(d_read, addr(sdr, stream, token, value_base + col)) : 0.0f;
      const float rv =
          valid ? load_value(read_values, addr(srv, stream, token, value_base + col)) : 0.0f;
      //: the VJP divides by the amplitude, which is exactly 0 at a TIE: substitute 1 and -- see docs/internals/entmax/entmax.md#safe
      const float safe = (sup && mid != 0.0f) ? mid : 1.0f;
      d_mid[i] = dr * read_gate + (sup ? dlw / safe : 0.0f);
      d_delta[i] = (dr * rv * write_gate - dlw * read_gate) * (sat ? 0.0f : 1.0f);
      //: the entmax rank-1's two sums accumulate HERE, so `s` never becomes an array.
      const float si = sup ? (ALPHA15 ? sqrtf(mid) : 1.0f) : 0.0f;
      partial.p += si * d_mid[i];
      partial.q += si;
    }

    // ---- the entmax rank-1: (numerator, denominator) in ONE fused reduction ----
    Pair red = WarpRedP(redp_store).Reduce(partial, PairAdd());
    float num = __shfl_sync(lmask, red.p, 0, LW);
    float den = __shfl_sync(lmask, red.q, 0, LW);
    den = den > 0.0f ? den : 1.0f;

#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + i * LW;
      if (col >= width) continue;
      const bool sup = (support_mask >> i) & 1u;
      const float mid = midpoint_values[addr(smid, stream, token, midpoint_base + col)];
      const float si = sup ? (ALPHA15 ? sqrtf(mid) : 1.0f) : 0.0f;
      const float d_ml = si * (d_mid[i] - num / den);
      store_value(d_read_logits, addr_split(sdrl, sb, sh, token, logit_base + col),
                  0.5f * d_ml + d_delta[i]);
      store_value(d_write_logits, addr_split(sdwl, sb, sh, token, logit_base + col),
                  0.5f * d_ml - d_delta[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// HOST DISPATCH -- the matcher (`common/plan/entmax_launch.h`) chooses the arm from the call's
// facts, so the kernel branches on nothing. One launch per union level. The arm's lane width and
// items a thread and why the launch count is unchanged from the Triton lowering:
// docs/internals/entmax/entmax.md#launch-shape
// ---------------------------------------------------------------------------
using UnionForward = plan::union_forward::Kernel;
using UnionBackward = plan::union_backward::Kernel;

//: THE ARM SETS, defined over the generated selection headers: the key `(lw, ipt, alpha15, logit, value)` of each kernel
FSLA_DEFINE_ARM_SET(UnionForwardArmSet, UNION_FORWARD);
FSLA_DEFINE_ARM_SET(UnionBackwardArmSet, UNION_BACKWARD);

template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
void launch_forward_at(const Tensor& rl, const Tensor& wl, Tensor& mid, Tensor& sup, Tensor& rv,
                       Tensor& wv, const LevelTable& table, int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/entmax.md#nstreams
  const int n_streams = rl.size(0) * heads, tokens = rl.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  union_forward_kernel<Bt, LW, IPT, ALPHA15, LogitT, OutT><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const LogitT*)rl.data_ptr(), (const LogitT*)wl.data_ptr(), mid.mutable_data_ptr<float>(),
      sup.mutable_data_ptr<int32_t>(), (OutT*)rv.data_ptr(), (OutT*)wv.data_ptr(), tokens, heads,
      table, five_logits(rl, heads), five_logits(wl, heads), five(mid), five(rv), five(wv),
      five(sup));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_forward_at` at the arm the matcher chose, at `rl`'s card's build: the one branch to the arm, and inside it the one branch to the build, outside every launch.
void launch_forward(const Tensor& rl, const Tensor& wl, Tensor& mid, Tensor& sup, Tensor& rv,
                    Tensor& wv, const LevelTable& table,
                    const plan::EntmaxLaunch<UnionForward>& launch, int heads) {
  const plan::EntmaxAlphaArm& arm = launch.arm;
  arm_switch<UnionForwardArmSet>(
      {{arm.lw, arm.ipt, arm.alpha15, arm.logit, arm.value}}, [&](auto built) {
        constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
        constexpr bool kAlpha15 = decltype(built)::values[2] != 0;
        using LogitT = DtypeOfCode<decltype(built)::values[3]>;
        using OutT = DtypeOfCode<decltype(built)::values[4]>;
        build_switch(card_of(rl.get_device_index()), [&](auto b) {
          constexpr ::denseref::arch::Build kB = decltype(b)::build;
          if constexpr (in_image(kB)) {
            using Bt = ::denseref::arch::BuildTag<kB>;
            launch_forward_at<Bt, kLane, kItems, kAlpha15, LogitT, OutT>(
                rl, wl, mid, sup, rv, wv, table, launch.levels, heads);
          }
        });
      });
}

template <class Bt, int LW, int IPT, bool ALPHA15, typename LogitT, typename OutT>
void launch_backward_at(const Tensor& rl, const Tensor& wl, const Tensor& mid, const Tensor& sup,
                        const Tensor& rv, const Tensor& wv, const Tensor& dr, const Tensor& dw,
                        Tensor& drl, Tensor& dwl, const LevelTable& table, int levels, int heads) {
  //: A token-major plane's stream is TWO axes, so the grid's stream extent is `B * H`; -- see docs/internals/entmax/entmax.md#nstreams-2
  const int n_streams = drl.size(0) * heads, tokens = drl.size(1);
  dim3 grid((tokens + WORD_TOKENS - 1) / WORD_TOKENS, n_streams, levels);
  auto stream = current_stream();
  union_backward_kernel<Bt, LW, IPT, ALPHA15, LogitT, OutT><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const LogitT*)rl.data_ptr(), (const LogitT*)wl.data_ptr(), mid.mutable_data_ptr<float>(),
      sup.mutable_data_ptr<int32_t>(), (const OutT*)rv.data_ptr(), (const OutT*)wv.data_ptr(),
      (const OutT*)dr.data_ptr(), (const OutT*)dw.data_ptr(), (LogitT*)drl.data_ptr(),
      (LogitT*)dwl.data_ptr(), tokens, heads, table, five_logits(rl, heads), five_logits(wl, heads),
      five(mid), five(rv), five(wv), five(dr), five(dw), five_logits(drl, heads),
      five_logits(dwl, heads), five(sup));
  FSLA_CUDA_LAUNCH_CHECK();
}

//: `launch_backward_at` at the arm the matcher chose, at `rl`'s card's build.
void launch_backward(const Tensor& rl, const Tensor& wl, const Tensor& mid, const Tensor& sup,
                     const Tensor& rv, const Tensor& wv, const Tensor& dr, const Tensor& dw,
                     Tensor& drl, Tensor& dwl, const LevelTable& table,
                     const plan::EntmaxLaunch<UnionBackward>& launch, int heads) {
  const plan::EntmaxAlphaArm& arm = launch.arm;
  arm_switch<UnionBackwardArmSet>(
      {{arm.lw, arm.ipt, arm.alpha15, arm.logit, arm.value}}, [&](auto built) {
        constexpr int kLane = decltype(built)::values[0], kItems = decltype(built)::values[1];
        constexpr bool kAlpha15 = decltype(built)::values[2] != 0;
        using LogitT = DtypeOfCode<decltype(built)::values[3]>;
        using OutT = DtypeOfCode<decltype(built)::values[4]>;
        build_switch(card_of(rl.get_device_index()), [&](auto b) {
          constexpr ::denseref::arch::Build kB = decltype(b)::build;
          if constexpr (in_image(kB)) {
            using Bt = ::denseref::arch::BuildTag<kB>;
            launch_backward_at<Bt, kLane, kItems, kAlpha15, LogitT, OutT>(
                rl, wl, mid, sup, rv, wv, dr, dw, drl, dwl, table, launch.levels, heads);
          }
        });
      });
}

}  // namespace detail

//: THE ARMS THIS BINARY CARRIES, as `[lw, ipt, alpha15, logit, value]` rows
std::vector<std::vector<int64_t>> union_forward_arms() {
  return detail::UnionForwardArmSet::rows();
}

std::vector<std::vector<int64_t>> union_backward_arms() {
  return detail::UnionBackwardArmSet::rows();
}

void union_forward(const Tensor& read_logits, const Tensor& write_logits, Tensor& midpoint_values,
                   Tensor& support_words, Tensor& read_values, Tensor& write_values,
                   const std::vector<int64_t>& logit_offsets,
                   const std::vector<int64_t>& value_offsets,
                   const std::vector<int64_t>& midpoint_offsets,
                   const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                   int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(read_logits);
  STD_TORCH_CHECK(read_logits.scalar_type() == write_logits.scalar_type(),
                  "read/write routing logits must share a dtype");
  STD_TORCH_CHECK(read_values.scalar_type() == write_values.scalar_type(),
                  "read/write route values must share a dtype");
  using K = detail::UnionForward;
  const int device = read_logits.get_device_index();
  const int logit = detail::dtype_code(read_logits.scalar_type());
  const int value = detail::dtype_code(read_values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_alpha_call<K>(width, levels, alpha, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths,
                                         launch.klass, midpoint_offsets);
  const DeviceGuard guard(device);
  detail::launch_forward(read_logits, write_logits, midpoint_values, support_words, read_values,
                         write_values, table, launch, (int)heads);
}

void union_backward(const Tensor& read_logits, const Tensor& write_logits,
                    const Tensor& midpoint_values, const Tensor& support_words,
                    const Tensor& read_values, const Tensor& write_values, const Tensor& d_read,
                    const Tensor& d_write, Tensor& d_read_logits, Tensor& d_write_logits,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& midpoint_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha) {
  check_arch_table();
  detail::check_logits(read_logits);
  STD_TORCH_CHECK(read_logits.scalar_type() == d_read_logits.scalar_type(),
                  "the union VJP writes logit gradients in the logits' own dtype");
  using K = detail::UnionBackward;
  const int device = read_logits.get_device_index();
  const int logit = detail::dtype_code(read_logits.scalar_type());
  const int value = detail::dtype_code(read_values.scalar_type());
  const int64_t width = detail::first_width(widths);
  const int64_t levels = static_cast<int64_t>(widths.size());
  const auto call = plan::entmax_alpha_call<K>(width, levels, alpha, logit, value);
  const auto launch = plan::entmax_launch<K>(plan::descriptor_of<K>(device, call));
  const auto table = detail::build_table(logit_offsets, value_offsets, support_offsets, widths,
                                         launch.klass, midpoint_offsets);
  const DeviceGuard guard(device);
  detail::launch_backward(read_logits, write_logits, midpoint_values, support_words, read_values,
                          write_values, d_read, d_write, d_read_logits, d_write_logits, table,
                          launch, (int)heads);
}

//: THE LEVEL BOUND and THE WIDTH CLASS, for the host: pure functions of the matcher, no device
int64_t max_levels() { return plan::kEntmaxMaxLevels; }

int64_t width_class(int64_t width) { return plan::entmax_padded_width(static_cast<int>(width)); }

}  // namespace entmax
}  // namespace fsla
