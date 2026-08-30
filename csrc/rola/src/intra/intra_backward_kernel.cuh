// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
//
// THE INTRA TERM'S REVERSE PASS.
//
// RECOMPUTE, NEVER SAVE, AT BOTH GRAINS.  The pairwise coefficient `A` is `[W x W]`
// per window -- the snapshot class this pass's memory contract forbids -- so it rebuilds
// the per-level Grams from the same staged slabs the forward reads, under the same
// slab-occupancy mask, and forms `A` in registers.  It is a SEPARATE kernel because the
// forward intra must stay a pure forward for inference: `dA` needs the seed, which
// exists only in training, and fusing would put a training-only operand in an inference
// kernel's signature.
//
// The four cotangents, in the order the kernel computes them (`t` a read token, `j` a
// write token, `e` a digit, `l` a level, `G_l` the level's Gram, `P = prod_l G_l`):
//
//     Gd'[t,j] = <d_num[t,:], v[j,:]> + d_den[t]      the ONE new pairwise object
//     dA[t,j]  = [j <= t] * Gd'[t,j]
//     d_v[j,:]   += sum_t A[t,j] * d_num[t,:]
//     d_gw[j]    += sum_t dA[t,j] * P[t,j]
//     dG_l[t,j]  = dA[t,j] * gw[j] * prod_{l' != l} G_l'[t,j]
//     d_pr_l[t,e] += sum_j dG_l[t,j] * pw_l[j,e]
//     d_pw_l[j,e] += sum_t dG_l[t,j] * pr_l[t,e]
//
// THE PER-LEVEL LEAVE-ONE-OUT SURVIVES HERE and is computed as a prefix/suffix product
// over the `D` Grams already in registers -- never as `A / G_l`, because `G_l` is
// exactly zero wherever the two tokens' level-`l` supports are disjoint, which under an
// ALT structure is the common case.  See docs/internals/intra/intra_backward.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/design.cuh"
#include "common/ops.cuh"
#include "intra/intra_kernel.cuh"

namespace rola {
namespace intra {
namespace detail {

//: the reverse intra's own device stamp: a fact only THIS binary can produce.
constexpr int64_t kIntraBwdMagic = 31040003;

constexpr int kSeedPitch = kValuePitch;
constexpr int kChanTiles = kValueWidth / kSlabDigits;

constexpr int warps_of(int window) { return threads_of(window) / 32; }

//: THE REVERSE STAGES HALF AS WIDE AS THE FORWARD. Its block carries the -- see docs/internals/intra/intra_backward_kernel.md#kbwdstagecap
constexpr int kBwdStageCap = 32;

constexpr int bwd_stage_digits(int width) {
  return level_digits(width) < kBwdStageCap ? level_digits(width) : kBwdStageCap;
}

constexpr int bwd_slabs_per_stage(int width) { return bwd_stage_digits(width) / kSlabDigits; }

constexpr int bwd_stages_per_level(int width) {
  return level_digits(width) / bwd_stage_digits(width);
}

constexpr int bwd_operand_pitch(int width) { return bwd_stage_digits(width) + 8; }

//: THE RELAY PLANE, one `[kMmaM x kSlabDigits]` tile per warp: every cotangent -- see docs/internals/intra/intra_backward_kernel.md#kxposerows
constexpr int kXposeRows = 16;
constexpr int kXposePitch = kSlabDigits + 2;

//: THE `dG_l` IMAGE: the tile pair's coefficient object, `[C reader tokens x -- see docs/internals/intra/intra_backward_kernel.md#kimagepitch
constexpr int kImagePitch = kTile + 8;

//: The forward's block footprint plus the ROW side's seed rows, the per-warp relay -- see docs/internals/intra/intra_backward_kernel.md#near-line-77
template <int D, int B, int W>
struct alignas(16) IntraBwdSmem {
  __nv_bfloat16 rslab[2][rows_of(W)][bwd_operand_pitch(B)];
  __nv_bfloat16 wslab[2][kTile][bwd_operand_pitch(B)];
  __nv_bfloat16 vslab[kTile][kValuePitch];
  __nv_bfloat16 dslab[rows_of(W)][kSeedPitch];
  __nv_bfloat16 gain[kTile];
  float dden[rows_of(W)];
  alignas(16) float xpose[warps_of(W)][kXposeRows][kXposePitch];
  uint32_t mask[tiles_of(W)][D][2][mask_words(B)];
  __nv_bfloat16 img[rows_of(W)][kImagePitch];
};

struct IntraBwdParams {
  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* v;
  const __nv_bfloat16* dnum;
  const float* dden;
  float* d_pr;
  float* d_pw;
  float* d_gw;
  float* d_v;
  int L;
  int row_width;
  int heads;
  int tokens;
  uint32_t modes;
};

//: the mask helpers read one CTA-local summary, so the reverse pass's two one-sided -- see docs/internals/intra/intra_backward_kernel.md#near-line-111
template <int D, int B, int W>
__device__ __forceinline__ uint32_t side_bits(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                              int ti, int l, int side, int slab) {
  const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
  if (((modes >> (2 * l)) & want) == 0u) return 0xFFFFu;
  return (sm.mask[ti][l][side][slab >> 1] >> ((slab & 1) * 16)) & 0xFFFFu;
}

template <int D, int B, int W>
__device__ __forceinline__ bool side_stage_live(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                                int ti, int l, int side, int stage) {
  bool any = false;
  //: §6: bwd_slabs_per_stage(B) is compile-time (B is this function's template -- see docs/internals/intra/intra_backward_kernel.md#kslabsperstage
  constexpr int kSlabsPerStage = bwd_slabs_per_stage(B);
#pragma unroll
  for (int sub = 0; sub < kSlabsPerStage; ++sub)
    any = any || side_bits<D, B, W>(sm, modes, ti, l, side, stage * kSlabsPerStage + sub) != 0u;
  return any;
}

//: THE GRAM PASS'S GATE IS THE INTERSECTION and the digit passes' is ONE SIDE. -- see docs/internals/intra/intra_backward_kernel.md#live-rows-side
template <int D, int B, int W>
__device__ __forceinline__ uint32_t live_rows_side(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                                   int rt0, int ct, int l, int stage) {
  uint32_t bits = 0u;
  //: §6: row_tiles_of(W) is compile-time (W is this function's template parameter) -- see docs/internals/intra/intra_backward_kernel.md#krowtiles
  constexpr int kRowTiles = row_tiles_of(W);
#pragma unroll
  for (int r = 0; r < kRowTiles; ++r)
    if (rt0 + r >= ct && side_stage_live<D, B, W>(sm, modes, rt0 + r, l, 0, stage)) bits |= 1u << r;
  return bits;
}

//: THE GRAM PASS'S OWN GATE: both sides' supports must meet in the digit slab, which is -- see docs/internals/intra/intra_backward_kernel.md#near-line-149
template <int D, int B, int W>
__device__ __forceinline__ bool pair_slab_live(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                               int rt, int ct, int l, int slab) {
  return (side_bits<D, B, W>(sm, modes, rt, l, 0, slab)
          & side_bits<D, B, W>(sm, modes, ct, l, 1, slab))
         != 0u;
}

template <int D, int B, int W>
__device__ __forceinline__ uint32_t live_rows_pair(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                                   int rt0, int ct, int l, int stage) {
  uint32_t bits = 0u;
  //: §6: row_tiles_of(W)/bwd_slabs_per_stage(B) are compile-time (D, B, W are this -- see docs/internals/intra/intra_backward_kernel.md#krowtiles-2
  constexpr int kRowTiles = row_tiles_of(W);
  constexpr int kSlabsPerStage = bwd_slabs_per_stage(B);
#pragma unroll
  for (int r = 0; r < kRowTiles; ++r) {
    if (rt0 + r < ct) continue;
    bool any = false;
#pragma unroll
    for (int sub = 0; sub < kSlabsPerStage; ++sub)
      any = any || pair_slab_live<D, B, W>(sm, modes, rt0 + r, ct, l, stage * kSlabsPerStage + sub);
    if (any) bits |= 1u << r;
  }
  return bits;
}

template <int D, int B, int W>
__device__ __forceinline__ void build_bwd_masks(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                                int bh, int wbase, int tid) {
  //: THE HALF-SPLIT CERTIFICATE PASS needs a level at least two words wide.  A -- see docs/internals/intra/intra_backward_kernel.md#near-line-183
  static_assert(mask_words(B) % 2 == 0, "the reverse certificate pass splits a level in half");
  constexpr int kTiles = tiles_of(W);
  constexpr int kThreads = threads_of(W);
  constexpr int kCohorts = kThreads / (2 * kTile);
  uint32_t* flat = &sm.mask[0][0][0][0];
  //: §6: both bounds are compile-time (kTiles/D are locals/template params, mask_words(B) -- see docs/internals/intra/intra_backward_kernel.md#maskwords
  constexpr int kFlatWords = kTiles * D * 2 * mask_words(B);
  for (int i = tid; i < kFlatWords; i += kThreads) flat[i] = 0u;
  __syncthreads();

  const int token = tid & (kTile - 1);
  const int half = (tid >> 6) & 1;
  const int cohort = tid >> 7;
#pragma unroll
  for (int l = 0; l < D; ++l) {
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
      if (((p.modes >> (2 * l)) & want) == 0u) continue;
      const __nv_bfloat16* plane = (side == 0) ? p.pread : p.pwrite;
      //: §6: mask_words(B)/2 is compile-time (B is this function's template parameter) -- see docs/internals/intra/intra_backward_kernel.md#kmaskwordshalf
      constexpr int kMaskWordsHalf = mask_words(B) / 2;
      for (int ti = cohort; ti < kTiles; ti += kCohorts) {
        const __nv_bfloat16* row =
            plane + (long long)(bh * p.L + wbase + ti * kTile + token) * p.row_width + l * B
            + half * (B / 2);
#pragma unroll
        for (int j = 0; j < kMaskWordsHalf; ++j) {
          const uint32_t bits = __reduce_or_sync(0xFFFFFFFFu, digit_union_word<32>(row + j * 32));
          if ((tid & 31) == 0) atomicOr(&sm.mask[ti][l][side][half * kMaskWordsHalf + j], bits);
        }
      }
    }
  }
  __syncthreads();
}

template <int D, int B, int W>
__device__ __forceinline__ void stage_bwd_slabs(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                                int bh, int wbase, int rt0, int ct, int l,
                                                int stage, int buf, int tid, uint32_t rows,
                                                bool cols) {
  const int digit = l * B + stage * bwd_stage_digits(B);
  const long long base = (long long)(bh * p.L + wbase) * p.row_width + digit;
  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColDigits = bwd_stage_digits(B) / kColThreads;
  //: §6: these are all compile-time (B, W are this function's template parameters) -- see docs/internals/intra/intra_backward_kernel.md#kcoldigitoctets
  constexpr int kColDigitOctets = kColDigits / 8;
  constexpr int kStageDigitsHex = bwd_stage_digits(B) / 16;
  if (cols) {
    const int ctoken = tid / kColThreads;
    const int cpart = (tid % kColThreads) * kColDigits;
    const __nv_bfloat16* wsrc =
        p.pwrite + base + (long long)(ct * kTile + ctoken) * p.row_width + cpart;
#pragma unroll
    for (int c = 0; c < kColDigitOctets; ++c)
      stage_tile(smem_addr(&sm.wslab[buf][ctoken][cpart + c * 8]), wsrc + c * 8);
  }
  const int rtoken = (tid >> 1) & (kTile - 1);
  const int rtile = tid / (2 * kTile);
  const int rpart = (tid & 1) * (bwd_stage_digits(B) / 2);
  if ((rows >> rtile) & 1u) {
    const __nv_bfloat16* rsrc =
        p.pread + base + (long long)((rt0 + rtile) * kTile + rtoken) * p.row_width + rpart;
#pragma unroll
    for (int c = 0; c < kStageDigitsHex; ++c)
      stage_tile(smem_addr(&sm.rslab[buf][rtile * kTile + rtoken][rpart + c * 8]), rsrc + c * 8);
  }
}

template <int D, int B, int W>
__device__ __forceinline__ void stage_bwd_values(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                                 int bh, int wbase, int ct, int tid) {
  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColChannels = kValueWidth / kColThreads;
  //: §6: kColChannels/8 is compile-time but the heuristic only reads a bare -- see docs/internals/intra/intra_backward_kernel.md#kcolchanneloctets
  constexpr int kColChannelOctets = kColChannels / 8;
  const int b = bh / p.heads;
  const int h = bh - b * p.heads;
  const int t0 = wbase + ct * kTile;
  const int token = tid / kColThreads;
  const int part = (tid % kColThreads) * kColChannels;
  const __nv_bfloat16* src =
      p.v + ((long long)(b * p.tokens + t0 + token) * p.heads + h) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < kColChannelOctets; ++c)
    stage_tile(smem_addr(&sm.vslab[token][part + c * 8]), src + c * 8);
  if (tid < kTile / 8)
    stage_tile(smem_addr(&sm.gain[tid * 8]), p.gwrite + (long long)bh * p.L + t0 + tid * 8);
}

//: THE SEED, staged ONCE PER OUTPUT BLOCK: its rows are the block's readers, which do -- see docs/internals/intra/intra_backward_kernel.md#near-line-284
template <int D, int B, int W>
__device__ __forceinline__ void stage_bwd_seed(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                               int bh, int wbase, int rt0, int tid) {
  constexpr int kThreads = threads_of(W);
  constexpr int kRows = rows_of(W);
  constexpr int kRowThreads = kThreads / kRows;
  constexpr int kRowChannels = kValueWidth / kRowThreads;
  //: §6: kRowChannels/8 is compile-time but the heuristic only reads a bare -- see docs/internals/intra/intra_backward_kernel.md#krowchanneloctets
  constexpr int kRowChannelOctets = kRowChannels / 8;
  const int t0 = wbase + rt0 * kTile;
  const int token = tid / kRowThreads;
  const int part = (tid % kRowThreads) * kRowChannels;
  const __nv_bfloat16* src = p.dnum + (long long)(bh * p.L + t0 + token) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < kRowChannelOctets; ++c)
    stage_tile(smem_addr(&sm.dslab[token][part + c * 8]), src + c * 8);
  if (tid < kRows) sm.dden[tid] = p.dden[(long long)bh * p.L + t0 + tid];
}

//: THE PUBLISH. One row-carved `[kWarpRows x kTile]` accumulator becomes bf16
//: rows of the image, at the rounding pass 2's A operand carried anyway. -- see docs/internals/intra/intra_backward_kernel.md#publish-image
template <int D, int B, int W>
__device__ __forceinline__ void publish_image(IntraBwdSmem<D, B, W>& sm,
                                              const float (&acc)[kTile / 8][4], int warp,
                                              int row_lo, int col_lo) {
  __nv_bfloat16* lo = &sm.img[warp * kWarpRows + row_lo][col_lo];
  __nv_bfloat16* hi = &sm.img[warp * kWarpRows + row_lo + 8][col_lo];
  //: §6: kTile/8 is compile-time (kTile is a file-level constexpr) but the -- see docs/internals/intra/intra_backward_kernel.md#ktileoctets
  constexpr int kTileOctets = kTile / 8;
#pragma unroll
  for (int n = 0; n < kTileOctets; ++n) {
    *reinterpret_cast<uint32_t*>(lo + n * 8) = pack_bf16x2(acc[n][0], acc[n][1]);
    *reinterpret_cast<uint32_t*>(hi + n * 8) = pack_bf16x2(acc[n][2], acc[n][3]);
  }
}

//: THE RELAY.  `acc` is one `[kMmaM x kSlabDigits]` accumulator in the MMA's lane map -- see docs/internals/intra/intra_backward_kernel.md#relaytile
__device__ __forceinline__ void relay_tile(float* put, const float (&acc)[2][4]) {
  __syncwarp();
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      *reinterpret_cast<float2*>(put + h * 8 * kXposePitch + nt * 8) =
          make_float2(acc[nt][2 * h], acc[nt][2 * h + 1]);
  __syncwarp();
}

//: THE ONE REDUCING PATH. `get` is the lane's cursor into the relayed plane and -- see docs/internals/intra/intra_backward_kernel.md#reduce-tile
__device__ __forceinline__ void reduce_tile(float* put, const float* get, int lane_off,
                                            const float (&acc)[2][4], float* base, int stride) {
  relay_tile(put, acc);
  int off = (lane_off >> 8) * stride + (lane_off & 0xFF);
  //: §6: kXposeRows/2 is compile-time (kXposeRows is a file-level constexpr) but the -- see docs/internals/intra/intra_backward_kernel.md#kxposerowshalf
  constexpr int kXposeRowsHalf = kXposeRows / 2;
#pragma unroll
  for (int q = 0; q < kXposeRowsHalf; ++q, off += 2 * stride)
    fan_in_add(base + off, get[q * 2 * kXposePitch]);
}

//: THE MAGNITUDE BITS, so a signed zero is off support exactly as an unsigned one is.
__device__ __forceinline__ unsigned support_bits(denseref::ops::SmemAddr a) {
  unsigned x;
  asm("ld.shared.u16 %0, [%1];" : "=r"(x) : "r"(a));
  return x & 0x7FFFu;
}

//: A PREDICATED reducing instruction rather than a branched one: the gate is a -- see docs/internals/intra/intra_backward_kernel.md#fan-in-add-if
__device__ __forceinline__ void fan_in_add_if(unsigned live, float* p, float x) {
  asm volatile("{ .reg .pred q; setp.ne.u32 q, %2, 0; @q red.global.add.f32 [%0], %1; }" ::"l"(p),
               "f"(x), "r"(live)
               : "memory");
}

//: THE SAME PATH, RESTRICTED TO THE DESTINATION PLANE'S OWN SUPPORT. -- see docs/internals/intra/intra_backward_kernel.md#reduce-tile-on-support
__device__ __forceinline__ void reduce_tile_on_support(float* put, const float* get, int lane_off,
                                                       const float (&acc)[2][4], float* base,
                                                       int stride, denseref::ops::SmemAddr sup,
                                                       int sup_step, bool restricted) {
  if (!restricted) {
    reduce_tile(put, get, lane_off, acc, base, stride);
    return;
  }
  relay_tile(put, acc);
  int off = (lane_off >> 8) * stride + (lane_off & 0xFF);
  //: §6: kXposeRows/2 is compile-time (kXposeRows is a file-level constexpr) but the -- see docs/internals/intra/intra_backward_kernel.md#kxposerowshalf-2
  constexpr int kXposeRowsHalf = kXposeRows / 2;
#pragma unroll 2
  for (int q = 0; q < kXposeRowsHalf; ++q, off += 2 * stride, sup += sup_step)
    fan_in_add_if(support_bits(sup), base + off, get[q * 2 * kXposePitch]);
}

//: THE SECOND LAUNCH BOUND, from the arm's own footprint. Without it ptxas -- see docs/internals/intra/intra_backward_kernel.md#kbwdregfloor
constexpr int kBwdRegFloor = 232;

template <int D, int B, int W>
constexpr int kBwdCtaTarget =
    denseref::design::cta_target((int)sizeof(IntraBwdSmem<D, B, W>), threads_of(W), kBwdRegFloor);

template <int D, int B, int W>
__global__ __launch_bounds__(threads_of(W), (kBwdCtaTarget<D, B, W>)) void intra_backward_kernel(
    IntraBwdParams p) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kRowTiles = row_tiles_of(W);
  constexpr int kWarps = warps_of(W);
  //: THE DIGIT CARVE. -- see docs/internals/intra/intra_backward_kernel.md#kparts
  constexpr int kParts = bwd_slabs_per_stage(B);
  constexpr int kGroups = kWarps / kParts;
  constexpr int kPrRows = rows_of(W) / kGroups;
  constexpr int kPrM = kPrRows / kWarpRows;
  constexpr int kPwCols = kTile / kGroups;
  constexpr int kPwM = kPwCols / kWarpRows;
  constexpr int kRowK = rows_of(W) / kWarpRows;
  constexpr int kColK = kTile / kWarpRows;
  //: `d_v` carves the same object by (write-token tile, channel group).
  constexpr int kDvGroups = kWarps / kColK;
  constexpr int kDvChans = kValueWidth / kDvGroups;
  constexpr int kDvN = kDvChans / kSlabDigits;
  static_assert(kTiles * kTile == W, "the window is a whole number of output tiles");
  static_assert(kTiles % kRowTiles == 0, "the window is a whole number of output blocks");
  static_assert(kValueWidth % kSlabDigits == 0, "the value axis stages like the digit axis");
  static_assert(kParts == 2, "the digit carve gives each warp ONE slab of the stage");
  static_assert(bwd_stages_per_level(B) % 2 == 0,
                "pass 2 unrolls the staging ring by its own period, so the ring's buffer "
                "parity is a property of the unrolled body");
  static_assert(kPrRows % kWarpRows == 0 && kPwCols % kWarpRows == 0,
                "the carve cuts both destination token axes on the MMA's m step");
  static_assert(kPrRows <= kTile, "a row group lies inside ONE output tile");
  static_assert(kWarps % kColK == 0 && kValueWidth % kDvGroups == 0,
                "the value axis carves over whole channel tiles");
  extern __shared__ __align__(16) char raw[];
  IntraBwdSmem<D, B, W>& sm = *reinterpret_cast<IntraBwdSmem<D, B, W>*>(raw);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bh = blockIdx.y;
  const int wbase = blockIdx.x * W;

  build_bwd_masks<D, B, W>(sm, p, bh, wbase, tid);

  const int mine = (warp / kWarpsPerTile) & (kRowTiles - 1);
  const int arow = warp * kWarpRows + (lane & 15);
  const int trow = (warp % kWarpsPerTile) * kWarpRows;
  const int acol = (lane >> 4) * 8;
  const int row_lo = lane >> 2;
  const int col_lo = (lane & 3) * 2;
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  //: the `.trans` lane map for an A operand read out of a `[k][m]` plane
  const int rT = lr + (lane >> 4) * 8, cT = (tl & 1) * 8;
  float* const relay_put = &sm.xpose[warp][lane >> 2][(lane & 3) * 2];
  const float* const relay_get = &sm.xpose[warp][lane >> 4][lane & (kSlabDigits - 1)];
  const int relay_off = ((lane >> 4) << 8) | (lane & (kSlabDigits - 1));
  const int dp = warp % kParts;
  const int grp = warp / kParts;
  //: the row group's own output tile: a group is `kPrRows <= kTile` rows wide, so it -- see docs/internals/intra/intra_backward_kernel.md#grptile
  const int grp_tile = (grp * kPrRows) / kTile;
  const int jt = warp % kColK;
  const int chan = (warp / kColK) * kDvChans;
  //: THE OPERAND BASES. -- see docs/internals/intra/intra_backward_kernel.md#kelem
  constexpr unsigned kElem = (unsigned)sizeof(__nv_bfloat16);
  constexpr unsigned kSlabRow = bwd_operand_pitch(B) * kElem;
  constexpr unsigned kImgRow = kImagePitch * kElem;
  constexpr unsigned kSeedRow = kSeedPitch * kElem;
  constexpr unsigned kRBuf = rows_of(W) * kSlabRow;
  constexpr unsigned kWBuf = kTile * kSlabRow;
  const denseref::ops::SmemAddr base_pw_r =
      smem_addr(&sm.rslab[0][rF][cF * 8]) + dp * kSlabDigits * kElem;
  const denseref::ops::SmemAddr base_pr_w =
      smem_addr(&sm.wslab[0][rF][cF * 8]) + dp * kSlabDigits * kElem;
  const denseref::ops::SmemAddr base_pr_a = smem_addr(&sm.img[grp * kPrRows + (lane & 15)][acol]);
  const denseref::ops::SmemAddr base_pw_a = smem_addr(&sm.img[rT][grp * kPwCols + cT]);
  const denseref::ops::SmemAddr base_dv_a = smem_addr(&sm.img[rT][jt * kWarpRows + cT]);
  const denseref::ops::SmemAddr base_dv_b = smem_addr(&sm.dslab[rF][chan + cF * 8]);
  //: THE OWN-SIDE CERTIFICATE'S per-cell address: a lane's eight destination cells are -- see docs/internals/intra/intra_backward_kernel.md#basesupr
  const denseref::ops::SmemAddr base_sup_r =
      smem_addr(&sm.rslab[0][grp * kPrRows + (lane >> 4)][dp * kSlabDigits + (lane & 15)]);
  const denseref::ops::SmemAddr base_sup_w =
      smem_addr(&sm.wslab[0][grp * kPwCols + (lane >> 4)][dp * kSlabDigits + (lane & 15)]);

  for (int rt0 = 0; rt0 < kTiles; rt0 += kRowTiles) {
    const int rt = rt0 + mine;
    float* const dpr_grp =
        p.d_pr + (long long)(bh * p.L + wbase + rt0 * kTile + grp * kPrRows) * p.row_width;
    __syncthreads();
    stage_bwd_seed<D, B, W>(sm, p, bh, wbase, rt0, tid);
    stage_commit();
    stage_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int ct = 0; ct < rt0 + kRowTiles; ++ct) {
      __syncthreads();
      stage_bwd_values<D, B, W>(sm, p, bh, wbase, ct, tid);
      uint32_t rows = live_rows_pair<D, B, W>(sm, p.modes, rt0, ct, 0, 0);
      if (rows) stage_bwd_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, 0, 0, 0, tid, rows, true);
      stage_commit();

      float gram[D][8][4];
#pragma unroll
      for (int l = 0; l < D; ++l)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) gram[l][n][i] = 0.f;

      //: §6: bwd_stages_per_level(B)/bwd_slabs_per_stage(B) are compile-time (B is -- see docs/internals/intra/intra_backward_kernel.md#kstagesperlevel
      constexpr int kStagesPerLevel = bwd_stages_per_level(B);
      constexpr int kSlabsPerStage = bwd_slabs_per_stage(B);

      // ---- PASS 1: the Grams, recomputed under the forward's own gate ----------
#pragma unroll
      for (int l = 0; l < D; ++l) {
#pragma unroll
        for (int stage = 0; stage < kStagesPerLevel; ++stage) {
          const int step = l * kStagesPerLevel + stage;
          const int nl = (stage + 1 == kStagesPerLevel) ? l + 1 : l;
          const int ns = (stage + 1 == kStagesPerLevel) ? 0 : stage + 1;
          uint32_t next_rows = 0u;
          if (nl < D) {
            next_rows = live_rows_pair<D, B, W>(sm, p.modes, rt0, ct, nl, ns);
            if (next_rows)
              stage_bwd_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, nl, ns, (step + 1) & 1, tid,
                                       next_rows, true);
          }
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          if ((rows >> mine) & 1u) {
            const int buf = step & 1;
#pragma unroll
            for (int sub = 0; sub < kSlabsPerStage; ++sub) {
              const int slab = stage * kSlabsPerStage + sub;
              if (!pair_slab_live<D, B, W>(sm, p.modes, rt, ct, l, slab)) continue;
              const int k0 = sub * kSlabDigits + acol;
              uint32_t af[4];
              load_frag(af, smem_addr(&sm.rslab[buf][arow][k0]));
#pragma unroll
              for (int jn = 0; jn < 4; ++jn) {
                uint32_t bf[4];
                load_frag(bf, smem_addr(&sm.wslab[buf][jn * 16 + (lane & 15)][k0]));
                const uint32_t b0[2] = {bf[0], bf[2]};
                const uint32_t b1[2] = {bf[1], bf[3]};
                mma(gram[l][2 * jn], af, b0);
                mma(gram[l][2 * jn + 1], af, b1);
              }
            }
          }
          rows = next_rows;
          __syncthreads();
        }
      }
      const bool active = rt >= ct;

      stage_bwd_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, 0, 0, 0, tid,
                               live_rows_side<D, B, W>(sm, p.modes, rt0, ct, 0, 0),
                               side_stage_live<D, B, W>(sm, p.modes, ct, 0, 1, 0));
      stage_commit();

      // ---- Gd', then A and dA, then dG_l ---------------------------------------
      float gd[8][4];
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) gd[n][i] = 0.f;
#pragma unroll
      for (int kb = 0; kb < kChanTiles; ++kb) {
        if (!active) break;
        const int c0 = kb * 16 + acol;
        uint32_t af[4];
        load_frag(af, smem_addr(&sm.dslab[arow][c0]));
#pragma unroll
        for (int jn = 0; jn < 4; ++jn) {
          uint32_t bf[4];
          load_frag(bf, smem_addr(&sm.vslab[jn * 16 + (lane & 15)][c0]));
          const uint32_t b0[2] = {bf[0], bf[2]};
          const uint32_t b1[2] = {bf[1], bf[3]};
          mma(gd[2 * jn], af, b0);
          mma(gd[2 * jn + 1], af, b1);
        }
      }

      //: `A`, `dA` and every `dG_l` are formed IN PLACE over the Gram registers, and -- see docs/internals/intra/intra_backward_kernel.md#note-l612
      float* const dgw_base = p.d_gw + (long long)(bh * p.L + wbase + ct * kTile) + lane * 2;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        float dgw[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int cl = n * 8 + col_lo + (i & 1);
          const int rl = trow + row_lo + (i >> 1) * 8;
          const float g = __bfloat162float(sm.gain[cl]);
          const bool live = !(rt == ct && cl > rl);
          //: the leave-one-out over LEVELS, as a prefix/suffix product and never as -- see docs/internals/intra/intra_backward_kernel.md#pre
          float loo[D], pre = 1.f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            loo[l] = pre;
            pre *= gram[l][n][i];
          }
          float suf = 1.f;
#pragma unroll
          for (int l = D - 1; l >= 0; --l) {
            loo[l] *= suf;
            suf *= gram[l][n][i];
          }
          const float da = live ? (gd[n][i] + sm.dden[mine * kTile + rl]) : 0.f;
          dgw[i & 1] += da * pre;
          gd[n][i] = live ? pre * g : 0.f;
#pragma unroll
          for (int l = 0; l < D; ++l) gram[l][n][i] = da * g * loo[l];
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float acc = dgw[j];
#pragma unroll
          for (int s = 4; s < 32; s <<= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, s);
          if (active && lane < 4) fan_in_add(dgw_base + n * 8 + j, acc);
        }
      }

      // ---- d_v = A^T . d_num, off the published image -------------------------
      //: THE IMAGE IS THE RE-CARVE. `A` and each `dG_l` are accumulated ROW-carved -- -- see docs/internals/intra/intra_backward_kernel.md#syncthreads
      __syncthreads();
      publish_image<D, B, W>(sm, gd, warp, row_lo, col_lo);
      __syncthreads();
      {
        float* dv_base = p.d_v
                         + (long long)(bh * p.L + wbase + ct * kTile + jt * kWarpRows) * kValueWidth
                         + chan;
#pragma unroll
        for (int nb = 0; nb < kDvN; ++nb) {
          float dv[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
          for (int kt = 0; kt < kRowK; ++kt) {
            uint32_t af[4];
            load_frag_t(af, base_dv_a + kt * kWarpRows * kImgRow);
            uint32_t bf[4];
            load_frag_t(bf, base_dv_b + kt * kWarpRows * kSeedRow + nb * kSlabDigits * kElem);
            const uint32_t b0[2] = {bf[0], bf[1]};
            const uint32_t b1[2] = {bf[2], bf[3]};
            mma(dv[0], af, b0);
            mma(dv[1], af, b1);
          }
          reduce_tile(relay_put, relay_get, relay_off, dv, dv_base + nb * kSlabDigits, kValueWidth);
        }
      }

      // ---- PASS 2: the two digit contractions against the re-staged slabs ------
      //: THE FIRST FILL IS ISSUED BEFORE THE SEED CONTRACTION ABOVE, so it lands under -- see docs/internals/intra/intra_backward_kernel.md#dpwbase
      float* const dpw_base =
          p.d_pw + (long long)(bh * p.L + wbase + ct * kTile + grp * kPwCols) * p.row_width;
      const bool pr_active = rt0 + grp_tile >= ct;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        //: A SOFTMAX DUTY IS DENSE BY THEOREM, so its cotangent plane is live on the -- see docs/internals/intra/intra_backward_kernel.md#prown
        const bool pr_own = ((p.modes >> (2 * l)) & kModeReadSparse) != 0u;
        const bool pw_own = ((p.modes >> (2 * l)) & kModeWriteSparse) != 0u;
        __syncthreads();
        publish_image<D, B, W>(sm, gram[l], warp, row_lo, col_lo);
        __syncthreads();
        //: §6: kStagesPerLevel is the same compile-time quantity aliased above -- see docs/internals/intra/intra_backward_kernel.md#near-line-679
#pragma unroll 2
        for (int stage = 0; stage < kStagesPerLevel; ++stage) {
          const int step = l * kStagesPerLevel + stage;
          const int buf = step & 1;
          const int nl = (stage + 1 == kStagesPerLevel) ? l + 1 : l;
          const int ns = (stage + 1 == kStagesPerLevel) ? 0 : stage + 1;
          if (nl < D)
            stage_bwd_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, nl, ns, (step + 1) & 1, tid,
                                     live_rows_side<D, B, W>(sm, p.modes, rt0, ct, nl, ns),
                                     side_stage_live<D, B, W>(sm, p.modes, ct, nl, 1, ns));
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          const int slab = stage * bwd_slabs_per_stage(B) + dp;
          const int gdig = l * B + stage * bwd_stage_digits(B) + dp * kSlabDigits;
          if (pr_active && side_bits<D, B, W>(sm, p.modes, ct, l, 1, slab) != 0u
              && side_bits<D, B, W>(sm, p.modes, rt0 + grp_tile, l, 0, slab) != 0u) {
#pragma unroll
            for (int mt = 0; mt < kPrM; ++mt) {
              float dpr[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
              for (int kt = 0; kt < kColK; ++kt) {
                uint32_t bf[4];
                load_frag_t(bf, base_pr_w + buf * kWBuf + kt * kWarpRows * kSlabRow);
                uint32_t af[4];
                load_frag(af, base_pr_a + mt * kWarpRows * kImgRow + kt * kWarpRows * kElem);
                const uint32_t b0[2] = {bf[0], bf[1]};
                const uint32_t b1[2] = {bf[2], bf[3]};
                mma(dpr[0], af, b0);
                mma(dpr[1], af, b1);
              }
              reduce_tile_on_support(
                  relay_put, relay_get, relay_off, dpr,
                  dpr_grp + (long long)mt * kWarpRows * p.row_width + gdig, p.row_width,
                  base_sup_r + buf * kRBuf + mt * kWarpRows * kSlabRow, 2 * kSlabRow, pr_own);
            }
          }
          //: THE READ-DIGIT contraction runs over the WHOLE block of readers, so it is gated per -- see docs/internals/intra/intra_backward_kernel.md#note-l741
          uint32_t rlive = 0u;
          if (side_bits<D, B, W>(sm, p.modes, ct, l, 1, slab) != 0u) {
#pragma unroll
            for (int r = 0; r < kRowTiles; ++r)
              if (rt0 + r >= ct && side_bits<D, B, W>(sm, p.modes, rt0 + r, l, 0, slab) != 0u)
                rlive |= 1u << r;
          }
          if (rlive) {
            float dpw[kPwM][2][4] = {};
#pragma unroll
            for (int r = 0; r < kRowTiles; ++r) {
              if (!((rlive >> r) & 1u)) continue;
#pragma unroll
              for (int kk = 0; kk < kColK; ++kk) {
                const int kt = r * kColK + kk;
                uint32_t bf[4];
                load_frag_t(bf, base_pw_r + buf * kRBuf + kt * kWarpRows * kSlabRow);
                const uint32_t b0[2] = {bf[0], bf[1]};
                const uint32_t b1[2] = {bf[2], bf[3]};
#pragma unroll
                for (int mt = 0; mt < kPwM; ++mt) {
                  uint32_t af[4];
                  load_frag_t(af, base_pw_a + kt * kWarpRows * kImgRow + mt * kWarpRows * kElem);
                  mma(dpw[mt][0], af, b0);
                  mma(dpw[mt][1], af, b1);
                }
              }
            }
#pragma unroll
            for (int mt = 0; mt < kPwM; ++mt)
              reduce_tile_on_support(
                  relay_put, relay_get, relay_off, dpw[mt],
                  dpw_base + (long long)mt * kWarpRows * p.row_width + gdig, p.row_width,
                  base_sup_w + buf * kWBuf + mt * kWarpRows * kSlabRow, 2 * kSlabRow, pw_own);
          }
          __syncthreads();
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace intra
}  // namespace rola
