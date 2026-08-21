// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
//
// K31 RUNG R4: THE INTRA TERM'S REVERSE PASS.
//
// RECOMPUTE, NEVER SAVE, AT BOTH GRAINS.  The pairwise coefficient `A` is `[W x W]`
// per window -- the snapshot class the R4 constraint forbids -- so this kernel rebuilds
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

using denseref::ops::transpose_frag_b16;

//: the reverse intra's own device stamp: a fact only THIS binary can produce.
constexpr int64_t kIntraBwdMagic = 31040002;

constexpr int kSeedPitch = kValuePitch;
constexpr int kChanTiles = kValueWidth / kSlabDigits;
constexpr int warps_of(int window) { return threads_of(window) / 32; }

//: THE RELAY PLANE, one `[kMmaM x kSlabDigits]` tile per warp: every cotangent this
//: kernel reduces has that shape, and the accumulator's lane map scatters a reducing
//: instruction's 32 values over eight destination rows a plane pitch apart.  The pitch
//: is EVEN because the accumulator's two column-adjacent values relay as one eight-byte
//: store.  See docs/internals/intra/intra_backward_kernel.md.
constexpr int kXposeRows = 16;
constexpr int kXposePitch = kSlabDigits + 2;

//: The forward's block footprint plus the ROW side's seed rows and the per-warp relay
//: plane.  The seed is staged once per output block, not once per column tile: its rows
//: are the block's readers.
template <int D, int B, int W>
struct alignas(16) IntraBwdSmem {
  __nv_bfloat16 rslab[2][rows_of(W)][operand_pitch(B)];
  __nv_bfloat16 wslab[2][kTile][operand_pitch(B)];
  __nv_bfloat16 vslab[kTile][kValuePitch];
  __nv_bfloat16 dslab[rows_of(W)][kSeedPitch];
  __nv_bfloat16 gain[kTile];
  float dden[rows_of(W)];
  alignas(16) float xpose[warps_of(W)][kXposeRows][kXposePitch];
  uint32_t mask[tiles_of(W)][D][2][mask_words(B)];
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

//: the mask helpers read one CTA-local summary, so the reverse pass's two one-sided
//: predicates cost nothing beyond the forward's intersection one.
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
#pragma unroll
  for (int sub = 0; sub < slabs_per_stage(B); ++sub)
    any = any || side_bits<D, B, W>(sm, modes, ti, l, side, stage * slabs_per_stage(B) + sub) != 0u;
  return any;
}

//: THE GRAM PASS'S GATE IS THE INTERSECTION and the digit passes' is ONE SIDE.  A
//: level's digit gradient drops that level's own factor, so `d_pr_l[t,e]` is live
//: wherever the WRITE side has support at `e` whatever the read side does there; gating
//: it on the intersection would drop real terms.
template <int D, int B, int W>
__device__ __forceinline__ uint32_t live_rows_side(const IntraBwdSmem<D, B, W>& sm, uint32_t modes,
                                                   int rt0, int ct, int l, int stage) {
  uint32_t bits = 0u;
#pragma unroll
  for (int r = 0; r < row_tiles_of(W); ++r)
    if (rt0 + r >= ct && side_stage_live<D, B, W>(sm, modes, rt0 + r, l, 0, stage)) bits |= 1u << r;
  return bits;
}

//: THE GRAM PASS'S OWN GATE: both sides' supports must meet in the digit slab, which is
//: the forward's intersection certificate read off this kernel's own mask summary.
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
#pragma unroll
  for (int r = 0; r < row_tiles_of(W); ++r) {
    if (rt0 + r < ct) continue;
    bool any = false;
#pragma unroll
    for (int sub = 0; sub < slabs_per_stage(B); ++sub)
      any = any
            || pair_slab_live<D, B, W>(sm, modes, rt0 + r, ct, l, stage * slabs_per_stage(B) + sub);
    if (any) bits |= 1u << r;
  }
  return bits;
}

template <int D, int B, int W>
__device__ __forceinline__ void build_bwd_masks(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                                int bh, int wbase, int tid) {
  //: THE HALF-SPLIT CERTIFICATE PASS needs a level at least two words wide.  A
  //: narrower level is a REFUSAL here rather than a second pass shape, because the
  //: reverse family is built at depth two only (`intra_backward.cu`).
  static_assert(mask_words(B) % 2 == 0, "the reverse certificate pass splits a level in half");
  constexpr int kTiles = tiles_of(W);
  constexpr int kThreads = threads_of(W);
  constexpr int kCohorts = kThreads / (2 * kTile);
  uint32_t* flat = &sm.mask[0][0][0][0];
  for (int i = tid; i < kTiles * D * 2 * mask_words(B); i += kThreads) flat[i] = 0u;
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
      for (int ti = cohort; ti < kTiles; ti += kCohorts) {
        const __nv_bfloat16* row =
            plane + (long long)(bh * p.L + wbase + ti * kTile + token) * p.row_width + l * B
            + half * (B / 2);
#pragma unroll
        for (int j = 0; j < mask_words(B) / 2; ++j) {
          const uint32_t bits = __reduce_or_sync(0xFFFFFFFFu, digit_union_word<32>(row + j * 32));
          if ((tid & 31) == 0)
            atomicOr(&sm.mask[ti][l][side][half * (mask_words(B) / 2) + j], bits);
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
  const int digit = l * B + stage * stage_digits(B);
  const long long base = (long long)(bh * p.L + wbase) * p.row_width + digit;
  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColDigits = stage_digits(B) / kColThreads;
  if (cols) {
    const int ctoken = tid / kColThreads;
    const int cpart = (tid % kColThreads) * kColDigits;
    const __nv_bfloat16* wsrc =
        p.pwrite + base + (long long)(ct * kTile + ctoken) * p.row_width + cpart;
#pragma unroll
    for (int c = 0; c < kColDigits / 8; ++c)
      stage_tile(smem_addr(&sm.wslab[buf][ctoken][cpart + c * 8]), wsrc + c * 8);
  }
  const int rtoken = (tid >> 1) & (kTile - 1);
  const int rtile = tid / (2 * kTile);
  const int rpart = (tid & 1) * (stage_digits(B) / 2);
  if ((rows >> rtile) & 1u) {
    const __nv_bfloat16* rsrc =
        p.pread + base + (long long)((rt0 + rtile) * kTile + rtoken) * p.row_width + rpart;
#pragma unroll
    for (int c = 0; c < stage_digits(B) / 16; ++c)
      stage_tile(smem_addr(&sm.rslab[buf][rtile * kTile + rtoken][rpart + c * 8]), rsrc + c * 8);
  }
}

template <int D, int B, int W>
__device__ __forceinline__ void stage_bwd_values(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                                 int bh, int wbase, int ct, int tid) {
  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColChannels = kValueWidth / kColThreads;
  const int b = bh / p.heads;
  const int h = bh - b * p.heads;
  const int t0 = wbase + ct * kTile;
  const int token = tid / kColThreads;
  const int part = (tid % kColThreads) * kColChannels;
  const __nv_bfloat16* src =
      p.v + ((long long)(b * p.tokens + t0 + token) * p.heads + h) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < kColChannels / 8; ++c)
    stage_tile(smem_addr(&sm.vslab[token][part + c * 8]), src + c * 8);
  if (tid < kTile / 8)
    stage_tile(smem_addr(&sm.gain[tid * 8]), p.gwrite + (long long)bh * p.L + t0 + tid * 8);
}

//: THE SEED, staged ONCE PER OUTPUT BLOCK: its rows are the block's readers, which do
//: not change with the column tile.
template <int D, int B, int W>
__device__ __forceinline__ void stage_bwd_seed(IntraBwdSmem<D, B, W>& sm, const IntraBwdParams& p,
                                               int bh, int wbase, int rt0, int tid) {
  constexpr int kThreads = threads_of(W);
  constexpr int kRows = rows_of(W);
  constexpr int kRowThreads = kThreads / kRows;
  constexpr int kRowChannels = kValueWidth / kRowThreads;
  const int t0 = wbase + rt0 * kTile;
  const int token = tid / kRowThreads;
  const int part = (tid % kRowThreads) * kRowChannels;
  const __nv_bfloat16* src = p.dnum + (long long)(bh * p.L + t0 + token) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < kRowChannels / 8; ++c)
    stage_tile(smem_addr(&sm.dslab[token][part + c * 8]), src + c * 8);
  if (tid < kRows) sm.dden[tid] = p.dden[(long long)bh * p.L + t0 + tid];
}

//: A `[16 x 64]` accumulator's TRANSPOSE, as the four A fragments of m-tile `jm`:
//: `m` becomes the accumulator's column axis and `k` its row axis.  This is the one
//: place the reverse pass needs an in-register tile transpose; the alternative is an
//: SMEM round trip of an object the warp already holds.
__device__ __forceinline__ void transpose_acc(const float (&acc)[8][4], int jm, uint32_t (&a)[4]) {
  a[0] = transpose_frag_b16(pack_bf16x2(acc[2 * jm][0], acc[2 * jm][1]));
  a[1] = transpose_frag_b16(pack_bf16x2(acc[2 * jm + 1][0], acc[2 * jm + 1][1]));
  a[2] = transpose_frag_b16(pack_bf16x2(acc[2 * jm][2], acc[2 * jm][3]));
  a[3] = transpose_frag_b16(pack_bf16x2(acc[2 * jm + 1][2], acc[2 * jm + 1][3]));
}

//: THE ONE REDUCING PATH.  `acc` is one `[kMmaM x kSlabDigits]` accumulator in the
//: MMA's lane map; `base` addresses its first destination row and `stride` the
//: destination's row pitch.  The relay through the warp's own plane costs two shared
//: accesses per value and buys a reducing instruction that spans two rows of
//: contiguous digits -- two sectors for 128 bytes where the direct form takes eight.
template <int D, int B, int W>
__device__ __forceinline__ void reduce_tile(IntraBwdSmem<D, B, W>& sm, const float (&acc)[2][4],
                                            float* base, long long stride, int warp, int lane) {
  float* x = &sm.xpose[warp][0][0];
  const int arow = lane >> 2;
  const int acol = (lane & 3) * 2;
  __syncwarp();
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      *reinterpret_cast<float2*>(&x[(arow + h * 8) * kXposePitch + nt * 8 + acol]) =
          make_float2(acc[nt][2 * h], acc[nt][2 * h + 1]);
  __syncwarp();
  const int col = lane & (kSlabDigits - 1);
  const int half = lane >> 4;
#pragma unroll
  for (int q = 0; q < kXposeRows / 2; ++q) {
    const int row = q * 2 + half;
    fan_in_add(base + row * stride + col, x[row * kXposePitch + col]);
  }
}

__device__ __forceinline__ void pack_acc(const float (&acc)[8][4], int kb, uint32_t (&a)[4]) {
  a[0] = pack_bf16x2(acc[2 * kb][0], acc[2 * kb][1]);
  a[1] = pack_bf16x2(acc[2 * kb][2], acc[2 * kb][3]);
  a[2] = pack_bf16x2(acc[2 * kb + 1][0], acc[2 * kb + 1][1]);
  a[3] = pack_bf16x2(acc[2 * kb + 1][2], acc[2 * kb + 1][3]);
}

//: THE SECOND LAUNCH BOUND, from the arm's own footprint.  Without it ptxas budgets
//: registers for the residency the THREAD COUNT suggests rather than the one this
//: block's shared memory leaves reachable, and the narrow-window arm pays the
//: difference in spill.  `kBwdRegFloor` is the measured live set of the built arms.
constexpr int kBwdRegFloor = 232;

template <int D, int B, int W>
constexpr int kBwdCtaTarget =
    denseref::design::cta_target((int)sizeof(IntraBwdSmem<D, B, W>), threads_of(W), kBwdRegFloor);

template <int D, int B, int W>
__global__ __launch_bounds__(threads_of(W), (kBwdCtaTarget<D, B, W>)) void intra_backward_kernel(
    IntraBwdParams p) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kRowTiles = row_tiles_of(W);
  static_assert(kTiles * kTile == W, "the window is a whole number of output tiles");
  static_assert(kTiles % kRowTiles == 0, "the window is a whole number of output blocks");
  static_assert(kValueWidth % kSlabDigits == 0, "the value axis stages like the digit axis");
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

  for (int rt0 = 0; rt0 < kTiles; rt0 += kRowTiles) {
    const int rt = rt0 + mine;
    float* const dpr_base =
        p.d_pr + (long long)(bh * p.L + wbase + rt * kTile + trow) * p.row_width;
    __syncthreads();
    stage_bwd_seed<D, B, W>(sm, p, bh, wbase, rt0, tid);
    stage_commit();
    stage_wait<0>();
    __syncthreads();

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

      // ---- PASS 1: the Grams, recomputed under the forward's own gate ----------
#pragma unroll
      for (int l = 0; l < D; ++l) {
#pragma unroll
        for (int stage = 0; stage < stages_per_level(B); ++stage) {
          const int step = l * stages_per_level(B) + stage;
          const int nl = (stage + 1 == stages_per_level(B)) ? l + 1 : l;
          const int ns = (stage + 1 == stages_per_level(B)) ? 0 : stage + 1;
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
            for (int sub = 0; sub < slabs_per_stage(B); ++sub) {
              const int slab = stage * slabs_per_stage(B) + sub;
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

      //: `A`, `dA` and every `dG_l` are formed IN PLACE over the Gram registers, and
      //: (I3)'s row sum is drained inside the same sweep so it never becomes an array.
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        float dgw[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int cl = n * 8 + col_lo + (i & 1);
          const int rl = trow + row_lo + (i >> 1) * 8;
          const float g = __bfloat162float(sm.gain[cl]);
          const bool live = !(rt == ct && cl > rl);
          //: the leave-one-out over LEVELS, as a prefix/suffix product and never as
          //: `A / G_l`: `G_l` is exactly zero on disjoint supports.
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
          if (active && lane < 4)
            fan_in_add(p.d_gw + (long long)bh * p.L + wbase + ct * kTile + n * 8 + lane * 2 + j,
                       acc);
        }
      }

      // ---- (I2) d_v = A^T . d_num, one 16-channel pass at a time ---------------
      uint32_t at[4][4];
#pragma unroll
      for (int jm = 0; jm < 4; ++jm) transpose_acc(gd, jm, at[jm]);
      float* dv_base = p.d_v + (long long)(bh * p.L + wbase + ct * kTile) * kValueWidth;
#pragma unroll
      for (int cb = 0; cb < kChanTiles; ++cb) {
        if (!active) break;
        uint32_t bf[4];
        load_frag_t(bf, smem_addr(&sm.dslab[warp * kWarpRows + rF][cb * kSlabDigits + cF * 8]));
        const uint32_t b0[2] = {bf[0], bf[1]};
        const uint32_t b1[2] = {bf[2], bf[3]};
#pragma unroll
        for (int jm = 0; jm < 4; ++jm) {
          float dv[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          mma(dv[0], at[jm], b0);
          mma(dv[1], at[jm], b1);
          reduce_tile<D, B, W>(sm, dv,
                               dv_base + (long long)jm * 16 * kValueWidth + cb * kSlabDigits,
                               kValueWidth, warp, lane);
        }
      }

      // ---- PASS 2: (I5) and (I6) against the re-staged slabs -------------------
      //: THE FIRST FILL IS ISSUED BEFORE THE SEED CONTRACTION ABOVE, so it lands under
      //: it; from there the ring is the forward's, in the buffers pass 1 already owns.
      float* dpw_base = p.d_pw + (long long)(bh * p.L + wbase + ct * kTile) * p.row_width;
#pragma unroll
      for (int l = 0; l < D; ++l) {
#pragma unroll
        for (int stage = 0; stage < stages_per_level(B); ++stage) {
          const int step = l * stages_per_level(B) + stage;
          const int buf = step & 1;
          const int nl = (stage + 1 == stages_per_level(B)) ? l + 1 : l;
          const int ns = (stage + 1 == stages_per_level(B)) ? 0 : stage + 1;
          if (nl < D)
            stage_bwd_slabs<D, B, W>(sm, p, bh, wbase, rt0, ct, nl, ns, (step + 1) & 1, tid,
                                     live_rows_side<D, B, W>(sm, p.modes, rt0, ct, nl, ns),
                                     side_stage_live<D, B, W>(sm, p.modes, ct, nl, 1, ns));
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          const uint32_t rrows = live_rows_side<D, B, W>(sm, p.modes, rt0, ct, l, stage);
          const int dbase = l * B + stage * stage_digits(B);
#pragma unroll
          for (int sub = 0; sub < slabs_per_stage(B); ++sub) {
            const int slab = stage * slabs_per_stage(B) + sub;
            const int k0 = sub * kSlabDigits;
            const int gdig = dbase + sub * kSlabDigits;
            if (active && side_bits<D, B, W>(sm, p.modes, ct, l, 1, slab) != 0u) {
              float dpr[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
              for (int jt = 0; jt < 4; ++jt) {
                uint32_t af[4];
                pack_acc(gram[l], jt, af);
                uint32_t bf[4];
                load_frag_t(bf, smem_addr(&sm.wslab[buf][jt * 16 + rF][k0 + cF * 8]));
                const uint32_t b0[2] = {bf[0], bf[1]};
                const uint32_t b1[2] = {bf[2], bf[3]};
                mma(dpr[0], af, b0);
                mma(dpr[1], af, b1);
              }
              reduce_tile<D, B, W>(sm, dpr, dpr_base + gdig, p.row_width, warp, lane);
            }
            if (active && ((rrows >> mine) & 1u)
                && side_bits<D, B, W>(sm, p.modes, rt, l, 0, slab) != 0u) {
              uint32_t bf[4];
              load_frag_t(bf, smem_addr(&sm.rslab[buf][warp * kWarpRows + rF][k0 + cF * 8]));
              const uint32_t b0[2] = {bf[0], bf[1]};
              const uint32_t b1[2] = {bf[2], bf[3]};
#pragma unroll
              for (int jm = 0; jm < 4; ++jm) {
                uint32_t af[4];
                transpose_acc(gram[l], jm, af);
                float dpw[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                mma(dpw[0], af, b0);
                mma(dpw[1], af, b1);
                reduce_tile<D, B, W>(sm, dpw, dpw_base + (long long)jm * 16 * p.row_width + gdig,
                                     p.row_width, warp, lane);
              }
            }
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
