// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"
#include "intra/intra.cuh"

namespace rola {
namespace intra {
namespace detail {

using denseref::ops::fan_in_add;
using denseref::ops::kBf16MagnitudeMask;
using denseref::ops::load_frag;
using denseref::ops::load_frag_t;
using denseref::ops::mma;
using denseref::ops::pack_bf16x2;
using denseref::ops::smem_addr;
using denseref::ops::stage_commit;
using denseref::ops::stage_tile;
using denseref::ops::stage_wait;

//: THE WINDOW IS AN ARM, so the tile counts it induces are functions of it and
//: never file-scope constants -- the carry's `W` template argument, mirrored.
constexpr int tiles_of(int window) { return window / kTile; }
//: THE OUTPUT BLOCK IS C = `kTile * row_tiles_of(W)` READER TOKENS.  The column
//: operands staged for one column tile serve every row tile of the block, which
//: is what takes the tile-pair grid's operand re-read down; a window with a
//: single tile has no pair to share with, so the block collapses onto it.
constexpr int row_tiles_of(int window) { return tiles_of(window) >= 2 ? 2 : 1; }
constexpr int rows_of(int window) { return row_tiles_of(window) * kTile; }
//: ONE WARP PER `kWarpRows` OF THE BLOCK: a warp holds the Gram of ONE tile pair,
//: so the block's accumulator budget per thread does not grow with C -- the row
//: tiles are shared across warps, not stacked in one warp's registers.  The CTA
//: width is therefore a function of the arm, like every other tile count here.
constexpr int threads_of(int window) { return 32 * (rows_of(window) / kWarpRows); }
constexpr int kWarpsPerTile = kTile / kWarpRows;
constexpr int kSlabsPerLevel = kLevelWidth / kSlabDigits;
constexpr int kMaskWords = kLevelWidth / 32;
//: THE PIPELINE STAGE IS WIDER THAN THE MMA's CONTRACTION STEP.  A stage is one
//: cp.async fill and one pair of barriers; four `kSlabDigits` steps ride it, so
//: the barrier count per tile-pair is the stage count, not the MMA-step count.
constexpr int kStageDigits = 64;
constexpr int kSlabsPerStage = kStageDigits / kSlabDigits;
constexpr int kStagesPerLevel = kLevelWidth / kStageDigits;
//: The operand rows are padded so the eight row addresses ldmatrix reads per
//: 8x8 tile land in eight distinct banks; the pad is in elements and keeps every
//: row base 16-byte aligned, which cp.async requires.
constexpr int kOperandPitch = kStageDigits + 8;
constexpr int kValuePitch = kValueWidth + 8;

struct IntraParams {
  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* v;
  float* o;
  float* den;
  int L;
  int row_width;
  int heads;
  int tokens;
  uint32_t modes;
};

//: THE OPERAND SLABS ARE PER-TILE-PAIR AND DO NOT GROW WITH THE WINDOW: only the
//: zero-certificate masks do, at one word per (tile, level, side, 32 digits).
//: The block's whole reader side is resident, which puts the footprint past the
//: 48 KB a kernel gets unasked, so the arm launches on opted-in dynamic shared
//: memory.
template <int D, int W>
struct alignas(16) IntraSmem {
  __nv_bfloat16 rslab[2][rows_of(W)][kOperandPitch];
  __nv_bfloat16 wslab[2][kTile][kOperandPitch];
  __nv_bfloat16 vslab[kTile][kValuePitch];
  __nv_bfloat16 gain[kTile];
  uint32_t mask[tiles_of(W)][D][2][kMaskWords];
};

__device__ __forceinline__ uint32_t nonzero_pair(uint32_t w) {
  const uint32_t m = w & kBf16MagnitudeMask;
  return ((m & 0xFFFFu) != 0u ? 1u : 0u) | ((m & 0xFFFF0000u) != 0u ? 2u : 0u);
}

__device__ __forceinline__ uint32_t digit_union_word(const __nv_bfloat16* row) {
  uint32_t bits = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 x = *reinterpret_cast<const uint4*>(row + q * 8);
    bits |= nonzero_pair(x.x) << (q * 8 + 0);
    bits |= nonzero_pair(x.y) << (q * 8 + 2);
    bits |= nonzero_pair(x.z) << (q * 8 + 4);
    bits |= nonzero_pair(x.w) << (q * 8 + 6);
  }
  return bits;
}

//: One bit per digit, unioned over a tile's 64 tokens -- the exact zero
//: certificate the slab gate reads.  A side the mode word does not declare
//: sparse is never summarized and answers all-ones, which is what makes the
//: gate self-deactivating rather than a second code path.
template <int D, int W>
__device__ void build_masks(IntraSmem<D, W>& sm, const IntraParams& p, int bh, int wbase, int tid) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kThreads = threads_of(W);
  constexpr int kCohorts = kThreads / (2 * kTile);
  uint32_t* flat = &sm.mask[0][0][0][0];
  for (int i = tid; i < kTiles * D * 2 * kMaskWords; i += kThreads) flat[i] = 0u;
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
            plane + (long long)(bh * p.L + wbase + ti * kTile + token) * p.row_width
            + l * kLevelWidth + half * (kLevelWidth / 2);
#pragma unroll
        for (int j = 0; j < kMaskWords / 2; ++j) {
          const uint32_t bits = __reduce_or_sync(0xFFFFFFFFu, digit_union_word(row + j * 32));
          if ((tid & 31) == 0) atomicOr(&sm.mask[ti][l][side][half * (kMaskWords / 2) + j], bits);
        }
      }
    }
  }
  __syncthreads();
}

template <int D, int W>
__device__ __forceinline__ uint32_t slab_bits(const IntraSmem<D, W>& sm, uint32_t modes, int ti,
                                              int l, int side, int slab) {
  const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
  if (((modes >> (2 * l)) & want) == 0u) return 0xFFFFu;
  return (sm.mask[ti][l][side][slab >> 1] >> ((slab & 1) * 16)) & 0xFFFFu;
}

template <int D, int W>
__device__ __forceinline__ bool slab_live(const IntraSmem<D, W>& sm, uint32_t modes, int rt, int ct,
                                          int l, int slab) {
  return (slab_bits(sm, modes, rt, l, 0, slab) & slab_bits(sm, modes, ct, l, 1, slab)) != 0u;
}

template <int D, int W>
__device__ __forceinline__ bool stage_live(const IntraSmem<D, W>& sm, uint32_t modes, int rt,
                                           int ct, int l, int stage) {
  bool any = false;
#pragma unroll
  for (int sub = 0; sub < kSlabsPerStage; ++sub)
    any = any || slab_live<D, W>(sm, modes, rt, ct, l, stage * kSlabsPerStage + sub);
  return any;
}

//: One bit per row tile of the block: the pairs of this stage with work in them.
//: A strictly-upper pair is never set, which is how the causal grid is walked.
template <int D, int W>
__device__ __forceinline__ uint32_t live_rows(const IntraSmem<D, W>& sm, uint32_t modes, int rt0,
                                              int ct, int l, int stage) {
  uint32_t bits = 0u;
#pragma unroll
  for (int r = 0; r < row_tiles_of(W); ++r)
    if (rt0 + r >= ct && stage_live<D, W>(sm, modes, rt0 + r, ct, l, stage)) bits |= 1u << r;
  return bits;
}

template <int D, int W>
__device__ __forceinline__ void stage_slabs(IntraSmem<D, W>& sm, const IntraParams& p, int bh,
                                            int wbase, int rt0, int ct, int l, int stage, int buf,
                                            int tid, uint32_t rows) {
  const int digit = l * kLevelWidth + stage * kStageDigits;
  const long long base = (long long)(bh * p.L + wbase) * p.row_width + digit;

  constexpr int kThreads = threads_of(W);
  constexpr int kColThreads = kThreads / kTile;
  constexpr int kColDigits = kStageDigits / kColThreads;
  const int ctoken = tid / kColThreads;
  const int cpart = (tid % kColThreads) * kColDigits;
  const __nv_bfloat16* wsrc =
      p.pwrite + base + (long long)(ct * kTile + ctoken) * p.row_width + cpart;
#pragma unroll
  for (int c = 0; c < kColDigits / 8; ++c)
    stage_tile(smem_addr(&sm.wslab[buf][ctoken][cpart + c * 8]), wsrc + c * 8);

  const int rtoken = (tid >> 1) & (kTile - 1);
  const int rtile = tid / (2 * kTile);
  const int rpart = (tid & 1) * (kStageDigits / 2);
  if ((rows >> rtile) & 1u) {
    const __nv_bfloat16* rsrc =
        p.pread + base + (long long)((rt0 + rtile) * kTile + rtoken) * p.row_width + rpart;
#pragma unroll
    for (int c = 0; c < kStageDigits / 16; ++c)
      stage_tile(smem_addr(&sm.rslab[buf][rtile * kTile + rtoken][rpart + c * 8]), rsrc + c * 8);
  }
}

template <int D, int W>
__device__ __forceinline__ void stage_values(IntraSmem<D, W>& sm, const IntraParams& p, int bh,
                                             int wbase, int ct, int tid) {
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
  if (tid < kTile / 8) {
    stage_tile(smem_addr(&sm.gain[tid * 8]), p.gwrite + (long long)bh * p.L + t0 + tid * 8);
  }
}

template <int D, int W>
__global__ __launch_bounds__(threads_of(W)) void intra_kernel(IntraParams p) {
  constexpr int kTiles = tiles_of(W);
  constexpr int kRowTiles = row_tiles_of(W);
  static_assert(kTiles * kTile == W, "the window is a whole number of output tiles");
  static_assert(kTiles % kRowTiles == 0, "the window is a whole number of output blocks");
  extern __shared__ __align__(16) char raw[];
  IntraSmem<D, W>& sm = *reinterpret_cast<IntraSmem<D, W>*>(raw);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bh = blockIdx.y;
  const int wbase = blockIdx.x * W;

  build_masks<D, W>(sm, p, bh, wbase, tid);

  //: The warp's row tile within the block, and its rows within that tile.
  const int mine = (warp / kWarpsPerTile) & (kRowTiles - 1);
  const int arow = warp * kWarpRows + (lane & 15);
  const int trow = (warp % kWarpsPerTile) * kWarpRows;
  const int acol = (lane >> 4) * 8;
  const int row_lo = lane >> 2;
  const int col_lo = (lane & 3) * 2;

  for (int rt0 = 0; rt0 < kTiles; rt0 += kRowTiles) {
    const int rt = rt0 + mine;
    float oacc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    float dacc[2] = {0.f, 0.f};

    for (int ct = 0; ct < rt0 + kRowTiles; ++ct) {
      __syncthreads();
      stage_values<D, W>(sm, p, bh, wbase, ct, tid);
      uint32_t rows = live_rows<D, W>(sm, p.modes, rt0, ct, 0, 0);
      if (rows) stage_slabs<D, W>(sm, p, bh, wbase, rt0, ct, 0, 0, 0, tid, rows);
      stage_commit();

      float gram[D][8][4];
#pragma unroll
      for (int l = 0; l < D; ++l)
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i) gram[l][n][i] = 0.f;

#pragma unroll
      for (int l = 0; l < D; ++l) {
#pragma unroll
        for (int stage = 0; stage < kStagesPerLevel; ++stage) {
          const int step = l * kStagesPerLevel + stage;
          const int nl = (stage + 1 == kStagesPerLevel) ? l + 1 : l;
          const int ns = (stage + 1 == kStagesPerLevel) ? 0 : stage + 1;
          uint32_t next_rows = 0u;
          if (nl < D) {
            next_rows = live_rows<D, W>(sm, p.modes, rt0, ct, nl, ns);
            if (next_rows)
              stage_slabs<D, W>(sm, p, bh, wbase, rt0, ct, nl, ns, (step + 1) & 1, tid, next_rows);
          }
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          if ((rows >> mine) & 1u) {
            const int buf = step & 1;
#pragma unroll
            for (int sub = 0; sub < kSlabsPerStage; ++sub) {
              const int slab = stage * kSlabsPerStage + sub;
              if (!slab_live<D, W>(sm, p.modes, rt, ct, l, slab)) continue;
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

      if (rt >= ct) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float a = gram[0][n][i];
#pragma unroll
            for (int l = 1; l < D; ++l) a *= gram[l][n][i];
            const int cl = n * 8 + col_lo + (i & 1);
            a *= __bfloat162float(sm.gain[cl]);
            if (rt == ct && cl > trow + row_lo + (i >> 1) * 8) a = 0.f;
            gram[0][n][i] = a;
          }
          dacc[0] += gram[0][n][0] + gram[0][n][1];
          dacc[1] += gram[0][n][2] + gram[0][n][3];
        }

#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint32_t av[4] = {pack_bf16x2(gram[0][2 * kb][0], gram[0][2 * kb][1]),
                                  pack_bf16x2(gram[0][2 * kb][2], gram[0][2 * kb][3]),
                                  pack_bf16x2(gram[0][2 * kb + 1][0], gram[0][2 * kb + 1][1]),
                                  pack_bf16x2(gram[0][2 * kb + 1][2], gram[0][2 * kb + 1][3])};
#pragma unroll
          for (int nv = 0; nv < 4; ++nv) {
            uint32_t bf[4];
            load_frag_t(bf, smem_addr(&sm.vslab[kb * 16 + (lane & 15)][nv * 16 + acol]));
            const uint32_t b0[2] = {bf[0], bf[1]};
            const uint32_t b1[2] = {bf[2], bf[3]};
            mma(oacc[nv * 2], av, b0);
            mma(oacc[nv * 2 + 1], av, b1);
          }
        }
      }
    }

    const int grow = wbase + rt * kTile + trow + row_lo;
    float* obase = p.o + (long long)(bh * p.L + grow) * kValueWidth;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        fan_in_add(obase + (i >> 1) * 8 * kValueWidth + n * 8 + col_lo + (i & 1), oacc[n][i]);
#pragma unroll
    for (int s = 1; s <= 2; s *= 2) {
      dacc[0] += __shfl_xor_sync(0xFFFFFFFFu, dacc[0], s);
      dacc[1] += __shfl_xor_sync(0xFFFFFFFFu, dacc[1], s);
    }
    if ((lane & 3) == 0) {
      float* dbase = p.den + (long long)bh * p.L + grow;
      fan_in_add(dbase, dacc[0]);
      fan_in_add(dbase + 8, dacc[1]);
    }
  }
}

}  // namespace detail
}  // namespace intra
}  // namespace rola
