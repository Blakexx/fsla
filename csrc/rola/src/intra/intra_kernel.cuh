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

constexpr int kThreads = 128;
constexpr int kTiles = kWindow / kTile;
constexpr int kSlabsPerLevel = kLevelWidth / kSlabDigits;
constexpr int kMaskWords = kLevelWidth / 32;
//: The operand rows are padded so the eight row addresses ldmatrix reads per
//: 8x8 tile land in eight distinct banks; the pad is in elements and keeps every
//: row base 16-byte aligned, which cp.async requires.
constexpr int kOperandPitch = kSlabDigits + 8;
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

template <int D>
struct alignas(16) IntraSmem {
  __nv_bfloat16 rslab[2][kTile][kOperandPitch];
  __nv_bfloat16 wslab[2][kTile][kOperandPitch];
  __nv_bfloat16 vslab[kTile][kValuePitch];
  __nv_bfloat16 gain[kTile];
  uint32_t mask[kTiles][D][2][kMaskWords];
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
template <int D>
__device__ void build_masks(IntraSmem<D>& sm, const IntraParams& p, int bh, int wbase, int tid) {
  uint32_t* flat = &sm.mask[0][0][0][0];
  for (int i = tid; i < kTiles * D * 2 * kMaskWords; i += kThreads) flat[i] = 0u;
  __syncthreads();

  const int token = tid & (kTile - 1);
  const int half = tid >> 6;
#pragma unroll
  for (int l = 0; l < D; ++l) {
#pragma unroll
    for (int side = 0; side < 2; ++side) {
      const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
      if (((p.modes >> (2 * l)) & want) == 0u) continue;
      const __nv_bfloat16* plane = (side == 0) ? p.pread : p.pwrite;
      for (int ti = 0; ti < kTiles; ++ti) {
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

template <int D>
__device__ __forceinline__ uint32_t slab_bits(const IntraSmem<D>& sm, uint32_t modes, int ti, int l,
                                              int side, int slab) {
  const uint32_t want = (side == 0 ? kModeReadSparse : kModeWriteSparse);
  if (((modes >> (2 * l)) & want) == 0u) return 0xFFFFu;
  return (sm.mask[ti][l][side][slab >> 1] >> ((slab & 1) * 16)) & 0xFFFFu;
}

template <int D>
__device__ __forceinline__ bool slab_live(const IntraSmem<D>& sm, uint32_t modes, int rt, int ct,
                                          int l, int slab) {
  return (slab_bits(sm, modes, rt, l, 0, slab) & slab_bits(sm, modes, ct, l, 1, slab)) != 0u;
}

template <int D>
__device__ __forceinline__ void stage_slabs(IntraSmem<D>& sm, const IntraParams& p, int bh,
                                            int wbase, int rt, int ct, int l, int slab, int buf,
                                            int tid) {
  const int token = tid >> 1;
  const int part = (tid & 1) * 8;
  const int digit = l * kLevelWidth + slab * kSlabDigits + part;
  const long long row = (long long)(bh * p.L + wbase + token) * p.row_width + digit;
  stage_tile(smem_addr(&sm.rslab[buf][token][part]),
             p.pread + row + (long long)rt * kTile * p.row_width);
  stage_tile(smem_addr(&sm.wslab[buf][token][part]),
             p.pwrite + row + (long long)ct * kTile * p.row_width);
}

template <int D>
__device__ __forceinline__ void stage_values(IntraSmem<D>& sm, const IntraParams& p, int bh,
                                             int wbase, int ct, int tid) {
  const int b = bh / p.heads;
  const int h = bh - b * p.heads;
  const int t0 = wbase + ct * kTile;
  const int token = tid >> 1;
  const int part = (tid & 1) * 32;
  const __nv_bfloat16* src =
      p.v + ((long long)(b * p.tokens + t0 + token) * p.heads + h) * kValueWidth + part;
#pragma unroll
  for (int c = 0; c < 4; ++c) stage_tile(smem_addr(&sm.vslab[token][part + c * 8]), src + c * 8);
  if (tid < kTile / 8) {
    stage_tile(smem_addr(&sm.gain[tid * 8]), p.gwrite + (long long)bh * p.L + t0 + tid * 8);
  }
}

template <int D>
__global__ __launch_bounds__(kThreads) void intra_kernel(IntraParams p) {
  __shared__ IntraSmem<D> sm;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int bh = blockIdx.y;
  const int wbase = blockIdx.x * kWindow;

  build_masks<D>(sm, p, bh, wbase, tid);

  const int arow = warp * kWarpRows + (lane & 15);
  const int acol = (lane >> 4) * 8;
  const int row_lo = lane >> 2;
  const int col_lo = (lane & 3) * 2;

  for (int rt = 0; rt < kTiles; ++rt) {
    float oacc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    float dacc[2] = {0.f, 0.f};

    for (int ct = 0; ct <= rt; ++ct) {
      __syncthreads();
      stage_values<D>(sm, p, bh, wbase, ct, tid);
      bool live = slab_live<D>(sm, p.modes, rt, ct, 0, 0);
      if (live) stage_slabs<D>(sm, p, bh, wbase, rt, ct, 0, 0, 0, tid);
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
        for (int slab = 0; slab < kSlabsPerLevel; ++slab) {
          const int step = l * kSlabsPerLevel + slab;
          const int nl = (slab + 1 == kSlabsPerLevel) ? l + 1 : l;
          const int ns = (slab + 1 == kSlabsPerLevel) ? 0 : slab + 1;
          bool next_live = false;
          if (nl < D) {
            next_live = slab_live<D>(sm, p.modes, rt, ct, nl, ns);
            if (next_live) stage_slabs<D>(sm, p, bh, wbase, rt, ct, nl, ns, (step + 1) & 1, tid);
          }
          stage_commit();
          stage_wait<1>();
          __syncthreads();
          if (live) {
            const int buf = step & 1;
            uint32_t af[4];
            load_frag(af, smem_addr(&sm.rslab[buf][arow][acol]));
#pragma unroll
            for (int jn = 0; jn < 4; ++jn) {
              uint32_t bf[4];
              load_frag(bf, smem_addr(&sm.wslab[buf][jn * 16 + (lane & 15)][acol]));
              const uint32_t b0[2] = {bf[0], bf[2]};
              const uint32_t b1[2] = {bf[1], bf[3]};
              mma(gram[l][2 * jn], af, b0);
              mma(gram[l][2 * jn + 1], af, b1);
            }
          }
          live = next_live;
          __syncthreads();
        }
      }

#pragma unroll
      for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float a = gram[0][n][i];
#pragma unroll
          for (int l = 1; l < D; ++l) a *= gram[l][n][i];
          const int cl = n * 8 + col_lo + (i & 1);
          a *= __bfloat162float(sm.gain[cl]);
          if (rt == ct && cl > warp * kWarpRows + row_lo + (i >> 1) * 8) a = 0.f;
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

    const int grow = wbase + rt * kTile + warp * kWarpRows + row_lo;
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
