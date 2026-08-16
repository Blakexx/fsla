// THE PRODUCER'S TILE-GEMM COMPONENT -- bf16 operands, fp32 accumulators, one CTA tile.
// Design of record: docs/internals/entmax/tile_gemm.md.
//
// One CTA computes the fp32 logit tile `[WORD_TOKENS, C]` for ONE (batch, head) stream:
// `logits[t, c] = sum_k x[t, k] * W[head, k, c] (+ bias[c])`, with the operands narrowed
// to bf16 and every accumulation in fp32 on the tensor cores (`mma.m16n8k16`). The tile
// is the support word's 32 tokens, which is what lets the solve that follows own its
// support bits with no atomic and no second pass.
//
// THE ACCUMULATOR LIVES IN REGISTERS and the tile never reaches DRAM: that is the whole
// point of the fusion, and it is why the n-tile count per warp `NT` is a template
// parameter -- a runtime trip count would force a fixed worst-case register array.
#pragma once

#include <cuda_bf16.h>

#include "entmax.cuh"

namespace rola {
namespace entmax {
namespace detail {

constexpr int GEMM_WARPS = 8;
constexpr int GEMM_THREADS = GEMM_WARPS * WARP_SIZE;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int TILE_K = 32;
constexpr int M_TILES = WORD_TOKENS / MMA_M;
//: `NT` is bounded by the widest packed router one CTA can hold; the host refuses more.
constexpr int MAX_N_TILES_PER_WARP = 8;
constexpr int MAX_PACKED_COLUMNS = MAX_N_TILES_PER_WARP * GEMM_WARPS * MMA_N;
constexpr int XS_STRIDE = TILE_K + 8;

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)d;
  (void)a;
  (void)b;
#endif
}

__device__ __forceinline__ __nv_bfloat16 to_bf16(const void* p, int64_t i, bool is_bf16) {
  return is_bf16 ? ((const __nv_bfloat16*)p)[i] : __float2bfloat16(((const float*)p)[i]);
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  uint32_t r;
  uint16_t l = *(const uint16_t*)&lo, h = *(const uint16_t*)&hi;
  r = ((uint32_t)h << 16) | (uint32_t)l;
  return r;
}

//: ONE CTA TILE. `xs`/`ws` are the K-block staging buffers and `out` the fp32 tile; the
//: caller ALIASES them (the staging is dead once the last `mma` retires) and inserts the
//: barrier -- docs/internals/entmax/tile_gemm.md#smem-alias
template <int NT>
__device__ __forceinline__ void tile_gemm(const void* __restrict__ x, bool x_bf16,
                                          int64_t x_token_stride, int64_t x_origin,
                                          const void* __restrict__ w, bool w_bf16, int64_t w_origin,
                                          const float* __restrict__ bias, int hidden, int columns,
                                          int rows, __nv_bfloat16* xs, __nv_bfloat16* ws,
                                          int ws_stride, float* out, int out_stride) {
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int group = lane >> 2;
  const int in_group = lane & 3;

  float acc[NT][M_TILES][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int m = 0; m < M_TILES; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[t][m][i] = 0.0f;

  const int x_elems = WORD_TOKENS * TILE_K;
  const int w_elems = TILE_K * ws_stride;
  for (int kb = 0; kb < hidden; kb += TILE_K) {
    for (int e = tid; e < x_elems; e += GEMM_THREADS) {
      const int row = e / TILE_K, k = e - row * TILE_K;
      const bool live = row < rows && (kb + k) < hidden;
      xs[row * XS_STRIDE + k] =
          live ? to_bf16(x, x_origin + (int64_t)row * x_token_stride + kb + k, x_bf16)
               : __float2bfloat16(0.0f);
    }
    //: The pad columns are ZEROED rather than skipped: an n-tile that runs past `columns`
    //: still has to be issued (an `mma` is a warp-wide collective), so its operands must
    //: be harmless -- docs/internals/entmax/tile_gemm.md#ragged-tile
    for (int e = tid; e < w_elems; e += GEMM_THREADS) {
      const int k = e / ws_stride, c = e - k * ws_stride;
      const bool live = (kb + k) < hidden && c < columns;
      ws[e] = live ? to_bf16(w, w_origin + (int64_t)(kb + k) * columns + c, w_bf16)
                   : __float2bfloat16(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < TILE_K / MMA_K; ++ks) {
      const int ka = ks * MMA_K + in_group * 2;
      uint32_t a[M_TILES][4];
#pragma unroll
      for (int m = 0; m < M_TILES; ++m) {
        const int r0 = m * MMA_M + group;
        a[m][0] = *(const uint32_t*)&xs[r0 * XS_STRIDE + ka];
        a[m][1] = *(const uint32_t*)&xs[(r0 + 8) * XS_STRIDE + ka];
        a[m][2] = *(const uint32_t*)&xs[r0 * XS_STRIDE + ka + 8];
        a[m][3] = *(const uint32_t*)&xs[(r0 + 8) * XS_STRIDE + ka + 8];
      }
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        //: WARP-UNIFORM, and it has to be: `mma` is a warp-wide collective, so a per-lane
        //: column bound cannot gate it. The ragged columns carry zeros instead.
        const int n0 = (t * GEMM_WARPS + warp) * MMA_N;
        if (n0 >= columns) continue;
        const int col = n0 + group;
        uint32_t b[2];
        b[0] = pack_bf16(ws[(ks * MMA_K + in_group * 2) * ws_stride + col],
                         ws[(ks * MMA_K + in_group * 2 + 1) * ws_stride + col]);
        b[1] = pack_bf16(ws[(ks * MMA_K + in_group * 2 + 8) * ws_stride + col],
                         ws[(ks * MMA_K + in_group * 2 + 9) * ws_stride + col]);
#pragma unroll
        for (int m = 0; m < M_TILES; ++m) mma_m16n8k16(acc[t][m], a[m], b);
      }
    }
    __syncthreads();
  }

  //: The staging buffers are dead here and `out` aliases them; the caller's barrier above
  //: is the last one that read `ws`.
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int c0 = (t * GEMM_WARPS + warp) * MMA_N + in_group * 2;
#pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
      const int r0 = m * MMA_M + group;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = r0 + (i >> 1) * 8;
        const int col = c0 + (i & 1);
        if (row < rows && col < columns)
          out[row * out_stride + col] = acc[t][m][i] + (bias != nullptr ? bias[col] : 0.0f);
      }
    }
  }
}

}  // namespace detail
}  // namespace entmax
}  // namespace rola
