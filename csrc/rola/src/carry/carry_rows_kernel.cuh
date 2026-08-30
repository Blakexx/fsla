// csrc/rola/src/carry/carry_rows_kernel.cuh -- THE OWNER BALLOT ROWS, at the
// addressing block's own grain: one bit per (digit run, token), one kernel per DEPTH
// because K50 makes the runs runtime. Why the carry owns this pass, and what it does
// NOT share with the facts family's bit emission:
// see docs/internals/carry/carry_rows_kernel.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/ops.cuh"
#include "box.cuh"
#include "geom.cuh"

namespace rola::carry {

using namespace denseref::ops;

//: one warp per token word, four warps per CTA.
constexpr int kRowsThreads = 128;

//: THE TABLE'S MINOR AXIS -- see docs/internals/carry/carry_rows_kernel.md#tok-words
constexpr int kTokBits = 32;
__host__ __device__ constexpr int tok_words(int L) { return (L + kTokBits - 1) / kTokBits; }

//: THE OR OF A CONTIGUOUS RUN OF RUNTIME WIDTH, aligned by the seam's own refusal.
//: -- see docs/internals/carry/carry_rows_kernel.md#or-run-rt
__device__ __forceinline__ uint32_t or_run_rt(const __nv_bfloat16* p, int n) {
  if (n == 1) return (uint32_t)*reinterpret_cast<const unsigned short*>(p);
  if (n == 2) return *reinterpret_cast<const uint32_t*>(p);
  if (n == 4) {
    const uint2 x = *reinterpret_cast<const uint2*>(p);
    return x.x | x.y;
  }
  uint32_t acc = 0u;
#pragma unroll 1
  for (int i = 0; i < n; i += 8) {
    const uint4 x = *reinterpret_cast<const uint4*>(p + i);
    acc |= x.x | x.y | x.z | x.w;
  }
  return acc;
}

//: THE TABLE -- see docs/internals/carry/carry_rows_kernel.md#carry-rows-kernel
template <int D>
__global__ __launch_bounds__(kRowsThreads) void carry_rows_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    uint32_t* __restrict__ tbl, int L, const CarryGeomRT geo) {
  const int words = tok_words(L);
  const int w = blockIdx.x * (kRowsThreads / kBallotLanes) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= words) return;  // a whole lockstep unit leaves together
  //: A TOKEN PAST THE END READS TOKEN ZERO AND VOTES DEAD.
  const int t = w * kTokBits + lane;
  const bool real = t < L;
  const long off = (long)blockIdx.y * L * geo.wtot + (long)(real ? t : 0) * geo.wtot;
  const long stride = words;
  const int rowsBoth = geo.r.rows + geo.w.rows;

  static_for<2>([&](auto Sc) {
    constexpr int side = decltype(Sc)::value;  // 0 = read, 1 = write
    const SideGeom& sg = side ? geo.w : geo.r;
    const __nv_bfloat16* const amp = (side ? pwrite : pread) + off;
    uint32_t* out = tbl + ((long)blockIdx.y * rowsBoth + (side ? geo.r.rows : 0)) * stride + w;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int span = sg.row_span[l];
      const __nv_bfloat16* p = amp + geo.col_base[l];
      uint32_t* o = out + (long)sg.row_prefix[l] * stride;
#pragma unroll 1
      for (int e = 0; e < sg.row_g[l]; ++e) {
        const uint32_t vote = lane_ballot(real && (or_run_rt(p, span) & kBf16MagnitudeMask) != 0u);
        if (lane == 0) *o = vote;
        p += span;
        o += stride;
      }
    });
  });
}

}  // namespace rola::carry
