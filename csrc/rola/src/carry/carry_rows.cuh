// csrc/rola/src/carry/carry_rows.cuh -- the OWNER BALLOT ROWS' launch. Torch-free
// (K36) and the carry's own pass since K50. See docs/internals/carry/carry_rows.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "carry_rows_kernel.cuh"
#include "geom.cuh"

namespace rola::carry {

//: THE SEAM'S CARVE DIAGNOSTIC (`owner_rows(carve=True)`), as a stream count.
static constexpr int kSeamCarveStreams = 4;

//: `[BH, rows_read + rows_write, ceil(L/32)]`, the READ side's rows first.
static constexpr int rows_entries(const CarryGeomRT& geo) { return geo.r.rows + geo.w.rows; }

template <int D>
static cudaError_t launch_rows(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                               int BH, int L, const CarryGeomRT& geo, cudaStream_t stream) {
  const int words = tok_words(L);
  const int per_cta = kRowsThreads / 32;
  const dim3 grid((unsigned)((words + per_cta - 1) / per_cta), (unsigned)BH);
  carry_rows_kernel<D><<<grid, kRowsThreads, 0, stream>>>(pr, pw, tbl, L, geo);
  return cudaGetLastError();
}

}  // namespace rola::carry
