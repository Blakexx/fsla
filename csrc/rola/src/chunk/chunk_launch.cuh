// csrc/rola/src/chunk/chunk_launch.cuh -- the chunk consumer's ARM SURFACE.
// Included ONLY by the instantiation shards; chunk.cu sees declarations
// alone (a dispatch TU satisfying an arm locally would leave the build green
// and nothing sharded -- the consumer family's rule, kept here).  TORCH-FREE
// on the same rule as consumer_launch.cuh: shards carry no host framework;
// every CUDA error is RETURNED and the dispatch TU raises through C10.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

#include "../facts/facts_launch.cuh"
#include "chunk_kernel.cuh"

namespace rola::chunk {

//: ONE ARM, ONE STATE VARIANT.  `STATE` is a kernel axis (chunk_kernel.cuh): the
//: variant that carries no state I/O compiles none of it, and which one a call
//: wants is a property of its POINTERS, decided by `chunk.cu`'s dispatch on the
//: host where a branch costs nothing.  The BACKING is not an axis: dense and
//: paged are one kernel and one base translation, told apart by whether the page
//: table is null (docs/internals/paging/paging.md).  Neither is LIVENESS: the
//: bitmap is a pointer the CTA exits on, and the grid is one CTA per block
//: whether it is present or not (docs/internals/chunk/chunk_launch.md#the-grid).
template <int C, int BC, int D, int B, bool STATE>
cudaError_t launch_chunk_arm(const ChunkParams& p, cudaStream_t stream) {
  constexpr size_t sm = Geom<C, BC>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        compacted_kernel<C, D, B, BC, STATE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  //: a state pointer -- or a table to resolve one through -- handed to the
  //: variant that has no state code would be read by nothing: the dispatch's
  //: choice and the pointers must agree.
  if (!STATE && (p.s_in != nullptr || p.s_out != nullptr || p.page_tbl != nullptr))
    return cudaErrorInvalidValue;
  const dim3 grid((unsigned)LevelPlan<D, B, BC>::blocks, (unsigned)p.BH);
  compacted_kernel<C, D, B, BC, STATE><<<grid, threads_for(BC), sm, stream>>>(
      p.pr, p.pw, p.gw, p.v, p.tbl, p.o, p.den, p.L, p.v_row_bytes, p.s_in, p.s_out, p.page_tbl,
      p.block_bits, p.run_table);
  return cudaGetLastError();
}

//: the census: everything build_stamp reports for an arm.  Lives HERE so the
//: dispatch TU never touches the kernel symbol.
template <int C, int BC, int D, int B, bool STATE>
std::array<int64_t, 9> query_chunk_census_arm() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, compacted_kernel<C, D, B, BC, STATE>) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  return {a.numRegs,
          (int64_t)a.sharedSizeBytes,
          a.binaryVersion,
          (int64_t)Geom<C, BC>::bytes,
          (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.localSizeBytes,
          (int64_t)Geom<C, BC>::stages,
          (int64_t)LevelPlan<D, B, BC>::blocks,
          (int64_t)LevelPlan<D, B, BC>::entries};
}

}  // namespace rola::chunk
