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

//: ONE PASS, BOTH ROW SETS.  `MASK` is the CALL CLASS and never a bool per fact:
//: the stateless class compiles no atom-grain emission at all, and within the
//: stateful one a call with no arena to admit into hands a null atom table -- the
//: runtime-null OUTPUT skip, not a second arm.
template <int D, int B, int BC, int MASK>
cudaError_t launch_chunk_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   uint32_t* atom_tbl, int BH, int L, cudaStream_t stream) {
  constexpr int kTblWarps = kFactsThreads / kBallotLanes;
  //: the variant with no atom-grain emission compiled in, handed an atom table,
  //: would return it unwritten: the class and the pointers must agree.
  if (MASK == kPrefillStateless && atom_tbl != nullptr) return cudaErrorInvalidValue;
  const dim3 grid((unsigned)((tok_words(L) + kTblWarps - 1) / kTblWarps), (unsigned)BH);
  chunk_facts_tables<D, B, BC, MASK><<<grid, kFactsThreads, 0, stream>>>(pr, pw, tbl, atom_tbl, L);
  return cudaGetLastError();
}

//: THE STATS PASS' REDUCTION ARM, keyed (D, B) alone: the atom rectangle is
//: `kAtomLeaves` by construction, so residency is BC-free and this arm list is
//: smaller than the consumer's (docs/internals/paging/stats.md).
template <int D, int B>
cudaError_t launch_chunk_atom_bitmap_arm(const uint32_t* tbl, uint8_t* atom_bits, int BH, int L,
                                         cudaStream_t stream) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int kInner = LP::clauses > 0 ? LP::groups_at(LP::clauses - 1) : 1;
  const dim3 grid((unsigned)(LP::blocks / kInner), (unsigned)BH);
  atom_bitmap_kernel<D, B><<<grid, kAtomThreads, 0, stream>>>(tbl, atom_bits, L);
  return cudaGetLastError();
}

//: THE LIVENESS EPILOGUE'S LAUNCH, keyed to (D, B, BC) like the union table it
//: reduces -- the block grain is the LAUNCH's, so unlike the stats pass this one
//: does carry a `BC` axis.
template <int D, int B, int BC>
cudaError_t launch_chunk_block_bitmap_arm(const uint32_t* tbl, const int32_t* page_tbl,
                                          uint8_t* block_bits, int BH, int L, cudaStream_t stream) {
  using LP = LevelPlan<D, B, BC>;
  constexpr int kInner = LP::clauses > 0 ? LP::groups_at(LP::clauses - 1) : 1;
  const dim3 grid((unsigned)(LP::blocks / block_bitmap_group(kInner)), (unsigned)BH);
  block_bitmap_kernel<D, B, BC><<<grid, kAtomThreads, 0, stream>>>(tbl, page_tbl, block_bits, L);
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
