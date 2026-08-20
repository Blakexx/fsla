// csrc/rola/src/facts/facts_launch.cuh -- the facts kernels' ARM SURFACE:
// the union-table pass, the stats reduction and the liveness epilogue, one
// launcher template each.  Included only by the instantiation shards, on the
// consumer family's rule (a dispatch TU satisfying an arm locally would leave
// the build green and nothing sharded).  TORCH-FREE: every CUDA error is
// RETURNED and the dispatch TU raises through C10.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "facts_kernel.cuh"

namespace rola::chunk {

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

}  // namespace rola::chunk
