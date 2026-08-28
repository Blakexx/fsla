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
template <class PLAN_R, class PLAN_W, int MASK>
cudaError_t launch_facts_plan(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                              uint32_t* atom_tbl, int BH, int L, cudaStream_t stream) {
  constexpr int kTblWarps = kFactsThreads / kBallotLanes;
  //: the variant with no atom-grain emission compiled in, handed an atom table,
  //: would return it unwritten: the class and the pointers must agree.
  if (MASK == kPrefillStateless && atom_tbl != nullptr) return cudaErrorInvalidValue;
  const dim3 grid((unsigned)((tok_words(L) + kTblWarps - 1) / kTblWarps), (unsigned)BH);
  chunk_facts_tables<PLAN_R, PLAN_W, MASK>
      <<<grid, kFactsThreads, 0, stream>>>(pr, pw, tbl, atom_tbl, L);
  return cudaGetLastError();
}

//: THE LEX ARM, unchanged as a SURFACE: the dispatch vocabulary is still
//: `(D, B, BC)` x call class, and the plan it names is the lex policy's.  The
//: SPAN arm below is the same launcher under a plan whose spans are given, which
//: is the whole of what K31-i-d needed.
template <int D, int B, int BC, int MASK>
cudaError_t launch_chunk_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   uint32_t* atom_tbl, int BH, int L, cudaStream_t stream) {
  return launch_facts_plan<LevelPlan<D, B, BC>, LevelPlan<D, B, BC>, MASK>(pr, pw, tbl, atom_tbl,
                                                                           BH, L, stream);
}

//: THE GIVEN-SPAN ARM -- the box lattice's owner rows, AT ONE PARTITION PER SIDE
//: (K35 S1b).  `R0..R3` are the READ side's per-level runs and `W0..W3` the WRITE
//: side's; a level past `D` passes `B` (the identity), and a level a side spans whole
//: emits no rows there.  The carry consumes these rows, and there is no atom grain
//: here: residency is the paging layer's question and the carry family does not ask it.
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream) {
  return launch_facts_plan<SpanPlan<D, B, R0, R1, R2, R3>, SpanPlan<D, B, W0, W1, W2, W3>, MASK>(
      pr, pw, tbl, nullptr, BH, L, stream);
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
