// csrc/rola/src/facts/facts_launch.cuh -- the facts kernels' ARM SURFACE: one
// launcher template each for the union-table pass, the stats reduction and the
// liveness epilogue. Included only by the instantiation shards; TORCH-FREE.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "facts_kernel.cuh"

namespace rola::chunk {

//: ONE PASS, BOTH ROW SETS. -- see docs/internals/facts/facts_launch.md#ktblwarps
template <class PLAN_R, class PLAN_W, int MASK>
cudaError_t launch_facts_plan(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                              uint32_t* atom_tbl, int BH, int L, cudaStream_t stream) {
  constexpr int kTblWarps = kFactsThreads / kBallotLanes;
  //: the variant with no atom-grain emission compiled in, handed an atom table, -- see docs/internals/facts/facts_launch.md#mask
  if (MASK == kPrefillStateless && atom_tbl != nullptr) return cudaErrorInvalidValue;
  const dim3 grid((unsigned)((tok_words(L) + kTblWarps - 1) / kTblWarps), (unsigned)BH);
  chunk_facts_tables<PLAN_R, PLAN_W, MASK>
      <<<grid, kFactsThreads, 0, stream>>>(pr, pw, tbl, atom_tbl, L);
  return cudaGetLastError();
}

//: THE LEX ARM, unchanged as a SURFACE: the dispatch vocabulary is still `(D, -- see docs/internals/facts/facts_launch.md#note-l34
template <int D, int B, int BC, int MASK>
cudaError_t launch_chunk_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   uint32_t* atom_tbl, int BH, int L, cudaStream_t stream) {
  return launch_facts_plan<LevelPlan<D, B, BC>, LevelPlan<D, B, BC>, MASK>(pr, pw, tbl, atom_tbl,
                                                                           BH, L, stream);
}

//: THE GIVEN-SPAN ARM -- the box lattice's owner rows, AT ONE PARTITION PER -- see docs/internals/facts/facts_launch.md#note-l45
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream) {
  return launch_facts_plan<SpanPlan<D, B, R0, R1, R2, R3>, SpanPlan<D, B, W0, W1, W2, W3>, MASK>(
      pr, pw, tbl, nullptr, BH, L, stream);
}

//: THE STATS PASS' REDUCTION ARM, keyed (D, B) alone: the atom rectangle is -- see docs/internals/facts/facts_launch.md#near-line-52
template <int D, int B>
cudaError_t launch_chunk_atom_bitmap_arm(const uint32_t* tbl, uint8_t* atom_bits, int BH, int L,
                                         cudaStream_t stream) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int kInner = LP::clauses > 0 ? LP::groups_at(LP::clauses - 1) : 1;
  const dim3 grid((unsigned)(LP::blocks / kInner), (unsigned)BH);
  atom_bitmap_kernel<D, B><<<grid, kAtomThreads, 0, stream>>>(tbl, atom_bits, L);
  return cudaGetLastError();
}

//: THE LIVENESS EPILOGUE'S LAUNCH, keyed to (D, B, BC) like the union table it -- see docs/internals/facts/facts_launch.md#near-line-65
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
