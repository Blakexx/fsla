// csrc/rola/src/carry/carry_rows.cuh -- the carry family's OWNER BALLOT ROWS: the
// row plans and the one call that materializes them, shared by the dispatch TU
// and every per-arm TU. Torch-free BY MEASUREMENT (K36: 33.7s/2.9GB avoided per
// arm TU) and instantiates no kernel -- the facts launcher stays a declaration.
// See docs/internals/carry/carry_rows.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "box.cuh"
#include "facts/facts_kernel.cuh"

//: THE OWNER BALLOT ROWS, DECLARED (K31-i-d). The facts family owns the pass -- see docs/internals/carry/carry_rows.md#note-l24
namespace rola::chunk {
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream);
}  // namespace rola::chunk

namespace rola::carry {

//: the plan the carry's rows are voted under, named ONCE: a `BoxPlan`'s spans as a -- see docs/internals/carry/carry_rows.md#near-line-36
template <int D, int B, int K, int M>
using OwnerRows =
    rola::chunk::SpanPlan<D, B, BoxPlan<D, B, K, M>::s(0), (D > 1 ? BoxPlan<D, B, K, M>::s(1) : B),
                          (D > 2 ? BoxPlan<D, B, K, M>::s(2) : B),
                          (D > 3 ? BoxPlan<D, B, K, M>::s(3) : B)>;

//: the plan ONE SIDE's rows are voted under (K35 S1b addendum): that side's own -- see docs/internals/carry/carry_rows.md#siderowplan
template <class SV>
using SideRowPlan =
    rola::chunk::SpanPlan<CarveRows<SV>::kLevels, CarveRows<SV>::kWidth, CarveRows<SV>::s(0),
                          CarveRows<SV>::s(1), CarveRows<SV>::s(2), CarveRows<SV>::s(3)>;

//: THE SINGLE-LEVEL LEAF SPLIT the seam's `owner_rows(carve=True)` diagnostic -- see docs/internals/carry/carry_rows.md#kseamcarvestreams
static constexpr int kSeamCarveStreams = 4;
template <int D, int B, int K, int M>
constexpr int owner_span(int l) {
  return l < D ? BoxPlan<D, B, K, M>::s(l) : B;
}
template <int D, int B, int K, int M>
constexpr int split_span(int l) {
  return owner_span<D, B, K, M>(l) / (l == D - 1 ? kSeamCarveStreams : 1);
}
template <int D, int B, int K, int M>
using SeamSplitPlan =
    rola::chunk::SpanPlan<D, B, split_span<D, B, K, M>(0), split_span<D, B, K, M>(1),
                          split_span<D, B, K, M>(2), split_span<D, B, K, M>(3)>;

//: THE TABLE: `[BH, entries_read + entries_write, L/32]` packed bits, the READ -- see docs/internals/carry/carry_rows.md#rows-entries
template <class RPR, class RPW>
static constexpr int rows_entries() {
  return (int)RPR::entries + (int)RPW::entries;
}

template <class RPR, class RPW>
static cudaError_t launch_rows(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                               int BH, int L, cudaStream_t stream) {
  return rola::chunk::launch_span_facts_arm<RPR::depth, RPR::kWidth, RPR::span_at(0),
                                            RPR::span_at(1), RPR::span_at(2), RPR::span_at(3),
                                            RPW::span_at(0), RPW::span_at(1), RPW::span_at(2),
                                            RPW::span_at(3), rola::chunk::kPrefillStateless>(
      pr, pw, tbl, BH, L, stream);
}

}  // namespace rola::carry
