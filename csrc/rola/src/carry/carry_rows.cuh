// csrc/rola/src/carry/carry_rows.cuh -- the carry family's OWNER BALLOT ROWS.
//
// The row plans and the one call that materializes them, in the one header BOTH
// the dispatch TU (`carry.cu`) and the per-arm TUs
// (`../instantiations/carry_arm_<i>.cu`) include.  Nothing here instantiates a
// carry kernel: the facts launcher is a DECLARATION, exactly as it was in the
// single TU, so this header costs the dispatch a parse and no codegen.
//
// IT IS ALSO FREE OF THE TENSOR LIBRARY, and that is a build-time requirement, not a
// taste: an arm TU that parses the extension header costs 33.7 s and 2.9 GB before it
// compiles a single instruction of the arm, against 6 s for the arm's own codegen
// (measured, K36).  Allocation is therefore the DISPATCH's job and this header speaks
// pointers.  docs/internals/carry/carry_rows.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "box.cuh"
#include "facts/facts_kernel.cuh"

//: THE OWNER BALLOT ROWS, DECLARED (K31-i-d).  The facts family owns the pass and
//: its instantiation shard owns the arms; this TU sees a DECLARATION, exactly as the
//: facts dispatch TU does, so nothing here can satisfy an arm locally.  The SPANS the
//: carry asks for are `BoxPlan::s(l)` -- the lattice's balanced owner runs, which is
//: the vocabulary the lex plan could not state.
namespace rola::chunk {
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream);
}  // namespace rola::chunk

namespace rola::carry {

//: the plan the carry's rows are voted under, named ONCE: a `BoxPlan`'s spans as a
//: `SpanPlan`.  `D` is at most four and a level past `D` carries `B`.
template <int D, int B, int K, int M>
using OwnerRows =
    rola::chunk::SpanPlan<D, B, BoxPlan<D, B, K, M>::s(0), (D > 1 ? BoxPlan<D, B, K, M>::s(1) : B),
                          (D > 2 ? BoxPlan<D, B, K, M>::s(2) : B),
                          (D > 3 ? BoxPlan<D, B, K, M>::s(3) : B)>;

//: the plan ONE SIDE's rows are voted under (K35 S1b addendum): that side's own
//: partition, at its own run size, with a level it spans whole emitting nothing.  The
//: two sides are independent -- each votes its bitlists from its own factors -- so a
//: stream's test reads exactly one row per tested level and only the owner's whole-run
//: OR spans several (`CarveRows`).
template <class SV>
using SideRowPlan =
    rola::chunk::SpanPlan<CarveRows<SV>::kLevels, CarveRows<SV>::kWidth, CarveRows<SV>::s(0),
                          CarveRows<SV>::s(1), CarveRows<SV>::s(2), CarveRows<SV>::s(3)>;

//: THE SINGLE-LEVEL LEAF SPLIT the seam's `owner_rows(carve=True)` diagnostic names --
//: the innermost level's owner run over `kSeamCarveStreams` streams, which is the grain
//: that entry point documents.  It is a DIAGNOSTIC's declared grain and not an arm's
//: carve, so it is spelled as a span vector here and derived from nothing.
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

//: THE TABLE: `[BH, entries_read + entries_write, L/32]` packed bits, the READ plan's
//: rows and then the WRITE plan's, one row per (level, run of THAT side's partition).
//: The two plans are independent (K35 S1b addendum): a side's bitlists are voted from
//: its own factors, so nothing is gained by refining one side's table by the other's
//: carve.  Where the two plans coincide the block IS the old `[BH, 2, E, L/32]`.
//: Also a SEPARATE ENTRY POINT, so the grain the kernel consumes and the grain the
//: seam names can be compared bit for bit (`tests/k31/test_owner_rows.py`).
//:
//: THE ALLOCATION IS THE CALLER'S (K36): this launches into a table the caller sized
//: with `rows_entries` and owns.
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
