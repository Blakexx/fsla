#pragma once

// THE PRODUCER-EPILOGUE ROUTING SUMMARY PASS (`THUNDER_RUN_SPEC.md` §4.1/§4.2), as an
// ahead-of-time CUDA kernel. This header owns the parameter block, the pass's declared
// bounds and its two entry points; `summaries.cu` owns the kernel.
//
// Three facts a reader must not miss:
//
//   1. IT IS A KERNEL AND NOT FUSED TORCH: no Inductor in the runtime (§6 P4), the
//      representation is BITS, and "one launch, one D2H, zero host syncs" is a property
//      only a hand-written pass can have. The eager-torch first implementation survives
//      in `rola/routing/summaries.py` as the ENUMERATED REFERENCE this kernel is gated
//      against -- a second derivation, not a fallback.
//   2. HAND-ROLL SEMANTICS, USE LIBRARIES FOR COMMODITY PRIMITIVES. The digit algebra,
//      the owner rectangle, the candidacy union and the leaf AND-chain are written out;
//      the scalar folds are `cub::BlockReduce`. CUB is NOT used for the histogram: its
//      bin count is `QPO + 1`, which invariant I1 forbids making a literal.
//   3. Liveness is EXACT NONZERO-NESS of the amplitude (spec §1.2) everywhere in this
//      pass -- never a threshold, and never a magnitude test on anything downstream.
//
// The three reasons in full, the per-output cost table (and why the `O(N)` leaf bitmap
// is irreducible), and how §4.2's owner condensation rides the SAME `atomicOr`:
// docs/internals/summaries.md

#include <cuda_runtime_api.h>

#include <cstdint>

#include <torch/types.h>

#include <vector>

//: For `kMmaKQuantum`, §6's MMA k-extent and therefore the quantum every gathered
//: extent rounds up to. Taken from the one home rather than restated here: this pass's
//: bin count and the engine's atom granularity are the SAME number, and the host
//: entry checks the derived bin count against the caller's so a drift fails loudly.
//: docs/internals/arch_table.md#mma-quantum
#include "arch_table.cuh"

namespace rola::summaries {

//: Token bits carried by one support word. A `BT` above this cannot be
//: represented and is REFUSED, never split into a second code path.
constexpr int kTokenBitsPerWord = 32;

//: Ceiling on `sum_l b_l`, the digit-row extent one CTA stages -- two `int32` arrays of
//: this length in shared memory, so a 16 KiB budget. A STRUCTURAL bound under the
//: standing "design for nc ~ 65K" rule: docs/internals/summaries.md#declared-bounds
constexpr int kMaxTotalDigits = 2048;

//: Ceiling on `D`. Matches `FeederFlags`' own `D <= 16` (§4.4 packs two bits per
//: level into one word), so the two limits cannot drift apart.
constexpr int kMaxLevels = 16;

//: EVERY ARGUMENT THE PASS TAKES, and nothing derived that the host can derive. The
//: amplitude pointers are the producer's FOLDED per-level allocations, `[BH, T, b_l]`
//: fp32 -- exactly what `RoutingViews.read_folded` / `.write_folded` hold.
//: docs/internals/summaries.md#params
struct SummaryParams {
  //: PASSED BY VALUE, in the kernel's PARAMETER SPACE, and that is deliberate: the
  //: alternative is four pageable H2D copies on a pass whose declared budget is one
  //: transfer and zero host syncs. docs/internals/summaries.md#params-by-value
  const float* read_amps[kMaxLevels];  //: `[BH, T, b_l]` fp32, folded, contiguous
  const float* write_amps[kMaxLevels];
  int level_width[kMaxLevels];        //: `b_l`
  int level_row_offset[kMaxLevels];   //: offset into the digit-row space
  int level_leaf_stride[kMaxLevels];  //: `prod(b_{l+1..D-1})`
  int level_fixed_radix[kMaxLevels];  //: `prod(b_{l+1..j-1})`; 1 on spanned levels

  int level_count;  //: `D`
  int batch_heads;  //: `BH`
  int token_count;  //: `T`
  int token_tile;   //: `BT`
  int n_tiles;
  int total_digits;  //: `sum_l b_l`
  int leaf_count;    //: `N`
  int state_block;   //: `BC`
  int owners_total;  //: `N / BC`
  int n_slices;
  int slice_digits;         //: `c`
  int first_spanned_level;  //: `j`
  int bins;                 //: `QPO + 1`, `QPO = BC / kMmaKQuantum`
  int leaf_words;           //: `ceil(N / 32)`
  int owner_words;          //: `ceil(owners_total / 32)`

  unsigned long long* occupancy;            //: [2 * D] live `(token, digit)` pairs
  unsigned long long* histogram;            //: [bins] candidate pairs per quantum
  unsigned long long* candidate_count;      //: scalar
  unsigned long long* written_owner_count;  //: scalar
  int* written_leaf_bitmap;                 //: [BH, leaf_words], §4.1's canonical truth
  int* written_owner_bitmap;                //: [BH, owner_words], the derived condensation
};

//: THE ENTRY POINT. One launch, no host synchronization, nothing allocated. Every
//: output tensor is caller-owned and MUST BE ZEROED beforehand: the kernel folds into
//: them with atomics, so it accumulates rather than assigns.
void launch_summaries(const SummaryParams& params, cudaStream_t stream);

//: The pybind surface (`rola_routing_summaries`). Validates shapes, builds the device
//: pointer arrays, launches. Returns nothing: the outputs are the caller's tensors, so
//: the Python side owns the ONE D2H transfer. docs/internals/summaries.md#entry-points
void rola_routing_summaries_cuda(
    const std::vector<torch::Tensor>& read_amps, const std::vector<torch::Tensor>& write_amps,
    const std::vector<int64_t>& level_widths, int64_t token_tile, int64_t state_block,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices, torch::Tensor occupancy,
    torch::Tensor histogram, torch::Tensor candidate_count, torch::Tensor written_owner_count,
    torch::Tensor written_leaf_bitmap, torch::Tensor written_owner_bitmap);

}  // namespace rola::summaries
