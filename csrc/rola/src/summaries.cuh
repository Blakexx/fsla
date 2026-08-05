#pragma once

// THE PRODUCER-EPILOGUE ROUTING SUMMARY PASS (`THUNDER_RUN_SPEC.md` §4.1/§4.2),
// as an ahead-of-time CUDA kernel.
//
// ---- WHY THIS IS A KERNEL AND NOT FUSED TORCH -------------------------------
//
// The first implementation was eager torch. It is retained, in
// `rola/routing/summaries.py`, as the ENUMERATED REFERENCE this kernel is gated
// against -- a legitimate second-derivation role, not a fallback: there is no
// runtime path that reaches it, and it refuses to be called as one.
//
// Three reasons the runtime pass is hand-written:
//
//   1. NO INDUCTOR IN THE RUNTIME. `THUNDER_RUN_SPEC.md` §6 P4 exists partly to
//      remove `torch.compile` as a runtime dependency of the planner; a fused
//      torch summary pass would put it straight back. Eager torch instead means
//      ~30 kernel launches and a DRAM round trip per intermediate, on a pass
//      whose whole justification is that it is microseconds.
//   2. THE REPRESENTATION IS BITS. Every step below is a word operation --
//      per-tile union words over DIGITS, `__popc`, `__ballot_sync`, an
//      atomicOr-and-count-new-bits. Torch has no vocabulary for that; it would
//      spell each one as a boolean tensor and a reduction.
//   3. ONE LAUNCH, ONE D2H, ZERO HOST SYNCS is a property of the pass, and the
//      only way to have it is to write the pass.
//
// ---- THE STANDING RULE THIS FOLLOWS -----------------------------------------
//
// HAND-ROLL SEMANTICS, USE LIBRARIES FOR COMMODITY PRIMITIVES. The digit
// algebra, the owner rectangle, the candidacy union and the leaf AND-chain are
// semantics and are written out. The scalar folds are commodity and are
// `cub::BlockReduce`. CUB is header-only and ships with the pinned toolkit, so
// it is covered by the closed-world codegen policy's toolchain pin rather than
// vendored separately.
//
// The one place CUB is NOT used is the histogram. `cub::BlockHistogram` needs a
// compile-time bin count, and the bin count here is `QPO + 1 = BC/16 + 1` --
// which invariant I1 forbids turning into a literal or a template ladder. SMEM
// atomics over a runtime bin count are the standard alternative and cost one
// `atomicAdd` per candidate pair on a block-local array.
//
// ---- WHAT THE PASS COMPUTES, AND AT WHAT COST -------------------------------
//
// Per `(bh, tile)` CTA:
//
//   words     : one `int32` per `(side, level, digit)`, bit `t` = token
//               `tile*BT + t` is live. `O(BT * sum_l b_l)` coalesced reads --
//               ONE pass over the routing, which the CTA has to make anyway.
//   occupancy : `__popc` of those words. Free.
//   K_hat     : per owner, the product over spanned levels of the owner-slice
//               popcount of the UNION word. `O(owners_total)`.
//   candidacy : per owner, per side, the fixed-digit AND and the slice OR.
//   histogram : `ceil(K_hat / MMA_K_QUANTUM)`, binned. `O(owners_total)`.
//   leaf bits : §4.1's canonical per-leaf WRITE bitmap. `O(N)`, and it is the
//               irreducible term -- a per-leaf union over tokens is an OR of
//               ANDs and cannot be condensed from any per-owner statistic.
//               Warp-packed with `__ballot_sync` so 32 leaves cost ONE
//               `atomicOr`, and skipped entirely for an owner that is not
//               write-candidate in this tile (an in-kernel decision on data the
//               CTA has already fetched -- invariant I3, not a plan field).
//
// The owner-level condensation §4.2 reports rides the SAME `atomicOr`: the
// return value says which bits were newly set, so `__popc` of them is the exact
// increment for the active-owner counter and no second pass over the bitmap is
// needed. That is what keeps the whole thing to one launch.

#include <cuda_runtime_api.h>

#include <cstdint>

#include <torch/types.h>

#include <vector>

namespace rola::summaries {

//: §6's MMA k-extent, and therefore the quantum every gathered extent rounds up
//: to. A SHAPE of the instruction the consumer issues, never a tuning threshold.
//: Mirrored from `rola.routing.summaries.MMA_K_QUANTUM`; the host entry point
//: below checks the derived bin count against the caller's, so a drift is a
//: failure and not a silent one.
constexpr int kMmaKQuantum = 16;

//: Token bits carried by one support word. A `BT` above this cannot be
//: represented and is REFUSED, never split into a second code path.
constexpr int kTokenBitsPerWord = 32;

//: Ceiling on `sum_l b_l`, the digit-row extent one CTA stages. Two `int32`
//: arrays of this length live in shared memory (read words and write words), so
//: the budget is `8 * kMaxTotalDigits` bytes = 16 KiB here. A STRUCTURAL bound
//: under the standing "design for nc ~ 65K" rule: at the parity topology
//: (`branches = 256, 256`) `sum_l b_l = 512`, so this admits 4x that with the
//: union words derived on the fly rather than stored.
constexpr int kMaxTotalDigits = 2048;

//: Ceiling on `D`. Matches `FeederFlags`' own `D <= 16` (§4.4 packs two bits per
//: level into one word), so the two limits cannot drift apart.
constexpr int kMaxLevels = 16;

//: EVERY ARGUMENT THE PASS TAKES, and nothing derived that the host can derive.
//:
//: `level_amps` is the producer's FOLDED per-level allocation, `[BH, T, b_l]`
//: fp32, exactly what `RoutingViews.read_folded` / `.write_folded` hold.
//: Liveness is EXACT nonzero-ness of the amplitude (spec §1.2: entmax's exact
//: support zeros are the scheduling authority) -- never a threshold, and never a
//: magnitude test on anything downstream of it.
struct SummaryParams {
  //: PASSED BY VALUE, in the kernel's parameter space, and that is deliberate: the
  //: alternative is four small device tensors staged with `from_blob(...).to(device)`,
  //: i.e. four pageable host-to-device copies on a pass whose declared budget is ONE
  //: transfer and ZERO host synchronizations. Kernel parameters land in constant
  //: memory with the launch, so this costs no transfer at all. `kMaxLevels * (2*8 +
  //: 4*4) = 512` bytes, far inside the 4 KiB parameter limit.
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

//: THE ENTRY POINT. One launch, no host synchronization, nothing allocated.
//: Every output tensor is caller-owned and must be ZEROED beforehand -- the
//: kernel folds into them with atomics, so it accumulates rather than assigns,
//: and that is what lets one launch serve every `(bh, tile)` pair.
void launch_summaries(const SummaryParams& params, cudaStream_t stream);

//: The pybind surface (`rola_routing_summaries`). Validates shapes, builds the
//: device pointer arrays, and launches. Returns nothing: the outputs are the
//: caller's tensors, so the Python side owns the ONE D2H transfer and can decide
//: what rides it.
void rola_routing_summaries_cuda(
    const std::vector<torch::Tensor>& read_amps, const std::vector<torch::Tensor>& write_amps,
    const std::vector<int64_t>& level_widths, int64_t token_tile, int64_t state_block,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices, torch::Tensor occupancy,
    torch::Tensor histogram, torch::Tensor candidate_count, torch::Tensor written_owner_count,
    torch::Tensor written_leaf_bitmap, torch::Tensor written_owner_bitmap);

}  // namespace rola::summaries
