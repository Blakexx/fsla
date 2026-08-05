#pragma once

// §6.4's per-segment tile-range mask, EXTRACTED from the REGISTER-RESIDENT
// consumer's kernel header when that consumer was deleted (item 6b, 2026-08-01).
// That header was also named `consumer_kernel.cuh` and is readable at
// `c3c2f845`; it is a different kernel from today's file of that name.
//
// WHY IT IS ITS OWN HEADER RATHER THAN INLINED INTO THE KERNEL. The mask is the
// arithmetic that decides which tiles a SEGMENT owns, and it is read by the
// consumer kernel and, at `P > 1`, must agree bit for bit with the host-side
// segment partition the schedule publishes. Keeping it at a named, single
// definition is what the sequence-parallel gates pin; folding it into the kernel
// body would make the pin a comment.
//
// **THE OTHER HALF OF THE BRIEF'S "SPLIT" IS NOT HERE, AND THAT IS A FINDING RATHER
// THAN AN OMISSION.** `detail::state_pitch<DV>() = DV + 1` was the second symbol the
// old header shared. Read back from the tree at the point of deletion: the consumer
// kernel declared `using rola::detail::state_pitch;` and NEVER CALLED IT --
// its own row pitch is `SP = DV + 4`, derived from the bf16 fragment map and proven
// by `banks_distinct`. `state_pitch` therefore dies with the kernel that used it,
// and the `state_pitch` CONDITIONAL MANDATE (keep the divergence documented while
// two kernels disagree on the pitch) DISCHARGES: there is one kernel and one pitch.

#include <cuda_runtime.h>
#include <cstdint>

namespace rola {
namespace detail {

//: §6.4: the bits of candidate word `w` that lie in the tile range `[lo, hi)`.
//: A segment owns a CONTIGUOUS tile range, so restricting the candidate walk to it
//: is one more word-AND, of exactly the shape §4's `j+1` candidate ANDs already
//: are. At `nseg == 1` the range is the whole tile axis, every word comes back
//: all-ones, and the AND is the identity -- which is one half of why `P = 1` is
//: bit-identical to the pre-`P` kernel.
__device__ __forceinline__ uint32_t segment_word_mask(int w, int lo, int hi) {
  const int base = w * 32;
  int a = lo - base;
  int b = hi - base;
  a = (a < 0) ? 0 : ((a > 32) ? 32 : a);
  b = (b < 0) ? 0 : ((b > 32) ? 32 : b);
  if (b <= a) return 0u;  // also the only way `a == 32` could reach the shift below
  const uint32_t upto = (b == 32) ? 0xffffffffu : ((1u << b) - 1u);
  return upto & ~((1u << a) - 1u);
}

}  // namespace detail
}  // namespace rola
