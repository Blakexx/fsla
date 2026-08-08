#pragma once

// THE ASYNCHRONOUS COPY PRIMITIVE -- `cp.async`, as the three operations this project
// uses and nothing more: a PREDICATED 16-byte global-to-shared copy, a commit, and a
// drain.
//
// The predicate is expressed as CUTLASS expresses it (`cutlass/arch/memory_sm80.h`,
// `cp_async_zfill`): the instruction's SRC-SIZE operand, not a branch. A false
// predicate copies zero bytes and the hardware ZERO-FILLS the destination, so a
// skipped chunk is not "unwritten" -- it is written with zeros, which is what lets a
// consumer read the destination unconditionally. The alternative (branching around
// the issue) leaves the destination holding the PREVIOUS window's bytes and makes
// every consumer's correctness depend on a mask it would then have to test.
//
// 16 bytes is the only width. It is the widest `cp.async` and the one the sector
// invariant is stated in (`smem_layout.cuh`: one atom of bf16 digits IS one 32-byte
// sector), and both ends must be 16-byte aligned -- a requirement the callers
// discharge structurally rather than at runtime.
//
// The completion model, the per-warp commit state and why the consumer's wait sits
// where it does: docs/internals/cp_async.md

#include <cuda_runtime.h>

#include "arch_table.cuh"

namespace rola {

static_assert(kConsumerArch >= 800,
              "cp.async is sm_80 and later; arch_table.cuh admits no earlier target");

//: ONE 16-BYTE ASYNCHRONOUS COPY, PREDICATED BY SRC-SIZE. `smem_dst` and `gmem_src`
//: must both be 16-byte aligned. `live == false` zero-fills the destination and reads
//: nothing from `gmem_src` -- but the address is still FORMED, so callers clamp rather
//: than letting an out-of-range index reach here.
//: `.ca` (cache in L1) rather than `.cg`: the amplitude sectors of neighbouring tokens
//: share cache lines under any routing that is not fully incoherent, and the streams
//: are re-read across owners.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src, bool live) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int bytes = live ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem_src),
               "r"(bytes));
}

//: ONE 4-BYTE ASYNCHRONOUS COPY, the same predicate discipline at the narrowest
//: width the instruction admits. Its stream is the ROW-SCALE SCALARS -- the write
//: gate and the fixed levels' amplitudes, one fp32 each per token per side -- which
//: are gathered by token and so have no wider contiguous run to copy.
__device__ __forceinline__ void cp_async_4(void* smem_dst, const void* gmem_src, bool live) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  const int bytes = live ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(gmem_src),
               "r"(bytes));
}

//: CLOSE THE CURRENT COMMIT GROUP. The group is per WARP: every `cp.async` this warp
//: issued since its last commit becomes one countable unit.
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

//: DRAIN EVERY OUTSTANDING GROUP OF THE CALLING WARP. Completion is visible to the
//: EXECUTING THREAD only; publishing the destination to other threads takes a barrier,
//: and the consumer places this call immediately before one that already exists.
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

//: WAIT UNTIL AT MOST `N` COMMIT GROUPS REMAIN OUTSTANDING -- the multi-stage form,
//: and the whole reason the link can run ahead: a visit consuming stage `v` waits
//: only for stage `v`'s group and leaves the `S - 1` later ones in flight.
//: `N` is an immediate, so the depth is a template parameter rather than a variable.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  static_assert(N >= 0, "cp.async.wait_group takes a non-negative immediate");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

}  // namespace rola
