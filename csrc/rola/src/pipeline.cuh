#pragma once

// THE CONSUMER'S ITEM-STREAM PIPELINE -- the intra-CTA phase protocol, and nothing
// else. A tile's phases are ordered by COUNTERS rather than by `__syncthreads()`, so
// a warp that reaches a phase edge early executes another WORK ITEM instead of
// stalling on a rendezvous. This header owns the control-word map and the three
// primitives that publish and observe it; the phases themselves are the kernel body's
// (`consumer_kernel.cuh`), and the SMEM the control block occupies is priced by
// `smem_layout.cuh`, which includes this file for exactly `ctl_words`.
//
// It sits BELOW the carve on purpose: how many counters exist is a pipeline fact, and
// the carve has to budget for it. Nothing here knows a tile shape.
//
// The three counters, the release/acquire discipline, the edge DAG and the deadlock
// argument that rests on them: docs/internals/pipeline.md

#include <cuda_runtime.h>

#include "arch_table.cuh"

namespace rola {
namespace consumer {

//: THE ITEM-STREAM CONTROL BLOCK -- named here because `consumer_smem_bytes` has to
//: account for it and the kernel body has to index it. `kCtlReadDone` gates the fold
//: (which WRITES `S`) on the readout (which READ it); `kCtlGramDone` gates phase C on
//: the Gram; `kCtlFold0 + w` is window `w`'s fold item-claim cursor, ONE PER WINDOW so
//: no cursor has to be reset mid-tile. What each counter means and the deadlock
//: argument that rests on them: docs/internals/pipeline.md#item-stream
constexpr int kCtlReadDone = 0;
constexpr int kCtlGramDone = 1;
constexpr int kCtlFold0 = 2;
constexpr int ctl_words(int BC) { return kCtlFold0 + BC / kMmaKQuantum; }

//: THE TILE'S INTRA-CTA PHASE ORDERING IS CARRIED BY COUNTERS RATHER THAN BY
//: `__syncthreads()`. The point is not to spell a barrier differently: a warp
//: that reaches a phase edge early goes and executes another WORK ITEM instead of
//: stalling on a rendezvous, which is what makes the fold a fluid pool. A WORK ITEM is
//: one `(GEMM, warp-tile)` pair with a WHOLE `k` extent, and full-`k` is a BIT-IDENTITY
//: requirement rather than a convenience. The item table is DERIVED, never stored.
//: The three counters, the release/acquire discipline and the deadlock argument that
//: rests on them: docs/internals/pipeline.md#item-stream
__device__ __forceinline__ void retire_item(int* counter, int lane) {
  //: HAZARD release-acquire -- docs/internals/pipeline.md#release-acquire
  __threadfence_block();
  if (lane == 0) atomicAdd(counter, 1);
}

__device__ __forceinline__ bool eligible(const volatile int* counter, int target) {
  //: HAZARD release-acquire -- docs/internals/pipeline.md#release-acquire
  const bool ok = (*counter >= target);
  if (ok) __threadfence_block();
  return ok;
}

//: The spin, written in terms of `eligible` so the release/acquire discipline lives in
//: exactly TWO functions and can be audited there rather than at every phase edge.
//: `__nanosleep` backs the polling warp off the issue port.
__device__ __forceinline__ void await_count(const volatile int* counter, int target) {
  while (!eligible(counter, target)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(32);
#endif
  }
}

//: THE ASYNC FETCH PRIMITIVES (sm_80+ `cp.async`), here because completion of an
//: async copy is a PHASE-ORDERING fact exactly like the counters above: the kernel
//: issues a tile's fetch at one program point and consumes it behind a later
//: barrier, and the wait/commit pair is what makes that edge real. `cg` bypasses L1
//: (the amplitude slab is read once per tile) and the 4th operand is the PTX
//: zero-fill form: the copy always writes 16 bytes of shared memory, reading only
//: `src_bytes` of global and zero-filling the rest -- branch-free ragged-tail
//: handling at a stream's end. docs/internals/pipeline.md#cp-async
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem, int src_bytes) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src_gmem),
               "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

//: Waits on THIS THREAD's committed groups only; cross-thread visibility of the
//: copied bytes still requires the barrier the caller already owns.
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;"); }

}  // namespace consumer
}  // namespace rola
