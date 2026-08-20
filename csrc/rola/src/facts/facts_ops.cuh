// csrc/rola/src/facts/facts_ops.cuh -- the facts kernels' device operations:
// the subset of the retired consumer baseline's operation catalogue
// (chunk/ops.cuh, deleted with the K31 batch; baseline = tag
// baseline/pre-k31) that the union-table pass and the two bitmap reductions
// use.  Bodies are verbatim -- the extraction must not move a single SASS
// body -- and the namespace stays `denseref::ops` so the kernels' `using`
// directive does not move either.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace denseref {
namespace ops {

//: A GATHERED ROW'S ADDRESS: a base formed once, displaced by a 32-bit BYTE
//: offset that is zero-extended rather than multiplied out.  The extension is
//: written here explicitly because the arithmetic is the contract: at element
//: type the compiler is entitled to re-derive `base + index` as a widening
//: multiply per access, and it does -- measured at 30 `IMAD.WIDE` in a loop that
//: the form below emits none for.  The caller owns the 32-bit bound.
//:
//: THE BASE IS A GLOBAL ADDRESS, AND SAYING SO IS PART OF THE OPERATION.  An
//: address the compiler cannot see through is an address whose storage class it
//: cannot infer, and the difference is not cosmetic: a generic pointer turns the
//: fan-in's reduction into an atomic read-modify-write and every gathered load
//: into a generic one.  The assumption restores what the arithmetic hid.
__device__ __forceinline__ const void* gather_ptr(const void* base, uint32_t off) {
  const void* p;
  asm("{ .reg .u64 t; cvt.u64.u32 t, %2; add.u64 %0, %1, t; }" : "=l"(p) : "l"(base), "r"(off));
  __builtin_assume(__isGlobal(p));
  return p;
}

//: A BASE THE COMPILER MUST KEEP RATHER THAN RE-DERIVE.  The value is unchanged;
//: what changes is that it becomes the result of an operation ptxas cannot see
//: through, so rematerialising it is not available and the base stays live.
template <class T>
__device__ __forceinline__ T* pin_address(T* p) {
  asm("" : "+l"(p));
  return p;
}

__device__ __forceinline__ void* gather_ptr(void* base, uint32_t off) {
  void* p;
  asm("{ .reg .u64 t; cvt.u64.u32 t, %2; add.u64 %0, %1, t; }" : "=l"(p) : "l"(base), "r"(off));
  __builtin_assume(__isGlobal(p));
  return p;
}

__device__ __forceinline__ void rendezvous() { __syncthreads(); }

//: THE LOCKSTEP UNIT'S WIDTH, and it is deliberately not `kMmaUnitThreads`.  The
//: two coincide on the architecture built here and do not in general: a
//: warpgroup issues ONE MMA across 128 threads and still votes in lanes of 32.
//: A caller that conflates them would silently read a quarter of a vote.
constexpr int kBallotLanes = 32;

//: ONE BIT PER LANE, IN LANE ORDER.  The unit's own agreement about a property
//: of the data its lanes hold, reached without a memory round trip and without
//: ordering anything.  Expressed as the intrinsic rather than as a named
//: instruction: what the operation promises is the bit vector, not the vote.
__device__ __forceinline__ uint32_t lane_ballot(bool pred) {
  return __ballot_sync(0xFFFFFFFFu, pred);
}

//: The magnitude bits of a packed bf16 pair.  A run of bf16 is entirely zero
//: exactly when the OR of its storage words has no magnitude bit set, since the
//: only zeros bf16 has are +0 and -0 and OR cannot clear a set bit.
constexpr uint32_t kBf16MagnitudeMask = 0x7FFF7FFFu;

//: THE OR OF A CONTIGUOUS RUN, AT THE WIDEST ACCESS THAT COVERS IT.  `n` is a
//: compile-time run length and the run is naturally aligned, so 1, 2, 4 and 8
//: amplitudes are one access each and longer runs are a whole number of 16-byte
//: accesses.  The reduction is bitwise because the CALLER's question is whether
//: anything in the run is nonzero, which needs no arithmetic and no ordering.
template <int n>
__device__ __forceinline__ uint32_t or_run(const __nv_bfloat16* p) {
  static_assert(n == 1 || n == 2 || n == 4 || (n >= 8 && n % 8 == 0),
                "a run is one access of 2, 4, 8 or 16 bytes, or a whole number "
                "of 16-byte accesses");
  if constexpr (n == 1) {
    return (uint32_t)*reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (n == 2) {
    return *reinterpret_cast<const uint32_t*>(p);
  } else if constexpr (n == 4) {
    const uint2 x = *reinterpret_cast<const uint2*>(p);
    return x.x | x.y;
  } else {
    uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < n / 8; ++i) {
      const uint4 x = *reinterpret_cast<const uint4*>(p + i * 8);
      acc |= x.x | x.y | x.z | x.w;
    }
    return acc;
  }
}

}  // namespace ops
}  // namespace denseref
