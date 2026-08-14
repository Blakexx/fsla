#pragma once

// LAYER 2b -- THE BASELINE OPERATIONS: how each of layer 2a's (`design.cuh`)
// steps is EXECUTED, as operations with contracts.  Every entry is defined by
// WHAT THE ALGORITHM NEEDS, never by what an instruction does; the kernel
// above this header names no architecture and issues no inline PTX.  The
// per-operation contract catalogue, the deliberate omissions, and the 2a/2b
// consistency rules live in docs/internals/chunk/ops.md.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "arch_caps.cuh"
#include "design.cuh"

namespace denseref {
namespace ops {

//: Bytes moved by one `stage_tile`.  A FACT OF THE OPERATION, not of the
//: kernel: it is the widest single asynchronous copy the staging implementation
//: issues, and the staging loop's shape is derived from it.
constexpr int kStageBytes = 16;

//: The MMA's shape, in the units the carve is expressed in.
constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;

//: THE BUILT `mma`'s UNIT: the threads that jointly issue one MMA and jointly
//: own its accumulator.  This is a property of the IMPLEMENTATION below, not of
//: any architecture -- the capability row states what the architecture's unit
//: is, and the consistency section at the foot of this file is where the two
//: have to agree.
constexpr int kMmaUnitThreads = 32;

//: The operand sources the built `mma` accepts.  A kernel whose accumulator
//: must land on a particular side asserts against THESE, at its own site: which
//: side is a property of that kernel and not of the baseline.
constexpr bool kMmaAcceptsRegisterA = true;
constexpr bool kMmaAcceptsRegisterB = true;
constexpr bool kMmaAcceptsTmemA = false;

//: The built `stage_tile` moves `kStageBytes` per issue.  A bulk (TMA) form
//: would satisfy the same contract with one descriptor issue for a whole tile;
//: it is not built.  See the announcement in the consistency section.
constexpr bool kStagingIsBulk = false;

using SmemAddr = uint32_t;

__device__ __forceinline__ SmemAddr smem_addr(const void* p) {
  return static_cast<SmemAddr>(__cvta_generic_to_shared(p));
}

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

// ------------------------------------------------------------------ staging

__device__ __forceinline__ void stage_tile(SmemAddr dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void stage_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int KeepInFlight>
__device__ __forceinline__ void stage_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(KeepInFlight));
}

__device__ __forceinline__ void rendezvous() { __syncthreads(); }

//: `rendezvous_group(id, n)` -- a barrier over `n` threads that arrive at the
//: SAME named barrier `id`, leaving the other warps of the CTA unwaited.  The
//: contract is rendezvous()'s restricted to a warp-aligned subset; id 0 is the
//: CTA barrier's and is not offered here.  sm_80/86/90 all provide sixteen.
__device__ __forceinline__ void rendezvous_group(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n));
}

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

//: THE TWO READINGS OF A VOTE.  A bit vector answers "how many" and "where
//: first"; together they turn a lockstep unit's agreement into POSITIONS, which
//: is what a caller needs when the lanes are producing a compacted list rather
//: than a single yes or no.
__device__ __forceinline__ int bit_count(uint32_t x) { return __popc(x); }

//: `kBallotLanes` for an empty vector, so "nowhere" compares greater than every
//: lane and needs no separate case at the call site.
__device__ __forceinline__ int first_set(uint32_t x) {
  const int f = __ffs((int)x);
  return f ? f - 1 : kBallotLanes;
}

// ----------------------------------------------------------------- operands

using Frag = uint32_t[4];

__device__ __forceinline__ void load_frag(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void load_frag_t(uint32_t (&r)[4], SmemAddr a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

//: THE ATOM IS CHOSEN BY SHAPE FAMILY, NOT BY NAME.  `kMmaM/N/K` is the shape
//: the carve and every fragment index are written in; an architecture whose
//: register-source family is shaped differently (Hopper's `_RS` atoms are
//: 64 x {8..256} x 16 with a K-major register A) substitutes its own member of
//: the family, and the constraints that keeps -- M a multiple of 64, A K-major
//: -- are properties of the family rather than of a hand-picked instruction.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -------------------------------------------------------- bf16 elementwise

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  static_assert(arch::kCaps.bf16_cvt_packed,
                "UNSUPPORTED ARCHITECTURE: without a packed fp32->bf16x2 convert "
                "the accumulator cannot become an operand in place");
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: One bf16 into BOTH halves of a packed pair, so that a scalar can be an
//: operand of `mul_bf16x2`.  The ladder's coefficient production multiplies a
//: whole run of amplitudes by a single per-digit amplitude; the packed operand
//: is what makes that one operation rather than a scalar special case, on every
//: architecture including the one where `mul_bf16x2` is a single instruction.
__device__ __forceinline__ uint32_t splat_bf16x2(__nv_bfloat16 x) {
  const __nv_bfloat162 p = __bfloat162bfloat162(x);
  return *reinterpret_cast<const uint32_t*>(&p);
}

//: THE ARCHETYPAL SELECTING ENTRY.  Both implementations satisfy one contract:
//: the elementwise product of two packed bf16 pairs, rounded to nearest-even in
//: bf16.  `Simd = true` is one instruction; `Simd = false` is the fp32 route
//: that consumer Ampere is left with, at 1 FMUL + 1/2 pack per element.
template <bool Simd = arch::kCaps.bf16_simd_arith>
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  static_assert(Simd <= arch::kCaps.bf16_simd_arith,
                "mul.rn.bf16x2 requires .target sm_90 or higher; this "
                "architecture's row says so and ptxas agrees");
  if constexpr (Simd) {
    uint32_t d;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
  } else {
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162 y = *reinterpret_cast<const __nv_bfloat162*>(&b);
    return pack_bf16x2(__bfloat162float(x.x) * __bfloat162float(y.x),
                       __bfloat162float(x.y) * __bfloat162float(y.y));
  }
}

// -------------------------------------------------------------------- stores

__device__ __forceinline__ void store_bf16x2(__nv_bfloat16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}

//: One element, rounded to nearest-even exactly as `pack_bf16x2` rounds each of
//: its two.  The single-element form exists because a transposed accumulator's
//: adjacent pair is adjacent in TOKEN, and a [token][d_v] buffer puts those in
//: different rows.
__device__ __forceinline__ void store_bf16(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

//: One fp32 element, stored where it will later be read as a summand rather than
//: as an mma operand.  There is no rounding here BY CONSTRUCTION: see
//: `fan_in_add`.
__device__ __forceinline__ void store_f32(float* p, float v) { *p = v; }

//: ONE CONTRIBUTION INTO A SHARED ACCUMULATOR, FIRE AND FORGET.  The result is
//: not returned, and that is the operation's substance rather than a convenience
//: -- an implementation that hands back the previous value has to keep the
//: issuing thread waiting for it, and the whole point of a reduction is that
//: nothing waits.  Expressed as the intrinsic rather than as a named instruction
//: so that an architecture whose best reduction is wider, or is performed by a
//: different unit, is served by its own compiler; the SASS gate for this kernel
//: checks that what was emitted is in fact the non-returning form.
__device__ __forceinline__ void fan_in_add(float* p, float x) { atomicAdd(p, x); }

//: The 16-byte quantum, synchronously.  `kStageBytes` is the width of every
//: asynchronous access this baseline issues, and these move the same width for
//: data the kernel PRODUCED rather than staged, so one bank analysis covers a
//: buffer whichever way its rows got there.
__device__ __forceinline__ uint4 load_vec16(const void* p) {
  return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void store_vec16(void* p, const uint4& v) {
  *reinterpret_cast<uint4*>(p) = v;
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

__device__ __forceinline__ void copy_shared_to_global_16b(void* dst, const void* src) {
  store_vec16(dst, load_vec16(src));
}

// ------------------------------------------------- 2a / 2b CONSISTENCY
// A SELECTION IS INVALID IF NO AVAILABLE OPERATION CAN SERVE IT.  That is what
// each clause below says, and it is deliberately not "this architecture lacks
// instruction X": the selection is the thing being checked, the operation set
// is what it is checked against, and an architecture number appears in neither.

//: `design.cuh` claims REGISTERS is a residency this baseline builds.  It is
//: built by an `mma` that reads an operand out of the issuing unit's own
//: registers, and by `pack_bf16x2`, which is the accumulator-to-operand
//: conversion that keeps the state out of shared memory.  If either went away
//: the claim would be false, and this is where that would be caught.
static_assert(!(design::kBuiltResidencies
                & design::residency_bit(design::StateResidency::REGISTERS))
                  || (kMmaAcceptsRegisterA && arch::kCaps.bf16_cvt_packed),
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency "
              "REGISTERS is selected, but no built `mma` accepts an operand from the "
              "issuing unit's registers (or no packed accumulator->operand conversion "
              "exists to feed it). Either build one or stop claiming the residency.");

static_assert(!(design::kBuiltResidencies & design::residency_bit(design::StateResidency::TMEM))
                  || kMmaAcceptsTmemA,
              "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency TMEM "
              "is claimed as built, but no built `mma` sources an operand from tensor "
              "memory.");

//: THE CARVE'S UNIT.  Not a capability and not an architecture clause: `mma`,
//: `load_frag` and the d_v carve above are written for a 32-thread unit owning
//: a 16-row slice of the state, and a 128-thread warpgroup unit owns 64 and
//: issues a differently shaped atom.  So this is a consistency clause between
//: the design's carve and the operations that execute it, and its failure means
//: NOT BUILT rather than UNSUPPORTED.
static_assert(arch::kCaps.mma_unit_threads == kMmaUnitThreads,
              "NOT BUILT FOR THIS ARCHITECTURE: no built implementation of "
              "`mma` and `load_frag` serves this architecture's MMA unit. The "
              "operations here are written for a 32-thread unit owning a 16-row "
              "slice of the state; a 128-thread warpgroup unit owns 64. Every "
              "structural clause holds -- this is an unwritten implementation, "
              "not a refused structure.");

//: THE ANNOUNCEMENT SITES.  Each is a case where the capability row records a
//: better implementation of a baseline operation than the one built here.  None
//: of them is an error: the built path is correct everywhere it compiles.  What
//: is forbidden is taking the slower path SILENTLY.
//:
//: `stage_tile` on a part with bulk asynchronous copy: TMA satisfies the same
//: contract with one descriptor issue instead of one per 16 bytes.
inline constexpr design::Announce<arch::kCaps.async_bulk_copy && !kStagingIsBulk>
    kStagingAnnouncement{};

}  // namespace ops
}  // namespace denseref
