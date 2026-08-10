#pragma once

// LAYER 2b -- THE BASELINE OPERATIONS.
//
// HOW EACH STEP IS PERFORMED.  Layer 2a (`design.cuh`) says what the design
// LOOKS LIKE on this architecture; this file says how each of its steps is
// EXECUTED, as operations with contracts.  Every entry is defined by WHAT THE
// ALGORITHM NEEDS, never by what an instruction does: "stage this tile and tell
// me when it is visible", not "issue a cp.async".  Get that level wrong and the
// abstraction leaks on the first new architecture -- TMA cannot implement "issue
// a cp.async", but it can implement `stage_tile`.
//
// The kernel above this header names no architecture and issues no inline PTX.
// It reads layer 2a for its design and calls this file to execute it.
//
// This file is also where THE CONSISTENCY BETWEEN 2a AND 2b is checked (see the
// section at the foot): a selection is invalid if no available operation can
// serve it, and that is what the assertion says -- "selection unsatisfiable
// given available operations", never "this architecture lacks instruction X".
// Where the capability row offers a better implementation than the one built,
// the gap is ANNOUNCED rather than taken silently.
//
// OPERATIONS AND CONTRACTS
// ------------------------
//   smem_addr(p)              the CTA-local address of a shared object; the
//                             address type every operation here consumes.
//   stage_tile(dst, src)      copy `kStageBytes` bytes global -> shared without
//                             occupying the issuing thread.  Membership of a
//                             group is implicit: every stage_tile since the last
//                             stage_commit.  Completion is SIGNALLED, not waited.
//   stage_commit()            close the current group.
//   stage_wait<n>()           all but the newest `n` groups have landed FOR THE
//                             ISSUING THREAD.  CTA-wide visibility of another
//                             thread's stage requires a following rendezvous();
//                             this is a property of the contract, not of the
//                             sm_80 implementation, and TMA/mbarrier satisfies
//                             it the same way.
//   rendezvous()              every thread of the CTA arrives; all prior shared
//                             accesses by any thread are ordered before all
//                             subsequent ones.  Also the WAR fence that lets a
//                             ring slot be overwritten.
//   load_frag / load_frag_t   four 8x8 b16 tiles from shared memory, landed in
//                             the register order `mma` consumes.  `_t`
//                             transposes each 8x8, which is what lets one shared
//                             tile serve as an operand of either handedness.
//   mma(acc, a, b)            acc += A(16x16) . B(16x8), bf16 operands, fp32
//                             accumulator, accumulator PRIVATE to the issuing
//                             unit, and BOTH operands read from that unit's
//                             registers.  That last property is what serves
//                             `STATE_RESIDENCY = REGISTERS`, which is layer 2a's
//                             selection and not a flag; which SIDE a given
//                             kernel's accumulator lands on is that kernel's
//                             clause, asserted against this operation at its own
//                             site.
//   pack_bf16x2(lo, hi)       two fp32 -> the b32 that `mma` reads as two
//                             consecutive k-elements of an operand.  This is
//                             the accumulator-to-operand conversion; it is the
//                             only reason the state never touches shared memory.
//   splat_bf16x2(x)           one bf16 into both halves of a packed pair, so a
//                             scalar can be an operand of `mul_bf16x2`.
//   mul_bf16x2(a, b)          elementwise product of two packed bf16 pairs.
//                             The archetypal SELECTING entry: sm_90 has one
//                             instruction for it, sm_80/86/89 must go through
//                             fp32 at 1 FMUL + 1/2 pack per element.
//   fan_in_add(p, x)          add ONE contribution into an accumulator that
//                             other units also contribute to.  The issuing
//                             thread does not consume a result and must not wait
//                             for one -- that is what makes this a reduction
//                             rather than an atomic read-modify-write, and it is
//                             the whole contract.  No ordering across
//                             contributors is defined and no contributor
//                             observes another's value, so a caller that needs a
//                             reproducible summation order needs a different
//                             operation.  Stated by the ladder's capacity carve,
//                             which is its first caller: the N/BC blocks of a
//                             slot axis each compute a complete partial output.
//   store_f32(p, x)           one fp32, at fp32 element addressing.  The
//                             counterpart of `store_bf16` for a buffer whose
//                             consumer is `fan_in_add` rather than an mma
//                             operand: a summand that will be added to N/BC
//                             others cannot be rounded to the operand format
//                             first without the fan-in's error exceeding the
//                             design's own.
//   store_bf16x2(p, v)        one packed bf16 pair, addressed as bf16 elements.
//   store_bf16(p, x)          one fp32 -> one bf16 element, same round-to-
//                             nearest-even as `pack_bf16x2` per element.  A
//                             TRANSPOSED output accumulator holds two
//                             token-adjacent values of ONE d_v column, and in a
//                             [token][d_v] buffer those are different rows, so
//                             the pair cannot be published as a b32.
//   load_vec16 / store_vec16  one naturally aligned 16-byte value, moved in a
//                             single access.  These are the SYNCHRONOUS pair of
//                             `stage_tile`: 16 B is `kStageBytes`, the widest
//                             access this baseline has anywhere, and a row that
//                             is COMPUTED rather than staged must be published at
//                             the same width the staged rows arrive at, or the
//                             two would need different bank analyses of the same
//                             buffer.  Stated by the ladder's routed coefficient
//                             production, which is their first caller.
//   copy_shared_to_global_16b one fully coalesced 128-bit store, shared -> global.
//                             The composition of the two above, kept as its own
//                             name because it is the whole of the epilogue's
//                             contract and reads as such at the call site.
//
// OPERATIONS THIS KERNEL DOES NOT HAVE, AND WHY NOT
// -------------------------------------------------
// `warp_reduce`, `compact` and a `publish`/`acquire` pair weaker than a full
// rendezvous are all absent, because nothing here needs them: the private d_v
// carve removes every cross-lane reduction, and the workload is dense so nothing
// is compacted.  A rung that needs one states its contract above first.  An
// operation invented before it has a caller cannot be defined by what the
// algorithm needs, because no algorithm needs it yet.
//
// WHAT IS DELIBERATELY NOT A CAPABILITY
// -------------------------------------
// The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
// tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
// layout knob in the capability table would make every future measurement
// unfalsifiable.  It is a free axis, and `design.cuh` says why it stays one.

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

__device__ __forceinline__ SmemAddr smem_addr(const void *p) {
  return static_cast<SmemAddr>(__cvta_generic_to_shared(p));
}

// ------------------------------------------------------------------ staging

__device__ __forceinline__ void stage_tile(SmemAddr dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void stage_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int KeepInFlight> __device__ __forceinline__ void stage_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(KeepInFlight));
}

__device__ __forceinline__ void rendezvous() { __syncthreads(); }

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
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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
  return *reinterpret_cast<const uint32_t *>(&p);
}

//: One bf16 into BOTH halves of a packed pair, so that a scalar can be an
//: operand of `mul_bf16x2`.  The ladder's coefficient production multiplies a
//: whole run of amplitudes by a single per-digit amplitude; the packed operand
//: is what makes that one operation rather than a scalar special case, on every
//: architecture including the one where `mul_bf16x2` is a single instruction.
__device__ __forceinline__ uint32_t splat_bf16x2(__nv_bfloat16 x) {
  const __nv_bfloat162 p = __bfloat162bfloat162(x);
  return *reinterpret_cast<const uint32_t *>(&p);
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
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162 *>(&a);
    const __nv_bfloat162 y = *reinterpret_cast<const __nv_bfloat162 *>(&b);
    return pack_bf16x2(__bfloat162float(x.x) * __bfloat162float(y.x),
                       __bfloat162float(x.y) * __bfloat162float(y.y));
  }
}

// -------------------------------------------------------------------- stores

__device__ __forceinline__ void store_bf16x2(__nv_bfloat16 *p, uint32_t v) {
  *reinterpret_cast<uint32_t *>(p) = v;
}

//: One element, rounded to nearest-even exactly as `pack_bf16x2` rounds each of
//: its two.  The single-element form exists because a transposed accumulator's
//: adjacent pair is adjacent in TOKEN, and a [token][d_v] buffer puts those in
//: different rows.
__device__ __forceinline__ void store_bf16(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

//: One fp32 element, stored where it will later be read as a summand rather than
//: as an mma operand.  There is no rounding here BY CONSTRUCTION: see
//: `fan_in_add`.
__device__ __forceinline__ void store_f32(float *p, float v) { *p = v; }

//: ONE CONTRIBUTION INTO A SHARED ACCUMULATOR, FIRE AND FORGET.  The result is
//: not returned, and that is the operation's substance rather than a convenience
//: -- an implementation that hands back the previous value has to keep the
//: issuing thread waiting for it, and the whole point of a reduction is that
//: nothing waits.  Expressed as the intrinsic rather than as a named instruction
//: so that an architecture whose best reduction is wider, or is performed by a
//: different unit, is served by its own compiler; the SASS gate for this rung
//: checks that what was emitted is in fact the non-returning form.
__device__ __forceinline__ void fan_in_add(float *p, float x) { atomicAdd(p, x); }

//: The 16-byte quantum, synchronously.  `kStageBytes` is the width of every
//: asynchronous access this baseline issues, and these move the same width for
//: data the kernel PRODUCED rather than staged, so one bank analysis covers a
//: buffer whichever way its rows got there.
__device__ __forceinline__ uint4 load_vec16(const void *p) {
  return *reinterpret_cast<const uint4 *>(p);
}

__device__ __forceinline__ void store_vec16(void *p, const uint4 &v) {
  *reinterpret_cast<uint4 *>(p) = v;
}

__device__ __forceinline__ void copy_shared_to_global_16b(void *dst, const void *src) {
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
static_assert(
    !(design::kBuiltResidencies &
      design::residency_bit(design::StateResidency::REGISTERS)) ||
        (kMmaAcceptsRegisterA && arch::kCaps.bf16_cvt_packed),
    "SELECTION UNSATISFIABLE GIVEN AVAILABLE OPERATIONS: state residency "
    "REGISTERS is selected, but no built `mma` accepts an operand from the "
    "issuing unit's registers (or no packed accumulator->operand conversion "
    "exists to feed it). Either build one or stop claiming the residency.");

static_assert(
    !(design::kBuiltResidencies &
      design::residency_bit(design::StateResidency::TMEM)) ||
        kMmaAcceptsTmemA,
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
