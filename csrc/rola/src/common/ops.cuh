#pragma once

// LAYER 2b -- THE BASELINE OPERATIONS: how each of layer 2a's (`design.cuh`)
// steps is EXECUTED, as operations with contracts.  Every entry is defined by
// WHAT THE ALGORITHM NEEDS, never by what an instruction does; the kernel
// above this header names no architecture and issues no inline PTX.  The
// per-operation contract catalogue, the deliberate omissions, and the 2a/2b
// consistency rules live in docs/internals/common/ops.md.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "facts/arch_caps.cuh"
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

//: THE SAME COPY AT A NARROWER QUANTUM.  `kStageBytes` is the widest the staging
//: issues; a run whose whole length is shorter than that is still ONE asynchronous
//: access, and a caller with a compile-time run length gets exactly one issue rather
//: than a padded 16-byte one that would read a neighbour's bytes.  The four- and
//: eight-byte forms are the cache-ALL variant because the sixteen-byte bypass has no
//: narrower member; both src and dst are naturally aligned, which is the caller's
//: contract and the reason no size is ever split here.
template <int Bytes>
__device__ __forceinline__ void stage_run(SmemAddr dst, const void* src) {
  static_assert(Bytes == 4 || Bytes == 8 || Bytes == kStageBytes,
                "an asynchronous run is 4, 8 or `kStageBytes` bytes");
  if constexpr (Bytes == kStageBytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes));
  }
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

//: THE OTHER END OF THE SAME VOTE, for a caller that consumes its lanes from
//: the high side.  `-1` for an empty vector, so "nowhere" compares BELOW every
//: lane and needs no separate case, exactly as `kBallotLanes` compares above.
__device__ __forceinline__ int last_set(uint32_t x) { return 31 - __clz((int)x); }

//: THE UNIT'S OWN RUNNING TOTAL.  `N` must be a power of two no wider than the
//: unit; lanes at or above it contribute whatever they hold and are the caller's
//: to mask.  Every lane executes every step, so the whole unit is converged at
//: each `__shfl_up_sync`.
template <int N>
__device__ __forceinline__ int lane_prefix_incl(int x) {
#pragma unroll
  for (int d = 1; d < N; d <<= 1) {
    const int up = __shfl_up_sync(0xFFFFFFFFu, x, d);
    if ((int)(threadIdx.x & (kBallotLanes - 1)) >= d) x += up;
  }
  return x;
}

//: one lane's value, read by all of them.
__device__ __forceinline__ int lane_broadcast(int x, int src) {
  return __shfl_sync(0xFFFFFFFFu, x, src);
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

//: AN 8x8 b16 TILE, TRANSPOSED WHERE IT ALREADY LIES.  The contract is about
//: MAPS and not about data: an accumulator spreads its two axes over the unit's
//: lanes one way and a `k`-side operand spreads its two the other way, so a
//: contraction over an accumulator's OWN m axis cannot read it without this.
//: MEASURED, not read off the ISA -- the probe and the refuted alternative are
//: in docs/internals/common/ops.md.
__device__ __forceinline__ uint32_t transpose_frag_b16(uint32_t x) {
  static_assert(arch::kCaps.reg_transpose_b16,
                "UNSUPPORTED ARCHITECTURE: without an in-register b16 tile "
                "transpose the adjoint's channel contraction has to route its "
                "operand through shared memory, which is a different algorithm "
                "and not a slower arm of this one");
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(x));
  return d;
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

//: THE ACCUMULATOR'S TOP HALVES, TAKEN AS A bf16 PAIR (K35).  A bf16 IS the top
//: sixteen bits of an fp32, so a pair of accumulator registers becomes an MMA
//: operand with ONE `prmt.b32` -- byte selector `0x7632` picks bytes 2,3 of the low
//: source and 6,7 of the high one -- against `pack_bf16x2`'s conversion pair.
//:
//: THE ARITHMETIC IS NOT THE SAME AND THAT IS DECLARED, NOT INCIDENTAL.  This
//: TRUNCATES where `pack_bf16x2` ROUNDS TO NEAREST EVEN, so the operand can differ
//: by one bf16 ulp (relative 2^-8 rather than 2^-9) and it is biased toward zero.
//: It is the K35 read path's declared rounding, chosen because the hi-plane hybrid
//: publishes exactly these bytes: the strip a warp writes and the fragment it would
//: have built on the fly are then BIT-IDENTICAL, so the hybrid is a placement change
//: and never an arithmetic one.  The oracle mirror carries the same truncation.
//: THE SAME TRUNCATION FOR ONE FLOAT.  The exchange places each leaf's channel on its
//: own, so a fragment element reaches shared memory as a 2-byte store rather than as
//: half of a packed pair; the value stored is bit-for-bit what `hi_bf16x2` would have
//: put in that half.
__device__ __forceinline__ uint16_t hi_bf16(float v) {
  return (uint16_t)(__float_as_uint(v) >> 16);
}

__device__ __forceinline__ uint32_t hi_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x7632;"
      : "=r"(d)
      : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return d;
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

//: the same splat over a raw bf16 BIT PATTERN.  A run of amplitudes held in a
//: register array must never be ADDRESS-TAKEN -- one `&run[i]` puts the whole run
//: in local memory, whatever the index's constness -- so a caller holding the bits
//: splats them by value rather than by reinterpreting a pointer.
__device__ __forceinline__ uint32_t splat_bf16x2_bits(unsigned short x) {
  return ((uint32_t)x << 16) | (uint32_t)x;
}

//: one raw bf16 bit pattern as a float, by value, for the same reason.
__device__ __forceinline__ float bf16_bits_to_float(unsigned short x) {
  const uint32_t w = (uint32_t)x << 16;
  return *reinterpret_cast<const float*>(&w);
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

//: THE STATE'S LO HALVES, PACKED.  The top 16 bits of an fp32 ARE its bf16
//: truncation, so a state word splits exactly into the `hi` the exchange publishes and
//: the `lo` its owner keeps.  Two lows ride one register; `join_hi_lo` puts a word
//: back together bit-for-bit, so the round trip is lossless by construction.
__device__ __forceinline__ uint32_t pack_lo(float a, float b) {
  return (__float_as_uint(a) & 0xFFFFu) | (__float_as_uint(b) << 16);
}

__device__ __forceinline__ float join_hi_lo(uint32_t hi, uint32_t lo, int k) {
  return __uint_as_float(k ? ((hi & 0xFFFF0000u) | (lo >> 16)) : ((hi << 16) | (lo & 0xFFFFu)));
}

//: A SEQUENTIAL SHARED STORE, ADDRESSED RATHER THAN INDEXED.  This is the store
//: side of the hazard the gather comment above names: `dst[k++]` through a
//: `uint16_t*` re-derives `dst + k` as a WIDENING MULTIPLY every iteration, so a
//: loop whose work is one `STS.U16` per live bit measures `IMAD:114, IADD3:69` and
//: no `STS` in its top opcodes at all -- the schedule walk's scatter, 447 warp
//: instructions per (owner, window) for about two stores a lane.  Carrying the
//: shared address in a 32-bit register and advancing it by the element size makes
//: the per-iteration cost one add.
__device__ __forceinline__ void store_shared_u16(SmemAddr a, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
}

//: A SHARED WORD ACCESS, ADDRESSED RATHER THAN INDEXED for the same reason
//: `store_shared_u16` is: a swizzled address is not an affine function of an index, so
//: indexed form would re-derive a widening multiply per access.
__device__ __forceinline__ void store_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: ONE CONTRIBUTION INTO A SHARED BITSET, FIRE AND FORGET -- `fan_in_add`'s
//: contract over the bitwise monoid.  The per-tile liveness mask has one writer per
//: (token, stream) row and one reader per stream, so what it needs is a reduction and
//: never a read-modify-write: nothing waits and no lane learns what anyone else set.
__device__ __forceinline__ void or_shared_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.or.b32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: ONE CONTRIBUTION INTO A SHARED FP32 ACCUMULATOR, FIRE AND FORGET -- `fan_in_add`'s
//: contract with the meeting held in shared memory instead of in the output plane.
//: The fan-in ring's whole point is that a token's partials MEET INSIDE THE CTA, so
//: what this needs is exactly what the global form needs: a reduction that returns
//: nothing and makes nothing wait.
__device__ __forceinline__ void red_shared_add_f32(SmemAddr a, float v) {
  asm volatile("red.shared.add.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}

//: THE SAME OVER THE COUNTING MONOID.  A ring row's arrival count is incremented once
//: per contributing warp-row and read by whoever wants to reclaim the row; the
//: incrementer never learns the total, so this is a reduction and not an exchange.
__device__ __forceinline__ void red_shared_add_u32(SmemAddr a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: THE ONE PLACE A SHARED WORD IS EXCHANGED RATHER THAN REDUCED.  Reclaiming a ring
//: row is a CLAIM -- exactly one of the warps that find the row stale may flush it --
//: so here the previous value IS the answer and the caller does wait for it.
__device__ __forceinline__ uint32_t cas_shared_u32(SmemAddr a, uint32_t cmp, uint32_t val) {
  uint32_t old;
  asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old)
               : "r"(a), "r"(cmp), "r"(val)
               : "memory");
  return old;
}

//: ONE FP32 CELL OF A SHARED ACCUMULATOR, ADDRESSED.  The ring's flush reads the row
//: it is about to hand to the output plane; the address is swizzled by the row, so an
//: indexed form would re-derive a widening multiply per access.
__device__ __forceinline__ float load_shared_f32(SmemAddr a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}

//: THE ORDER BETWEEN A UNIT'S SHARED WRITES AND THE FLAG THAT PUBLISHES THEM, and
//: nothing wider: the ring's producers and consumers are warps of ONE CTA talking
//: through shared memory, so the scope that has to be fenced is the block's.  A
//: device-scope fence here would order traffic no participant can observe.
__device__ __forceinline__ void publish_fence() { __threadfence_block(); }

//: the 16-byte quantum into shared memory, addressed.  The segment's dead rows are
//: cleared at this width so a whole slot row is a constant number of stores.
__device__ __forceinline__ void store_shared_vec16(SmemAddr a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

//: one amplitude out of a staged factor row, by ADDRESS.  Its index is a digit, so
//: the access is dynamic and an indexed form would re-derive a widening multiply.
__device__ __forceinline__ unsigned short load_shared_u16(SmemAddr a) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t load_shared_u32(SmemAddr a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
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
