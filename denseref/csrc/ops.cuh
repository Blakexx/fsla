#pragma once

// THE BASELINE OPERATIONS.
//
// This is the structure the kernel is written against.  Every entry is defined
// by WHAT THE ALGORITHM NEEDS, never by what an instruction does: "stage this
// tile and tell me when it is visible", not "issue a cp.async".  Get that level
// wrong and the abstraction leaks on the first new architecture -- TMA cannot
// implement "issue a cp.async", but it can implement `stage_tile`.
//
// The kernel below this header names no architecture and issues no inline PTX.
// Which implementation of an operation it gets is decided by `arch::kCaps`.
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
//                             unit, and B READ FROM THAT UNIT'S REGISTERS.  The
//                             last clause is the design's pivot and is a
//                             REQUIRED capability, not a flag.
//   pack_bf16x2(lo, hi)       two fp32 -> the b32 that `mma` reads as two
//                             consecutive k-elements of a B operand.  This is
//                             the accumulator-to-operand conversion; it is the
//                             only reason the state never touches shared memory.
//   widen_bf16x2(v, lo, hi)   the EXACT inverse of `pack_bf16x2`: a packed pair
//                             back to two fp32, by bit placement.
//   splat_bf16x2(x)           one bf16 into both halves of a packed pair, so a
//                             scalar can be an operand of `mul_bf16x2`.
//   mul_bf16x2(a, b)          elementwise product of two packed bf16 pairs.
//                             The archetypal SELECTING entry: sm_90 has one
//                             instruction for it, sm_80/86/89 must go through
//                             fp32 at 1 FMUL + 1/2 pack per element.
//   store_bf16x2(p, v)        one packed bf16 pair, addressed as bf16 elements.
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
// `warp_reduce`, `compact`, `emit_partial` and a `publish`/`acquire` pair
// weaker than a full rendezvous are all absent, because the dense baseline
// needs none of them: the private d_v carve removes every cross-lane reduction,
// the workload is dense so nothing is compacted, and one CTA owning a whole
// (b,h) removes every fan-in.  A rung that needs one states its contract here
// first.  An operation invented before it has a caller cannot be defined by
// what the algorithm needs, because no algorithm needs it yet.
//
// WHAT IS DELIBERATELY NOT A CAPABILITY
// -------------------------------------
// The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
// tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
// layout knob in the capability table would make every future measurement
// unfalsifiable.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "arch_caps.cuh"

namespace denseref {
namespace ops {

//: Bytes moved by one `stage_tile`.  A FACT OF THE OPERATION, not of the
//: kernel: it is the widest single asynchronous copy the staging implementation
//: issues, and the staging loop's shape is derived from it.
constexpr int kStageBytes = 16;

//: The MMA's shape, in the units the carve is expressed in.
constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;
constexpr int kMmaUnitThreads = arch::kCaps.mma_unit_threads;

using SmemAddr = uint32_t;

__device__ __forceinline__ SmemAddr smem_addr(const void *p) {
  return static_cast<SmemAddr>(__cvta_generic_to_shared(p));
}

// ------------------------------------------------------------------ staging

template <bool Bulk = arch::kCaps.async_bulk_copy>
__device__ __forceinline__ void stage_tile(SmemAddr dst, const void *src) {
  static_assert(!Bulk, "the bulk (TMA) implementation of stage_tile is not built; "
                       "it is a SELECTING capability with no ratified arch here");
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

//: EXACT widening of a packed bf16 pair to two fp32.  bf16 is the top half of an
//: fp32, so this is bit placement, not a rounding operation, and it is the
//: inverse of `pack_bf16x2` on every architecture.  It exists as its own entry
//: because an elementwise product whose two operands have DIFFERENT reuse
//: lifetimes -- one amplitude reused across a whole run of coefficients -- must
//: be able to pay for the widening once rather than once per product.  See the
//: rung's two production arms, and which one measurement chose.
__device__ __forceinline__ void widen_bf16x2(uint32_t v, float &lo, float &hi) {
  lo = __int_as_float(v << 16);
  hi = __int_as_float(v & 0xffff0000u);
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

}  // namespace ops
}  // namespace denseref
