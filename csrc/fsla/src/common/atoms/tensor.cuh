// csrc/fsla/src/common/atoms/tensor.cuh -- the hopper tensor step (`wgmma`) and a warpgroup's registers, one PTX instruction each.
// see docs/internals/common/atoms/tensor.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/atoms/address.cuh"

namespace denseref {
namespace ops {

//: A STEP'S SHARED-MEMORY OPERAND, unswizzled -- see docs/internals/common/atoms/tensor.md#operand
__device__ __forceinline__ uint64_t step_operand_lines(uint32_t line, uint32_t lbo, uint32_t sbo) {
  //: FIXED by the instruction's descriptor: 14 bits a field, the two offsets at bits 16 and 32
  constexpr uint32_t kField = 0x3fffu;
  return uint64_t(line & kField) | (uint64_t(lbo & kField) << 16) | (uint64_t(sbo & kField) << 32);
}

__device__ __forceinline__ uint64_t step_operand(SmemAddr a, uint32_t lbo, uint32_t sbo) {
  return step_operand_lines(a >> 4, lbo, sbo);
}

//: A STEP'S OPERAND OF 16 ROWS, each one K entry's 64 values in 128 bytes, read through the 128-byte swizzle; `a` is on a 1,024-byte boundary -- see docs/internals/common/atoms/tensor.md#rows
__device__ __forceinline__ uint64_t step_operand_rows(SmemAddr a) {
  //: FIXED by the instruction's descriptor: 8 rows are 64 lines, and layout type 1 at bit 62 is the 128-byte swizzle
  constexpr uint32_t kHalf = 64;
  constexpr uint64_t kSwizzle128 = uint64_t{1} << 62;
  return step_operand(a, kHalf, kHalf) | kSwizzle128;
}

//: A K-MAJOR STEP OPERAND THROUGH THE 128-BYTE SWIZZLE, a row 128 bytes -- see docs/internals/common/atoms/tensor.md#k-rows
__device__ __forceinline__ uint64_t step_operand_k128(SmemAddr a) {
  constexpr uint32_t kAtomLines = 64;
  constexpr uint64_t kSwizzle128 = uint64_t{1} << 62;
  return step_operand(a, 1, kAtomLines) | kSwizzle128;
}

//: WHERE A ROW'S PIECE LIES under that swizzle: piece `piece` of the row at `row` -- see docs/internals/common/atoms/tensor.md#rows
__device__ __forceinline__ SmemAddr step_row_piece(SmemAddr row, uint32_t piece) {
  //: FIXED by the swizzle: a row is 128 bytes of 8 pieces, and a piece's place is turned by the row's own three bits
  return row + ((piece ^ ((row >> 7) & 7u)) << 4);
}

//: THE EDGES OF A GROUP OF STEPS -- see docs/internals/common/atoms/tensor.md#edges
__device__ __forceinline__ void steps_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void steps_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

//: wait until at most `InFlight` committed groups are in flight
template <int InFlight>
__device__ __forceinline__ void steps_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(InFlight) : "memory");
}

//: A REGISTER A STEP WROTE, held behind the wait the compiler cannot see it depends on
__device__ __forceinline__ void step_hold(float& r) { asm volatile("" : "+f"(r) : : "memory"); }

#define FSLA_STEP_D8(d) \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])

//: 64 BY 16 BY 16 with A M|N-major (rows, or bricks) and B K-major -- see docs/internals/common/atoms/tensor.md#steps
__device__ __forceinline__ void step_n16_rows(float (&d)[8], uint64_t a, uint64_t b, bool add) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
      "%8, %9, p, 1, 1, 1, 0; }"
      : FSLA_STEP_D8(d)
      : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

//: 64 BY 16 BY 16, bf16 into fp32, A K-major and B K-major: `d` += A [64 x 16] B [16 x 16], or `d` = A B -- see docs/internals/common/atoms/tensor.md#steps
__device__ __forceinline__ void step_n16(float (&d)[8], uint64_t a, uint64_t b, bool add) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, "
      "%8, %9, p, 1, 1, 0, 0; }"
      : FSLA_STEP_D8(d)
      : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

//: 64 BY 64 BY 16, both operands M|N-major, the accumulator four 16-column sets end to end -- see docs/internals/common/atoms/tensor.md#steps
__device__ __forceinline__ void step_n64(float (&d0)[8], float (&d1)[8], float (&d2)[8],
                                         float (&d3)[8], uint64_t a, uint64_t b, bool add) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
      "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 1, 1; }"
      : FSLA_STEP_D8(d0), FSLA_STEP_D8(d1), FSLA_STEP_D8(d2), FSLA_STEP_D8(d3)
      : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

#undef FSLA_STEP_D8

//: A WARPGROUP'S REGISTERS A THREAD, lowered or raised -- see docs/internals/common/atoms/tensor.md#registers
template <int Count>
__device__ __forceinline__ void registers_down() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(Count));
}

template <int Count>
__device__ __forceinline__ void registers_up() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(Count));
}

}  // namespace ops
}  // namespace denseref
