// measure/lab/lab.cuh -- THE LAB'S KIT: what every experiment shares -- a warp's cycles over a region, a sink the
// compiler cannot drop, the atoms as PTX, and the host runner that launches a variant, repeats it, prints one JSON line
// a result and declares the SASS each kernel must contain (checked by `tools/lab.py` before any number is kept).
// see docs/internals/tools/lab.md
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/atoms/barrier.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/tensor.cuh"
#include "common/atoms/warp.cuh"
#include "common/hardware.cuh"
#include "common/structure_switch.cuh"
#include "warpline/host.hpp"
#include "warpline/stamp.cuh"

namespace lab {

//: THE BUILD an experiment compiles at: its one architecture's own, the same in both passes (`tools/lab.py` passes
//: `FSLA_LAB_CC`; docs/internals/tools/lab.md#build)
constexpr denseref::arch::Build kLabBuild{FSLA_LAB_CC};

// ------------------------------------------------------------------------------------------------- device

//: THE PRODUCTION CONTROL ATOMS, included not copied (`tools/lab.py` puts `csrc/fsla/src` on the include path): a branch
//: on a per-thread value is serialized, so a switch over a CTA-uniform value is `uniform_switch<N>(v, body)`
//: (`common/structure_switch.cuh`), a warp's index as a value ptxas knows uniform is `uniform_warp<Warps>()`, and a
//: value made opaque at its use is `opaque(v)` (`common/atoms/warp.cuh`, docs/internals/common/atoms/warp.md)
using denseref::ops::opaque;
using denseref::ops::uniform_warp;
using fsla::uniform_switch;

//: THE WARP'S INDEX as the thread index shifted: PER-THREAD to ptxas's analysis, which `uniform_warp` is not -- a loop
//: or switch bounded by this one is divergent. Kept beside `uniform_warp` because it differs in meaning; a kernel that
//: branches on the warp calls `uniform_warp`
__device__ __forceinline__ int warp_id() { return static_cast<int>(threadIdx.x) >> 5; }

__device__ __forceinline__ int lane_id() { return static_cast<int>(threadIdx.x) & 31; }

//: A WARP'S CYCLES over a region, lane 0's `clock64`, stored at `(block, warp)`.
struct WarpClock {
  long long t0 = 0;

  __device__ __forceinline__ void start() { t0 = clock64(); }

  __device__ __forceinline__ void stop(long long* out) {
    const long long dt = clock64() - t0;
    if (lane_id() == 0) out[blockIdx.x * (blockDim.x >> 5) + warp_id()] = dt;
  }
};

//: A WARP'S SPAN, its start and end `clock64` (the SM's counter), stored at `2 (block, warp)`: a block's span -- its
//: last warp's end less its first warp's start -- is the throughput when warps finish apart, where the mean warp's
//: cycles undercount.
struct WarpSpan {
  long long t0 = 0;

  __device__ __forceinline__ void start() { t0 = clock64(); }

  __device__ __forceinline__ void stop(long long* out) {
    const long long t1 = clock64();
    const int at = 2 * (blockIdx.x * (blockDim.x >> 5) + warp_id());
    if (lane_id() == 0) {
      out[at] = t0;
      out[at + 1] = t1;
    }
  }
};

//: THE SAME SPAN WITH NOTHING HELD: the start stored as it is read, for a role whose registers hold no count
struct WarpMark {
  __device__ __forceinline__ static void start(long long* out) {
    const long long t = clock64();
    if (lane_id() == 0) out[2 * (blockIdx.x * (blockDim.x >> 5) + warp_id())] = t;
  }

  __device__ __forceinline__ static void stop(long long* out) {
    const long long t = clock64();
    if (lane_id() == 0) out[2 * (blockIdx.x * (blockDim.x >> 5) + warp_id()) + 1] = t;
  }
};

//: a mark's entries
constexpr uint32_t kMarkEntries = 2;
//: A WARP'S TIMELINE: stamps -- the SM's clock's low word beside an event's, its kind and its site -- written by the
//: warp's first lane, one predicated store a stamp to the warp's row in global memory, the row's head its count and
//: the CTA's shared base. A stamp is a clock read, its predicate and one store, priced a stamp by `stamp_cost` as the
//: lab's trace. `ON` false: nothing (docs/internals/tools/lab.md#trace)
template <bool ON>
struct Trace {
  //: a warp's stamps, and its row's words: a line of head, then the stamps
  static constexpr int kCap = 8192, kRow = 32 + kCap;
  //: warpline's cursor over the warp's row past the lab's head (the stamp is warpline's, `warpline/stamp.cuh`; the
  //: switch `ON` and the row's head are the lab's)
  warpline::Cursor<kCap> row = {nullptr, 0, 0};

  //: the warp's row; `after`: the entries a tracer of this warp bound earlier has already written
  __device__ __forceinline__ void bind(unsigned long long* rows, uint32_t after = 0) {
    if constexpr (ON) {
      row.to = reinterpret_cast<warpline::Entry*>(
          rows + static_cast<long long>(blockIdx.x * (blockDim.x >> 5) + warp_id()) * kRow + 32);
      row.n = after;
      row.mine = lane_id() == 0;
    }
  }

  //: an event: `kind` one of warpline's kinds (`warpline/entry.hpp`), `site` the experiment's -- a barrier's shared address over 8 (below 0x8000), or
  //: its own
  __device__ __forceinline__ void stamp(uint32_t kind, uint32_t site) {
    if constexpr (ON) row.stamp(warpline::id(kind, site));
  }
  //: A MARK: a stamp on BOTH clocks -- the stamp, then an entry of kind `warpline::kTime` at the same site whose clock word is
  //: the nanosecond timer's low word: one moment tied between the cycle counter and time
  __device__ __forceinline__ void mark(uint32_t kind, uint32_t site) {
    if constexpr (ON) row.mark(warpline::id(kind, site));
  }
  //: an event at the barrier at shared address `bar`
  __device__ __forceinline__ void at(uint32_t kind, uint32_t bar) { stamp(kind, site(bar)); }
  //: a barrier's site: its shared address over 8
  __device__ __forceinline__ static uint32_t site(uint32_t bar) { return (bar >> 3) & 0x7fffu; }

  //: the count, and the shared base the barriers' sites are read against
  __device__ __forceinline__ void close(uint32_t base) {
    if constexpr (ON) {
      if (lane_id() == 0) {
        auto* const head = reinterpret_cast<unsigned long long*>(row.to) - 32;
        head[0] = row.n;
        head[1] = base;
      }
    }
  }
};

//: THE SINK: a fold of everything a thread computed, stored unconditionally. A result tested against a value the
//: compiler can fold lets ptxas delete the work behind it -- the failure the expectations exist to catch.
__device__ __forceinline__ void sink(int* out, uint32_t v) {
  out[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<int>(v);
}

__device__ __forceinline__ uint32_t bits(float x) { return __float_as_uint(x); }

__device__ __forceinline__ uint32_t bits(uint32_t x) { return x; }

__device__ __forceinline__ uint32_t bits(int x) { return static_cast<uint32_t>(x); }

//: THE OPERAND PIECES the kernels use: `ldmatrix.x4` and its `.trans`, a packed f16 product, two fp32 to an f16 pair.
__device__ __forceinline__ void ldsm(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

//: four 8 x 8 matrices of 16-bit pairs STORED TRANSPOSED: register k is matrix k's fragment (a thread's row `lane / 4`,
//: its columns `2 (lane % 4)` and the next), and lane `8 k + x` names the address of the stored row x, column x of the
//: fragment as 16 bytes (sm_90 up)
__device__ __forceinline__ void stsm_t(uint32_t a, const uint32_t (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};"
               ::"r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

__device__ __forceinline__ void ldsm_t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

//: `v` HELD in a register across the loops that use it: an empty asm hides its derivation, so ptxas cannot
//: rematerialize it (a lane's constants otherwise come back from `S2R` and their arithmetic at every use). Kept beside
//: the production `opaque`, which differs in meaning: `held` emits no instruction (an empty `asm volatile`, value in and
//: out) and takes 32 or 64 bits unsigned, `opaque` is a `mov.b32` of an `int` formed at its use
__device__ __forceinline__ uint32_t held(uint32_t v) {
  asm volatile("" : "+r"(v));
  return v;
}

//: the same for a 64-bit value (a wgmma operand descriptor)
__device__ __forceinline__ uint64_t held(uint64_t v) {
  asm volatile("" : "+l"(v));
  return v;
}

//: THE ASYNCHRONOUS COPIES (`cp.async`, sm_80+) as `ops.cuh` spells them: 16 bytes bypassing L1, or 4 through it; a
//: group committed and waited, or an mbarrier arrival when all of the thread's prior copies land.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

//: AN OPERAND'S K-STEP STAGED BY ONE WARP into wgmma's unswizzled K-major bricks (`wg_desc(at, 8, 16)` reads a k16
//: slice of it): `R` rows of 64 bf16 from `src`, `pitch` elements a row; an instruction 16 rows of 32 bytes (two lanes a
//: row's sector), which land over all 32 banks (4 wavefronts an instruction); a k16 slice R * 32 bytes, a brick of 8
//: rows 256, the brick's K 8-15 128 on
template <int R, class T>
__device__ __forceinline__ void cp_async_bricks(uint32_t dst, const T* src, long long pitch, int lane) {
  static_assert(sizeof(T) == 2, "the bricks hold 16-bit values");
  const int row = lane >> 1, half = lane & 1;
  const uint32_t to = dst + (row >> 3) * 256 + half * 128 + (row & 7) * 16;
  const T* from = src + row * pitch + half * 8;
#pragma unroll
  for (int rb = 0; rb < R / 16; ++rb)
#pragma unroll
    for (int kp = 0; kp < 4; ++kp) cp_async16(to + kp * (R * 32) + rb * 512, from + rb * 16 * pitch + kp * 16);
}

//: THE SHARED-MEMORY BARRIER (`mbarrier`, sm_80+), as `ops.cuh` spells it: `count` arrivals complete a phase, a phase's
//: parity flips at each completion, a wait polls until the phase of `parity` has completed.
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t count) { ::denseref::ops::mbar_init(a, count); }

__device__ __forceinline__ void mbar_arrive_when_landed(uint32_t a) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t a) { ::denseref::ops::mbar_arrive(a); }

__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) { ::denseref::ops::mbar_wait(a, parity); }

//: the same wait by the instruction that may SUSPEND the thread until the phase completes or a time limit of the
//: system's passes (`try_wait`); `mbar_wait`'s `test_wait` never suspends: its loop issues every trip it waits
__device__ __forceinline__ void mbar_wait_suspended(uint32_t a, uint32_t parity) { ::denseref::ops::mbar_wait_suspended(a, parity); }

//: A NAMED BARRIER (ids 1-15; 0 is `__syncthreads`'s): `count` threads, a multiple of 32, complete it; `bar_sync`
//: arrives and waits, `bar_arrive` arrives and goes on
__device__ __forceinline__ void bar_sync(int id, int count) { ::denseref::ops::rendezvous_group(id, count); }

__device__ __forceinline__ void bar_arrive(int id, int count) { ::denseref::ops::rendezvous_arrive(id, count); }

//: the same, asked once: has the phase of parity `parity` completed?
__device__ __forceinline__ bool mbar_test(uint32_t a, uint32_t parity) { return ::denseref::ops::mbar_test(a, parity); }

// ------------------------------------------------------------------------------------------- hopper atoms
// each one PTX instruction of sm_90a; an experiment that calls one compiles for sm_90a only
// see docs/internals/tools/lab.md#wgmma

//: WGMMA'S SHARED-MEMORY OPERAND, unswizzled: the tile at `a`, rows (A) or columns (B) in bricks of 8. K-MAJOR (an
//: atom's plain form): a line of 16 bytes is one row's or column's K 0-7, a brick 8 such lines, its K 8-15 `lbo`
//: lines on, the next brick `sbo` lines on. M|N-MAJOR (an `_mn` atom): a line is one K's 8 rows or columns, a
//: brick's K 0-7 8 lines, its K 8-15 `lbo` lines on, the next brick `sbo` lines on
__device__ __forceinline__ uint64_t wg_desc_lines(uint32_t line, uint32_t lbo, uint32_t sbo) {
  return ::denseref::ops::step_operand_lines(line, lbo, sbo);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t a, uint32_t lbo, uint32_t sbo) { return wg_desc_lines(a >> 4, lbo, sbo); }

__device__ __forceinline__ void wg_fence() { ::denseref::ops::steps_fence(); }

__device__ __forceinline__ void wg_commit() { ::denseref::ops::steps_commit(); }

//: wait until at most `N` committed groups are in flight
template <int N>
__device__ __forceinline__ void wg_wait() {
  ::denseref::ops::steps_wait<N>();
}

//: A REGISTER AN ASYNCHRONOUS MMA WROTE, held behind the wait the compiler cannot see it depends on
__device__ __forceinline__ void wg_hold(float& r) { ::denseref::ops::step_hold(r); }
__device__ __forceinline__ void wg_hold(int& r) { asm volatile("" : "+r"(r) : : "memory"); }

//: m64n16k16, bf16 x bf16 into fp32, both operands shared memory: `d` += A [64 x 16] B [16 x 16]
__device__ __forceinline__ void wg_bf16_n16_ss(float (&d)[8], uint64_t a, uint64_t b, bool add = true) {
  ::denseref::ops::step_n16(d, a, b, add);
}

//: m64n16k16 with A M|N-major (a line one K's 8 rows) and B K-major: `d` += A [64 x 16] B [16 x 16], or with
//: `add` false `d` = A B -- an accumulator's first step, which needs no register cleared
__device__ __forceinline__ void wg_bf16_n16_ss_mna(float (&d)[8], uint64_t a, uint64_t b, bool add = true) {
  ::denseref::ops::step_n16_rows(d, a, b, add);
}

//: m64n16k16 with BOTH operands M|N-major: `d` += A [64 x 16] B [16 x 16]; a line is one K's 8 rows or columns
__device__ __forceinline__ void wg_bf16_n16_ss_mn(float (&d)[8], uint64_t a, uint64_t b, bool add = true) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 1, 1; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

//: m64n8k16 with both operands M|N-major: `d` += A [64 x 16] B [16 x 8]
__device__ __forceinline__ void wg_bf16_n8_ss_mn(float (&d)[4], uint64_t a, uint64_t b, bool add = true) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 1, 1; }"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

#define LAB_WG_D32(d)                                                                                              \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),      \
      "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),       \
      "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),      \
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
//: an integer accumulator of 32 as a step's operands
#define LAB_WG_D32_INT(d)                                                                                          \
  "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),      \
      "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),       \
      "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),      \
      "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
#define LAB_WG_D32_LIST                                                                                            \
  "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, " \
  "%24, %25, %26, %27, %28, %29, %30, %31}"

//: m64n64k16, bf16 x bf16 into fp32, both operands shared memory: `d` += A [64 x 16] B [16 x 64]
__device__ __forceinline__ void wg_bf16_n64_ss(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " LAB_WG_D32_LIST ", %32, %33, p, 1, 1, 0, 0; }"
               : LAB_WG_D32(d)
               : "l"(a), "l"(b), "r"(1));
}

//: m64n64k16, both operands M|N-major, its accumulator FOUR 16-column sets end to end (set `s` is columns `16 s`
//: on: the registers a 16-column step of that set writes): `d` += A [64 x 16] B [16 x 64], or written whole
__device__ __forceinline__ void wg_bf16_n64_ss_mn4(float (&d0)[8], float (&d1)[8], float (&d2)[8], float (&d3)[8],
                                                   uint64_t a, uint64_t b, bool add = true) {
  ::denseref::ops::step_n64(d0, d1, d2, d3, a, b, add);
}

//: m64n64k16 with BOTH operands M|N-major (the transposed form; `wg_desc` states the tile): `d` += A [64 x 16]
//: B [16 x 64]
__device__ __forceinline__ void wg_bf16_n64_ss_mn(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " LAB_WG_D32_LIST ", %32, %33, p, 1, 1, 1, 1; }"
               : LAB_WG_D32(d)
               : "l"(a), "l"(b), "r"(1));
}

//: m64n256k16, the widest step, both operands M|N-major: `d` += A [64 x 16] B [16 x 256]
__device__ __forceinline__ void wg_bf16_n256_ss_mn(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 1, 1; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
               : "l"(a), "l"(b), "r"(1));
}

//: the same with A from the warpgroup's registers: a thread's 8 bf16 of A as 4 words, in the layout of an m64n16
//: accumulator's 8 fp32 (word `j` its values `2j`, `2j + 1`); `d` added to, or written whole
__device__ __forceinline__ void wg_bf16_n64_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t b, bool add = true) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " LAB_WG_D32_LIST
               ", {%32, %33, %34, %35}, %36, p, 1, 1, 0; }"
               : LAB_WG_D32(d)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(static_cast<int>(add)));
}

//: m64n64k16 with A from the warpgroup's registers and B M|N-major in shared memory (a line one K's 8 columns):
//: `d` += A [64 x 16] B [16 x 64], or written whole
__device__ __forceinline__ void wg_bf16_n64_rs_mn(float (&d)[32], const uint32_t (&a)[4], uint64_t b, bool add = true) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " LAB_WG_D32_LIST
               ", {%32, %33, %34, %35}, %36, p, 1, 1, 1; }"
               : LAB_WG_D32(d)
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(static_cast<int>(add)));
}

//: the same 72 columns wide: `d` += A [64 x 16] B [16 x 72], B a K-major tile of 9 bricks
__device__ __forceinline__ void wg_bf16_n72_rs(float (&d)[36], const uint32_t (&a)[4], uint64_t b, bool add = true) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}"
               ", {%36, %37, %38, %39}, %40, p, 1, 1, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(static_cast<int>(add)));
}

//: m64n64k32, u8 x u8 into s32 (the toolkit's `IGMMA`), both operands shared memory and K-major -- a line a row's
//: or a column's 16 digits: `d` += A [64 x 32] B [32 x 64], or written whole
__device__ __forceinline__ void wg_u8_n64_ss(int (&d)[32], uint64_t a, uint64_t b, bool add = true) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.u8 " LAB_WG_D32_LIST ", %32, %33, p; }"
               : LAB_WG_D32_INT(d)
               : "l"(a), "l"(b), "r"(static_cast<int>(add)));
}

//: the same with B M|N-major: a line one K's 8 columns
__device__ __forceinline__ void wg_bf16_n72_rs_mn(float (&d)[36], const uint32_t (&a)[4], uint64_t b, bool add = true) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}"
               ", {%36, %37, %38, %39}, %40, p, 1, 1, 1; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(static_cast<int>(add)));
}

//: two fp32 narrowed to a bf16 pair by TRUNCATION, each value's high 16 bits in one byte permute, and ROUNDED to
//: nearest in one conversion: the instructions of `common/atoms/math.cuh`'s `hi_bf16x2` and `pack_bf16x2`, kept beside
//: them because they differ in meaning -- the atoms are plain code (`asm` and `__floats2bfloat162_rn`) that nvcc may
//: merge, hoist or schedule among their neighbours, and these are `asm volatile`, so a probe's narrowing stays where its
//: row wrote it -- see docs/internals/tools/lab.md
__device__ __forceinline__ uint32_t bf16x2_trunc(float lo, float hi) {
  uint32_t r;
  asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(r) : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi)));
  return r;
}

__device__ __forceinline__ uint32_t bf16x2_round(float lo, float hi) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

//: A WARPGROUP'S REGISTERS A THREAD, lowered or raised (sm_90): 24 to 256 in eights; a raise waits for registers
//: another warpgroup of the CTA released. Honoured only in a kernel whose `__launch_bounds__` give the count at entry,
//: each role's whole body in its own branch (docs/lab/reg_trade.md)
template <int N>
__device__ __forceinline__ void regs_down() {
  ::denseref::ops::registers_down<N>();
}

template <int N>
__device__ __forceinline__ void regs_up() {
  ::denseref::ops::registers_up<N>();
}

//: A BULK COPY IN (sm_90): `bytes` of global memory at `src` into shared memory at `dst`, asynchronous; the bytes
//: complete the barrier `bar`'s phase once its one arrival has stated them (`mbar_expect`). Multiples of 16 bytes
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :
               : "r"(dst), "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

//: A TENSOR COPY IN (sm_90): the box at (`c0`, `c1`) of the 2-d tensor `map` describes, into shared memory at
//: `dst`, its bytes completing the barrier `bar`; `map` a kernel's `__grid_constant__` argument
__device__ __forceinline__ void tensor_load_2d(uint32_t dst, const void* map, int c0, int c1, uint32_t bar) {
  ::denseref::ops::stage_box_2d(dst, map, c0, c1, bar);
}

//: A 3-D TENSOR COPY IN (sm_90): the box at (`c0`, `c1`, `c2`) of the 3-d tensor `map` describes, into shared memory at
//: `dst`, its bytes completing the barrier `bar`
__device__ __forceinline__ void tensor_load_3d(uint32_t dst, const void* map, int c0, int c1, int c2, uint32_t bar) {
  ::denseref::ops::stage_box_3d(dst, map, c0, c1, c2, bar);
}

//: the same MULTICAST: one copy landed at `dst` in every CTA of the cluster that `mask` names (a bit a rank), its
//: bytes completing each one's own barrier at `bar`
__device__ __forceinline__ void tensor_cast_2d(uint32_t dst, const void* map, int c0, int c1, uint32_t bar,
                                               uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
               "[%0], [%1, {%2, %3}], [%4], %5;"
               :
               : "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask)
               : "memory");
}

//: an arrival that also states the bytes a bulk copy will complete the phase with
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  ::denseref::ops::mbar_expect_bytes(bar, bytes);
}

//: A BULK REDUCTION OUT (sm_90): `bytes` of shared memory at `src` ADDED into global `dst`, fp32 by fp32,
//: asynchronous, in this thread's bulk group; both addresses and the size multiples of 16 bytes
__device__ __forceinline__ void bulk_add_f32(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
               :
               : "l"(dst), "r"(src), "r"(bytes)
               : "memory");
}

//: A BULK COPY OUT (sm_90): `bytes` of shared memory at `src` to global `dst`, asynchronous, in this thread's bulk
//: group; both addresses and the size multiples of 16 bytes
__device__ __forceinline__ void bulk_store(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :
               : "l"(dst), "r"(src), "r"(bytes)
               : "memory");
}

//: THE WRITERS' EDGE to an asynchronous reader of shared memory -- a bulk copy out, a wgmma in another warp: a
//: thread's ordinary stores made visible to it
__device__ __forceinline__ void bulk_fence() { ::denseref::ops::async_fence(); }

__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

//: wait until at most `N` of this thread's committed bulk groups are in flight
template <int N>
__device__ __forceinline__ void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

//: one pair of f16 added into global memory by its thread, each half its own sum (no flush to zero)
__device__ __forceinline__ void red_add_f16x2(uint32_t* dst, uint32_t v) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" : : "l"(dst), "r"(v) : "memory");
}
//: the same as ONE VECTOR ADD of four f16 pairs: 16 bytes a lane at `dst`, each element its own atomic add (sm_90 up)
__device__ __forceinline__ void red_add_f16x2_v4(uint32_t* dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("red.global.v4.f16x2.add.noftz [%0], {%1, %2, %3, %4};" : : "l"(dst), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

//: A LINE ASKED INTO THE CACHE: `prefetch.global.L2` of the line at `p` (the toolkit's `CCTL.E.PF2`), which its thread
//: does not wait for (docs/internals/tools/lab.md)
__device__ __forceinline__ void touch(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" : : "l"(p) : "memory");
}

//: one fp32 added into global memory by its thread
__device__ __forceinline__ void red_add_f32(float* dst, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" : : "l"(dst), "f"(v) : "memory");
}

//: one 32-bit and one 64-bit integer added into global memory by its thread
__device__ __forceinline__ void red_add_u32(uint32_t* dst, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" : : "l"(dst), "r"(v) : "memory");
}
__device__ __forceinline__ void red_add_u64(uint64_t* dst, uint64_t v) {
  asm volatile("red.global.add.u64 [%0], %1;" : : "l"(dst), "l"(v) : "memory");
}

// ------------------------------------------------------------------------------------------ hopper clusters
// a cluster's CTAs reach each other's shared memory; a kernel of them carries `__cluster_dims__`
// see docs/internals/tools/lab.md#cluster

//: this CTA's rank in its cluster
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

//: the shared address `a` of this CTA, as CTA `rank` of the cluster holds it
__device__ __forceinline__ uint32_t cluster_addr(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}

//: one fp32 stored into a cluster-mate's shared memory
__device__ __forceinline__ void cluster_store_f32(uint32_t a, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" : : "r"(a), "f"(v) : "memory");
}

//: one fp32 ADDED into a cluster-mate's shared memory by its thread
__device__ __forceinline__ void cluster_add_f32(uint32_t a, float v) {
  asm volatile("red.shared::cluster.add.f32 [%0], %1;" : : "r"(a), "f"(v) : "memory");
}

//: A BULK COPY INTO A CLUSTER-MATE'S SHARED MEMORY: `bytes` of this CTA's shared memory at `src` to the mate's at
//: `dst`, asynchronous; the bytes complete the MATE'S barrier `bar` (its arrival states them: `mbar_expect`).
//: Multiples of 16 bytes
__device__ __forceinline__ void cluster_bulk_copy(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :
               : "r"(dst), "r"(src), "r"(bytes), "r"(bar)
               : "memory");
}

//: the same ADDED into the mate's shared memory, 64-bit integer by 64-bit integer
__device__ __forceinline__ void cluster_bulk_add_u64(uint32_t dst, uint32_t src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 "
               "[%0], [%1], %2, [%3];"
               :
               : "r"(dst), "r"(src), "r"(bytes), "r"(bar)
               : "memory");
}

//: THE CLUSTER'S OWN BARRIER: every thread of every CTA arrives and waits -- before any CTA touches a mate's shared
//: memory (it may not have started) and before any exits (a mate may still reach in)
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  asm volatile("barrier.cluster.wait;" ::: "memory");
}

//: an arrival at a cluster-mate's barrier, RELEASING to the cluster: what this CTA wrote into the mate's shared memory
//: before it is ordered before the mate's wait
__device__ __forceinline__ void cluster_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" : : "r"(a) : "memory");
}

//: this CTA's barrier waited on ACQUIRING from the cluster (polled), the pair of `cluster_arrive`
__device__ __forceinline__ void cluster_wait(uint32_t a, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nLAB_CWAIT_%=:\n"
               "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra LAB_CWAIT_%=;\n}"
               :
               : "r"(a), "r"(parity)
               : "memory");
}

//: this thread's stores and adds into a mate's shared memory ordered for the cluster, before the hand-over's barrier
__device__ __forceinline__ void cluster_fence() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
//: a barrier's initialization made visible to the cluster, before any mate arrives at it
__device__ __forceinline__ void cluster_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

//: one 64-bit integer added into a cluster-mate's shared memory by its thread (an unsigned add: the same bits)
__device__ __forceinline__ void cluster_add_u64(uint32_t a, long long v) {
  asm volatile("red.shared::cluster.add.u64 [%0], %1;" : : "r"(a), "l"(v) : "memory");
}

// --------------------------------------------------------------------------------------------------- host

//: THE TRACED ROWS AS WARPLINE'S NATIVE RECORD (`warpline/host.hpp`; docs/internals/tools/lab.md#trace): the rows of CTAs
//: `ctas` copied back -- `rows[i]` the `warps` rows of CTA `ctas[i]`, each `Trace<true>::kRow` words: the lab's head (the
//: count of entries issued, the CTA's shared base) and the stamps -- one record of one scope a CTA (`cta<N>`) and one lane
//: a warp (`warp<W>`, its role from `roles`), the sites named by `legend` (a barrier's site read against the row's base).
//: The lab's head is not warpline's (a row's head there is `kHead` entries with the count in its first), so each row is
//: given by its entries and its issued count (`Record::add_row`'s head-less overload), and the entries are the lab's own
//: but in three places:
//:  - the sites `entry` and `exit` are the launch's edges: a phase-begin mark there is `kLaunchBegin` / `kLaunchEnd`;
//:  - a barrier's site is its shared address over 8, read against the row's base (a name of the legend's, whichever CTA);
//:  - the barriers' facts, which the record needs and the host header cannot infer, are the lab's from the stamps: a wait
//:    on a site named `... steps` is its warp's own steps (a count of 0), a site a copy is issued at is released by the
//:    copy (`Record::operation`: one arrival, not stamped, the copy its issue, its `done` the barrier's wait's end), a named
//:    barrier's (0x8000 to 0xBFFF) arrival is not stamped (the warp's reaching it is its wait's begin: `Record::rendezvous`,
//:    `kRendezvousSkew`), and any other's count is the warps that arrive at it, one each a phase.
//: the skew a rendezvous' wait ends are accepted with, in cycles (docs/internals/tools/lab.md#warpline)
constexpr uint32_t kRendezvousSkew = 128;

inline std::string trace_record(const std::string& variant, const std::vector<int>& ctas,
                                const std::vector<std::vector<unsigned long long>>& rows, int warps,
                                const std::vector<std::string>& roles,
                                const std::vector<std::pair<uint32_t, std::string>>& legend) {
  constexpr int kRow = Trace<true>::kRow;
  constexpr uint32_t kNamedLow = 0x8000u, kMarksLow = 0xC000u;
  std::map<uint32_t, std::string> names(legend.begin(), legend.end());
  const auto name_of = [&](uint32_t site) {
    const auto it = names.find(site);
    if (it != names.end()) return it->second;
    char buf[32];
    std::snprintf(buf, sizeof buf, "site_0x%x", site);
    return std::string(buf);
  };
  const auto is_edge = [&](uint32_t site, const char* edge) { return name_of(site) == edge; };
  warpline::host::Record record;
  for (const auto& [site, name] : legend) record.name(site, name);
  //: per site, per scope: the warps that wait, the warps that arrive, and whether a copy is issued there
  std::map<uint32_t, std::map<int, std::set<int>>> waiting, arriving;
  std::set<uint32_t> issued;
  std::vector<std::vector<warpline::Entry>> kept;
  for (size_t i = 0; i < ctas.size(); ++i) {
    for (int w = 0; w < warps; ++w) {
      const unsigned long long* row = rows[i].data() + static_cast<size_t>(w) * kRow;
      const unsigned long long n = std::min<unsigned long long>(row[0], Trace<true>::kCap);
      const uint32_t origin = static_cast<uint32_t>(row[1] >> 3);
      std::vector<warpline::Entry> entries(n);
      for (unsigned long long k = 0; k < n; ++k) {
        const unsigned long long x = row[32 + k];
        uint32_t event = static_cast<uint32_t>(x >> 32);
        uint32_t kind = warpline::kind_of(event), site = warpline::site_of(event);
        if (site < kNamedLow) site = (site - origin) & 0xffffu;
        if (kind == warpline::kPhaseBegin && is_edge(site, "entry")) kind = warpline::kLaunchBegin;
        if (kind == warpline::kPhaseBegin && is_edge(site, "exit")) kind = warpline::kLaunchEnd;
        if (kind == warpline::kWaitBegin) waiting[site][ctas[i]].insert(w);
        if (kind == warpline::kArrive) arriving[site][ctas[i]].insert(w);
        if (kind == warpline::kIssue) issued.insert(site);
        entries[k] = {static_cast<uint32_t>(x), warpline::id(kind, site)};
      }
      record.add_row("cta" + std::to_string(ctas[i]), "warp" + std::to_string(w), roles[w], entries.data(), static_cast<uint32_t>(row[0]),
                     Trace<true>::kCap);
    }
  }
  const auto fewest = [](const std::map<int, std::set<int>>& per_scope) {
    size_t least = SIZE_MAX;
    for (const auto& [scope, lanes] : per_scope) least = std::min(least, lanes.size());
    return static_cast<uint32_t>(least == SIZE_MAX ? 0 : least);
  };
  std::set<uint32_t> sites;
  for (const auto& [site, per_scope] : waiting) sites.insert(site);
  for (const auto& [site, per_scope] : arriving) sites.insert(site);
  for (const uint32_t site : sites) {
    const std::string name = name_of(site);
    if (name.size() > 6 && name.compare(name.size() - 6, 6, " steps") == 0) {
      record.barrier(name, 0);
    } else if (issued.count(site)) {
      record.operation(name, site, name);
    } else if (site >= kNamedLow && site < kMarksLow) {
      record.rendezvous(name, std::max(fewest(waiting[site]), 1u), kRendezvousSkew);
    } else {
      record.barrier(name, std::max(fewest(arriving[site]), 1u));
    }
  }
  record.variant(variant);
  return record.json();
}

//: THE DRIVER'S TENSOR-MAP ENCODER (`cuTensorMapEncodeTiled`), from the running driver's library (no toolkit is
//: linked for it); an experiment that cannot reach it exits 2, its name in the refusal
using TensorMapEncode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                     const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                     CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline TensorMapEncode tensor_map_encoder(const char* experiment) {
  void* const driver = dlopen("libcuda.so.1", RTLD_NOW);
  const auto encode = driver ? reinterpret_cast<TensorMapEncode>(dlsym(driver, "cuTensorMapEncodeTiled")) : nullptr;
  if (!encode) {
    std::fprintf(stderr, "%s: the driver's cuTensorMapEncodeTiled is not reachable\n", experiment);
    std::exit(2);
  }
  return encode;
}

//: A KERNEL'S DEMANGLED NAME AS AN EXPECTATION'S PATTERN: `kernel("tax", {3, 17})` matches
//: `ns::tax<(int)3, (int)17>(...)`, the spelling `cu++filt` gives integer template arguments.
inline std::string kernel(const std::string& name, std::initializer_list<long long> args) {
  std::string out = name + "<";
  int i = 0;
  //: a template's integer argument, as the demangler writes an `int`'s or a `long long`'s
  for (long long a : args) out += std::string(i++ ? ", " : "") + "\\((?:int|long long)\\)" + std::to_string(a);
  return out + ">\\(";
}
//: a kernel and those that share its leading template arguments (a detail row's beside its row's); `name` a regex
inline std::string kernel_family(const std::string& name, std::initializer_list<long long> args) {
  const std::string one = kernel(name, args);
  return one.substr(0, one.size() - 3) + "(?:, [^>]*)?>\\(";
}
inline std::string kernel(const std::string& name, std::initializer_list<int> args) {
  std::vector<long long> wide(args.begin(), args.end());
  std::string out = name + "<";
  for (size_t i = 0; i < wide.size(); ++i) out += std::string(i ? ", " : "") + "\\((?:int|long long)\\)" + std::to_string(wide[i]);
  return out + ">\\(";
}

#define LAB_CHECK(call)                                                                       \
  do {                                                                                        \
    const cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "lab: %s failed: %s\n", #call, cudaGetErrorString(e_));            \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

//: THE HOST'S f16: a half's value, and a nonnegative integer as a conversion to f16 rounds it (to nearest, ties to
//: even; under 65,520)
inline float half_to_float(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
  return h & 0x8000 ? -mag : mag;
}
inline long long f16_of(long long v) {
  if (v < 2048) return v;
  int e = 0;
  while ((v >> e) >= 2048) ++e;
  long long kept = v >> e;
  const long long rest = v & ((1ll << e) - 1), half = 1ll << (e - 1);
  kept += rest > half || (rest == half && (kept & 1));
  return kept << e;
}

//: A RESULT'S SAMPLES, one a repetition.
struct Samples {
  std::vector<double> v;

  double median() const {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s.empty() ? 0.0 : s[s.size() / 2];
  }
};

//: THE RUNNER. `--expect` prints the declared SASS and runs nothing; otherwise each `variant` warms up once, repeats
//: `--reps` times, and `record` prints a result as one JSON line `tools/lab.py` reads.
class Bench {
 public:
  static constexpr int kMaxWarps = 64;
  int sms = 0, reps = 7, blocks = 0;
  bool expect_only = false;
  long long* times = nullptr;
  int* sinks = nullptr;

  Bench(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--expect")) expect_only = true;
      if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
      if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = std::atoi(argv[++i]);
    }
    if (expect_only) return;
    LAB_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    //: `--blocks N`: the grid an experiment launches over `sms` held to N CTAs (a sanitizer's run at one CTA)
    if (blocks > 0 && blocks < sms) sms = blocks;
    LAB_CHECK(cudaMalloc(&times, sizeof(long long) * sms * kMaxWarps));
    LAB_CHECK(cudaMalloc(&sinks, sizeof(int) * sms * kMaxWarps * 32));
    cudaDeviceProp prop{};
    LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"lab\": \"device\", \"name\": \"%s\", \"cc\": %d%d, \"sms\": %d}\n", prop.name,
                prop.major, prop.minor, sms);
  }

  //: a kernel whose demangled name matches `kernel` must hold `op` `count` times (`op` an opcode prefix, `cmp` one of
  //: ">=", "==", "<=")
  void expect(const std::string& kernel, const std::string& op, const char* cmp, int count) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf(
        "{\"lab\": \"expect\", \"kernel\": \"%s\", \"op\": \"%s\", \"cmp\": \"%s\", \"count\": %d}\n",
        escaped.c_str(), op.c_str(), cmp, count);
  }

  //: THE INSTRUCTION BUDGET (KERNEL_STANDARDS section 29): a kernel whose demangled name matches `kernel` executes at
  //: most `max` instructions of opcode class `cls` (`tools/sass.py`'s `OPCODE_CLASSES`) per tensor-pipe instruction,
  //: stated from the design's operation count before its code; `tools/lab.py run` reads the executed counts (ncu) and
  //: refuses to record a variant over it
  void budget(const std::string& kernel, const char* cls, double max) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"budget\", \"kernel\": \"%s\", \"class\": \"%s\", \"max\": %.3f}\n", escaped.c_str(),
                cls, max);
  }

  //: AN MMA KERNEL WITHOUT A BUDGET, on purpose (KERNEL_STANDARDS section 29): a rate probe or an isolation, its reason
  void exempt(const std::string& kernel, const char* reason) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"exempt\", \"kernel\": \"%s\", \"reason\": \"%s\"}\n", escaped.c_str(), reason);
  }

  //: A VARIANT'S OUTPUTS CHECKED against a host reference (KERNEL_STANDARDS section 29): `tools/lab.py run` records no
  //: variant whose check failed, nor one with neither a check nor `unchecked`; `detail` says what was compared
  void check(const std::string& variant, bool ok, const std::string& detail) const {
    std::printf("{\"lab\": \"check\", \"variant\": \"%s\", \"ok\": %s, \"detail\": \"%s\"}\n", variant.c_str(),
                ok ? "true" : "false", detail.c_str());
  }

  //: a variant whose outputs are not checked, and why (an isolation whose outputs fold into a sink)
  void unchecked(const std::string& variant, const char* reason) const {
    std::printf("{\"lab\": \"unchecked\", \"variant\": \"%s\", \"reason\": \"%s\"}\n", variant.c_str(), reason);
  }

  //: THE UNIT a kernel without MMAs is budgeted per: its first launch does `n` of them (a box-window colored, say), so
  //: `tools/lab.py run` divides its executed counts by `n` where an MMA kernel's divide by its MMAs
  void units(const std::string& kernel, const char* unit, long long n) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"units\", \"kernel\": \"%s\", \"unit\": \"%s\", \"n\": %lld}\n", escaped.c_str(),
                unit, n);
  }

  //: the per-(block, warp) cycles of the last launch, `warps` a block
  std::vector<long long> cycles(int warps) const {
    std::vector<long long> h(static_cast<size_t>(sms) * warps);
    LAB_CHECK(cudaMemcpy(h.data(), times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    return h;
  }

  //: the mean over blocks of the block's span (`WarpSpan`), `warps` a block, `blocks` launched (one an SM if omitted)
  //: SPAN_RECORDED: a timed launch starts from cleared clocks (`variant`), so a warp that never stopped its own
  //: leaves a zero -- no block's span is read over one
  static void span_recorded(long long lo, long long hi, int block, int warps) {
    if (lo > 0 && hi >= lo && hi - lo < (1ll << 40)) return;
    std::fprintf(stderr, "lab: block %d's span is not a measurement: a warp of its %d did not record its clock\n",
                 block, warps);
    std::exit(3);
  }

  double span(int warps, int blocks = 0) const {
    const int n = blocks > 0 ? blocks : sms;
    std::vector<long long> h(static_cast<size_t>(n) * warps * 2);
    LAB_CHECK(cudaMemcpy(h.data(), times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    double s = 0;
    for (int b = 0; b < n; ++b) {
      long long lo = h[2 * b * warps], hi = h[2 * b * warps + 1];
      for (int w = 1; w < warps; ++w) {
        lo = std::min(lo, h[2 * (b * warps + w)]);
        hi = std::max(hi, h[2 * (b * warps + w) + 1]);
      }
      span_recorded(lo, hi, b, warps);
      s += double(hi - lo);
    }
    return s / n;
  }

  //: the mean over blocks of warps `[first, first + count)`'s cycles
  static double mean(const std::vector<long long>& t, int warps, int first, int count) {
    double s = 0;
    const size_t blocks = t.size() / warps;
    for (size_t b = 0; b < blocks; ++b)
      for (int w = first; w < first + count; ++w) s += double(t[b * warps + w]);
    return s / double(blocks * count);
  }

  //: one variant's samples: `launch()` runs it once, `measure()` reads the value from its cycles
  template <class Launch, class Measure>
  Samples variant(Launch launch, Measure measure) const {
    Samples out;
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
      LAB_CHECK(cudaMemset(times, 0, sizeof(long long) * sms * kMaxWarps));
      launch();
      LAB_CHECK(cudaDeviceSynchronize());
      out.v.push_back(measure());
    }
    return out;
  }

  //: THE TRACES of a launch's CTAs `ctas`, as one JSON line: the variant's name and its warpline record
  //: (`trace_record`); `legend` names the sites. A cycle of the record is relative to its CTA's first, a nanosecond to the
  //: record's first tie, both unwrapped past their 32-bit words.
  void trace(const std::string& variant, const unsigned long long* rows, int warps, std::initializer_list<int> ctas,
             const std::vector<std::string>& roles, const std::vector<std::pair<uint32_t, std::string>>& legend) const {
    constexpr int kRow = Trace<true>::kRow;
    std::vector<std::vector<unsigned long long>> host;
    for (int cta : ctas) {
      host.emplace_back(static_cast<size_t>(warps) * kRow);
      LAB_CHECK(cudaMemcpy(host.back().data(), rows + static_cast<size_t>(cta) * warps * kRow, host.back().size() * 8,
                           cudaMemcpyDeviceToHost));
    }
    std::string quoted;
    for (char c : variant) quoted += c == '"' || c == '\\' ? std::string("\\") + c : std::string(1, c);
    const std::string doc = trace_record(variant, std::vector<int>(ctas), host, warps, roles, legend);
    std::printf("{\"lab\": \"trace\", \"variant\": \"%s\", \"record\": %s}\n", quoted.c_str(), doc.c_str());
  }

  //: a result, as one JSON line
  void record(const std::string& name, const char* unit, const Samples& s) const {
    std::string vals;
    for (size_t i = 0; i < s.v.size(); ++i) vals += (i ? ", " : "") + std::to_string(s.v[i]);
    std::printf("{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"%s\", \"samples\": [%s]}\n",
                name.c_str(), unit, vals.c_str());
  }
};

}  // namespace lab
