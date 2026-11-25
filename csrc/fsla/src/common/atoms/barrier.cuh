// csrc/fsla/src/common/atoms/barrier.cuh -- a concern of the baseline operations (`common/ops.cuh`).
// see docs/internals/common/atoms/barrier.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

#include "common/atoms/address.cuh"

namespace denseref {
namespace ops {

//: A SHARED-MEMORY BARRIER OBJECT (`mbarrier`, sm_80+) -- see docs/internals/common/atoms/barrier.md#mbarrier
__device__ __forceinline__ void mbar_init(SmemAddr a, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(SmemAddr a) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(a) : "memory");
}

__device__ __forceinline__ void mbar_wait(SmemAddr a, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "MBAR_WAIT_%=:\n"
      "mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra MBAR_WAIT_%=;\n}\n" ::"r"(a),
      "r"(parity)
      : "memory");
}

//: THE SAME WAIT, SUSPENDING -- see docs/internals/common/atoms/barrier.md#suspended
__device__ __forceinline__ void mbar_wait_suspended(SmemAddr a, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "MBAR_TRY_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra MBAR_TRY_%=;\n}\n" ::"r"(a),
      "r"(parity)
      : "memory");
}

//: the same, asked once
__device__ __forceinline__ bool mbar_test(SmemAddr a, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(a), "r"(parity)
      : "memory");
  return ok != 0u;
}

__device__ __forceinline__ void rendezvous() { __syncthreads(); }

//: A RENDEZVOUS WHERE `cond`, CTA-uniform -- see docs/internals/common/atoms/barrier.md#rendezvous-if
__device__ __forceinline__ void rendezvous_if(bool cond) {
  asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %0, 0;\n @p bar.sync 0;\n}\n" ::"r"(
                   static_cast<uint32_t>(cond))
               : "memory");
}

//: THE WARP'S OWN EDGE, stated as the instruction -- see docs/internals/common/atoms/barrier.md#warp-sync
__device__ __forceinline__ void warp_sync() {
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
}

//: A BARRIER OVER `n` WARP-ALIGNED THREADS at named barrier `id` -- see docs/internals/common/atoms/barrier.md#rendezvous-group
__device__ __forceinline__ void rendezvous_group(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

//: AN ARRIVE AT A NAMED BARRIER, without waiting -- see docs/internals/common/atoms/barrier.md#rendezvous-group
__device__ __forceinline__ void rendezvous_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

//: THE ORDER BETWEEN A UNIT'S SHARED WRITES AND THE FLAG THAT PUBLISHES THEM -- see docs/internals/common/atoms/barrier.md#publish-fence
__device__ __forceinline__ void publish_fence() { __threadfence_block(); }

//: THE ORDER BETWEEN THREADS' SHARED WRITES AND AN ASYNCHRONOUS OPERATION'S READ OF THEM -- see docs/internals/common/atoms/barrier.md#async-fence
__device__ __forceinline__ void async_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

}  // namespace ops
}  // namespace denseref
