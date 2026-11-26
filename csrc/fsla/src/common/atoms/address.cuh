// csrc/fsla/src/common/atoms/address.cuh -- THE ADDRESSES: shared addresses, gathered and pinned bases, the state
// plane's page translation and the facts' per-atom bits; a concern of the baseline operations (`common/ops.cuh`). see
// docs/internals/common/atoms/address.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/arch_caps.cuh"

namespace denseref {
namespace ops {

using SmemAddr = uint32_t;

__device__ __forceinline__ SmemAddr smem_addr(const void* p) {
  return static_cast<SmemAddr>(__cvta_generic_to_shared(p));
}

//: A GATHERED ROW'S ADDRESS: a base formed once, displaced by a 32-bit BYTE -- see docs/internals/common/atoms/address.md#gather-ptr
__device__ __forceinline__ const void* gather_ptr(const void* base, uint32_t off) {
  const void* p;
  asm("{ .reg .u64 t; cvt.u64.u32 t, %2; add.u64 %0, %1, t; }" : "=l"(p) : "l"(base), "r"(off));
  __builtin_assume(__isGlobal(p));
  return p;
}

//: A BASE THE COMPILER MUST KEEP RATHER THAN RE-DERIVE.  The value is unchanged; -- see docs/internals/common/atoms/address.md#t
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

}  // namespace ops
}  // namespace denseref
