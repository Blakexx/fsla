// csrc/rola/src/common/window.cuh -- the window W, a card's. see docs/internals/common/window.md
#pragma once

#include "common/arch_caps.cuh"

namespace rola {

//: W by compute capability (its own build's), a slot's token bits and query heads at W. -- see docs/internals/common/window.md
__host__ __device__ constexpr int window_for(int cc) {
  return ::denseref::arch::traits_of(::denseref::arch::Build{cc, 0}).window;
}

__host__ __device__ constexpr int window_bits(int w) { return w == 1024 ? 10 : w == 512 ? 9 : 0; }

__host__ __device__ constexpr int window_heads(int w) { return 1 << (15 - window_bits(w)); }

//: the compiling card's W (the pinned card's in the host pass): host code reads `device_window` (`arch_runtime.cuh`)
constexpr int kWindow = window_for(::denseref::arch::kArch);
static_assert(kWindow > 0,
              "an untabulated compute capability has no window: add its row to window_for");

}  // namespace rola
