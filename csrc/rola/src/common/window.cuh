// csrc/rola/src/common/window.cuh -- the window W, a card's. see docs/internals/common/window.md
#pragma once

#include "common/arch_caps.cuh"

namespace rola {

//: W by compute capability; a union slot's token bits and query heads at W. -- see docs/internals/common/window.md
__host__ __device__ constexpr int window_for(int cc) {
  return cc == 800 ? 1024 : cc == 860 ? 1024 : cc == 870 ? 512 : cc == 890 ? 512 : 0;
}

__host__ __device__ constexpr int window_bits(int w) { return w == 1024 ? 10 : w == 512 ? 9 : 0; }

__host__ __device__ constexpr int window_heads(int w) { return 1 << (15 - window_bits(w)); }

//: the compiling card's W (the pinned card's in the host pass), and a device's for host code
constexpr int kWindow = window_for(::denseref::arch::kArch);
static_assert(kWindow > 0,
              "an untabulated compute capability has no window: add its row to window_for");

int device_window(int device);

}  // namespace rola
