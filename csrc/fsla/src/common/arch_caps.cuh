#pragma once

// THE CARD'S FACTS at the compiling target: the rows `common/hardware.cuh` generates from the registry's device parts
// (hardware/parts/device/), the compiling architecture and its row. see docs/internals/common/arch_caps.md

#include <cuda_runtime.h>

#include "common/hardware.cuh"

namespace denseref {
namespace arch {

// THE COMPILING TARGET; inert in the host pass, which is the hopper's.
#ifdef __CUDA_ARCH__
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 900;
#endif

static_assert(tabulated(kArch),
              "untabulated compute capability: add its row to caps_of() rather "
              "than letting an operation guess which implementation it has");

//: THE ONE FAMILY BUILT: every other family is refused by name, here and nowhere else
static_assert(traits_of(Build{kArch}).family == Family::kHopper,
              "only the hopper family is built: build for sm_90a (--//:archs)");

constexpr Caps kCaps = caps_of(kArch);

}  // namespace arch
}  // namespace denseref
