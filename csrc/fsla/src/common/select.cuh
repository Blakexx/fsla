// csrc/fsla/src/common/select.cuh -- THE SELECTION LAYER: a device's card (its build), the builds this
// binary carries, and the host's branch to the card's build at compile time. see docs/internals/common/select.md
#pragma once

#include <utility>

#include "builds.inc"
#include "common/arch_caps.cuh"

namespace fsla {

using ::denseref::arch::Build;
using ::denseref::arch::BuildTag;
using ::denseref::arch::Traits;
using ::denseref::arch::traits_of;

//: A DEVICE'S CARD: its compute capability, its SMs (the planner's read), the build it runs and that build's traits.
struct Card {
  int device, cc, sms;
  Build build;
  Traits traits;
};

//: `device`'s card, admitted once a process -- see docs/internals/common/select.md#card-of
const Card& card_of(int device);
const Card& current_card();

//: TRUE where build `b`'s kernels compile -- see docs/internals/common/select.md#in-image
constexpr bool in_image(Build b) {
#ifdef __CUDA_ARCH__
  return b.cc == __CUDA_ARCH__;
#else
  return b.cc > 0;
#endif
}

//: ONE HOST BRANCH TO THE CARD'S BUILD -- see docs/internals/common/select.md#build-switch
template <class Body>
inline void build_switch(const Card& c, Body&& body) {
#define FSLA_BUILD_CASE(CC_)                      \
  if (c.build.cc == (CC_)) {                      \
    body(BuildTag<Build{CC_}>{});                 \
    return;                                       \
  }
  FSLA_BUILDS(FSLA_BUILD_CASE)
#undef FSLA_BUILD_CASE
}

}  // namespace fsla
