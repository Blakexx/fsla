// csrc/rola/src/common/structure_switch.cuh -- R9's four enforcement layers, once.
// See docs/internals/common/structure_switch.md
#pragma once

#include <type_traits>

namespace rola {

namespace structure_switch_detail {

//: the primitive's OWN compile-time loop -- see docs/internals/common/structure_switch.md#uniform-switch
template <int N, class F>
__device__ __forceinline__ void unroll(F&& f) {
  if constexpr (N > 0) {
    unroll<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

}  // namespace structure_switch_detail

//: ONE CTA-UNIFORM BRANCH OVER A GENERATED SET. -- see docs/internals/common/structure_switch.md#uniform-switch
template <int NA, class F>
__device__ __forceinline__ void uniform_switch(int value, F&& body) {
  static_assert(NA >= 1, "a generated structure set is never empty");
  bool hit = false;
  structure_switch_detail::unroll<NA>([&](auto Ac) {
    if (value == decltype(Ac)::value) {
      body(Ac);
      hit = true;
    }
  });
  if (!hit) __trap();
}

}  // namespace rola
