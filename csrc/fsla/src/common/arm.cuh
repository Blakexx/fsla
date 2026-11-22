// csrc/fsla/src/common/arm.cuh -- an arm's key and constants, any family's, torch-free; see docs/internals/common/arm.md
#pragma once

namespace fsla {

template <int N>
struct ArmKey {
  int v[N];
};

template <int INDEX, int... VALUES>
struct Arm {
  static constexpr int index = INDEX;
  static constexpr int size = sizeof...(VALUES);
  static constexpr int values[sizeof...(VALUES)] = {VALUES...};
};

}  // namespace fsla
