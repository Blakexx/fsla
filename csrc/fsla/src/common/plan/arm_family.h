// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE ARM GENERATOR'S VIEW OF A FAMILY (docs/internals/common/plan/arms_gen.md): for each kernel whose arms the build generates, its facts, its
// arm and the arm's key, and the matcher's arm half over a context. One generator (`arms_gen.cpp`) serves every family through it.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "plan/decode/contract.h"
#include "plan/decode_launch.h"
#include "plan/entmax_launch.h"
#include "plan/factor_backward/contract.h"
#include "plan/factor_forward/contract.h"
#include "plan/softmax_backward/contract.h"
#include "plan/softmax_forward/contract.h"
#include "plan/union_backward/contract.h"
#include "plan/union_forward/contract.h"

namespace fsla::plan {

//: a family of kernel `K` (its generated definition): `Facts`, `Arm`, `Reach`, the key's `fields` and `key(arm)`, the matcher's `reach` over a context's
//: facts and `arm` over a point's, the matcher's file (named where a context reaches no arm), `read` (the call's facts as text) and `launch_reflects`
//: (whether the launch of a point answers that point's facts)
template <typename K>
struct ArmFamily;

template <>
struct ArmFamily<decode::Kernel> {
  using Facts = decode::Facts;
  using Arm = DecodeArm;
  using Reach = DecodeReach;

  static constexpr std::array<const char*, 3> fields = {"d_v", "levels", "decay"};
  static constexpr const char* matcher = "csrc/fsla/src/common/plan/decode_launch.h";

  static constexpr std::array<int, 3> key(const Arm& a) { return {a.d_v, a.levels, a.decay}; }

  static constexpr Reach reach(const Facts& d) { return decode_reach(d); }

  static constexpr Arm arm(const Facts& d) { return decode_arm(d); }

  static std::string read(const Facts& d) { return decode_facts_read(d); }

  static bool launch_reflects(const Facts& d) { return decode_args_reflect(d, decode_launch(d)); }
};

//: the family of each of the six entmax kernels, over `entmax_launch.h`'s matcher
template <typename K>
struct EntmaxFamily {
  using Facts = typename K::Descriptor;
  using Arm = EntmaxArmOf<K>;
  using Reach = EntmaxReach<K>;

  static constexpr auto fields = Arm::fields;
  static constexpr const char* matcher = "csrc/fsla/src/common/plan/entmax_launch.h";

  static constexpr auto key(const Arm& a) { return a.key(); }

  static constexpr Reach reach(const Facts& d) { return entmax_reach<K>(d); }

  static constexpr Arm arm(const Facts& d) { return entmax_arm<K>(d); }

  static std::string read(const Facts& d) {
    std::string out;
    for (std::string_view name : K::call_fact_names) {
      Interval v{};
      K::Reader::get(d, name, v);
      out += (out.empty() ? "" : " ") + std::string(name) + "=" + spell_interval(v);
    }
    return out;
  }

  static bool launch_reflects(const Facts& d) {
    const EntmaxLaunch<K> launch = entmax_launch<K>(d);
    return launch.arm == arm(d) && launch.levels == K::Reader::levels(d).lo
           && launch.klass == launch.arm.lw * launch.arm.ipt;
  }
};

template <>
struct ArmFamily<union_forward::Kernel> : EntmaxFamily<union_forward::Kernel> {};

template <>
struct ArmFamily<union_backward::Kernel> : EntmaxFamily<union_backward::Kernel> {};

template <>
struct ArmFamily<factor_forward::Kernel> : EntmaxFamily<factor_forward::Kernel> {};

template <>
struct ArmFamily<factor_backward::Kernel> : EntmaxFamily<factor_backward::Kernel> {};

template <>
struct ArmFamily<softmax_forward::Kernel> : EntmaxFamily<softmax_forward::Kernel> {};

template <>
struct ArmFamily<softmax_backward::Kernel> : EntmaxFamily<softmax_backward::Kernel> {};

}  // namespace fsla::plan
