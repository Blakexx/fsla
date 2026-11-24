// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE ENTMAX KERNELS' MATCHER (docs/internals/common/plan/entmax_launch.md): the arm to launch from a call's
// facts, one function for the four kernels of `csrc/fsla/src/entmax/`, written over the kernel's generated
// definition `K` as `planner.h` is. The arm half is evaluable over ranges, so the build asks it what each
// contract reaches.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "plan/planner.h"
#include "plan/rules.h"

namespace fsla::plan {

//: THE WIDTH LIMIT, the widest level the solves hold (`MAX_BRANCH_WIDTH`, fsla/routing/topology.py), and THE LEVEL BOUND, the levels one launch's
//: table holds
inline constexpr int kEntmaxMaxWidth = 256;
inline constexpr int kEntmaxMaxLevels = 16;

//: the threads of one logical warp at most: a wider class gives each lane more items
inline constexpr int kEntmaxWarpLanes = 32;

//: THE FACT CODES: the dtypes of the logit and value planes, and alpha as tenths; the code a call states for what the contract has none for
inline constexpr int kEntmaxFp32 = 0;
inline constexpr int kEntmaxBf16 = 1;
inline constexpr int kEntmaxNoDtype = -1;
inline constexpr int kEntmaxAlpha15 = 15;
inline constexpr int kEntmaxAlpha20 = 20;
inline constexpr int kEntmaxNoAlpha = 0;

inline constexpr std::size_t kMaxEntmaxArms = 64;

//: THE PADDED WIDTH: the width class, the next power of two from 2
constexpr int entmax_padded_width(int width) {
  int w = 2;
  while (w < width) w <<= 1;
  return w;
}

//: alpha as the contract's code: exactly 1.5 or 2.0, else the code no context holds
constexpr int entmax_alpha_code(double alpha) {
  if (alpha == 1.5) return kEntmaxAlpha15;
  return alpha == 2.0 ? kEntmaxAlpha20 : kEntmaxNoAlpha;
}

namespace detail {

constexpr int entmax_class_count() {
  int n = 0;
  for (int w = 2; w <= kEntmaxMaxWidth; w <<= 1) ++n;
  return n;
}

}  // namespace detail

//: THE WIDTH CLASSES, ascending: the padded widths up to the limit
inline constexpr auto kEntmaxClasses = [] {
  std::array<int, static_cast<std::size_t>(detail::entmax_class_count())> classes{};
  int at = 0;
  for (int w = 2; w <= kEntmaxMaxWidth; w <<= 1) classes[static_cast<std::size_t>(at++)] = w;
  return classes;
}();

inline constexpr int kEntmaxDtypeUpto[] = {kEntmaxFp32, kEntmaxBf16};

//: an arm of the two kernels that have an alpha: the key of `<kernel>_kernel<Build, LW, IPT, ALPHA15, LogitT, OutT>`; the dtypes are the codes
struct EntmaxAlphaArm {
  int lw;
  int ipt;
  int alpha15;
  int logit;
  int value;

  static constexpr std::array<const char*, 5> fields = {"lw", "ipt", "alpha15", "logit", "value"};

  constexpr std::array<int, 5> key() const { return {lw, ipt, alpha15, logit, value}; }

  constexpr bool operator==(const EntmaxAlphaArm&) const = default;

  constexpr bool operator<(const EntmaxAlphaArm& o) const { return key() < o.key(); }
};

//: an arm of the two softmax kernels, which have no alpha
struct EntmaxPlainArm {
  int lw;
  int ipt;
  int logit;
  int value;

  static constexpr std::array<const char*, 4> fields = {"lw", "ipt", "logit", "value"};

  constexpr std::array<int, 4> key() const { return {lw, ipt, logit, value}; }

  constexpr bool operator==(const EntmaxPlainArm&) const = default;

  constexpr bool operator<(const EntmaxPlainArm& o) const { return key() < o.key(); }
};

//: whether kernel `K` has an alpha: its facts have one
template <typename K>
concept EntmaxHasAlpha = requires(const typename K::Descriptor& d) {
  K::Reader::alpha_tenths(d);
};

template <typename K>
using EntmaxArmOf = std::conditional_t<EntmaxHasAlpha<K>, EntmaxAlphaArm, EntmaxPlainArm>;

template <typename K>
using EntmaxReach = Reach<EntmaxArmOf<K>, kMaxEntmaxArms>;

//: what the launch takes besides the arm: the padded width the launch's levels share, and the levels (the grid's depth)
template <typename K>
struct EntmaxLaunch {
  EntmaxArmOf<K> arm;
  int klass;
  int levels;
};

//: THE ARM HALF, a decision tree on Truth, exact over ranges (docs/internals/common/plan/rules.md#exact): the levels within the bound, the width's
//: class, the alpha, each dtype. The class gives the lane width `min(class, 32)` and the items a thread `class / lane`
template <typename K>
constexpr EntmaxReach<K> entmax_reach(const typename K::Descriptor& d) {
  using R = typename K::Reader;
  EntmaxReach<K> r;
  const auto hole = [&] { r.hole(); };

  const auto arm = [&](int lane, int items, int alpha15, int logit, int value) {
    if constexpr (EntmaxHasAlpha<K>) {
      r.arm({lane, items, alpha15, logit, value});
    } else {
      r.arm({lane, items, logit, value});
    }
  };

  const auto dtypes = [&](int lane, int items, int alpha15) {
    const auto logit = [&](std::size_t l) {
      const auto value = [&](std::size_t v) {
        arm(lane, items, alpha15, static_cast<int>(l), static_cast<int>(v));
      };
      ladder(R::value_dtype(d), kEntmaxFp32, kEntmaxDtypeUpto, std::size(kEntmaxDtypeUpto), hole,
             value);
    };
    ladder(R::logit_dtype(d), kEntmaxFp32, kEntmaxDtypeUpto, std::size(kEntmaxDtypeUpto), hole,
           logit);
  };

  const auto alphas = [&](int lane, int items) {
    if constexpr (EntmaxHasAlpha<K>) {
      const auto is15 = [&] { dtypes(lane, items, 1); };
      const auto is20 = [&] { dtypes(lane, items, 0); };
      const auto not15 = [&] { fork(eq(R::alpha_tenths(d), kEntmaxAlpha20), is20, hole); };
      fork(eq(R::alpha_tenths(d), kEntmaxAlpha15), is15, not15);
    } else {
      dtypes(lane, items, 0);
    }
  };

  const auto leaf = [&](std::size_t c) {
    const int klass = kEntmaxClasses[c];
    const int lane = klass < kEntmaxWarpLanes ? klass : kEntmaxWarpLanes;
    alphas(lane, klass / lane);
  };

  const auto widths = [&] {
    ladder(R::width(d), kEntmaxClasses[0], kEntmaxClasses.data(), kEntmaxClasses.size(), hole,
           leaf);
  };

  fork(in(R::levels(d), 1, kEntmaxMaxLevels), widths, hole);
  return r;
}

//: the arm of a point: the same walk, which over a point visits one leaf; a range, or a point no arm serves, aborts
template <typename K>
constexpr EntmaxArmOf<K> entmax_arm(const typename K::Descriptor& d) {
  return entmax_reach<K>(d).one();
}

//: THE LAUNCH of a point: its arm, the padded width its levels share and its level count
template <typename K>
constexpr EntmaxLaunch<K> entmax_launch(const typename K::Descriptor& d) {
  return {entmax_arm<K>(d), entmax_padded_width(static_cast<int>(K::Reader::width(d).lo)),
          static_cast<int>(K::Reader::levels(d).lo)};
}

//: THE CALL'S FACTS of a kernel without an alpha, in the form `describe` takes: the first level's width, the levels and each dtype as its code
template <typename K>
Named entmax_call(std::int64_t width, std::int64_t levels, int logit, int value) {
  return {{"width", width}, {"levels", levels}, {"logit_dtype", logit}, {"value_dtype", value}};
}

//: THE CALL'S FACTS of a kernel with an alpha: the same, and alpha as its code
template <typename K>
Named entmax_alpha_call(std::int64_t width, std::int64_t levels, double alpha, int logit,
                        int value) {
  Named call = entmax_call<K>(width, levels, logit, value);
  call.emplace_back("alpha_tenths", entmax_alpha_code(alpha));
  return call;
}

//: THE ONE-WIDTH-CLASS RULE: a launch's levels share the class `klass` of its first; empty when they do, else the refusal naming the level
inline std::string entmax_class_refusal(const std::vector<std::int64_t>& widths, int klass) {
  for (std::size_t i = 0; i < widths.size(); ++i) {
    if (entmax_padded_width(static_cast<int>(widths[i])) != klass) {
      return "a batched solve must be one width class; level " + std::to_string(i) + " has width "
             + std::to_string(widths[i]) + " against class " + std::to_string(klass);
    }
  }
  return {};
}

}  // namespace fsla::plan
