// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE RULE LANGUAGE (docs/internals/common/plan/rules.md): what a matcher is written in. Header-only and
// constexpr, so the build evaluates a matcher over the contract's ranges and the run time over a
// device's points with the same code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>

namespace fsla::plan {

//: a test over a range: Yes when every value of the range passes, No when none does, Maybe when
//: the range straddles
enum class Truth : std::uint8_t { No, Yes, Maybe };

//: lo <= hi; a point is lo == hi
struct Interval {
  std::int64_t lo;
  std::int64_t hi;

  constexpr bool point() const { return lo == hi; }

  constexpr bool operator==(const Interval&) const = default;
};

//: a set of up to 64 members of one fact's enum, a bit each
struct EnumSet {
  std::uint64_t bits;

  constexpr bool operator==(const EnumSet&) const = default;
};

constexpr EnumSet enum_set(std::initializer_list<int> members) {
  EnumSet s{0};
  for (int m : members) s.bits |= std::uint64_t{1} << m;
  return s;
}

constexpr Truth lt(Interval x, std::int64_t c) {
  if (x.hi < c) return Truth::Yes;
  return x.lo >= c ? Truth::No : Truth::Maybe;
}

constexpr Truth le(Interval x, std::int64_t c) {
  if (x.hi <= c) return Truth::Yes;
  return x.lo > c ? Truth::No : Truth::Maybe;
}

constexpr Truth ge(Interval x, std::int64_t c) {
  if (x.lo >= c) return Truth::Yes;
  return x.hi < c ? Truth::No : Truth::Maybe;
}

constexpr Truth eq(Interval x, std::int64_t c) {
  if (c < x.lo || c > x.hi) return Truth::No;
  return x.point() ? Truth::Yes : Truth::Maybe;
}

//: x lies inside [lo, hi]
constexpr Truth in(Interval x, std::int64_t lo, std::int64_t hi) {
  if (x.lo >= lo && x.hi <= hi) return Truth::Yes;
  return x.hi < lo || x.lo > hi ? Truth::No : Truth::Maybe;
}

//: the set holds the member (a set is one value, never a range of them)
constexpr Truth has(EnumSet s, int member) {
  return (s.bits >> member) & 1 ? Truth::Yes : Truth::No;
}

//: No when any test is No, Yes when every test is Yes, else Maybe
template <typename... T>
constexpr Truth all(T... tests) {
  bool any_no = false, all_yes = true;
  for (Truth t : {tests...}) {
    any_no |= t == Truth::No;
    all_yes &= t == Truth::Yes;
  }
  if (any_no) return Truth::No;
  return all_yes ? Truth::Yes : Truth::Maybe;
}

//: Yes when any test is Yes, No when every test is No, else Maybe
template <typename... T>
constexpr Truth any(T... tests) {
  bool any_yes = false, all_no = true;
  for (Truth t : {tests...}) {
    any_yes |= t == Truth::Yes;
    all_no &= t == Truth::No;
  }
  if (any_yes) return Truth::Yes;
  return all_no ? Truth::No : Truth::Maybe;
}

//: the arms a matcher returns, in the order they were added, each once; empty is NONE
template <typename Arm, std::size_t Cap>
struct ArmSet {
  Arm arms[Cap]{};

  std::size_t count = 0;

  constexpr bool none() const { return count == 0; }

  constexpr std::size_t size() const { return count; }

  constexpr void add(const Arm& a) {
    for (std::size_t i = 0; i < count; ++i)
      if (arms[i] == a) return;
    if (count == Cap) std::abort();
    arms[count++] = a;
  }
};

//: WHAT A DECISION TREE REACHES: the arms of the leaves visited, `gap` when a NONE leaf was, and the leaves visited
template <typename Arm, std::size_t Cap>
struct Reach {
  ArmSet<Arm, Cap> arms;

  bool gap = false;

  std::size_t leaves = 0;

  constexpr void arm(const Arm& a) {
    arms.add(a);
    ++leaves;
  }

  constexpr void hole() {
    gap = true;
    ++leaves;
  }

  //: THE ARM AT A POINT: the one leaf a point reaches; aborts on a range that reached other than one leaf, or on NONE
  constexpr Arm one() const {
    if (leaves != 1 || gap) std::abort();
    return arms.arms[0];
  }
};

//: a branch: `then` when the test can be Yes, `otherwise` when it can be No, both on Maybe
template <typename Then, typename Otherwise>
constexpr void fork(Truth t, Then then, Otherwise otherwise) {
  if (t != Truth::No) then();
  if (t != Truth::Yes) otherwise();
}

namespace detail {

template <typename Hole, typename Leaf>
constexpr void ladder_from(Interval x, const int* upto, std::size_t n, std::size_t i,
                           const Hole& hole, const Leaf& leaf) {
  if (i == n) {
    hole();
    return;
  }
  const auto here = [&] { leaf(i); };
  const auto above = [&] { ladder_from(x, upto, n, i + 1, hole, leaf); };
  fork(le(x, upto[i]), here, above);
}

}  // namespace detail

//: THE LADDER over an int fact (docs/internals/common/plan/rules.md#exact): `leaf(i)` for the values from `first` up to `upto[i]`
//: (`upto` ascends, `first <= upto[0]`); `hole()` below `first` or above `upto[n - 1]`
template <typename Hole, typename Leaf>
constexpr void ladder(Interval x, int first, const int* upto, std::size_t n, const Hole& hole,
                      const Leaf& leaf) {
  const auto rungs = [&] { detail::ladder_from(x, upto, n, 0, hole, leaf); };
  fork(ge(x, first), rungs, hole);
}

namespace detail {

//: THE POINT GUARANTEE, at compile time: over a point every primitive returns Yes or No, never Maybe
constexpr bool primitives_decide_at_points() {
  for (std::int64_t x = -3; x <= 3; ++x) {
    const Interval p{x, x};
    for (std::int64_t c = -3; c <= 3; ++c) {
      if (lt(p, c) == Truth::Maybe || le(p, c) == Truth::Maybe) return false;
      if (ge(p, c) == Truth::Maybe || eq(p, c) == Truth::Maybe) return false;
      for (std::int64_t hi = -3; hi <= 3; ++hi)
        if (in(p, c, hi) == Truth::Maybe) return false;
    }
  }
  for (int m = 0; m < 64; ++m)
    for (std::uint64_t bits : {std::uint64_t{0}, ~std::uint64_t{0}, std::uint64_t{1} << 5})
      if (has(EnumSet{bits}, m) == Truth::Maybe) return false;
  for (Truth a : {Truth::No, Truth::Yes})
    for (Truth b : {Truth::No, Truth::Yes})
      for (Truth c : {Truth::No, Truth::Yes})
        if (all(a, b, c) == Truth::Maybe || any(a, b, c) == Truth::Maybe) return false;
  return true;
}

//: and so a ladder over a point reaches exactly one leaf
constexpr bool a_point_reaches_one_leaf() {
  constexpr int kUpto[] = {2, 5, 9};
  for (std::int64_t x = -3; x <= 12; ++x) {
    Reach<int, 4> r;
    const auto hole = [&] { r.hole(); };
    const auto leaf = [&](std::size_t i) { r.arm(static_cast<int>(i)); };
    ladder(Interval{x, x}, 1, kUpto, 3, hole, leaf);
    if (r.leaves != 1 || r.gap != (x < 1 || x > 9)) return false;
  }
  return true;
}

static_assert(primitives_decide_at_points(), "a primitive over a point must be Yes or No");
static_assert(a_point_reaches_one_leaf(), "a tree over a point must reach exactly one leaf");

}  // namespace detail

}  // namespace fsla::plan
