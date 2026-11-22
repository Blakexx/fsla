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

}  // namespace fsla::plan
