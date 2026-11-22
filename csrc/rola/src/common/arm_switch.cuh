// csrc/rola/src/common/arm_switch.cuh -- the HOST dispatch over the shipped arm set,
// the host twin of common/structure_switch.cuh's `uniform_switch`. CONSTITUTION.md §R9's four layers,
// host side: see docs/internals/common/arm_switch.md
#pragma once

#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/arm.cuh"

namespace rola {

//: ONE FAMILY'S HOST ARM SET, defined from the family's generated selection header (`<family>_selection.inc`, included
//: before this macro is used): `ROLA_DEFINE_ARM_SET(DecodeArmSet, DECODE)` defines the struct `arm_switch` takes -- what THIS
//: BINARY carries and what the declaration declares (both, because the refusal tells an unbuilt arm from an undeclared one),
//: the visit over the built arms, the built set spelled and as rows, and the field that has no member at all.
//: -- see docs/internals/common/arm_switch.md#arm-set
#define ROLA_ARM_VISIT(I, ...)                                                          \
  if (matches(k, ::rola::Arm<I, __VA_ARGS__>::values)) {                                       \
    ++hits;                                                                             \
    if (hits == 1) body(::rola::Arm<I, __VA_ARGS__>{});                                         \
  }
#define ROLA_ARM_NAME(I, ...)                    \
  out += (out.empty() ? "" : ", ");              \
  out += spell(::rola::Arm<I, __VA_ARGS__>::values);
#define ROLA_ARM_ROW(I, ...) \
  out.emplace_back(std::begin(::rola::Arm<I, __VA_ARGS__>::values), std::end(::rola::Arm<I, __VA_ARGS__>::values));
#define ROLA_ARM_FIELD(I, ...)                                          \
  for (int i = 0; i < arity; ++i) seen[i] = seen[i] || k.v[i] == ::rola::Arm<I, __VA_ARGS__>::values[i];

#define ROLA_DEFINE_ARM_SET(NAME, UPPER)                                                              \
  struct NAME {                                                                                       \
    static constexpr int count = ROLA_##UPPER##_ARM_COUNT;                                            \
    static constexpr int declared = ROLA_##UPPER##_ARM_DECLARED;                                      \
    static constexpr const char* family = ROLA_##UPPER##_ARM_FAMILY;                                  \
    static constexpr const char* key = ROLA_##UPPER##_ARM_KEY;                                        \
    static constexpr const char* fields[] = {ROLA_##UPPER##_ARM_FIELDS};                              \
    static constexpr int arity = sizeof(fields) / sizeof(fields[0]);                                  \
    using Key = ::rola::ArmKey<arity>;                                                                        \
                                                                                                      \
    static bool matches(const Key& k, const int (&row)[arity]) {                                      \
      for (int i = 0; i < arity; ++i)                                                                 \
        if (k.v[i] != row[i]) return false;                                                           \
      return true;                                                                                    \
    }                                                                                                 \
    static std::string spell(const int (&row)[arity]) {                                               \
      std::string out = "(";                                                                          \
      for (int i = 0; i < arity; ++i) out += (i ? ", " : "") + std::to_string(row[i]);                \
      return out + ")";                                                                               \
    }                                                                                                 \
                                                                                                      \
    template <class Body>                                                                             \
    static int visit(const Key& k, Body&& body) {                                                     \
      (void)k;                                                                                        \
      (void)body;                                                                                     \
      int hits = 0;                                                                                   \
      ROLA_##UPPER##_ARM_SET_X(ROLA_ARM_VISIT)                                                        \
      return hits;                                                                                    \
    }                                                                                                 \
                                                                                                      \
    /* THE BUILT SET, SPELLED, so a refusal is one a caller can act on. */                            \
    static std::string members() {                                                                    \
      std::string out;                                                                                \
      ROLA_##UPPER##_ARM_SET_X(ROLA_ARM_NAME)                                                         \
      return out.empty() ? std::string("none") : out;                                                 \
    }                                                                                                 \
                                                                                                      \
    /* THE BUILT SET AS ROWS, each an arm's key: the binary's own answer to what it carries. */       \
    static std::vector<std::vector<int64_t>> rows() {                                                 \
      std::vector<std::vector<int64_t>> out;                                                          \
      ROLA_##UPPER##_ARM_SET_X(ROLA_ARM_ROW)                                                          \
      return out;                                                                                     \
    }                                                                                                 \
                                                                                                      \
    /* WHICH FIELD HAS NO MEMBER AT ALL -- see docs/internals/common/arm_switch.md#offending-field */ \
    static std::string offending_field(const Key& k) {                                                \
      (void)k;                                                                                        \
      bool seen[arity] = {};                                                                          \
      ROLA_##UPPER##_ARM_SET_X(ROLA_ARM_FIELD)                                                        \
      for (int i = 0; i < arity; ++i)                                                                 \
        if (!seen[i]) return fields[i];                                                               \
      return "the combination";                                                                       \
    }                                                                                                 \
  }

//: ONE HOST BRANCH OVER A GENERATED ARM SET; an empty set refuses every call by name.
//: -- see docs/internals/common/arm_switch.md#arm-switch
template <class ArmSet, class Body>
inline void arm_switch(const typename ArmSet::Key& k, Body&& body) {
  static_assert(ArmSet::count >= 0, "an arm set carries a member count");
  static_assert(ArmSet::count <= ArmSet::declared,
                "a build carries a subset of the declared arms; a set larger than the "
                "declaration means the selection and the declaration disagree");
  const int hits = ArmSet::visit(k, std::forward<Body>(body));
  const std::string at = [&] {
    std::string out = "(";
    for (int i = 0; i < ArmSet::arity; ++i) out += (i ? ", " : "") + std::to_string(k.v[i]);
    return out + ")";
  }();
  const std::string where = std::string(ArmSet::family) + " arm for " + ArmSet::key + " = " + at;
  STD_TORCH_CHECK(hits <= 1, "the ", where, " is covered ", hits, " times; an arm set is a set");
  STD_TORCH_CHECK(hits == 1, "this build carries no ", where, "; the field that misses is ",
                  ArmSet::offending_field(k), ". It carries ", ArmSet::count, " of ",
                  ArmSet::declared, " declared arms: ", ArmSet::members());
}

}  // namespace rola
