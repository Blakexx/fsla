#pragma once

// HOST-SIDE COMPILE-TIME DISPATCH, AS FUNCTION TEMPLATES TAKING A GENERIC LAMBDA.
//
// A kernel with `N` structural template axes has to be selected at runtime from `N`
// runtime values, and the shape that selection takes is a real engineering choice.
// The form this project refuses is the MACRO LADDER: a `#define` per axis whose body
// restates the launch, so an `N`-axis dispatch spells the launch `2^N` times inside a
// preprocessor expansion the compiler cannot type-check, the debugger cannot step and
// a reader cannot see. It was measured to be exactly the place a `<<<...>>>` argument
// drifts between arms.
//
// The form here is flash-attention's `BOOL_SWITCH` idiom with the macro removed: the
// axis walker is a function template, the launch is written ONCE as a generic lambda,
// and each axis arrives as an `integral_constant` whose `::value` is a compile-time
// constant at the point of use.
//
// WHAT THIS DOES AND DOES NOT CHANGE. The set of DEVICE instantiations is identical --
// the same template arguments are named, so the fatbin carries the same entry points
// and the ratification manifest is unmoved, which is a gate and not a hope. The HOST
// selection code is genuinely different (a comparison chain over the candidate set
// rather than a `switch` statement), it is not device code, and no SASS comparison can
// speak for it; what covers it is the CUDA suites, which exercise every arm.
//
// The idiom, what it replaced, and the axis-order rule: docs/internals/dispatch_switch.md

#include <type_traits>
#include <utility>

#include <torch/extension.h>

namespace rola {

//: ONE BOOLEAN AXIS. `body` is called with `std::true_type` or `std::false_type`, so
//: `decltype(FLAG)::value` is a template argument at the call site.
template <typename Body>
inline void bool_switch(bool value, Body&& body) {
  if (value) {
    body(std::true_type{});
  } else {
    body(std::false_type{});
  }
}

//: ONE INTEGER AXIS OVER A CLOSED SET. The candidate values are template arguments,
//: so the set is part of the call and an unmatched runtime value is a REFUSAL rather
//: than a silently missing arm -- `what` names the axis in the message.
template <int... VALUES, typename Body>
inline void int_switch(int value, const char* what, Body&& body) {
  bool matched = false;
  //: An expansion over the candidate set, evaluated left to right: exactly one arm
  //: runs. The sets in this engine are two to six values wide, so the chain is not a
  //: jump table and does not need to be.
  (void)std::initializer_list<int>{
      (value == VALUES ? (body(std::integral_constant<int, VALUES>{}), matched = true, 0) : 0)...};
  TORCH_CHECK(matched, what, " is outside the instantiated set, got ", value);
}

}  // namespace rola
