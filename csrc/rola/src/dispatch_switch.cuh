#pragma once

// HOST-SIDE COMPILE-TIME DISPATCH, AS FUNCTION TEMPLATES TAKING A GENERIC
// LAMBDA.
// see docs/internals/dispatch_switch.md#note-l3

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
