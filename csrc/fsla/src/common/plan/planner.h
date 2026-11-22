// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PLANNER'S HOST HALF (docs/internals/common/plan/planner.md): a descriptor from a device's read facts and a call's, checked against
// the contract, or a refusal naming the facts. Plain C++: the device read (environment.cu) and the synthetic environments of the tests
// (plan_tool.cpp) both come through it.
#pragma once

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "plan/contract.h"
#include "plan/facts.h"
#include "plan/rules.h"

namespace fsla::plan {

using Descriptor = Facts;

//: integer facts by name, the form a device's read and a call's facts arrive in
using Named = std::vector<std::pair<std::string_view, std::int64_t>>;

//: a descriptor, or the refusal that names why there is none
struct Planned {
  Descriptor descriptor;
  std::string refusal;

  bool ok() const { return refusal.empty(); }
};

//: which facts a check covers: the device's alone (a device read before any call), or every fact
enum class Scope { Device, Whole };

inline std::string spell_interval(Interval v) {
  return v.point() ? std::to_string(v.lo) : std::to_string(v.lo) + ".." + std::to_string(v.hi);
}

//: the facts of `d` in `scope` as `name=value`, a set as its bits in hex
inline std::string spell(const Descriptor& d, Scope scope = Scope::Whole) {
  std::string out;
  for (std::string_view name : kFactNames) {
    if (scope == Scope::Device && (is_call(name) || is_set(name))) continue;
    Interval v{};
    EnumSet s{};
    std::string cell(name);
    if (Reader::get(d, name, v)) {
      cell += "=" + spell_interval(v);
    } else if (Reader::get(d, name, s)) {
      char hex[24];
      std::snprintf(hex, sizeof hex, "0x%llx", static_cast<unsigned long long>(s.bits));
      cell += std::string("=") + hex;
    }
    out += (out.empty() ? "" : " ") + cell;
  }
  return out;
}

//: the facts of `d` the context `c` does not hold, as `name=value (the contract: lo..hi)`; the sets are compared only over the whole scope
inline std::vector<std::string> outside(const Descriptor& d, const Descriptor& c, Scope scope) {
  std::vector<std::string> out;
  for (std::string_view name : kFactNames) {
    if (is_set(name)) {
      EnumSet have{}, held{};
      Reader::get(d, name, have);
      Reader::get(c, name, held);
      if (scope == Scope::Whole && have != held) {
        out.push_back(std::string(name) + " differs from the contract's");
      }
      continue;
    }
    if (scope == Scope::Device && is_call(name)) continue;
    Interval have{}, held{};
    Reader::get(d, name, have);
    Reader::get(c, name, held);
    if (have.lo < held.lo || have.hi > held.hi) {
      std::string cell = std::string(name) + "=" + spell_interval(have);
      out.push_back(cell + " (the contract: " + spell_interval(held) + ")");
    }
  }
  return out;
}

//: `d` inside the contract: itself where one context holds it (a device's sets, which no driver reports, are the context's), else the
//: refusal naming the facts that lie outside the nearest context
inline Planned settle(const Descriptor& d, Scope scope, std::string_view what,
                      std::string_view model) {
  const ContractContext* nearest = nullptr;
  std::vector<std::string> nearest_outside;
  for (const ContractContext& context : contract()) {
    std::vector<std::string> out = outside(d, context.facts, scope);
    if (out.empty()) {
      Planned held{d, {}};
      if (scope == Scope::Device) {
        for (std::string_view name : kFactNames) {
          EnumSet set{};
          if (Reader::get(context.facts, name, set)) held.descriptor.assign(name, set);
        }
      }
      return held;
    }
    if (nearest == nullptr || out.size() < nearest_outside.size()) {
      nearest = &context;
      nearest_outside = out;
    }
  }
  std::string text = std::string(what) + " lies outside FSLA's contract. The nearest context '";
  text += std::string(nearest->id) + "' does not hold:";
  for (const std::string& fact : nearest_outside) text += " " + fact + ";";
  text += " the facts read: " + spell(d, scope);
  if (!model.empty()) text += "; the device: " + std::string(model);
  return {d, text};
}

//: THE DEVICE'S DESCRIPTOR from the facts a driver reports (`read`, each DEVICE integer fact once), refused when no context holds them
inline Planned device_descriptor(const Named& read, std::string_view model) {
  Descriptor d;
  std::vector<std::string_view> seen;
  for (const auto& [name, value] : read) {
    if (is_set(name) || is_call(name) || !d.assign(name, Interval{value, value})) {
      const std::string fact(name);
      const std::string why = "', which is no integer fact the device supplies";
      return {d, "the device read names '" + fact + why};
    }
    seen.push_back(name);
  }
  for (std::string_view name : kFactNames) {
    bool supplied = false;
    for (std::string_view s : seen) supplied |= s == name;
    if (!is_set(name) && !is_call(name) && !supplied) {
      const std::string fact(name);
      return {d, "the device read states no value for the schema's fact '" + fact + "'"};
    }
  }
  return settle(d, Scope::Device, "the device", model);
}

//: A CALL'S DESCRIPTOR: `device`'s facts and the call's (`call`, each CALL fact once), refused when no context holds them all
inline Planned describe(const Descriptor& device, const Named& call, std::string_view model) {
  Descriptor d = device;
  std::vector<std::string_view> seen;
  for (const auto& [name, value] : call) {
    const std::string fact(name);
    if (!is_call(name) || is_set(name)) {
      return {d, "the call names '" + fact + "', which is no fact a call supplies"};
    }
    for (std::string_view s : seen) {
      if (s == name) return {d, "the call supplies '" + std::string(name) + "' twice"};
    }
    d.assign(name, Interval{value, value});
    seen.push_back(name);
  }
  for (std::string_view name : kFactNames) {
    bool supplied = false;
    for (std::string_view s : seen) supplied |= s == name;
    if (is_call(name) && !supplied) {
      return {d, "the call states no value for '" + std::string(name) + "'"};
    }
  }
  return settle(d, Scope::Whole, "the call", model);
}

//: a descriptor as its words, one a fact in the schema's order: an integer fact as its value, a set as its bits (what the op carries)
inline std::vector<std::int64_t> words(const Descriptor& d) {
  std::vector<std::int64_t> out;
  for (std::string_view name : kFactNames) {
    Interval v{};
    EnumSet s{};
    if (Reader::get(d, name, v)) out.push_back(v.lo);
    if (Reader::get(d, name, s)) out.push_back(static_cast<std::int64_t>(s.bits));
  }
  return out;
}

//: the descriptor of `w` (`words`), refused when it has the wrong count or holds a context the contract does not
inline Planned from_words(const std::vector<std::int64_t>& w) {
  Descriptor d;
  if (w.size() != std::size(kFactNames)) {
    const std::string want = std::to_string(std::size(kFactNames));
    const std::string have = std::to_string(w.size());
    return {d, "a descriptor is " + want + " words, one a fact; this one has " + have};
  }
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (is_set(kFactNames[i])) {
      d.assign(kFactNames[i], EnumSet{static_cast<std::uint64_t>(w[i])});
    } else {
      d.assign(kFactNames[i], Interval{w[i], w[i]});
    }
  }
  return settle(d, Scope::Whole, "the descriptor", {});
}

//: the DEVICE facts `d` holds that differ from `device`'s, as `name=value (this device: value)`; empty where they are the same device's
inline std::vector<std::string> device_differs(const Descriptor& d, const Descriptor& device) {
  std::vector<std::string> out;
  for (std::string_view name : kFactNames) {
    if (is_call(name)) continue;
    Interval have{}, held{};
    EnumSet have_set{}, held_set{};
    if (Reader::get(d, name, have)) {
      Reader::get(device, name, held);
      if (have != held) {
        std::string cell = std::string(name) + "=" + spell_interval(have);
        out.push_back(cell + " (this device: " + spell_interval(held) + ")");
      }
    } else if (Reader::get(d, name, have_set)) {
      Reader::get(device, name, held_set);
      if (have_set != held_set) out.push_back(std::string(name) + " differs from this device's");
    }
  }
  return out;
}

}  // namespace fsla::plan
