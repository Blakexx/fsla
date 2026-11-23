// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE PLANNER'S HOST HALF (docs/internals/common/plan/planner.md): a descriptor from a device's read facts and a call's, checked against
// a kernel's contract, or a refusal naming the facts. Written once over a kernel's generated definition `K` (`plan/<kernel>/contract.h`:
// `K::Descriptor`, `K::Reader`, its fact names, its contract and owed contexts), so every kernel is served by the same code. Plain C++: the device read
// (environment.cu) and the synthetic environments of the tests (plan_tool.cpp) both come through it.
#pragma once

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "plan/rules.h"

namespace fsla::plan {

//: a context of a kernel's contract: its id and its facts (`F` the kernel's `Facts`)
template <typename F>
struct ContractContext {
  const char* id;
  F facts;
};

//: a context a kernel's contract owes: its id, its facts and the kernels the call waits on
template <typename F>
struct OwedContext {
  const char* id;
  F facts;
  std::vector<const char*> kernels;
};

//: integer facts by name, the form a device's read and a call's facts arrive in
using Named = std::vector<std::pair<std::string_view, std::int64_t>>;

//: a descriptor of kernel `K`, or the refusal that names why there is none; `owed` names the kernels the call waits on when the refusal is that
//: the call lies inside an owed descriptor (empty for any other refusal)
template <typename K>
struct Planned {
  typename K::Descriptor descriptor;
  std::string refusal;
  std::vector<std::string> owed;

  bool ok() const { return refusal.empty(); }
};

//: which facts a check covers: the device's alone (a device read before any call), or every fact
enum class Scope { Device, Whole };

inline std::string spell_interval(Interval v) {
  return v.point() ? std::to_string(v.lo) : std::to_string(v.lo) + ".." + std::to_string(v.hi);
}

//: the facts of `d` in `scope` as `name=value`, a set as its bits in hex
template <typename K>
std::string spell(const typename K::Descriptor& d, Scope scope = Scope::Whole) {
  std::string out;
  for (std::string_view name : K::fact_names) {
    if (scope == Scope::Device && (K::is_call(name) || K::is_set(name))) continue;
    Interval v{};
    EnumSet s{};
    std::string cell(name);
    if (K::Reader::get(d, name, v)) {
      cell += "=" + spell_interval(v);
    } else if (K::Reader::get(d, name, s)) {
      char hex[24];
      std::snprintf(hex, sizeof hex, "0x%llx", static_cast<unsigned long long>(s.bits));
      cell += std::string("=") + hex;
    }
    out += (out.empty() ? "" : " ") + cell;
  }
  return out;
}

//: the facts of `d` the context `c` does not hold, as `name=value (the contract: lo..hi)`; the sets are compared only over the whole scope
template <typename K>
std::vector<std::string> outside(const typename K::Descriptor& d, const typename K::Descriptor& c,
                                 Scope scope) {
  std::vector<std::string> out;
  for (std::string_view name : K::fact_names) {
    if (K::is_set(name)) {
      EnumSet have{}, held{};
      K::Reader::get(d, name, have);
      K::Reader::get(c, name, held);
      if (scope == Scope::Whole && have != held) {
        out.push_back(std::string(name) + " differs from the contract's");
      }
      continue;
    }
    if (scope == Scope::Device && K::is_call(name)) continue;
    Interval have{}, held{};
    K::Reader::get(d, name, have);
    K::Reader::get(c, name, held);
    if (have.lo < held.lo || have.hi > held.hi) {
      std::string cell = std::string(name) + "=" + spell_interval(have);
      out.push_back(cell + " (the contract: " + spell_interval(held) + ")");
    }
  }
  return out;
}

//: `d` inside the kernel's contract: itself where one context holds it (a device's sets, which no driver reports, are the context's), else the
//: refusal naming the facts that lie outside the nearest context. A device alone is held by a context of the contract or of an owed descriptor: the
//: device is the same whether the kernel's call is built or owed
template <typename K>
Planned<K> settle(const typename K::Descriptor& d, Scope scope, std::string_view what,
                  std::string_view model) {
  const typename K::Descriptor* nearest = nullptr;
  const char* nearest_id = "";
  std::vector<std::string> nearest_outside;

  const auto consider = [&](const char* id, const typename K::Descriptor& facts) {
    std::vector<std::string> out = outside<K>(d, facts, scope);
    if (!out.empty()) {
      if (nearest == nullptr || out.size() < nearest_outside.size()) {
        nearest = &facts;
        nearest_id = id;
        nearest_outside = out;
      }
      return false;
    }
    return true;
  };
  const auto held_by = [&](const typename K::Descriptor& context) {
    Planned<K> held{d, {}, {}};
    if (scope == Scope::Device) {
      for (std::string_view name : K::fact_names) {
        EnumSet set{};
        if (K::Reader::get(context, name, set)) held.descriptor.assign(name, set);
      }
    }
    return held;
  };

  for (const ContractContext<typename K::Descriptor>& context : K::contract()) {
    if (consider(context.id, context.facts)) return held_by(context.facts);
  }

  //: the owed contexts: a device alone is held by one as by a context of the contract; a whole call inside one is owed, and one inside none of them is outside
  //: them as it is outside the contract
  std::vector<std::string> kernels;
  std::string ids;
  for (const OwedContext<typename K::Descriptor>& context : K::owed()) {
    if (!consider(context.id, context.facts)) continue;
    if (scope == Scope::Device) return held_by(context.facts);
    ids += (ids.empty() ? "'" : ", '") + std::string(context.id) + "'";
    for (const char* kernel : context.kernels) {
      bool listed = false;
      for (const std::string& k : kernels) listed |= k == kernel;
      if (!listed) kernels.push_back(kernel);
    }
  }
  if (!kernels.empty()) {
    std::string names;
    for (const std::string& k : kernels) names += (names.empty() ? "" : ", ") + k;
    return {d,
            std::string(what) + " lies inside the contract's OWED descriptors (" + ids
                + "): the kernels it waits on are not built: " + names
                + ". Nothing guarantees it until they are",
            kernels};
  }

  std::string text = std::string(what) + " lies outside FSLA's " + std::string(K::name) + " ";
  if (K::narrowed) {
    text += "NARROWED contract '" + std::string(K::contract_name)
            + "' (a development build; the full contract is enforced in the gate and in releases)";
  } else {
    text += "contract";
  }
  if (nearest == nullptr) return {d, text + ", which holds no context"};
  text += ". The nearest context '" + std::string(nearest_id) + "' does not hold:";
  for (const std::string& fact : nearest_outside) text += " " + fact + ";";
  text += " the facts read: " + spell<K>(d, scope);
  if (!model.empty()) text += "; the device: " + std::string(model);
  return {d, text};
}

//: THE DEVICE'S DESCRIPTOR from the facts a driver reports (`read`, each DEVICE integer fact once), refused when no context holds them
template <typename K>
Planned<K> device_descriptor(const Named& read, std::string_view model) {
  typename K::Descriptor d;
  std::vector<std::string_view> seen;
  for (const auto& [name, value] : read) {
    if (K::is_set(name) || K::is_call(name) || !d.assign(name, Interval{value, value})) {
      const std::string fact(name);
      const std::string why = "', which is no integer fact the device supplies";
      return {d, "the device read names '" + fact + why};
    }
    seen.push_back(name);
  }
  for (std::string_view name : K::fact_names) {
    bool supplied = false;
    for (std::string_view s : seen) supplied |= s == name;
    if (!K::is_set(name) && !K::is_call(name) && !supplied) {
      const std::string fact(name);
      return {d, "the device read states no value for the schema's fact '" + fact + "'"};
    }
  }
  return settle<K>(d, Scope::Device, "the device", model);
}

//: A CALL'S DESCRIPTOR: `device`'s facts and the call's (`call`, each CALL fact once), refused when no context holds them all
template <typename K>
Planned<K> describe(const typename K::Descriptor& device, const Named& call,
                    std::string_view model) {
  typename K::Descriptor d = device;
  std::vector<std::string_view> seen;
  for (const auto& [name, value] : call) {
    const std::string fact(name);
    if (!K::is_call(name) || K::is_set(name)) {
      return {d, "the call names '" + fact + "', which is no fact a " + std::string(K::name)
                     + " call supplies"};
    }
    for (std::string_view s : seen) {
      if (s == name) return {d, "the call supplies '" + std::string(name) + "' twice"};
    }
    d.assign(name, Interval{value, value});
    seen.push_back(name);
  }
  for (std::string_view name : K::fact_names) {
    bool supplied = false;
    for (std::string_view s : seen) supplied |= s == name;
    if (K::is_call(name) && !supplied) {
      return {d, "the call states no value for '" + std::string(name) + "'"};
    }
  }
  return settle<K>(d, Scope::Whole, "the call", model);
}

//: a descriptor as its words, one a fact in the kernel's order: an integer fact as its value, a set as its bits (what the op carries)
template <typename K>
std::vector<std::int64_t> words(const typename K::Descriptor& d) {
  std::vector<std::int64_t> out;
  for (std::string_view name : K::fact_names) {
    Interval v{};
    EnumSet s{};
    if (K::Reader::get(d, name, v)) out.push_back(v.lo);
    if (K::Reader::get(d, name, s)) out.push_back(static_cast<std::int64_t>(s.bits));
  }
  return out;
}

//: the descriptor of `w` (`words`), refused when it has the wrong count or holds a context the kernel's contract does not
template <typename K>
Planned<K> from_words(const std::vector<std::int64_t>& w) {
  typename K::Descriptor d;
  if (w.size() != std::size(K::fact_names)) {
    const std::string want = std::to_string(std::size(K::fact_names));
    const std::string have = std::to_string(w.size());
    return {d, "a " + std::string(K::name) + " descriptor is " + want
                   + " words, one a fact; this one has " + have};
  }
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (K::is_set(K::fact_names[i])) {
      d.assign(K::fact_names[i], EnumSet{static_cast<std::uint64_t>(w[i])});
    } else {
      d.assign(K::fact_names[i], Interval{w[i], w[i]});
    }
  }
  return settle<K>(d, Scope::Whole, "the descriptor", {});
}

//: the DEVICE facts `d` holds that differ from `device`'s, as `name=value (this device: value)`; empty where they are the same device's
template <typename K>
std::vector<std::string> device_differs(const typename K::Descriptor& d,
                                        const typename K::Descriptor& device) {
  std::vector<std::string> out;
  for (std::string_view name : K::fact_names) {
    if (K::is_call(name)) continue;
    Interval have{}, held{};
    EnumSet have_set{}, held_set{};
    if (K::Reader::get(d, name, have)) {
      K::Reader::get(device, name, held);
      if (have != held) {
        std::string cell = std::string(name) + "=" + spell_interval(have);
        out.push_back(cell + " (this device: " + spell_interval(held) + ")");
      }
    } else if (K::Reader::get(d, name, have_set)) {
      K::Reader::get(device, name, held_set);
      if (have_set != held_set) out.push_back(std::string(name) + " differs from this device's");
    }
  }
  return out;
}

}  // namespace fsla::plan
