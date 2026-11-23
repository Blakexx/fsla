# The planner's host half (`csrc/fsla/src/common/plan/planner.h`)

Plain C++ written ONCE over a kernel's generated definition `K` (`plan/<kernel>/contract.h`: `K::Descriptor` the kernel's `Facts`, `K::Reader`, `K::fact_names`, `K::contract()`,
`K::owed()`; [`../../hardware/registry.md`](../../hardware/registry.md)), so every kernel's descriptors are made and checked by the same functions, as templates, and no kernel has a copy of them.
Templates because the facts type stays a concrete generated class, which the `constexpr` matchers read, and the planner needs no run-time dispatch over kernels. It turns what a driver reports
and what a call states into a descriptor of the kernel, or refuses.

- `device_descriptor<K>(read, model)`: the device's descriptor from the integer facts a driver reports (each DEVICE integer fact once, named). The
  sets (`tensor`, `instructions`) are not reported by any driver: the device's are those of the context that holds its numbers. A device no
  context of the kernel's contract or owed descriptors holds is refused.
- `describe<K>(device, call, model)`: the call's descriptor: the device's facts and the call's (each of the kernel's CALL facts once: a missing, doubled or foreign
  name is refused by name), then every fact checked against the contract.
- `words<K>` / `from_words<K>`: a descriptor as one word a fact in the kernel's order (an integer fact its value, a set its bits), which is what the op
  carries; `from_words` refuses a wrong count and a context outside the contract.
- `device_differs<K>`: the device facts a descriptor holds that differ from a device's: a descriptor is for the device it was read on.

<a id="the-contract-check"></a>
**THE CHECK IS BY FACTS, NEVER BY NAME.** A descriptor lies inside a kernel's contract when one context holds every fact: an integer fact within the
context's range, a set equal. Otherwise the refusal names the kernel and the nearest context (fewest facts outside, owed contexts counted) and each fact outside it, `name=value (the
contract: lo..hi)`, then the facts read and the model key. The model key (compute capability, PCI device id and name, the strings a driver
reports) is printed to help a reader tell a card from a slice or another SKU; it is never the match, because a name is text and a PCI id is
shared by slices.

<a id="the-reader"></a>
**THE READER.** A kernel's `Facts` keeps its members private; `Reader`, generated beside it, is the only friend. The functions of `common/plan/` read
through it and nothing else does: `tools/lint/lint_standards.py`'s `check_facts_reader_confined` refuses a file under `csrc/` outside
`common/plan/` that names it. A launcher holds the launch function's result, so a hardware fact reaches a launch only through the planner.

`plan_tool.cpp` is the planner on a synthetic environment ([`plan_tool.md`](plan_tool.md)); `environment.cu` is the device read
([`environment.md`](environment.md)).
