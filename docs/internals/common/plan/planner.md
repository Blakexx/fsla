# The planner's host half (`csrc/fsla/src/common/plan/planner.h`)

Plain C++ over the generated `Facts` ([`../../hardware/registry.md`](../../hardware/registry.md)) and the contract (`plan/contract.h`, generated
from the registry: `contract()`, a context each). It turns what a driver reports and what a call states into a descriptor, or refuses.

- `device_descriptor(read, model)`: the device's descriptor from the integer facts a driver reports (each DEVICE integer fact once, named). The
  sets (`tensor`, `instructions`) are not reported by any driver: the device's are those of the context that holds its numbers. A device no
  context holds is refused.
- `describe(device, call, model)`: the call's descriptor: the device's facts and the call's (each CALL fact once: a missing, doubled or foreign
  name is refused by name), then every fact checked against the contract.
- `words` / `from_words`: a descriptor as one word a fact in the schema's order (an integer fact its value, a set its bits), which is what the op
  carries; `from_words` refuses a wrong count and a context outside the contract.
- `device_differs`: the device facts a descriptor holds that differ from a device's: a descriptor is for the device it was read on.

<a id="the-contract-check"></a>
**THE CHECK IS BY FACTS, NEVER BY NAME.** A descriptor lies inside the contract when one context holds every fact: an integer fact within the
context's range, a set equal. Otherwise the refusal names the nearest context (fewest facts outside) and each fact outside it, `name=value (the
contract: lo..hi)`, then the facts read and the model key. The model key (compute capability, PCI device id and name, the strings a driver
reports) is printed to help a reader tell a card from a slice or another SKU; it is never the match, because a name is text and a PCI id is
shared by slices.

<a id="the-reader"></a>
**THE READER.** `Facts` keeps its members private; `Reader`, generated beside it, is the only friend. The functions of `common/plan/` read
through it and nothing else does: `tools/lint/lint_standards.py`'s `check_facts_reader_confined` refuses a file under `csrc/` outside
`common/plan/` that names it. A launcher holds the launch function's result, so a hardware fact reaches a launch only through the planner.

`plan_tool.cpp` is the planner on a synthetic environment ([`plan_tool.md`](plan_tool.md)); `environment.cu` is the device read
([`environment.md`](environment.md)).
