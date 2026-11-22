# The run-time environment (`csrc/fsla/src/common/plan/environment.{h,cu}`, `plan_api.cuh`)

The one place the CUDA runtime is asked for a hardware fact.

`device_facts(device)` reads `cudaGetDeviceProperties` and the two shared-memory attributes (per multiprocessor, per block opt-in) ONCE per
device and keeps the result (a map under a mutex). The read goes through `device_descriptor` ([`planner.md`](planner.md)): a device outside the
contract is refused there, by name, with the model key (`cc`, `pciDeviceID`, `name`) beside the facts. The result carries the descriptor of
the device's facts, the model key, and the two numbers the library's card keeps (`cc`, `sms`).

`descriptor_of(device, call)` is the descriptor of a call on a device; `checked_descriptor(words, device)` re-checks the words an op carries
against the contract and refuses a descriptor read on another device than the one the call runs on.

<a id="entries"></a>
## The entries

`plan_api.cuh` declares what `fsla_api.cpp` binds: `fsla_call_facts` (the call's fact names, comma-separated, in the order the values go) and
`fsla_descriptor(device, call)` (the descriptor's words). `fsla_decode_launch(descriptor, device)` is decode's
([`../../decode/decode_api.md`](../../decode/decode_api.md)). `fsla/ops/descriptor.py` is their Python face: `fsla.descriptor(d_v=, levels=,
decay=)` builds the validated, hashable `Descriptor` (cached by device and call), `fsla.workspace_bytes(descriptor)` the scratch a call needs,
and a caller's own `workspace=` is checked for type, size, alignment and device before a launch.

`select.cu`'s admission of a card takes this read of the device (`device_facts`) and checks that the binary carries its build; the contract
is the one table of the driver's numbers.
