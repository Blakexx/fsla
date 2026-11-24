# The run-time environment (`csrc/fsla/src/common/plan/environment.{h,cu}`, `plan_api.cuh`)

The one place the CUDA runtime is asked for a hardware fact.

`device_facts(device)` reads `cudaGetDeviceProperties` and the two shared-memory attributes (per multiprocessor, per block opt-in) ONCE per
device and keeps the result (a map under a mutex): the integer facts by name, the model key (`cc`, `pciDeviceID`, `name`) and the two numbers the library's card
keeps (`cc`, `sms`). The read is settled through `device_descriptor<K>` ([`planner.md`](planner.md)) against EVERY kernel's contract, and a device outside one is refused there, by name,
with the model key beside the facts. Each kernel's device descriptor is that one read settled against that kernel's contract, once per device.

`descriptor_of<K>(device, call)` is the descriptor of a call of kernel `K` on a device; `checked_descriptor<K>(words, device)` re-checks the words an op carries
against `K`'s contract and refuses a descriptor read on another device than the one the call runs on. Both are templates over a kernel's generated definition, instantiated in
`environment.cu` for `decode` and `prefill` (`checked_descriptor`, `descriptor_of`) and, for the four entmax kernels, `descriptor_of`: each entry builds its kernel's facts from its own arguments, and the device read is settled against all six contracts.

<a id="entries"></a>
## The entries

`plan_api.cuh` declares what `fsla_api.cpp` binds, each taking the kernel's name: `fsla_call_facts(kernel)` (the kernel's call fact names, comma-separated, in the order the values go),
`fsla_descriptor(kernel, device, call)` (the descriptor's words) and `fsla_owed_kernels(kernel, device, call)` (the kernels a call inside an owed descriptor waits on). `fsla_decode_launch(descriptor, device)`, `fsla_decode_lattice(widths)` and `fsla_decode_lattice_box(widths, k, m)` are decode's
([`../../decode/decode_api.md`](../../decode/decode_api.md)). `fsla/ops/descriptor.py` is their Python face: `fsla.ops.descriptor.descriptor(kernel=, d_v=, levels=, decay=, batch_heads=, width_bits_1= .. width_bits_4=, ...)`
(the call facts of `hardware/kernels/<kernel>/schema.json`; `fsla.ops.produce._decode_descriptor` and `_prefill_descriptor` state them from a declaration) builds the validated, hashable `Descriptor`, which says which kernel it is of
(cached by kernel, device and call; a context
the kernel's contract holds only as OWED is refused as `KernelNotBuilt` naming its kernels, [`../../hardware/registry.md`](../../hardware/registry.md#the-owed-folder)). The calls build their descriptors themselves (`fsla.ops.produce`'s cached builders); none of these is on `fsla`'s top level.

`select.cu`'s admission of a card takes this read of the device (`device_facts`) and checks that the binary carries its build; the contract
is the one table of the driver's numbers.
