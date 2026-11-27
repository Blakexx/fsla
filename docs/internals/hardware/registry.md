# The contract registry (`hardware/`)

`hardware/` holds the PUBLIC CONTRACT: every run-time context FSLA accepts, a fact each (the device's and the call's), and nothing else
is supported. A context is never shaped by what a kernel can or cannot do: support is the matchers' business, and the build refuses a
contract context the arm half of a launch function maps to no arm ([`arms_gen.md`](../common/plan/arms_gen.md)). So a run-time refusal means one thing: the caller's
context is outside the contract.

THE FORM. The owner's rulings (2026-11-23): "a kernel is a thing that has compiled arms that we run, the desciproits describe the public contract that enters into
those arms, the matchers map that contract to arms. anything that compiles with a contract should have a descriptor"; "if the descriptors are created per kernel, then they should also be described per kernel right. sicne each kernel ahs
its own contract. so each kenrel gets its own descriptor registry"; "the descriptor is runtime context derived from hardware facts and user input. the matchers contain our logic that
distills that into an arm." And (2026-11-22) "each kernel just matahces on the fields it cares about on these immutable glboal hardware facts": the hardware facts are ONE shared table.
So the registry is one shared table of the DEVICE's facts and, beside it, a registry a kernel under `hardware/kernels/<kernel>/`:

```
hardware/schema.json                      the device's facts (the shared table's schema)
hardware/parts/device/                    the device parts: an architecture each
hardware/kernels/<kernel>/schema.json     the kernel's CALL facts
hardware/kernels/<kernel>/parts/          the kernel's parts (a call part, a depth part each)
hardware/kernels/<kernel>/descriptors/    the kernel's descriptors, with dev/ (narrowed contracts) and owed/
```

Seven kernels are registered: `decode` (built), `prefill` (OWED entire: its kernels are not built, so every descriptor of it is in `owed/` and its contract holds no context), `mask_prepare` (the prefill's mask-preparation pass, built: its call facts are `width`, the widest masked level-side of the call, 2 to 256, and `levels`, the density's depth, 1 to 4; its one `call` part and its one Hopper descriptor state the contract, [`../mask_prepare/mask_prepare.md#the-kernel`](../mask_prepare/mask_prepare.md#the-kernel); it declares no arm family, `mask_prepare` checks the call against its contract by `descriptor_of`) and the producer's four solves, one registry for each kernel template (`factor_forward`, `factor_backward`, `softmax_forward`, `softmax_backward`: "One per kernel", the owner, 2026-11-23). A solve's call facts are `width`, `levels`, `alpha_tenths` (not the softmax kernels), `logit_dtype` and `value_dtype`; its one `call` part and its one Hopper descriptor state the contract ([`../entmax/entmax.md#the-contract`](../entmax/entmax.md#the-contract)).

<a id="schema"></a>

## The schemas

`hardware/schema.json` declares the DEVICE facts, each its name, its type (`int`, or `set<enum>` with its members), its unit and its read (`DEVICE`: the driver's attributes). A kernel's
`schema.json` declares that kernel's CALL facts in the same form, each read `CALL` (the call's own data): `decode` states `d_v`, `levels`, `decay`, `batch_heads`, `width_bits_1..4` and
`write_listed_1..4`; `prefill` states `d_v`, `levels`, `decay`, `tokens`, `batch_heads`, `stateful`, `width_bits_1..4`. A fact belongs to the table or to
a kernel, never both. `hardware/registry.py` refuses a schema with a duplicate name, an unknown type or read, no unit, a set without members, or a fact of the other side's read. A kernel's CONTEXT holds the device's facts and then
its own call facts (`registry.facts(kernel)`), and `emit.py --kernel <k>` writes the generated `plan/<k>/facts.h`: the kernel's `Facts` class, one member per fact (an `Interval`, an `EnumSet`), its reader and its fact names, so the arm half of
a launch function is written over the names its kernel's registry states ([`../common/plan/rules.md`](../common/plan/rules.md)). A fact a kernel does not state is a fact its matcher cannot read: decode's descriptor carries no read-side walk and no token count.

`sm_count` is a DEVICE fact of the Hopper context as a range (114 to 132, the H100 PCIe and SXM parts' counts, from NVIDIA's
documentation); the first reviewed device-probe part ([below](#probe)) restates it as measured. `l2_bytes` is not a fact of the device table
yet: a context states a value for every fact and the contract must build at every commit, so the table gains it in the same commit as
that reviewed part. The probe reads both already and a candidate part carries them.

<a id="parts"></a>

## Parts, descriptors and contexts

A PART states facts, each `{"value", <provenance>}` with exactly one of
`measured_by` (a lab record), `validated_by` (the test or check that holds it), `documented_by` or `declared_by` (the family's own declaration).
A value is a point, a range `{"lo", "hi"}`, a set `{"set": [members]}`, or a list of those: a list is ALTERNATIVES, each its own context. A DEVICE part (`parts/device/hopper_sm90a.json`, named `device/hopper_sm90a`) states device facts; a KERNEL's part
(`kernels/decode/parts/call.json`, `depth1.json`, named `call`, `depth1`) states that kernel's call facts, and a part stating a fact of the other side is refused.

A DESCRIPTOR (`kernels/decode/descriptors/hopper_decode_d1.json`) names the parts it composes, ONE device part of the shared table and parts of its own kernel, and the GPUs it
covers (`gpus`, the README's label); the architecture it is built for is its one device part's (`registry.contexts(kernel)` derives it, and refuses a descriptor whose parts hold no device part or two). `registry.contexts(kernel)` expands every descriptor of a kernel into FULL CONTEXTS: the parts' facts merged (a fact stated by two
parts, a fact neither schema names, a fact of the kernel's context no part states: refused), the alternatives crossed. A matcher and the run time see
contexts only. Two contexts of a kernel identical in every fact are refused, whatever descriptors made them.

THE DECODE CALL'S FACTS (all ints; the schemas hold each one's unit and doc). Its extents are `d_v`, `batch_heads` (B times H) and `levels`; its dial is `decay`; the state is the op's own argument (`fsla.dense_state()`) and the memory budget the layer's, so neither is a fact. A decode step is one token over a state in and out, which is not a fact of its contract. Its per-level facts are the log2 of each level's width, `width_bits_1` .. `width_bits_4` (a level's width is a
power of two, `fsla/_state.py`'s `StateFormat`, so a range of the log2 states every width exactly; 0 for a level the call does not have) and the write side's walk of each level,
`write_listed_1..4` (1 when that side of that level is WALKED BY ITS LISTS, 0 when every digit is walked: an action, chosen by the producer's rule `fsla.routing.types.activation_walks` from the declaration -- a side whose activation has exact zeros is listed -- and any valid walk is lawful, a wrong one costing time and never the result). There is no `write_dense` fact: decode's launch
function derives it (the innermost level's `write_listed` is 0). The prefill's call states in addition `tokens` (the sequence length T) and `stateful` (a state in or out), and no walk of any level: the densities of the two sides (the share of a level's digits a token is live on) are the call's own input beside the masks, so the host never reads them and the prefill's descriptor does not state them ([`../common/plan/prefill_launch.md`](../common/plan/prefill_launch.md)).
The depth and the per-level facts are correlated and a part's lists
are crossed, so the correlation is stated by composition: a kernel's `call` part holds what does not depend on the depth, and its `depth1` .. `depth4` each state `levels`
as a point, the four `width_bits` (a range from the width floor, `MIN_LEVEL_WIDTH`, up to the widest level the decode path holds, `kDecodeMaxLevelWidth`, for a level it has, the point 0 beyond) and the
walk facts (a range 0..1 for a level the depth has, the point 0 beyond); each kernel has depth parts of its own. The decode
contract is four descriptors, `hopper_decode_d1` .. `hopper_decode_d4`, each composing the device part, `call` and its depth part. A call whose fact is outside its range is refused when the
descriptor is made, naming the fact (`width_bits_1=3 (the contract: 4..8)`), by the same check as every other fact ([`../common/plan/planner.md`](../common/plan/planner.md#the-contract-check)).
A decode step serves `batch_heads` from 1 to 536870911 (`docs/api.md#batch-heads`); the planner test holds both ends.

<a id="architectures"></a>A DEVICE PART STATES AN ARCHITECTURE WHOLE: beside its facts it names its architecture (`arch`, `sm_90a`, the name it is compiled as, whose compute
capability its `cc` is) and its family (`family`, `hopper`) -- fields of the part, not facts. THE ONE SOURCE: the registry is the only place a device fact, an architecture
or its family is stated, ruled by the owner (2026-11-22): "isnt the descritpr a thing we will want to reuse for other kernels, not just carry? each kernel just
matahces on the fields it cares about on these immutable glboal hardware facts". The contract is what a device is admitted against
(`common/plan/environment.cu`); the same parts generate the kernels' header `common/hardware.cuh` (`registry.arch_header`, `emit.py --arch-header`: one row an
architecture, the `Caps` fields a kernel source reads, the family and the build's traits, [`../common/hardware.md`](../common/hardware.md)). An architecture is BUILT when a device part states it
(`registry.built`, with `cc_of`, `arch_of`, `device_arch` and `builds_header`, which the build's flags, the SASS tools and the record read); an architecture no part
states is neither built nor tabulated, and adding one is adding a device part ([`../../bringup.md`](../../bringup.md)). The environment repository that computes the built architectures watches the registry's modules and
the device-parts directory as a tree, so a device part added or changed is a new set of built architectures (`MODULE.bazel`). The window is the prefill matcher's, not a fact of a part
([`../common/window.md`](../common/window.md)). The measured rates and the products are measurement inputs, `measure/rates.json`.

MIG and MPS register no context: neither is documented to report a slice's own SM count, L2 share or shared memory, so a context for them waits
for a measurement on a real slice and until then the run time refuses them.

<a id="probe"></a>

## Onboarding a device: the probe

A device enters the contract by a MEASURED part, in four steps (the reviewer also names its `arch` and `family`):

1. RUN THE PROBE on the device: `bazel run //measure/lab:device_probe.run`, which stores its record, then
   `bazel run //tools:lab -- page device_probe --instance LABEL` ([`device_probe`](../../lab/device_probe.md)). The probe reads every DEVICE fact of the device table
   (the table's DEVICE facts, and `sm_count` and `l2_bytes` for the table's coming entries) through the properties and the attributes, checks the two routes
   and the device's own registers agree, and the page step writes `hardware/parts/device/<id>.candidate.json` from the record. The `<id>` is the compute capability and the
   driver's name (`cc900_nvidia_h100_80gb_hbm3`); the PCI slot the driver reports (`pciDeviceID` is the slot, not the product id) is the machine's, so it is not in the key.
2. REVIEW THE CANDIDATE. Each value cites its row of the record (`measured_by`); `tensor` is the MMA kinds of `measure/rates.json`'s architecture of the same compute capability and `instructions` the registry's device part of it
   (`documented_by`), and each is left out, named in `unestablished`, when there is none: the reviewer supplies them from a lab experiment or a document. MIG and MPS
   slices are not probed (see above).
3. COMMIT THE PART: rename `<id>.candidate.json` to `<id>.json` (`hardware/parts/device/`), add to each kernel that serves the device a descriptor composing it with that kernel's parts, and commit. THE REGISTRY REFUSES
   A CANDIDATE FILE (`registry.CANDIDATE`): `registry.check()` and `registry.contexts(kernel)` fail naming it, so a candidate never reaches the contract unreviewed.
4. THE BUILD'S CONTRACT CHECK THEN COVERS IT: the part's contexts are in `//hardware:decode_contexts`, the kernel's matcher is evaluated over them, and a context mapped to no arm
   fails the build ([`arms_gen.md`](../common/plan/arms_gen.md)).

<a id="narrowed"></a>

## A narrowed contract (development builds)

A development build may compile a subset of ONE KERNEL by naming a DECLARED NARROWED CONTRACT: a descriptor of that kernel's registry under `kernels/<kernel>/descriptors/dev/<name>.json`, marked
`"narrowed": true`, composing a subset of the kernel's call parts (`kernels/decode/parts/call_narrow_d32.json` and one depth part, `depth2`) beside the device part: a descriptor holds one depth, so the narrowed contract
`decode_d32_l2` is depth 2 alone. The build setting
`--//:contract=<name>` (default `full`) chooses it: the kernel whose registry declares the name (`registry.narrowing_kernel`) is narrowed, and `//hardware:decode_contexts` and `//hardware:decode_contract` are then written for it
(`hardware/contract.bzl`; `registry.contexts(kernel, contract=<name>)`), so `arms_gen` compiles exactly the arms the narrowed contract reaches and the run time,
holding the narrowed contract, refuses a call outside it, naming it (`kContractName`, `planner.h`'s refusal). Every other kernel keeps its full contract. The registry refuses a name no kernel declares or that several declare, a descriptor not marked narrowed, and a narrowed context outside every full context of its kernel: a narrowed contract promises less than
the full one, never more. A narrowed descriptor is never part of the full contract (a kernel's `descriptors/*.json` alone are), so the README's table and the
duplicate check read the full one. The full contract is enforced by `//:gate` (its `full_contract` test fails, naming the contract, under any other
setting) and by releases (the build record says `iteration: True`, which `bazel run //:wheels` refuses). `tests/unit/test_narrowed_contract.py` holds all of it.

<a id="the-owed-folder"></a>

## The owed folder: a descriptor that exists before its arm

`kernels/<kernel>/descriptors/owed/<name>.json` is a THIRD folder beside the kernel's full contract (`descriptors/*.json`) and its narrowed ones (`descriptors/dev/`). A descriptor there composes parts exactly as any
other (the prefill's `call` and a depth part: `hopper_prefill_d1` .. `hopper_prefill_d4`) and `registry.check` validates it against the kernel's schema, but it is NEVER in the full contract, docs/support.md's contract tables,
a narrowed contract or the arms the build compiles. It has one more field, `"owed": ["<kernel>", ...]`: the kernels its call waits on, the names `fsla.ops._ext.not_built` and the tests'
`requires_kernel` use. The registry refuses an owed descriptor reachable from its kernel's full contract (a context in both), so an owed call cannot be promised and owed at once. A kernel whose descriptors are all owed, as the prefill's are, holds no built context.

At run time `//hardware:<kernel>_contract` generates `plan/<kernel>/contract.h` from the registry (no hand-typed copy), owed contexts beside the contract, and the planner, finding a context outside the kernel's contract but inside one of its owed descriptors, refuses it
naming each owed kernel instead of saying it lies outside the contract; `fsla.ops.descriptor.descriptor` turns that into `KernelNotBuilt` (through the host op `fsla_owed_kernels`), so a test marked
`requires_kernel("prefill")` is a strict xfail through the surface. A context in neither is refused as every call is, naming the facts. A device alone is held by a context of the contract or of an owed descriptor, so the prefill's device descriptor settles against its owed contexts. A prefill descriptor leaves the owed folder when the prefill launch
function and its arm are built (`docs/open-work.md` rows A-f and A-h).

<a id="outputs"></a>

## What reads it

| reader | reads |
|---|---|
| `//hardware:<kernel>_facts` | the device table and the kernel's schema: `plan/<kernel>/facts.h` |
| `//hardware:<family>_contexts` | the family's contract under the build's `--//:contract` (decode and the four entmax kernels, the `ARM_FAMILIES`), a line a context (`registry.contexts_text`): `arms_gen`'s input |
| `//hardware:<kernel>_contract` | the kernel's contract and owed contexts as tables the run time holds, and its `Kernel` definition: `plan/<kernel>/contract.h` (`registry.contract_header`), which the planner checks every descriptor against |
| `tools/supported.py` | each kernel's contract: the README's summary, a row a kernel, architecture and toolchain in a reader's terms (the card, the depth, a level's width, the value width, the activations, the toolchain, built or not), and docs/support.md's tables, a table a kernel with its call facts as columns |
