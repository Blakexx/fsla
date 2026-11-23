# The contract registry (`hardware/`)

`hardware/` holds the PUBLIC CONTRACT: every run-time context FSLA accepts, a fact each (the device's and the call's), and nothing else
is supported. A context is never shaped by what a kernel can or cannot do: support is the matchers' business, and the build refuses a
contract context the arm half of a launch function maps to no arm ([`arms_gen.md`](../decode/arms_gen.md)). So a run-time refusal means one thing: the caller's
context is outside the contract.

<a id="schema"></a>

## The schema

`hardware/schema.json` declares every fact: its name, its type (`int`, or `set<enum>` with its members), its unit and where it is read at run
time, the `DEVICE` (the driver's attributes) or the `CALL` (the call's own data). `hardware/registry.py` refuses a schema with a duplicate
name, an unknown type or read, no unit, or a set without members. `emit.py` writes the generated `plan/facts.h`: the `Facts` struct, one member
per fact (an `Interval`, an `EnumSet`), so the arm half of a launch function is written over the same names the registry states
([`../common/plan/rules.md`](../common/plan/rules.md)).

`sm_count` is a DEVICE fact of the Hopper context as a range (114 to 132, the H100 PCIe and SXM parts' counts, from NVIDIA's
documentation); the first reviewed device-probe part ([below](#probe)) restates it as measured. `l2_bytes` is not a fact of the schema
yet: a context states a value for every fact and the contract must build at every commit, so the schema gains it in the same commit as
that reviewed part. The probe reads both already and a candidate part carries them.

<a id="parts"></a>

## Parts, descriptors and contexts

A PART (`parts/device/hopper_sm90a.json`, `parts/calls/decode.json`, `parts/calls/depth1.json`) states facts, each `{"value", <provenance>}` with exactly one of
`measured_by` (a lab record), `validated_by` (the test or check that holds it), `documented_by` or `declared_by` (the family's own declaration).
A value is a point, a range `{"lo", "hi"}`, a set `{"set": [members]}`, or a list of those: a list is ALTERNATIVES, each its own context.

A DESCRIPTOR (`descriptors/hopper_decode_d1.json`) names the parts it composes and the GPUs it
covers (`gpus`, the README's label); the architecture it is built for is its one device part's (`registry.contexts()` derives it, and refuses a descriptor whose parts hold no device part or two). `registry.contexts()` expands every descriptor into FULL CONTEXTS: the parts' facts merged (a fact stated by two
parts, a fact the schema does not name, a schema fact no part states: refused), the alternatives crossed. A matcher and the run time see
contexts only. Two contexts identical in every fact are refused, whatever descriptors made them.

THE CALL'S FACTS (all ints; the schema holds each one's unit and doc). The call's KIND is `prefill` (0 a decode step, 1 a prefill). Its extents are `d_v`,
`tokens` (the sequence length T; decode's point 1), `batch_heads` (B times H) and `levels`; its dials are `decay` and `stateful` (a state in or out); the state's backing is the state's own (`fsla.state(backing=)`) and the memory budget the layer's, so neither is a fact. Its per-level facts are the log2 of each level's width, `width_bits_1` .. `width_bits_4` (a level's width is a
power of two, `fsla/_state.py`'s `StateFormat`, so a range of the log2 states every width exactly; 0 for a level the call does not have) and each side's walk of each level,
`read_listed_1..4` and `write_listed_1..4` (1 when that side of that level is WALKED BY ITS LISTS, 0 when every digit is walked: an action, chosen by the producer's rule `fsla.ops.produce.activation_walks` from the declaration -- a side whose activation has exact zeros is listed -- and any valid walk is lawful, a wrong one costing time and never the result). There is no `write_dense` fact: decode's launch
function derives it (the innermost level's `write_listed` is 0). The depth and the per-level facts are correlated and a part's lists
are crossed, so the correlation is stated by composition: `calls/decode` holds what does not depend on the depth, and `calls/depth1` .. `calls/depth4` each state `levels`
as a point, the four `width_bits` (a range from the width floor, `MIN_LEVEL_WIDTH`, up to the widest level the decode path holds, `kDecodeMaxLevelWidth`, for a level it has, the point 0 beyond) and the
sparse facts (a range 0..1 for a level the depth has, the point 0 beyond). The
contract is four descriptors, `hopper_decode_d1` .. `hopper_decode_d4`, each composing the device part, `calls/decode` and its depth part. A call whose fact is outside its range is refused when the
descriptor is made, naming the fact (`width_bits_1=3 (the contract: 4..8)`), by the same check as every other fact ([`../common/plan/planner.md`](../common/plan/planner.md#the-contract-check)).
A decode step serves `batch_heads` from 1 to 536870911 (`docs/api.md#batch-heads`); the planner test holds both ends.

<a id="architectures"></a>A DEVICE PART STATES AN ARCHITECTURE WHOLE: beside its facts it names its architecture (`arch`, `sm_90a`, the name it is compiled as, whose compute
capability its `cc` is) and its family (`family`, `hopper`) -- fields of the part, not facts. THE ONE SOURCE: the registry is the only place a device fact, an architecture
or its family is stated, ruled by the owner (2026-11-22): "isnt the descritpr a thing we will want to reuse for other kernels, not just carry? each kernel just
matahces on the fields it cares about on these immutable glboal hardware facts". The contract is what a device is admitted against
(`common/plan/environment.cu`); the same parts generate the kernels' header `common/hardware.cuh` (`registry.arch_header`, `emit.py --arch-header`: one row an
architecture, the `Caps` fields a kernel source reads, the family and the build's traits, [`../common/hardware.md`](../common/hardware.md)). An architecture is BUILT when a device part states it
(`registry.built`, with `cc_of`, `arch_of`, `device_arch` and `builds_header`, which the build's flags, the SASS tools and the record read); an architecture no part
states is neither built nor tabulated, and adding one is adding a device part ([`../../bringup.md`](../../bringup.md)). The window is the prefill matcher's, not a fact of a part
([`../common/window.md`](../common/window.md)). The measured rates and the products are measurement inputs, `measure/rates.json`.

MIG and MPS register no context: neither is documented to report a slice's own SM count, L2 share or shared memory, so a context for them waits
for a measurement on a real slice and until then the run time refuses them.

<a id="probe"></a>

## Onboarding a device: the probe

A device enters the contract by a MEASURED part, in four steps (the reviewer also names its `arch` and `family`):

1. RUN THE PROBE on the device: `bazel run //tools:lab -- pack device_probe --arch sm_90a --into DIR`, then `bazel run //tools:lab -- run device_probe --record`, then
   `bazel run //tools:lab -- page device_probe --instance LABEL` ([`device_probe`](../../lab/device_probe.md)). The probe reads every DEVICE fact of the schema
   (the schema's DEVICE facts, and `sm_count` and `l2_bytes` for the schema's coming entries) through the properties and the attributes, checks the two routes
   and the device's own registers agree, and the page step writes `hardware/parts/device/<id>.candidate.json` from the record. The `<id>` is the compute capability and the
   driver's name (`cc900_nvidia_h100_80gb_hbm3`); the PCI slot the driver reports (`pciDeviceID` is the slot, not the product id) is the machine's, so it is not in the key.
2. REVIEW THE CANDIDATE. Each value cites its row of the record (`measured_by`); `tensor` is the MMA kinds of `measure/rates.json`'s architecture of the same compute capability and `instructions` the registry's device part of it
   (`documented_by`), and each is left out, named in `unestablished`, when there is none: the reviewer supplies them from a lab experiment or a document. MIG and MPS
   slices are not probed (see above).
3. COMMIT THE PART: rename `<id>.candidate.json` to `<id>.json` (`hardware/parts/device/`), add a descriptor composing it with a calls part, and commit. THE REGISTRY REFUSES
   A CANDIDATE FILE (`registry.CANDIDATE`): `registry.check()` and `registry.contexts()` fail naming it, so a candidate never reaches the contract unreviewed.
4. THE BUILD'S CONTRACT CHECK THEN COVERS IT: the part's contexts are in `//hardware:contexts`, every family's matcher is evaluated over them, and a context mapped to no arm
   fails the build ([`arms_gen.md`](../decode/arms_gen.md)).

<a id="narrowed"></a>

## A narrowed contract (development builds)

A development build may compile a subset by naming a DECLARED NARROWED CONTRACT: a descriptor of its own under `descriptors/dev/<name>.json`, marked
`"narrowed": true`, composing a subset of the call parts (`parts/calls/decode_narrow_d32.json` and one depth part, `calls/depth2`) beside the device part: a descriptor holds one depth, so the narrowed contract
`decode_d32_l2` is depth 2 alone. The build setting
`--//:contract=<name>` (default `full`) chooses it: `//hardware:contexts` and `//hardware:contract` are then written for it
(`hardware/contract.bzl`; `registry.contexts(contract=<name>)`), so `arms_gen` compiles exactly the arms the narrowed contract reaches and the run time,
holding the narrowed contract, refuses a call outside it, naming it (`kContractName`, `planner.h`'s refusal). The registry refuses a name no descriptor of
`descriptors/dev/` declares, a descriptor not marked narrowed, and a narrowed context outside every full context: a narrowed contract promises less than
the full one, never more. A narrowed descriptor is never part of the full contract (`descriptors/*.json` alone are), so the README's table and the
duplicate check read the full one. The full contract is enforced by `//:gate` (its `full_contract` test fails, naming the contract, under any other
setting) and by releases (the build record says `iteration: True`, which `bazel run //:wheels` refuses). `tests/unit/test_narrowed_contract.py` holds all of it.

<a id="the-owed-folder"></a>

## The owed folder: a descriptor that exists before its arm

`hardware/descriptors/owed/<name>.json` is a THIRD folder beside the full contract (`descriptors/*.json`) and the narrowed ones (`descriptors/dev/`). A descriptor there composes parts exactly as any
other (`calls/prefill` and a depth part: `hopper_prefill_d1` .. `hopper_prefill_d4`) and `registry.check` validates it against the schema, but it is NEVER in the full contract, the README's support table,
a narrowed contract or the arms the build compiles. It has one more field, `"owed": ["<kernel>", ...]`: the kernels its call waits on, the names `fsla.ops._ext.not_built` and the tests'
`requires_kernel` use. The registry refuses an owed descriptor reachable from the full contract (a context in both), so an owed call cannot be promised and owed at once.

At run time `//hardware:owed` generates `plan/owed.h` from the registry (no hand-typed copy) and the planner, finding a context outside the contract but inside an owed descriptor, refuses it
naming each owed kernel instead of saying it lies outside the contract; `fsla.ops.descriptor.descriptor` turns that into `KernelNotBuilt` (through the host op `fsla_owed_kernels`), so a test marked
`requires_kernel("prefill")` is a strict xfail through the surface. A context in neither is refused as every call is, naming the facts. The build's contract check fails, naming the context,
on a full-contract context whose `prefill` fact reaches 1, rather than evaluating it against decode's launch function. A prefill descriptor leaves the owed folder when the prefill launch
function and its arm are built (`docs/open-work.md` rows A-f and A-h).

<a id="outputs"></a>

## What reads it

| reader | reads |
|---|---|
| `//hardware:facts` | the schema: `plan/facts.h` |
| `//hardware:contexts` | the build's contract (`--//:contract`), a line a context (`registry.contexts_text`): `arms_gen`'s input |
| `//hardware:contract` | the contract as a table the run time holds: `plan/contract.h` (`registry.contract_header`), which the planner checks every descriptor against |
| `tools/supported.py` | the contract: the README's support table |
