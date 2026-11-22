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

A PART (`parts/device/hopper_sm90a.json`, `parts/calls/decode.json`) states facts, each `{"value", <provenance>}` with exactly one of
`measured_by` (a lab record), `validated_by` (the test or check that holds it), `documented_by` or `declared_by` (the family's own declaration).
A value is a point, a range `{"lo", "hi"}`, a set `{"set": [members]}`, or a list of those: a list is ALTERNATIVES, each its own context.

A DESCRIPTOR (`descriptors/hopper_decode.json`) names the parts it composes, the architecture it is built for (`arch`) and the GPUs it
covers (`gpus`, the README's label). `registry.contexts()` expands every descriptor into FULL CONTEXTS: the parts' facts merged (a fact stated by two
parts, a fact the schema does not name, a schema fact no part states: refused), the alternatives crossed. A matcher and the run time see
contexts only. Two contexts identical in every fact are refused, whatever descriptors made them.

The hopper device part holds the hardware table's sm_90a values for the facts the table names, and
`tests/unit/test_hardware_registry.py` holds them equal: the table is what `hardware.cuh` is generated from, the contract is what a device is
admitted against (`common/plan/environment.cu`).

MIG and MPS register no context: neither is documented to report a slice's own SM count, L2 share or shared memory, so a context for them waits
for a measurement on a real slice and until then the run time refuses them.

<a id="probe"></a>

## Onboarding a device: the probe

A device enters the contract by a MEASURED part, in four steps:

1. RUN THE PROBE on the device: `bazel run //tools:lab -- pack device_probe --arch sm_90a --into DIR`, then `bazel run //tools:lab -- run device_probe --record`, then
   `bazel run //tools:lab -- page device_probe --instance LABEL` ([`device_probe`](../../lab/device_probe.md)). The probe reads every DEVICE fact of the schema
   (the schema's DEVICE facts, and `sm_count` and `l2_bytes` for the schema's coming entries) through the properties and the attributes, checks the two routes
   and the device's own registers agree, and the page step writes `hardware/parts/device/<id>.candidate.json` from the record. The `<id>` is the compute capability and the
   driver's name (`cc900_nvidia_h100_80gb_hbm3`); the PCI slot the driver reports (`pciDeviceID` is the slot, not the product id) is the machine's, so it is not in the key.
2. REVIEW THE CANDIDATE. Each value cites its row of the record (`measured_by`); `tensor` and `instructions` are the architecture table's row of the same compute capability
   (`documented_by`), and are left out, named in `unestablished`, when the table has none: the reviewer supplies them from a lab experiment or a document. MIG and MPS
   slices are not probed (see above).
3. COMMIT THE PART: rename `<id>.candidate.json` to `<id>.json` (`hardware/parts/device/`), add a descriptor composing it with a calls part, and commit. THE REGISTRY REFUSES
   A CANDIDATE FILE (`registry.CANDIDATE`): `registry.check()` and `registry.contexts()` fail naming it, so a candidate never reaches the contract unreviewed.
4. THE BUILD'S CONTRACT CHECK THEN COVERS IT: the part's contexts are in `//hardware:contexts`, every family's matcher is evaluated over them, and a context mapped to no arm
   fails the build ([`arms_gen.md`](../decode/arms_gen.md)).

<a id="narrowed"></a>

## A narrowed contract (development builds)

A development build may compile a subset by naming a DECLARED NARROWED CONTRACT: a descriptor of its own under `descriptors/dev/<name>.json`, marked
`"narrowed": true`, composing a subset of the call parts (`parts/calls/decode_narrow_d32_l2.json`) beside the device part. The build setting
`--//:contract=<name>` (default `full`) chooses it: `//hardware:contexts` and `//hardware:contract` are then written for it
(`hardware/contract.bzl`; `registry.contexts(contract=<name>)`), so `arms_gen` compiles exactly the arms the narrowed contract reaches and the run time,
holding the narrowed contract, refuses a call outside it, naming it (`kContractName`, `planner.h`'s refusal). The registry refuses a name no descriptor of
`descriptors/dev/` declares, a descriptor not marked narrowed, and a narrowed context outside every full context: a narrowed contract promises less than
the full one, never more. A narrowed descriptor is never part of the full contract (`descriptors/*.json` alone are), so the README's table and the
duplicate check read the full one. The full contract is enforced by `//:gate` (its `full_contract` test fails, naming the contract, under any other
setting) and by releases (the build record says `iteration: True`, which `bazel run //:wheels` refuses). `tests/unit/test_narrowed_contract.py` holds all of it.

<a id="outputs"></a>

## What reads it

| reader | reads |
|---|---|
| `//hardware:facts` | the schema: `plan/facts.h` |
| `//hardware:contexts` | the build's contract (`--//:contract`), a line a context (`registry.contexts_text`): `arms_gen`'s input |
| `//hardware:contract` | the contract as a table the run time holds: `plan/contract.h` (`registry.contract_header`), which the planner checks every descriptor against |
| `tools/supported.py` | the contract: the README's support table |
