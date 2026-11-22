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

`sm_count` and `l2_bytes` are not facts yet: no probe has measured either, and a context states a value for every fact. They enter with
the device probe on the H100.

<a id="parts"></a>

## Parts, descriptors and contexts

A PART (`parts/device/hopper_sm90a.json`, `parts/calls/decode.json`) states facts, each `{"value", <provenance>}` with exactly one of
`measured_by` (a lab record), `validated_by` (the test or check that holds it), `documented_by` or `declared_by` (the family's own declaration).
A value is a point, a range `{"lo", "hi"}`, a set `{"set": [members]}`, or a list of those: a list is ALTERNATIVES, each its own context.

A DESCRIPTOR (`descriptors/hopper_decode.json`) names the parts it composes, the architecture it is built for (`arch`) and the GPUs it
covers (`gpus`, the README's label). `registry.contexts()` expands every descriptor into FULL CONTEXTS: the parts' facts merged (a fact stated by two
parts, a fact the schema does not name, a schema fact no part states: refused), the alternatives crossed. A matcher and the run time see
contexts only. Two contexts identical in every fact are refused, whatever descriptors made them.

The hopper device part holds the hardware table's sm_90a values for the facts the schema names, and
`tests/unit/test_hardware_registry.py` holds them equal: the table is still what `hardware.cuh` is generated from (the run-time card's
admission reads the driver against it) until the run-time planner replaces that read.

MIG and MPS register no context: neither is documented to report a slice's own SM count, L2 share or shared memory, so a context for them waits
for a measurement on a real slice and until then the run time refuses them.

<a id="outputs"></a>

## What reads it

| reader | reads |
|---|---|
| `//hardware:facts` | the schema: `plan/facts.h` |
| `//hardware:contexts` | the contract, a line a context (`registry.contexts_text`): `arms_gen`'s input |
| `tools/supported.py` | the contract: the README's support table |
| `tools/lint/lint_standards.py`'s `check_arch_coverage` | the contract's architectures: each must be built |
