# `carry/carry_arm_abi.cuh` — one arm's ABI

Mirrors `csrc/rola/src/carry/carry_arm_abi.cuh`: pointers and scalars only, so a per-arm translation unit never
parses the torch headers and the dispatch unit never parses a kernel body.

<a id="form"></a>
## The form

`CarryForm` is an arm's plan on the current device's card (`plan.md#forms`): its shared bytes, the image's parts, a
readout pass's and a fold unit's slots, a read and a write pool row's bytes. The facts pass sizes each side's passes
and entry offsets by it (`rola.ops.carry.form`), so the lists it emits are the ones the carry walks.

<a id="census"></a>
## The census

`ArmCensus` is what the device build reports for one arm: registers a thread, local bytes (zero, or the SASS gate is
red), the block's thread limit, the binary version and its form. `carry_census` returns one row an arm, its index and
key first, the form's fields after the binary version. `ROLA_CARRY_ARM_DECLARE` declares an arm's launch and census for the dispatch unit.
