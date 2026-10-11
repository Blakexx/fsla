# `carry/carry_arm_abi.cuh` — one arm's ABI

Mirrors `csrc/rola/src/carry/carry_arm_abi.cuh`: pointers and scalars only, so a per-arm translation unit never
parses the torch headers and the dispatch unit never parses a kernel body.

<a id="census"></a>
## The census

`ArmCensus` is what the device build reports for one arm: registers a thread, local bytes (zero, or the SASS gate is
red), the plan's shared bytes, the block's thread limit and the binary version. `carry_census` returns one row an
arm, its index and key first. `ROLA_CARRY_ARM_DECLARE` declares an arm's launch and census for the dispatch unit.
