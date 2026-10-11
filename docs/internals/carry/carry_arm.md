# `carry/carry_arm.cuh` — one arm's launch

Mirrors `csrc/rola/src/carry/carry_arm.cuh`, included only by the generated per-arm units
(`csrc/rola/src/instantiations/`, `tools/gen_shards.py`).

<a id="launch"></a>
## The launch

`carry_launch<Arm>` launches `carry_kernel<Arm>` on `(boxes, batch-heads)` with 256 threads and the plan's dynamic
shared memory (`plan.md#plan`), opting in above 48 KB once an arm behind a function-local flag. The launch bound is
`(256, 1)`: one CTA an SM is the design on every architecture. `carry_census<Arm>` reads the function's attributes
back (registers, local bytes, its maximum block) beside the plan's shared bytes. `ROLA_CARRY_ARM_DEFINE` stamps
both for one row of the shipped set, keyed by its six fields.
