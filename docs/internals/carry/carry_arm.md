# `carry/carry_arm.cuh` — one arm's launch

Mirrors `csrc/rola/src/carry/carry_arm.cuh`, included only by the generated per-arm units
(`csrc/rola/src/instantiations/`, `tools/gen_shards.py`).

<a id="plan"></a>
## The plan's size

`carry_smem_bytes<Arm>` is the plan's shared bytes on the current device's card (`plan.md#forms`), the host's own
instance of the plan for that card. The kernel compiled for a card holds that card's plan, and traps at entry when the
launched dynamic size is not its plan's (`%dynamic_smem_size`): code compiled for another card, or a host sized for
one, never runs.

<a id="launch"></a>
## The launch

`carry_launch<Arm>` launches `carry_kernel<Arm>` on `(boxes, batch-heads)` with 256 threads and the plan's
dynamic shared memory, opting in once an arm behind a function-local flag. The launch bound is `(256, 1)`: one CTA an
SM on every architecture. `carry_census<Arm>` reads the function's attributes back (registers, local bytes, its
maximum block) beside the plan's shared bytes. `ROLA_CARRY_ARM_DEFINE` stamps both for one row of the shipped
set, keyed by its six fields.
