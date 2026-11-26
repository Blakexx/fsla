# `plan_api.cuh`

The planner's host-facing declarations, the only header of `common/plan/` that `csrc/fsla/fsla_api.cpp` includes: `fsla_call_facts`, `fsla_descriptor` and
`fsla_owed_kernels`, each over a kernel's name, and `fsla_segment_plan`, `fsla_step_bytes` and `fsla_step_tokens`, the segment plan of a call and a step's bytes and tokens on a device ([`segments.md`](segments.md#entries)). What they are and how Python reaches them: [`environment.md#entries`](environment.md#entries).
