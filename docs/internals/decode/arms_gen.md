# Decode's arm generator (`csrc/fsla/src/decode/arms_gen.cpp`)

A host tool the graph builds (`//csrc/fsla:decode_arms_gen`) and runs (`//csrc/fsla:decode_plan`) over the contract
(`//hardware:contexts`, [`../hardware/registry.md`](../hardware/registry.md)). For every context it evaluates `plan::decode_arms`
([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)) and writes:

- `decode_arm_rows.json`: the arms some context reaches, in canonical order (`d_v`, then `levels`, then `decay`), which is an arm's
  index. `arm_family` (`bazel/arms.bzl`) takes it as its rows file: the selection header, the Python table and the declarations are written
  from it by `bazel/arm_emit.py`. The family has no per-arm unit, so the arms need not be known at load time.
- `decode_built.h`: the same arms as a constexpr array, which the run-time lookup will read when the planner reaches the run time.
- `decode_selection_record.txt`: each context and the arm index and key it reaches, kept as a build output for review.

THE CONTRACT IS GUARANTEED: a context `decode_arms` maps to NONE fails the tool with the context's id and the facts the matcher reads, so
the build fails; the fix is an arm, or a narrower contract in the same commit. A point that reaches other than one arm, or a corner of a
range (each ranged fact at its lo or its hi) that reaches an arm the range does not, fails it too.
`tests/unit/test_decode_arms_gen.py` holds the failure, with a contract that admits `d_v` 128, `levels` 5 or `decay` 2.

`--//:decode_arms` still names a subset of the generated list, by index, for an iteration build that is never a wheel.
