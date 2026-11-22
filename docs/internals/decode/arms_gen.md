# Decode's arm generator (`csrc/fsla/src/decode/arms_gen.cpp`)

A host tool the graph builds (`//csrc/fsla:decode_arms_gen`) and runs (`//csrc/fsla:decode_plan`) over the contract
(`//hardware:contexts`, [`../hardware/registry.md`](../hardware/registry.md)). For every context it evaluates `plan::decode_arms`
([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)) and writes:

- `decode_arm_rows.json`: the arms some context reaches, in canonical order (`d_v`, then `levels`, then `decay`), which is an arm's
  index. `arm_family` (`bazel/arms.bzl`) takes it as its rows file: the selection header, the Python table and the declarations are written
  from it by `bazel/arm_emit.py`. The family has no per-arm unit, so the arms need not be known at load time.
- `decode_selection_record.txt`: each context and the arm index and key it reaches, kept as a build output for review.

<a id="the-contract"></a>
THE CONTRACT IS GUARANTEED: a context `decode_arms` maps to NONE fails the tool with the context's id and the facts the launch function reads, so
the build fails; the fix is an arm, or a narrower contract in the same commit. Decode's contract is `hardware/parts/calls/decode.json`: `d_v`
from 2 (the value width `fsla/_state.py`'s `StateFormat` admits) to 64, one to four `levels`, `decay` 0 or 1. The value width is a RANGE
because the seam pads any width up to the arm that holds it, and the launch function maps it ([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)).

THE POINT PROPERTY, tried at run-time shapes: a launch function over a point reaches exactly one arm, and the points of a range reach arms
the range's arms hold. The points tried are every combination of the ranged facts' ends and each ranged fact swept through its whole range
(the others at their lowest). A point that reaches other than one arm, or an arm the range does not, fails the tool; so does a context with more
ranged facts than the corners are tried over (`kMaxRanged`): the tool refuses what it cannot check rather than reporting agreement.
`tests/unit/test_decode_arms_gen.py` holds the failures: a contract that admits `d_v` 128, `levels` 5 or `decay` 2, a range wider than the arms,
a context with too many ranged facts.

`--//:decode_arms` names a subset of the generated list, by index, for an iteration build that is never a wheel.
