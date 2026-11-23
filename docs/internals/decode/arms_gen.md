# Decode's arm generator (`csrc/fsla/src/decode/arms_gen.cpp`)

A host tool the graph builds (`//csrc/fsla:decode_arms_gen`) and runs (`//csrc/fsla:decode_plan`) over the contract
(`//hardware:decode_contexts`, [`../hardware/registry.md`](../hardware/registry.md)). For every context it walks `plan::decode_reach`
([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)) and writes:

- `decode_arm_rows.json`: the arms some context reaches, in canonical order (`d_v`, then `levels`, then `decay`), which is an arm's
  index. `arm_family` (`bazel/arms.bzl`) takes it as its rows file: the selection header, the Python table and the declarations are written
  from it by `bazel/arm_emit.py`. The family has no per-arm unit, so the arms need not be known at load time.
- `decode_selection_record.txt`: each context and the arm index and key it reaches, kept as a build output for review.

<a id="the-contract"></a>
THE CONTRACT IS GUARANTEED: a context whose `decode_reach` walk reaches NONE fails the tool with the context's id and the facts the launch function reads, so
the build fails; the fix is an arm, or a narrower contract in the same commit. Decode's contract is decode's own registry (`hardware/kernels/decode/`): its `call` part and one depth part each of the four hierarchy depths beside it
(`hardware/kernels/decode/parts`; the registry page names them): `d_v`
from 2 (the value width `fsla/_state.py`'s `StateFormat` admits) to 64, one to four `levels`, `decay` 0 or 1 (and the level widths, the per-level write walks, `batch_heads` and the rest of the call's facts, which the arm half does not read: the arm key stays `d_v`, `levels`, `decay`). The value width is a RANGE
because the seam pads any width up to the arm that holds it, and the launch function maps it ([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)).

<a id="coverage"></a>
COVERAGE IS DECIDED EXACTLY, NEVER SAMPLED: `plan::decode_reach` ([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)) walks the arm half's decision tree over a context's ranges
([`../common/plan/rules.md#exact`](../common/plan/rules.md#exact)). It reaches every arm some point of the ranges reaches, and the NONE leaf exactly when some point reaches
no arm; the count of ranged facts costs nothing (the walk visits at most the tree's leaves). A context whose walk reaches the NONE leaf fails the tool,
naming the context and a point of it that reaches no arm (found by halving each ranged fact toward the half where the walk still reaches NONE: the walk
being exact, one half always does); `tests/unit/test_decode_arms_gen.py` holds a hole at either end of a range, a hole in `cc`, nine ranged facts, and the
walk against every point of seeded ranges. The other half, a point reaches exactly one arm, holds by construction: over a point every primitive returns Yes or No
(`static_assert`s in `rules.h`), so the tree takes one branch at each fork. `--sample <points> --seed <n>` is the sanity check of that argument, not its guarantee: seeded random
points inside each context, each of which must reach, through `plan::decode_arm`, an arm of the context's reach.

`//hardware:decode_contexts` is the contract the build names (`--//:contract`, default `full`): under a narrowed contract decode's registry declares
([`../hardware/registry.md#narrowed`](../hardware/registry.md#narrowed)) the tool reaches only the arms that contract does (`decode_d32_l2`: 2 of the 16), and the
build is an iteration build that is never a wheel. The first lines of the contexts file name the contract and the kernel (`# kernel: decode`, which `arms_gen` checks); the selection record repeats the contract.
`tests/unit/test_narrowed_contract.py` holds the counts.
