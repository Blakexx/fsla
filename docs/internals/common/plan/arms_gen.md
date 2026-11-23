# The arm generator (`csrc/fsla/src/common/plan/arms_gen.cpp`)

A host tool the graph builds (`//csrc/fsla:arms_gen`) and runs for each arm family (`//csrc/fsla:<family>_plan`, one for each member of `ARM_FAMILIES`) over the family's
contract (`//hardware:<family>_contexts`, [`../../hardware/registry.md`](../../hardware/registry.md)). ONE SOURCE SERVES EVERY FAMILY: the family is the kernel the contexts
file names (`# kernel: <family>`), and the tool is `Generator<K>` over the family's view ([`arm_family.md`](arm_family.md): its `Facts`, its arm and the arm's key, its matcher's
reach over a context and its arm over a point). For every context it walks the family's reach (`plan::decode_reach`, [`decode_launch.md`](decode_launch.md); `plan::entmax_reach<K>`,
[`entmax_launch.md`](entmax_launch.md)) and writes:

- `<family>_arm_rows.json`: the arms some context reaches, in canonical order (the key's fields in order), which is an arm's index. `arm_family` (`bazel/arms.bzl`) takes it as its rows
  file: the selection header, the Python table and the declarations are written from it by `bazel/arm_emit.py`. No family has a per-arm unit, so the arms need not be known at load time.
- `<family>_selection_record.txt`: each context and the arm index and key it reaches, kept as a build output for review.

<a id="the-contract"></a>
THE CONTRACT IS GUARANTEED: a context whose reach walk reaches NONE fails the tool with the context's id and the facts the matcher reads, so the build fails; the fix is an arm, or a
narrower contract in the same commit. A family's contract is its own registry (`hardware/kernels/<family>/`). Decode's: its `call` part and one depth part each of the four hierarchy
depths beside it (`hardware/kernels/decode/parts`; the registry page names them): `d_v` from 2 (the value width `fsla/_state.py`'s `StateFormat` admits) to 64, one to four `levels`,
`decay` 0 or 1 (and the level widths, the per-level write walks, `batch_heads` and the rest of the call's facts, which the arm half does not read: the arm key stays `d_v`, `levels`,
`decay`). The value width is a RANGE because the seam pads any width up to the arm that holds it, and the launch function maps it ([`decode_launch.md`](decode_launch.md)). Each of the
six entmax kernels' contract is its `call` part: every width, both alphas where the kernel has one, the four dtype pairs, one to sixteen levels
([`../../entmax/entmax.md#the-contract`](../../entmax/entmax.md#the-contract)).

<a id="coverage"></a>
COVERAGE IS DECIDED EXACTLY, NEVER SAMPLED: the family's reach ([`decode_launch.md`](decode_launch.md), [`entmax_launch.md`](entmax_launch.md)) walks the arm half's decision tree over a
context's ranges ([`rules.md#exact`](rules.md#exact)). It reaches every arm some point of the ranges reaches, and the NONE leaf exactly when some point reaches
no arm; the count of ranged facts costs nothing (the walk visits at most the tree's leaves). A context whose walk reaches the NONE leaf fails the tool,
naming the context and a point of it that reaches no arm (found by halving each ranged fact toward the half where the walk still reaches NONE: the walk
being exact, one half always does); `tests/unit/test_decode_arms_gen.py` holds a hole at either end of a range, a hole in `cc`, nine ranged facts, and the
walk against every point of seeded ranges. The other half, a point reaches exactly one arm, holds by construction: over a point every primitive returns Yes or No
(`static_assert`s in `rules.h`), so the tree takes one branch at each fork. `--sample <points> --seed <n>` is the sanity check of that argument, not its guarantee: seeded random
points inside each context, each of which must reach, through the family's arm over a point, an arm of the context's reach. The family's launch function is also run on a sample point of each
context, and its arguments must be that point's own facts.

`//hardware:<family>_contexts` is the contract the build names (`--//:contract`, default `full`): under a narrowed contract decode's registry declares
([`../../hardware/registry.md#narrowed`](../../hardware/registry.md#narrowed)) the tool reaches only the arms that contract does (`decode_d32_l2`: 2 of the 16), and the
build is an iteration build that is never a wheel. The first lines of the contexts file name the contract and the kernel (`# kernel: <family>`, which picks the family); the selection record repeats the contract.
`tests/unit/test_narrowed_contract.py` holds the counts; `tests/unit/test_entmax_matcher.py` holds each entmax family's.
