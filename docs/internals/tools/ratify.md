# `tools/ratify.py` — the census, and the derivation's goldens

Mirrors `tools/ratify.py`. The census, its one refusal, its self-test and the closed-world rules are
[`docs/ratification.md`](../../ratification.md); this page covers the two things the tool reads and the mode that reads the
graph's build.

## Two records, two lifetimes

| artifact | scope | written by | checked by |
|---|---|---|---|
| `tools/manifests/derivation.json` | arch-independent | `--write-derivation` | `--derivation` (`bazel test //:derivation`) |
| `build/generated/*.inc` | per run, never checked in | `bazel run //:library` (the graph's header and stamp actions) | — (they are inputs) |

The census (`--census`) writes nothing: it prints, and `--json PATH` writes the raw per-instantiation record where the
caller names it. `--write-derivation` is its own mode because the derivation is host code measured by one small compile.

## The headers are the graph's, not this tool's

The census writes no header: the selection headers and the stamp are the graph's header and stamp actions' outputs, and
the units it reads were compiled against them. A census that measured against headers it wrote itself would be
measuring a tree nobody builds, so the scope is the build's parameters (`--graph`, below).

## What the derivation record does not cover

The derivation's code hash is over the files `derivation_cases.json` DECLARES
(`common/geom.cuh`), not over `csrc/`. Widening it to the whole tree would expire the
record on every kernel edit, and a record that expires constantly is one people
re-write without reading.

## <a id="graph"></a>On the graph's build: `--graph`

`--graph CENSUS --runfiles DIR` (both required) reads the graph's build and compiles nothing. Each `cuda_unit` keeps
ptxas's report beside its object, so reading the two measures the binary itself, by the two readers (`_parse_report`,
`sass_scan`): there is one compile, not a second one the census must agree with. The census record
(`bazel/defs.bzl`'s `ratify_census`, every path relative to `DIR`) names each unit the library links, the graph's ptxas
and disassemblers, the derivation harness the graph built, and the build's parameters; the readers read with the graph's
tools.

- THE SCOPE IS THE BUILD'S PARAMETERS (`--//:archs`, `--//:carry_arms`, `--//:decode_arms`): an iteration build's
  census is the measurement of its subset.
- THE MODES are `--census [--json PATH]` (the table, exit 1 on a discarded launch bound), `--derivation` (the goldens
  against the harness), `--write-derivation` (re-measure and file them) and `--self-test`.
- THE DERIVATION HARNESS is built by the graph: `tools/derivation.py` renders it from the declared cases with the
  standard library only, so a build action can run it before any venv, and tools/BUILD.bazel's
  `derivation_harness` compiles it as the extension's units are compiled and links it. The derivation check runs that
  binary and refuses it unless its source is this tree's rendering.
- `compiled_entries()` is the mangled name of every entry function the units compiled: what
  `bazel test //csrc/rola:unused_instantiations` reads ([`lint.md`](lint.md)).

`bazel run //:ratify -- --census` runs in the checkout ([`build-graph.md`](../../build-graph.md#census)); `bazel test
//:derivation` and the template lint run on the census in a copy of the tree.
