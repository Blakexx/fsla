# `tools/ratify.py` — the census

Mirrors `tools/ratify.py`. The census, its one refusal, its self-test and the closed-world rules are
[`docs/ratification.md`](../../ratification.md); this page covers what the tool writes and the mode that reads the graph's build.

## What it writes

The census (`--census`) writes nothing: it prints, and `--json PATH` writes the raw per-instantiation record where the
caller names it. The build's generated headers (`build/generated/*.inc`, a run's, never checked in) are the graph's
(`bazel run //:library`), and are its inputs.

## The headers are the graph's, not this tool's

The census writes no header: the selection headers and the stamp are the graph's header and stamp actions' outputs, and
the units it reads were compiled against them. A census that measured against headers it wrote itself would be
measuring a tree nobody builds, so the scope is the build's parameters (`--graph`, below).

## <a id="graph"></a>On the graph's build: `--graph`

`--graph CENSUS --runfiles DIR` (both required) reads the graph's build and compiles nothing. Each `cuda_unit` keeps
ptxas's report beside its object, so reading the two measures the binary itself, by the two readers (`_parse_report`,
`sass_scan`): there is one compile, not a second one the census must agree with. The census record
(`bazel/defs.bzl`'s `ratify_census`, every path relative to `DIR`) names each unit the library links, the graph's ptxas
and disassemblers, and the build's parameters; the readers read with the graph's
tools.

- THE SCOPE IS THE BUILD'S PARAMETERS (`--//:archs` and `--//:contract`): an iteration build's
  census is the measurement of its subset.
- THE MODES are `--census [--json PATH]` (the table, exit 1 on a discarded launch bound) and `--self-test`.
- `compiled_entries()` is the mangled name of every entry function the units compiled: what
  `bazel test //csrc/fsla:unused_instantiations` reads ([`lint.md`](lint.md)).

`bazel run //:ratify -- --census` runs in the checkout ([`build-graph.md`](../../build-graph.md#census)); the template lint runs on the census in a
copy of the tree.
