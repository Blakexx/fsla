# `tools/ratify.py` — the census

Mirrors `tools/ratify.py`. The census, its one refusal and the closed-world rules are
[`docs/ratification.md`](../../ratification.md); this page covers what the tool writes and the mode that reads the graph's build.

## What it writes

The census (`--census`) writes nothing: it prints, and `--json PATH` writes the raw per-instantiation record where the
caller names it. The build's generated headers (`build/generated/*.inc`, a run's, never checked in) are the graph's
(`bazel run //:library`), and are its inputs.

## The headers are the graph's, not this tool's

The census writes no header: the selection headers and the stamp are the graph's header and stamp actions' outputs, and
the library whose manifests it reads was compiled against them. A census that measured against headers it wrote itself would be
measuring a tree nobody builds, so the scope is the build's parameters (`--graph`, below).

## <a id="graph"></a>On the graph's build: `--graph`

`--graph CENSUS --runfiles DIR` (both required) reads the graph's build and compiles and disassembles nothing. It reads
the STATIC MANIFEST of each built architecture, the SASS gate's own (warpline's, `//:static_manifest`,
[`sass_gate.md`](sass_gate.md#manifest)), with warpline's reader of the form (`manifests`, through `tools/checked.py`): one
reading of the binary, the gate's, not a second one the census must agree with. A kernel's figures are its registers, the
spill bytes `ptxas -v` reported (`spill_stores`, `spill_loads`: the gate reads the build's notices, so a kernel without
them is refused), its local accesses in all and inside a natural loop (`local_loads`, `local_stores`, `local_in_loops`);
a discarded launch bound is the checker's `launch_bound_discarded` at the kernel. The register-side CTAs an SM are this
tool's arithmetic over the registers (`_regs_ctas`, at 256 threads). An instantiation two units emit is the gate's
refusal (warpline's listing reader refuses a function named twice). The census record (`bazel/defs.bzl`'s
`ratify_census`, every path relative to `DIR`) names each architecture's manifest, the graph's ptxas and readers (whose
versions are checked against the toolchain record before the first read), the directory of warpline's package, and the
build's parameters.

- THE SCOPE IS THE BUILD'S PARAMETERS (`--//:archs` and `--//:contract`): an iteration build's
  census is the measurement of its subset.
- THE MODE is `--census [--json PATH]` (the table, exit 1 on a discarded launch bound). The readers' falsifiability is
  their own tests': warpline's (the manifest's figures, the notices' rows) and the body hash's (`tests/unit/test_sass_bodies.py`).
- `compiled_entries()` is the mangled name of every kernel the manifests name: what
  `bazel test //csrc/fsla:unused_instantiations` reads ([`lint.md`](lint.md)).

`bazel run //:ratify -- --census` runs in the checkout ([`build-graph.md`](../../build-graph.md#census)); the template lint runs on the census in a
copy of the tree.
