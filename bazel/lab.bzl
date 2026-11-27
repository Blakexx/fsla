"""THE LAB'S EXPERIMENTS AS THE GRAPH'S TARGETS (docs/internals/tools/lab.md#graph): each `measure/lab/experiments/<name>.cu` is compiled by
the graph's own CUDA rule (`bazel/cuda.bzl`'s `cuda_unit`, the rule that builds the library) for sm_90a and linked into its
binary; the graph's disassemblers read the binary once (`tools/lab_facts.py`); and the checks -- the SASS checker, the notices'
ledger, the declarations' (`tools/lab_check.py`) -- are a validation action over those files, so a refusal fails the build. Each
experiment's RUN TARGET, `<name>.run`, is the lab (`tools/lab.py`) with that build in its runfiles: it runs and stores the
experiment, or reads its build, and runs no compiler, no disassembler and no nested build."""

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//bazel:cuda.bzl", "cuda_unit")
load("//bazel:defs.bzl", "lab_run", "warpline_root")

#: the lab's architecture: the one of this tree (`tools/lab.py`'s `ARCH`), and `lab.cuh`'s `kLabBuild` compute capability (`hardware/registry.py`'s
#: `cc_of`: x 10, as `__CUDA_ARCH__` reads it)
_ARCH = "sm_90a"
_CC = "900"

#: the experiment's compile: the lab's own flags. ptxas's own options the build switches on (`ptxas --help`, passed through nvcc's
#: `-Xptxas`): the warning where a function spills registers to local memory, the warning where a function uses local memory (its
#: stack frame's size in the text) and the verbose per-function summary
_COPTS = [
    "-O3",
    "-std=c++20",
    "-DFSLA_LAB_CC=" + _CC,
    "-lineinfo",
    "-Xptxas",
    "--warn-on-spills,--warn-on-local-memory-usage,--verbose",
    "--expt-relaxed-constexpr",
]

#: what the facts action writes beside the binary, each `<name>.<arch>.<part>` (`tools/lab.py`'s `PARTS`)
_FACTS = ["cubin", "sass", "usage", "graph", "lines", "names", "toolkit"]

def _lab_experiment_impl(ctx):
    stem = "%s.%s" % (ctx.label.name, _ARCH)
    binary = ctx.executable.binary
    if binary.basename != stem:
        fail("%s: its binary is %s, not %s" % (ctx.label, binary.basename, stem))
    ptxas = ctx.attr.unit[OutputGroupInfo].ptxas_reports.to_list()
    if len(ptxas) != 1:
        fail("%s: its unit compiles %d sources, not one" % (ctx.label, len(ptxas)))
    parts = {part: ctx.actions.declare_file("%s.%s" % (stem, part)) for part in _FACTS + ["ptxas"]}
    ctx.actions.run_shell(
        inputs = [ptxas[0]],
        outputs = [parts["ptxas"]],
        command = "cp \"$1\" \"$2\"",
        arguments = [ptxas[0].path, parts["ptxas"].path],
        mnemonic = "LabDiagnostics",
        progress_message = "The diagnostics of %s's compile" % ctx.label.name,
    )

    #: THE FACTS: the graph's disassemblers over the binary, once
    args = ctx.actions.args()
    args.add(ctx.file._facts)
    args.add("--binary", binary)
    args.add("--ptxas", parts["ptxas"])
    args.add("--cuobjdump", ctx.file._cuobjdump)
    args.add("--nvdisasm", ctx.file._nvdisasm)
    args.add("--cuxxfilt", ctx.file._cuxxfilt)
    args.add("--nvcc", ctx.file._nvcc)
    for part in _FACTS:
        args.add("--" + part, parts[part])
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = [binary, parts["ptxas"], ctx.file._facts, ctx.file._cuobjdump, ctx.file._nvdisasm, ctx.file._cuxxfilt, ctx.file._nvcc, ctx.file._sass],
        outputs = [parts[part] for part in _FACTS],
        mnemonic = "LabFacts",
        progress_message = "Reading %s's code with the graph's disassemblers" % ctx.label.name,
    )

    #: THE CHECKS, a validation: the lab's check over the files above; a refusal fails the build with the lab's text, and
    #: `--norun_validations` builds the target without them (the lab's other commands print the findings themselves)
    verdict = ctx.actions.declare_file("%s.check" % stem)
    check = ctx.actions.args()
    check.add(ctx.file._check)
    check.add("--built", binary.dirname)
    check.add("--experiment", ctx.label.name)
    check.add("--arch", _ARCH)
    check.add("--verdict", verdict)
    package, warpline = warpline_root(ctx, ctx.attr._warpline)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [check],
        inputs = [binary, ctx.file.source, ctx.file._check, ctx.file._ledger, ctx.file._rules] + parts.values() + ctx.files._lab_sources + warpline,
        outputs = [verdict],
        env = {"FSLA_WARPLINE_PYTHON": package.dirname.removesuffix("/warpline")},
        mnemonic = "LabCheck",
        progress_message = "Checking %s's code against what it declares" % ctx.label.name,
    )
    files = [binary] + parts.values()
    return [
        DefaultInfo(files = depset(files), runfiles = ctx.runfiles(files = files)),
        OutputGroupInfo(_validation = depset([verdict])),
    ]

_lab_experiment = rule(
    implementation = _lab_experiment_impl,
    attrs = {
        "binary": attr.label(mandatory = True, executable = True, cfg = "target", doc = "the experiment's binary, linked from its unit"),
        "unit": attr.label(mandatory = True, doc = "the experiment's compile (`cuda_unit`), whose diagnostics the target keeps"),
        "source": attr.label(mandatory = True, allow_single_file = [".cu"], doc = "the experiment's source, which its checks read (its header, its comments)"),
        "_facts": attr.label(default = "//tools:lab_facts.py", allow_single_file = True),
        "_sass": attr.label(default = "//tools:sass.py", allow_single_file = True),
        "_check": attr.label(default = "//tools:lab_check.py", allow_single_file = True),
        "_lab_sources": attr.label(default = "//tools:lab_sources"),
        "_ledger": attr.label(default = "//tools:notice_ledger.json", allow_single_file = True),
        "_rules": attr.label(default = "//:docs/rules.json", allow_single_file = True),
        "_warpline": attr.label(default = "@warpline//:checker"),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True, cfg = "exec"),
        "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True, cfg = "exec"),
        "_cuxxfilt": attr.label(default = "@cuda_tools//:bin/cu++filt", allow_single_file = True, cfg = "exec"),
        "_nvcc": attr.label(default = "@cuda_tools//:bin/nvcc", allow_single_file = True, cfg = "exec"),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "One lab experiment's outputs: its binary, and beside it what the graph's disassemblers read of it (the cubin, the " +
          "listings, the demangled names, the toolkit's version) and the compile's diagnostics; its checks are the target's validation.",
)

def lab_experiment(name, src, includes = [], deps = []):
    """The lab experiment `name` (`src`, its source): its unit, its binary, the outputs the lab reads and its run target
    `<name>.run` (the package's `:stamp`, `workspace_stamp`, the commit its records name). `includes`: the sources of
    other experiments it includes by name (`call_pair` is built of two others'). `deps`: the kernels' headers it includes beyond the
    harness's own (`//csrc/fsla:harness_headers`), so that an edit to a kernel's header rebuilds only the experiments that name it."""
    cuda_unit(
        name = name + "_unit",
        srcs = [src],
        hdrs = includes,
        archs = ["compute_90a:" + _ARCH],
        copts = _COPTS,
        deps = [
            "//csrc/fsla:harness_headers",
            ":lab_headers",
            "@warpline//:host",
            "@warpline//:stamp",
        ] + deps,
    )

    #: the binary is named as its pack names it (`<name>.<arch>`)
    cc_binary(
        name = "%s.%s" % (name, _ARCH),
        deps = [
            ":%s_unit" % name,
            "@cuda_tools//:cuda_runtime_static",
        ],
    )
    _lab_experiment(
        name = name,
        binary = ":%s.%s" % (name, _ARCH),
        unit = ":%s_unit" % name,
        source = src,
    )

    #: THE EXPERIMENT'S RUN TARGET: `bazel run //measure/lab:<name>.run` measures and stores it (`lab run`), and
    #: `-- <command> ...` is any other lab command that reads its build (`sass`, `ncu`, `ledger`, `waits`, `steps`, `pack`)
    lab_run(
        name = name + ".run",
        experiment = ":" + name,
        library = "@warpline//:warpline",
        stamp = ":stamp",
        workspace = native.module_name(),
    )

_CHECKED_TEST = """\
#!/bin/sh
# the lab's checks passed to build these verdicts (a check that refuses fails its action, so a verdict exists only for an experiment that passed): each is there and says so
status=0
for verdict in {verdicts}; do
  if ! test -s "$TEST_SRCDIR/_main/$verdict"; then
    echo "no verdict: $verdict" >&2
    status=1
  fi
done
exit $status
"""

def _lab_checked_test_impl(ctx):
    verdicts = []
    for experiment in ctx.attr.experiments:
        files = experiment[OutputGroupInfo]._validation.to_list()
        verdicts.extend(files)
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _CHECKED_TEST.format(verdicts = " ".join(["'%s'" % v.short_path for v in verdicts])), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = verdicts))]

lab_checked_test = rule(
    implementation = _lab_checked_test_impl,
    test = True,
    attrs = {"experiments": attr.label_list(mandatory = True, providers = [OutputGroupInfo], doc = "the lab's experiments (`lab_experiment`), each whose verdict (its LabCheck action's output) the test depends on")},
    doc = "The gate's lab build: depends on every experiment's verdict, so each experiment is compiled and read by the graph's disassemblers " +
          "and refused by the lab's checks in the commit that breaks it (the lab build is cached; `bazel build //measure/lab/...` runs the same).",
)
