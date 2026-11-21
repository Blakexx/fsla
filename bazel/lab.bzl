"""THE LAB'S EXPERIMENTS AS THE GRAPH'S TARGETS (docs/internals/tools/lab.md#graph): each `measure/lab/experiments/<name>.cu` is compiled by
the graph's own CUDA rule (`bazel/cuda.bzl`'s `cuda_unit`, the rule that builds the library) for sm_90a and linked into its
binary; the graph's disassemblers read the binary once (`tools/lab_facts.py`); and the checks -- the SASS checker, the notices'
ledger, the declarations' (`tools/lab_check.py`) -- are a validation action over those files, so a refusal fails the build. The lab
(`tools/lab.py`) builds the target and reads what it holds: it runs no compiler and no disassembler."""

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//bazel:cuda.bzl", "cuda_unit")

#: the lab's architecture: the one of this tree (`tools/lab.py`'s `ARCH`), and `lab.cuh`'s `kLabBuild` compute capability (`measure/hardware.py`'s
#: `cc_of`: x 10, as `__CUDA_ARCH__` reads it)
_ARCH = "sm_90a"
_CC = "900"

#: the experiment's compile: the lab's own flags. ptxas's own options the build switches on (`ptxas --help`, passed through nvcc's
#: `-Xptxas`): the warning where a function spills registers to local memory, the warning where a function uses local memory (its
#: stack frame's size in the text) and the verbose per-function summary
_COPTS = [
    "-O3",
    "-std=c++20",
    "-DROLA_LAB_CC=" + _CC,
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
    package = [f for f in ctx.files._warpline if f.path.endswith("/warpline/__init__.py")]
    if len(package) != 1:
        fail("%s: %s names %d `warpline/__init__.py`, not one" % (ctx.label, ctx.attr._warpline.label, len(package)))
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [check],
        inputs = [binary, ctx.file.source, ctx.file._check, ctx.file._ledger, ctx.file._rules] + parts.values() + ctx.files._lab_sources + ctx.files._warpline,
        outputs = [verdict],
        env = {"ROLA_WARPLINE_PYTHON": package[0].dirname.removesuffix("/warpline")},
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
        "_warpline": attr.label(default = "@warpline//:warpline"),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True, cfg = "exec"),
        "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True, cfg = "exec"),
        "_cuxxfilt": attr.label(default = "@cuda_tools//:bin/cu++filt", allow_single_file = True, cfg = "exec"),
        "_nvcc": attr.label(default = "@cuda_tools//:bin/nvcc", allow_single_file = True, cfg = "exec"),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "One lab experiment's outputs: its binary, and beside it what the graph's disassemblers read of it (the cubin, the " +
          "listings, the demangled names, the toolkit's version) and the compile's diagnostics; its checks are the target's validation.",
)

def lab_experiment(name, src, includes = []):
    """The lab experiment `name` (`src`, its source): its unit, its binary and the outputs the lab reads. `includes`: the sources of
    other experiments it includes by name (`call_pair` is built of two others')."""
    cuda_unit(
        name = name + "_unit",
        srcs = [src],
        hdrs = includes,
        archs = ["compute_90a:" + _ARCH],
        copts = _COPTS,
        host_cc = "@host_cc//:identity.txt",
        deps = [
            "//csrc/rola:headers",
            ":lab_headers",
            "@warpline//:host",
            "@warpline//:stamp",
        ],
    )

    #: the binary is named as its pack names it (`<name>.<arch>`)
    cc_binary(
        name = "%s.%s" % (name, _ARCH),
        deps = [
            ":%s_unit" % name,
            "@cuda//:cuda_runtime_static",
        ],
    )
    _lab_experiment(
        name = name,
        binary = ":%s.%s" % (name, _ARCH),
        unit = ":%s_unit" % name,
        source = src,
    )
