"""THE GRAPH'S RULES: the generated headers for a build's parameters, and the library
-- `fsla_cu13/_C.abi3.so` linked by the pinned mold beside its build record, which stay in the graph's output (`.bazel/bin`)
(docs/build-graph.md#compile); the checks -- over the tree's files, of the commit, and over each unit's own compile
(docs/build-graph.md#checks); and the timing entries and sessions, the instruments and the records' stores
(docs/measurement.md)."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@fsla_devtools//bazel:hermetic.bzl", _hermetic_test = "hermetic_test")
load(
    "@fsla_devtools//bazel:measure.bzl",
    _diff_side = "diff_side",
    _instrument = "instrument",
    _interleaved_timing = "interleaved_timing",
    _measure_memory = "measure_memory",
    _results_store = "results_store",
    _timing_bundle = "timing_bundle",
    _timing_entry = "timing_entry",
    _workspace_stamp = "workspace_stamp",
)
load(
    "@fsla_devtools//bazel:rules.bzl",
    _checkout_test = "checkout_test",
    _device_lane = "device_lane",
    _docrefs_test = "docrefs_test",
    _environment = "environment",
    _pytest_tier = "pytest_tier",
    _tool_run = "tool_run",
    _tool_test = "tool_test",
    _tree_part = "tree_part",
    _tree_test = "tree_test",
    _venv_line = "venv_line",
)
load("@fsla_env//:flags.bzl", "BUILT_ARCHS", "VENV")
load("@fsla_tree//:tree.bzl", "TRACKED")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@warpline//:checker.bzl", "WARPLINE_CHECK_ATTRS", "warpline_check_actions")
load(":arms.bzl", "ARM_FAMILIES")
load(":cuda.bzl", "CudaUnitInfo")

#: THE GRAPH'S ENVIRONMENT BY LABEL (`@fsla_env//:flags.bzl`'s `VENV`, fsla-devtools' `environment`): what a rule here that
#: runs on the environment takes, its completion mark resolved to the store venv where the script or action runs
_VENV = {"_venv": attr.label(default = VENV, allow_files = True)}

#: the build's parameters (the root package's settings), as every rule here reads them
_SETTINGS = {"_archs": "//:archs", "_contract": "//:contract"}

def _setting_attrs():
    return {name: attr.label(default = label) for name, label in _SETTINGS.items()}

def _value(ctx, name):
    return getattr(ctx.attr, name)[BuildSettingInfo].value

def _parameters(ctx, args):
    args.add("--archs", ",".join(_value(ctx, "_archs")))
    args.add("--contract", _value(ctx, "_contract"))

#: the header `bazel/headers.py` writes: the builds the binary carries (`hardware/registry.py`'s)
_BUILDS = "builds.inc"

def _headers_impl(ctx):
    builds = ctx.actions.declare_file("generated/" + _BUILDS)
    args = ctx.actions.args()
    args.add(ctx.file._script)
    args.add("--out-dir", builds.dirname)
    args.add("--archs", ",".join(_value(ctx, "_archs")))
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files._sources + [ctx.file._script],
        outputs = [builds],
        mnemonic = "FslaHeaders",
        progress_message = "Writing the build's carried builds",
    )
    #: each family's selection header and Python table (bazel/arms.bzl's `arm_family`), beside it
    arm_files = [f for target in ctx.attr.arm_families for f in target[DefaultInfo].files.to_list()]
    headers = [builds] + [f for f in arm_files if f.extension == "inc"]
    compilation = cc_common.create_compilation_context(headers = depset(headers), includes = depset([builds.dirname]))
    return [DefaultInfo(files = depset([builds] + arm_files)), CcInfo(compilation_context = compilation)]

fsla_headers = rule(
    implementation = _headers_impl,
    attrs = _setting_attrs() | {
        "arm_families": attr.label_list(allow_files = True, doc = "each arm family's selection header and Python table (`arm_selection`, `arm_table`)"),
        "_script": attr.label(default = "//bazel:headers.py", allow_single_file = True),
        "_sources": attr.label(default = "//bazel:generator_sources"),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "The carried builds for the build's architectures, and the arm families' selection headers and tables beside them.",
)

def _include_root_impl(ctx):
    roots = ["%s/%s" % (ctx.label.package, root) for root in ctx.attr.roots]
    own = cc_common.create_compilation_context(headers = depset(ctx.files.hdrs), includes = depset(roots))
    return [CcInfo(compilation_context = cc_common.merge_compilation_contexts(
        compilation_contexts = [own] + [dep[CcInfo].compilation_context for dep in ctx.attr.deps],
    ))]

include_root = rule(
    implementation = _include_root_impl,
    attrs = {
        "hdrs": attr.label_list(allow_files = True),
        "roots": attr.string_list(doc = "the include roots, relative to the package"),
        "deps": attr.label_list(providers = [CcInfo], doc = "the generated headers beside them (`common/hardware.cuh`)"),
    },
    doc = "Source headers under `-I` roots, searched before any dependency's (docs/build-graph.md#compile).",
)

#: the build's architectures as the compile's (`//bazel:cuda_archs`): the units carry the runtime's headers alone, the library links
#: the runtime itself, so no unit carries a dynamic library whose search path the link would write into the binary
def _cuda_settings(archs):
    return {
        #: one spec an architecture, each its own virtual one: a comma list would compile every card from the first's
        str(Label("//bazel:cuda_archs")): ";".join(["compute_%s:sm_%s" % (a, a) for a in archs]),
    }

def _archs_transition_impl(settings, _attr):
    return _cuda_settings(settings[str(Label("//:archs"))])

_archs_transition = transition(
    implementation = _archs_transition_impl,
    inputs = [str(Label("//:archs"))],
    outputs = [str(Label("//bazel:cuda_archs"))],
)

#: the libraries the extension links by path: torch's two, the CUDA runtime and the driver's stub (the real driver at
#: load); each NEEDED by its soname, and no search path written
_CUDA_LINKED = ("libcudart.so.13", "libcuda.so")

_INSTALL = """\
#!/bin/bash
# the graph's library stays a build output (`.bazel/bin/fsla_cu13/`, which the checkout's venv imports from, and which
# the wheels, the tools and the box's copy read): nothing a commit does not hold is written into the source tree. What
# is written is the build's generated headers, where the tools that compile the tree's units read them (build/generated,
# ignored: tools/life_ranges.py): copies, never links into Bazel's output tree, each through a temporary file renamed
# into place, so a reader never sees half a header
set -euo pipefail
: "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
generated="$BUILD_WORKSPACE_DIRECTORY/build/generated"
mkdir -p "$generated"
for f in {headers}; do
  to="$generated/$(basename "$f")"
  install -m 0644 "$f" "$to.graph" && mv -f "$to.graph" "$to"
done
echo "the library is .bazel/bin/fsla_cu13/_C.abi3.so; installed the build's generated headers from the graph"
"""

def _link(ctx, so):
    """Link `deps` into the shared library `so`: mold, torch's libraries and the CUDA runtime by path, each NEEDED by
    its soname."""
    cc_toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    ld = [f for f in ctx.files._linker if f.basename == "ld.mold"][0]
    linked = ctx.files.shared_libraries + [f for f in ctx.files._runtime_libs if f.basename in _CUDA_LINKED]
    if len(linked) != len(ctx.files.shared_libraries) + len(_CUDA_LINKED):
        fail("the CUDA runtime's files lack one of %s" % (_CUDA_LINKED,))
    cc_common.link(
        actions = ctx.actions,
        feature_configuration = features,
        cc_toolchain = cc_toolchain,
        linking_contexts = [dep[CcInfo].linking_context for dep in ctx.attr.deps],
        name = ctx.label.name,
        output_type = "dynamic_library",
        main_output = so,
        user_link_flags = ["-B" + ld.dirname + "/", "-fuse-ld=mold"] + [f.path for f in linked],
        additional_inputs = ctx.files._linker + linked,
    )

def _gated(ctx, probes):
    """What the library's units state for the gate, from `deps`: the ptxas report of every unit it links (the build's notices), and each
    built architecture's declarations file (`bazel/gate_declarations.py`: the declarations beside the kernels, the arm families', each
    arm's pattern and, but for a library of `probes`, each source kernel as a presence, fsla's notes, strict). A declaration with no
    architecture attaches to every one the build carries."""
    infos = [dep[CudaUnitInfo] for dep in ctx.attr.deps if CudaUnitInfo in dep]
    reports = [unit.report for unit in depset(transitive = [i.units for i in infos]).to_list()]
    archs = ["sm_" + a for a in _value(ctx, "_archs")]
    stated = {arch: [] for arch in archs}
    for d in depset(transitive = [i.declarations for i in infos]).to_list():
        for arch in ([d.arch] if d.arch else archs):
            if arch in stated:
                stated[arch].append({"kernel": d.kernel, "rule": d.rule, "allow": d.allow, "reason": d.reason})
    stated_file = ctx.actions.declare_file(ctx.label.name + "_declarations_stated.json")
    ctx.actions.write(stated_file, json.encode_indent({"architectures": stated}) + "\n")
    #: an arm family whose arms the build computes (`arm_declarations`) wrote its declarations and arms as files
    files = depset(transitive = [i.declaration_files for i in infos]).to_list()
    declared = {arch: ctx.actions.declare_file("%s_%s.declarations.json" % (ctx.label.name, arch)) for arch in archs}
    args = ctx.actions.args()
    args.add(ctx.file._gate_declarations)
    args.add("--stated", stated_file)
    args.add_all(files, before_each = "--file")
    args.add("--rules", ctx.file._rules)
    if probes:
        args.add("--probes")
    args.add_all(["%s=%s" % (arch, f.path) for arch, f in declared.items()], before_each = "--out")
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = [stated_file, ctx.file._gate_declarations, ctx.file._rules] + files + ctx.files._gate_sources,
        outputs = declared.values(),
        mnemonic = "FslaDeclarations",
        progress_message = "Writing the SASS gate's declarations for %{label}",
    )
    return declared, reports

def warpline_root(ctx, target):
    """The directory that holds the `warpline` package of `target` (a warpline module's target: the package's `__init__.py` is the
    shared part's, which a module's runfiles carry), as the resolved path `FSLA_WARPLINE_PYTHON` names, and the files it holds."""
    files = target[DefaultInfo].default_runfiles.files.to_list()
    package = [f for f in files if f.path.endswith("/warpline/__init__.py")]
    if len(package) != 1:
        fail("%s: %s names %d `warpline/__init__.py`, not one" % (ctx.label, target.label, len(package)))
    return package[0], files

#: CLOSED-WORLD RULE 2 on the linked binary: no PTX image a driver could just-in-time compile for an architecture nobody measured
#: (`cuobjdump --list-ptx` lists none); the verdict is the action's output
_PTX_GATE = """\
set -euo pipefail
listed="$("$1" --list-ptx "$2")"
if printf '%s\\n' "$listed" | grep '^PTX file'; then
  echo "RED  rule 2: the binary carries the PTX image(s) above -- a binary carries cubins alone (docs/build.md#what-the-build-checks)" >&2
  exit 1
fi
echo "rule 2: no PTX image, the binary carries cubins alone" > "$3"
"""

def _sass_gate(ctx, so, stem, probes = False):
    """THE SASS GATE (KERNEL_STANDARDS section 20) on the linked library `so`: warpline's check (`warpline_check_actions`, its manifest
    and its comparison) once a built architecture, on the graph's disassemblers, with the units' ptxas reports as the build's notices,
    the project's ledger and the architecture's declarations file (`_gated`: fsla's declarations, its arms and source kernels as
    presences, its notes, strict); and closed-world rule 2 (`_PTX_GATE`). A refusal fails its action, so the library's build. Returns
    the verdicts (each check's comparison text and rule 2's, `<stem>.<arch>.check.txt` and `<stem>.ptx.txt`) and the static manifests
    (`<stem>.<arch>.manifest.json`)."""
    declared, reports = _gated(ctx, probes)
    toolkit = [ctx.file._cuobjdump, ctx.file._nvdisasm, ctx.file._cuxxfilt]
    checks = [warpline_check_actions(
        ctx,
        name = "%s.%s" % (stem, arch),
        binary = so,
        declarations = declared[arch],
        arch = arch,
        toolkit = toolkit,
        notices = reports,
        ledger = ctx.file._ledger,
    ) for arch in sorted(declared)]
    ptx = ctx.actions.declare_file(stem + ".ptx.txt")
    ctx.actions.run_shell(
        inputs = [so, ctx.file._cuobjdump],
        outputs = [ptx],
        arguments = [ctx.file._cuobjdump.path, so.path, ptx.path],
        command = _PTX_GATE,
        mnemonic = "FslaPtxGate",
        progress_message = "Closed-world rule 2 on %{label}'s library: no PTX image",
    )
    return [c.report for c in checks] + [ptx], [c.manifest for c in checks]

#: the SASS gate's inputs, every library rule's: warpline's check (`WARPLINE_CHECK_ATTRS`), the graph's disassemblers, the project's
#: ledger, and what the declarations are written from
_SASS_GATE_ATTRS = WARPLINE_CHECK_ATTRS | {
    "_gate_declarations": attr.label(default = "//bazel:gate_declarations.py", allow_single_file = True),
    "_gate_sources": attr.label(default = "//bazel:gate_sources"),
    "_rules": attr.label(default = "//:docs/rules.json", allow_single_file = True),
    "_ledger": attr.label(default = "//tools:notice_ledger.json", allow_single_file = True),
    "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True, cfg = "exec"),
    "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True, cfg = "exec"),
    "_cuxxfilt": attr.label(default = "@cuda_tools//:bin/cu++filt", allow_single_file = True, cfg = "exec"),
    "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
}

def _extension_impl(ctx):
    so = ctx.actions.declare_file("fsla_cu13/_C.abi3.so")
    _link(ctx, so)

    #: THE BUILD GATES AND THE RECORD: rules 1, 3 and 5, then the record (bazel/record.py)
    record = ctx.actions.declare_file("fsla_cu13/_build_config.py")
    tables = {f.basename.removesuffix("_arms.py"): f for f in ctx.files._headers if f.basename.endswith("_arms.py")}
    ptxas = ctx.file._ptxas
    args = ctx.actions.args()
    args.add(ctx.file._record)
    args.add("--out", record)
    args.add("--toolchain", ctx.attr.toolchain)
    args.add("--ptxas", ptxas)
    args.add("--cuobjdump", ctx.file._cuobjdump)
    args.add("--nvdisasm", ctx.file._nvdisasm)
    args.add("--torch", ctx.attr.torch)
    args.add("--torch-cuda", ctx.attr.torch_cuda)
    args.add("--cxx11-abi", str(ctx.attr.cxx11_abi))
    for family in ARM_FAMILIES:
        args.add("--table", "%s=%s" % (family, tables[family].path))
    _parameters(ctx, args)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files._record_sources + ctx.files._sources + tables.values() + [ctx.file._record, ptxas, ctx.file._cuobjdump, ctx.file._nvdisasm],
        outputs = [record],
        mnemonic = "FslaRecord",
        progress_message = "Gating the build (rules 1, 3, 5) and writing its record",
    )

    #: THE BINARY GATES, on the linked library: the SASS gate (with rule 2), and the arm table with its coverage sweep (the binary
    #: asked what it carries)
    sass_verdicts, manifests = _sass_gate(ctx, so, "fsla_cu13/sass_gate")
    arm_verdict = ctx.actions.declare_file("fsla_cu13/arm_gate.txt")
    headers = {family: [f for f in ctx.files._headers if f.basename == "%s_selection.inc" % family][0] for family in ARM_FAMILIES}
    stub = [f for f in ctx.files._runtime_libs if f.path.endswith("stubs/libcuda.so")][0]
    arm_args = ctx.actions.args()
    arm_args.add(ctx.file._arm_gate)
    arm_args.add("--so", so)
    for family in ARM_FAMILIES:
        arm_args.add("--header", "%s=%s" % (family, headers[family].path))
        arm_args.add("--table", "%s=%s" % (family, tables[family].path))
    #: the environment's interpreter by its execution-root path, through the environment repository's link to the venv
    mark, env = _environment(ctx, ctx.attr._venv)
    arm_args.add("--python", mark.dirname + "/bin/python3")
    arm_args.add("--driver-stub", stub)
    arm_args.add("--out", arm_verdict)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [arm_args],
        inputs = ctx.files._sources + [so, stub, ctx.file._arm_gate] + headers.values() + [tables[family] for family in ARM_FAMILIES] + env,
        outputs = [arm_verdict],
        mnemonic = "FslaArmGate",
        progress_message = "The arm table and its coverage sweep on %{label}'s library",
        #: the environment's interpreter reads its packages from the store, which the sandbox does not hold; the
        #: environment's key, an input, keys the action by it
        execution_requirements = {"no-sandbox": "1"},
    )

    verdicts = sass_verdicts + [arm_verdict]
    install = ctx.actions.declare_file(ctx.label.name + "_install.sh")
    installed = [f for f in ctx.files._headers if f.extension in ("inc", "py")]
    ctx.actions.write(install, _INSTALL.format(headers = " ".join([f.short_path for f in installed])), is_executable = True)
    #: the verdicts are runfiles too: `bazel run` installs nothing a gate has not passed
    return [
        DefaultInfo(files = depset([so, record] + verdicts), executable = install,
                    runfiles = ctx.runfiles(files = [so, record] + verdicts + installed)),
        #: THE STATIC MANIFEST of each built architecture (warpline's, docs/internals/tools/sass_gate.md#manifest): `//:static_manifest`
        OutputGroupInfo(manifest = depset(manifests)),
        #: the units it links, for the census (`ratify_census`)
        CudaUnitInfo(
            units = depset(transitive = [dep[CudaUnitInfo].units for dep in ctx.attr.deps if CudaUnitInfo in dep]),
            declarations = depset(transitive = [dep[CudaUnitInfo].declarations for dep in ctx.attr.deps if CudaUnitInfo in dep]),
            declaration_files = depset(transitive = [dep[CudaUnitInfo].declaration_files for dep in ctx.attr.deps if CudaUnitInfo in dep]),
        ),
    ]

fsla_extension = rule(
    implementation = _extension_impl,
    cfg = _archs_transition,
    executable = True,
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    attrs = _setting_attrs() | _SASS_GATE_ATTRS | _VENV | {
        "deps": attr.label_list(providers = [CcInfo], doc = "the units and the arm set"),
        "shared_libraries": attr.label_list(allow_files = [".so"], doc = "torch's libraries the extension links"),
        "torch": attr.string(mandatory = True, doc = "the environment's torch version (@fsla_env//:flags.bzl)"),
        "torch_cuda": attr.string(mandatory = True, doc = "the CUDA release the environment's torch was built for"),
        "toolchain": attr.string(mandatory = True, doc = "the toolchain's name (its record)"),
        "cxx11_abi": attr.bool(mandatory = True, doc = "the environment's torch ABI flag"),
        "_linker": attr.label(default = "@mold//:linker"),
        "_ptxas": attr.label(default = "@cuda_tools//:bin/ptxas", allow_single_file = True),
        "_runtime_libs": attr.label(default = "@cuda_tools//:runtime_libs"),
        "_record": attr.label(default = "//bazel:record.py", allow_single_file = True),
        "_record_sources": attr.label(default = "//bazel:record_sources"),
        "_sources": attr.label(default = "//bazel:generator_sources"),
        "_arm_gate": attr.label(default = "//bazel:gates.py", allow_single_file = True),
        "_headers": attr.label(default = "//csrc/fsla:generated"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
    doc = "The extension `fsla_cu13/_C.abi3.so` and its build record, left in the graph's output (`.bazel/bin`); `bazel run` copies the generated headers.",
)

_INSTALL_INSTRUMENT = """\
#!/bin/bash
# an instrument library stays a build output, beside the extension in `.bazel/bin/fsla_cu13/` (bazel/defs.bzl's fsla_instrument)
set -euo pipefail
echo "the library is .bazel/bin/fsla_cu13/$(basename {so})"
"""

def _instrument_impl(ctx):
    so = ctx.actions.declare_file("fsla_cu13/%s.abi3.so" % ctx.attr.module)
    _link(ctx, so)
    #: the SASS gate, as on the extension, its probes keeping only the signatures their units' declarations name and owing none of
    #: the source's kernels; its verdicts runfiles, so `bazel run` installs nothing it has not passed
    verdicts, manifests = _sass_gate(ctx, so, "fsla_cu13/%s.sass_gate" % ctx.attr.module, probes = True)
    install = ctx.actions.declare_file(ctx.label.name + "_install.sh")
    ctx.actions.write(install, _INSTALL_INSTRUMENT.format(so = shell.quote(so.short_path)), is_executable = True)
    return [
        DefaultInfo(files = depset([so] + verdicts), executable = install, runfiles = ctx.runfiles(files = [so] + verdicts)),
        OutputGroupInfo(manifest = depset(manifests)),
    ]

fsla_instrument = rule(
    implementation = _instrument_impl,
    cfg = _archs_transition,
    executable = True,
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    attrs = _setting_attrs() | _SASS_GATE_ATTRS | {
        "module": attr.string(mandatory = True, doc = "the module's name in fsla_cu13 (`_C_parts`)"),
        "deps": attr.label_list(providers = [CcInfo], doc = "its units, whose declarations are the signatures its probes keep on purpose (docs/internals/tools/sass_gate.md#probes)"),
        "shared_libraries": attr.label_list(allow_files = [".so"], doc = "torch's libraries it links"),
        "_linker": attr.label(default = "@mold//:linker"),
        "_runtime_libs": attr.label(default = "@cuda_tools//:runtime_libs"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
    doc = "An instrument's library, `fsla_cu13/<module>.abi3.so`, linked as the extension is, for the architectures " +
          "`--//:archs` names, and SASS-gated as it is, its probes keeping the signatures its units declare; never " +
          "shipped, so no record or arm gate; `bazel run` copies it into the checkout.",
)

def _unit_sass_impl(ctx):
    found = [u for u in ctx.attr.library[CudaUnitInfo].units.to_list() if u.source.rsplit("/", 1)[-1].removesuffix(".cu") == ctx.attr.unit]
    if not found:
        fail("%s: %s links no unit named %s" % (ctx.label, ctx.attr.library.label, ctx.attr.unit))
    return [DefaultInfo(files = depset([u.sass for u in found]))]

unit_sass = rule(
    implementation = _unit_sass_impl,
    attrs = {
        "library": attr.label(mandatory = True, providers = [CudaUnitInfo], doc = "the library that links the unit"),
        "unit": attr.string(mandatory = True, doc = "the unit's source, by its name without `.cu`"),
    },
    doc = "One unit's SASS as text, by the graph's own disassembler, for every architecture the build carries: `bazel build //csrc/fsla:<unit>_sass`, " +
          "so a SASS diff never needs a disassembler typed by hand (the file is `_objs/<target>/<unit>.sass` under `.bazel/bin`).",
)

def _census_impl(ctx):
    #: the static manifests of the library's SASS gate, `<stem>.<arch>.manifest.json`: the census reads their figures
    manifests = {f.basename.split(".")[-3]: f for f in ctx.attr.library[OutputGroupInfo].manifest.to_list()}
    tools = {
        "ptxas": ctx.file._ptxas,
        "cuobjdump": ctx.file._cuobjdump,
        "nvdisasm": ctx.file._nvdisasm,
    }
    package, warpline = warpline_root(ctx, ctx.attr._warpline)
    record = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(record, json.encode_indent({
        "manifests": {arch: f.short_path for arch, f in manifests.items()},
        "tools": {name: f.short_path for name, f in tools.items()},
        "warpline": package.short_path.rsplit("/", 2)[0],
        "parameters": {"archs": ",".join(_value(ctx, "_archs")), "contract": _value(ctx, "_contract")},
    }) + "\n")
    files = [record] + tools.values() + manifests.values() + warpline
    return [DefaultInfo(files = depset([record]), runfiles = ctx.runfiles(files = files))]

ratify_census = rule(
    implementation = _census_impl,
    attrs = _setting_attrs() | {
        "library": attr.label(mandatory = True, providers = [OutputGroupInfo], doc = "the library whose static manifests are read"),
        "_ptxas": attr.label(default = "@cuda_tools//:bin/ptxas", allow_single_file = True),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True),
        "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True),
        "_warpline": attr.label(default = "@warpline//:checker"),
    },
    doc = "The census record (tools/ratify.py's `--graph`): the static manifest of each architecture the library's SASS gate " +
          "wrote (warpline's), the graph's assembler and readers, the directory of warpline's package (whose reader of the " +
          "manifest the census uses) and the build's parameters; every path relative to the runfiles' root, which holds them all.",
)

_CENSUS_TEST = """\
#!/bin/bash
# a tool on the graph's census (bazel/defs.bzl's census_test): `tools/{script}` in a copy of the tree, so a tool that
# resolves a path reads the copy (rule 14); the census record's paths are relative to the runfiles
set -euo pipefail
{venv}tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
export PATH="$venv/bin:/usr/bin:/bin"
exec python3 tools/{script} --graph {census} --runfiles "$TEST_SRCDIR/_main" {args}
"""

def _census_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    mark, env = _environment(ctx, ctx.attr._venv)
    ctx.actions.write(script, _CENSUS_TEST.format(
        venv = _venv_line(mark),
        script = ctx.attr.script,
        census = shell.quote(ctx.file.census.short_path),
        args = " ".join([shell.quote(arg) for arg in ctx.attr.args_]),
    ), is_executable = True)
    runfiles = ctx.runfiles(files = ctx.files.data + env).merge(ctx.attr.census[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = script, runfiles = runfiles)]

census_test = rule(
    implementation = _census_test_impl,
    test = True,
    attrs = _VENV | {
        "script": attr.string(mandatory = True, doc = "the tool under tools/ that reads the census (`--graph`, `--runfiles`)"),
        "args_": attr.string_list(doc = "its other arguments"),
        "census": attr.label(mandatory = True, allow_single_file = True, doc = "the census record (`ratify_census`)"),
        "data": attr.label_list(allow_files = True, doc = "the tree the tool reads: its sources and tools"),
    },
    doc = "A check over the graph's build: a tool under tools/ run on the census in a copy of the tree, cached on the " +
          "census and the tree.",
)

_RATIFY = """\
#!/bin/bash
# `tools/ratify.py --graph` on the graph's build, in the checkout (bazel/defs.bzl's ratify_run): `--census` reports; the
# census record's paths are relative to the runfiles, where `bazel run` starts
set -euo pipefail
runfiles="$PWD"
{venv}cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec "$venv/bin/python3" tools/ratify.py --graph "$runfiles/{census}" --runfiles "$runfiles" "$@"
"""

def _ratify_run_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    mark, env = _environment(ctx, ctx.attr._venv)
    ctx.actions.write(script, _RATIFY.format(venv = _venv_line(mark), census = ctx.file.census.short_path), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = env).merge(ctx.attr.census[DefaultInfo].default_runfiles))]

ratify_run = rule(
    implementation = _ratify_run_impl,
    executable = True,
    attrs = _VENV | {"census": attr.label(mandatory = True, allow_single_file = True, doc = "the census record (`ratify_census`)")},
    doc = "`bazel run` the census on the graph's build in the checkout: its arguments are ratify's (`--census`, `--json PATH`).",
)

_WHEELS = """\
#!/bin/bash
# the wheels of the graph's library (bazel/defs.bzl's fsla_wheels): `tools/wheels.py` in the checkout, on a clean
# commit; the library's and the record's paths are relative to the runfiles
set -euo pipefail
runfiles="$PWD"
{venv}cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec "$venv/bin/python3" tools/wheels.py --library "$runfiles/{so}" --record "$runfiles/{record}" "$@"
"""

def _wheels_impl(ctx):
    files = ctx.attr.library[DefaultInfo].files.to_list()
    so = [f for f in files if f.basename == "_C.abi3.so"][0]
    record = [f for f in files if f.basename == "_build_config.py"][0]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    mark, env = _environment(ctx, ctx.attr._venv)
    ctx.actions.write(script, _WHEELS.format(venv = _venv_line(mark), so = so.short_path, record = record.short_path), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = [so, record] + env))]

fsla_wheels = rule(
    implementation = _wheels_impl,
    executable = True,
    attrs = _VENV | {
        "library": attr.label(mandatory = True, doc = "the extension (`fsla_extension`), its library and record"),
    },
    doc = "`bazel run` the two wheels of the graph's library into the checkout's dist/ (docs/build.md#wheels): a " +
          "clean commit, the record a shippable build's.",
)

_LOCK_UPDATE = """\
#!/bin/bash
# requirements.lock from requirements.in, by the graph's pinned uv (bazel/defs.bzl's lock_update): in the checkout, never
# cached, since the compile reads the package indexes
set -euo pipefail
uv="$(readlink -f {uv})"
cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec "$uv" pip compile requirements.in --python-version 3.11 --generate-hashes \\
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \\
  --custom-compile-command "bazel run //:lock_update" -o requirements.lock "$@"
"""

def _lock_update_impl(ctx):
    uv = ctx.file._uv
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _LOCK_UPDATE.format(uv = shell.quote(uv.short_path)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = [uv]))]

lock_update = rule(
    implementation = _lock_update_impl,
    executable = True,
    attrs = {"_uv": attr.label(default = "@uv//:uv", allow_single_file = True, cfg = "exec")},
    doc = "`bazel run` writes requirements.lock from requirements.in with the pinned uv (the environment's lock, " +
          "docs/build-graph.md#environment); a compile reads the package indexes, so it is never a test.",
)

_COMMITTED_UPDATE = """\
#!/bin/bash
# writes a committed file from what the graph wrote (bazel/defs.bzl's committed_file): a copy into the checkout, through a
# temporary file renamed into place
set -euo pipefail
to="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"/{path}
install -m 0644 {generated} "$to.graph" && mv -f "$to.graph" "$to"
echo "wrote "{path}" from the graph"
"""

def _committed_update_impl(ctx):
    files = ctx.attr.generated[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("%s: %s names %d files, not one" % (ctx.label, ctx.attr.generated.label, len(files)))
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _COMMITTED_UPDATE.format(path = shell.quote(ctx.attr.path), generated = shell.quote(files[0].short_path)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = files))]

_committed_update = rule(
    implementation = _committed_update_impl,
    executable = True,
    attrs = {
        "generated": attr.label(mandatory = True, doc = "the target whose one file is what the committed file holds"),
        "path": attr.string(mandatory = True, doc = "the committed file's path in the checkout"),
    },
)

def committed_file(name, committed, generated):
    """A COMMITTED FILE THE GRAPH WRITES (docs/build-graph.md#committed-files): `bazel run :<name>_update` writes the
    committed file `committed` (a label of a file in the checkout) from `generated` (the target of the graph's action whose
    one output is what the file holds), and the test `:<name>` fails when the committed file differs from that output."""
    label = native.package_relative_label(committed)
    _committed_update(
        name = name + "_update",
        generated = generated,
        path = (label.package + "/" if label.package else "") + label.name,
    )
    tool_test(
        name = name,
        size = "small",
        args_ = [
            "--committed",
            "$(rootpath %s)" % committed,
            "--graph",
            "$(rootpath %s)" % generated,
            "--update",
            "//%s:%s_update" % (native.package_name(), name),
        ],
        data = [committed, generated],
        tool = Label("//tools:committed_file.py"),
    )

#: THE GENERIC RULES are fsla-devtools' (`@fsla_devtools//bazel:rules.bzl`), each handed this workspace's environment:
#: its environment by label (VENV), its interpreter, the tracked tree (TRACKED), the pinned toolkit and the devices' identity

def tree_test(**kwargs):
    """A check over the tree's tracked files, the environment's tools on PATH: one test, cached on what it reads."""
    _tree_test(venv = VENV, **kwargs)

def tool_test(**kwargs):
    """A check that is one run of a tree tool over files the graph names, on the environment's base interpreter."""
    _tool_test(python = Label("@fsla_env//:python"), **kwargs)

def checkout_test(name, cmd, **kwargs):
    """A check of the commit, run in the checkout, outside the sandbox, never cached."""
    _checkout_test(name = name, cmd = cmd, venv = VENV, anchor = Label("//:MODULE.bazel"), **kwargs)

def docrefs_test(**kwargs):
    """The doc references (fsla-devtools' `fsla_devtools.docrefs`), on the environment's interpreter, in the checkout."""
    _docrefs_test(python = Label("@fsla_env//:python"), tool = Label("@docrefs//:docrefs.py"),
                  anchor = Label("//:MODULE.bazel"), **kwargs)

_DEV_RUN = """\
#!/bin/bash
# a tools/dev.py command as a run target (bazel/defs.bzl's dev_run): in the checkout, on the environment's interpreter,
# the environment's tools named by their resolved paths (the runfiles' links do not outlive the run)
set -euo pipefail
{venv}{resolve}cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec "$venv/bin/python3" tools/dev.py {command} "$@"
"""

def _dev_run_impl(ctx):
    resolve, words = [], [shell.quote(w) for w in ctx.attr.command]
    if ctx.attr.checkout:
        words += ["--checkout", '"$BUILD_WORKSPACE_DIRECTORY"']
    if ctx.attr.env:
        words += ["--env", '"$venv"']
    tools = []
    for i, (target, flag) in enumerate(ctx.attr.tools.items()):
        found = target[DefaultInfo].files.to_list()
        if len(found) != 1:
            fail("%s: %s names %d files, not one" % (ctx.label, target.label, len(found)))
        tools += found
        resolve.append('tool%d="$(readlink -f %s)"\n' % (i, shell.quote(found[0].short_path)))
        words += [flag, '"$tool%d"' % i]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    mark, env = _environment(ctx, ctx.attr._venv)
    ctx.actions.write(script, _DEV_RUN.format(resolve = "".join(resolve), venv = _venv_line(mark), command = " ".join(words)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = tools + env))]

dev_run = rule(
    implementation = _dev_run_impl,
    executable = True,
    attrs = _VENV | {
        "command": attr.string_list(mandatory = True, doc = "the dev.py command and its fixed arguments"),
        "checkout": attr.bool(doc = "pass the checkout (`--checkout`, $BUILD_WORKSPACE_DIRECTORY)"),
        "env": attr.bool(doc = "pass the graph's environment, its store venv (`--env`)"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "each tool and the flag that names its path"),
    },
    doc = "A tools/dev.py command run by `bazel run` in the checkout (docs/build-graph.md#use): its arguments follow `--`.",
)

def tool_run(**kwargs):
    """A tree tool run by `bazel run` in the checkout, on its own venv or the environment's interpreter
    (docs/build-graph.md#use): its arguments follow `--`."""
    _tool_run(venv = VENV, **kwargs)

def hermetic_test(**kwargs):
    """THE HERMETIC CHECK (fsla-devtools' `hermetic_test`, docs/build-graph.md#rules): the gate's targets read whole, on
    the environment's interpreter; the tree compiles C++, so its one toolchain is the pinned one."""
    _hermetic_test(venv = VENV, cc = True, **kwargs)

#: THIS TREE'S BUNDLE LAYOUT (docs/measurement.md#entries): the arm tables laid out where `measure/provider.py` reads a
#: tree's (`build/generated`); `fsla` and the extension must load from inside the bundle
_ARM_TABLES = {Label("//csrc/fsla:%s_table" % family): "build/generated/%s_arms.py" % family for family in ARM_FAMILIES}

def timing_bundle(name, **kwargs):
    """A BUNDLE of this tree (docs/measurement.md#entries): the Python its executors import and the library with its
    build record, the arm tables laid out where `measure/provider.py` reads them, on the environment's store venv."""
    _timing_bundle(name = name, layout = _ARM_TABLES, venv = VENV, python = Label("@fsla_env//:python"), **kwargs)

def timing_entry(name, **kwargs):
    """A timing entry of this tree: an executor of `measure/provider.py` on cells (`@fsla_cells//:<cell>`), or the clock
    reader (`kind = "clock"`), in this tree's bundle (`//measure:bundle`)."""
    _timing_entry(name = name, bundle = Label("//measure:bundle"), **kwargs)

def diff_side(name, **kwargs):
    """A DIFF SIDE of this tree (docs/measurement.md#the-diffs): an executor of `tests/oracle/sides.py` on cells, in the
    sides' bundle (`//measure:side_bundle`)."""
    _diff_side(name = name, bundle = Label("//measure:side_bundle"), **kwargs)

def interleaved_timing(name, **kwargs):
    """An interleaved session over timing entries, or their null gate, stored by `bazel run` at its own label
    (docs/measurement.md#sessions)."""
    _interleaved_timing(name = name, workspace = native.module_name(), venv = VENV, **kwargs)

def measure_memory(name, **kwargs):
    """A memory pass over timing entries, stored by `bazel run` at its own label (docs/measurement.md#sessions)."""
    _measure_memory(name = name, workspace = native.module_name(), venv = VENV, **kwargs)

def instrument(name, device = False, **kwargs):
    """AN INSTRUMENT (docs/measurement.md#instruments): a tool's JSON over the files it reads, on the environment's
    interpreter; `device`, on the machine's device, keyed on the devices' identity and run by its store target."""
    _instrument(name = name, venv = VENV, devices = Label("@host_gpu//:identity.txt") if device else None, **kwargs)

def results_store(name, **kwargs):
    """A BUILD OUTPUT'S RECORD (docs/measurement.md#instruments): `bazel run` stores each of its files at this target's
    label unless the store already holds it."""
    _results_store(name = name, workspace = native.module_name(), venv = VENV, python = Label("@fsla_env//:python"), **kwargs)

def workspace_stamp(name):
    """The workspace's stamp as `<name>.stamp.json`: the commit a program run by `bazel run` stores what it measured
    against (the lab's runs, `lab_run`)."""
    _workspace_stamp(name = name, python = Label("@fsla_env//:python"))

_LAB_RUN = """\
#!/bin/bash
# a tool that reads compiled code through warpline or the lab's records, as a run target (bazel/defs.bzl's lab_run): {tool} in the checkout, on the
# checkout's own venv, handed the pinned data's root (`FSLA_DATA`, measure/data.py), the directory that holds the `warpline` package (the checker, and
# the library that reads the traced rows) and, for an experiment's run, the directory its build lies in and the workspace's stamp, each by its
# resolved path (the runfiles' links do not outlive the run)
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
data="$(dirname "$(readlink -f {data})")"
{package}{built}venv="$(git -C "$checkout" config --get fsla.venv || true)"
if [ ! -x "$venv/bin/python" ]; then
  echo "$checkout has no venv of its own (git config fsla.venv)" >&2
  exit 1
fi
cd "$checkout"
export FSLA_DATA="$data"
{warpline}exec "$venv/bin/python" {tool} {experiment}"$@"
"""

def _lab_run_impl(ctx):
    package, files = warpline_root(ctx, ctx.attr.library) if ctx.attr.library else (None, [])
    experiment, built = "", ""
    if ctx.attr.experiment:
        name = ctx.attr.experiment.label.name
        binary = [f for f in ctx.files.experiment if f.basename.startswith(name + ".") and "." not in f.basename[len(name) + 1:]]
        if len(binary) != 1 or not ctx.file.stamp or not ctx.attr.workspace:
            fail("%s: an experiment's run takes its target (`lab_experiment`, which builds `<name>.<arch>`), the stamp and its workspace" % ctx.label)
        built = 'built="$(dirname "$(readlink -f %s)")"\nstamp="$(readlink -f %s)"\n' % (
            shell.quote(binary[0].short_path),
            shell.quote(ctx.file.stamp.short_path),
        )
        label = "@%s//%s:%s" % (ctx.attr.workspace, ctx.label.package, ctx.label.name)
        experiment = '--built "$built" --stamp "$stamp" --label %s %s ' % (shell.quote(label), shell.quote(name))
        files = files + ctx.files.experiment + [ctx.file.stamp]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _LAB_RUN.format(
        tool = shell.quote(ctx.attr.tool),
        data = shell.quote(ctx.file.data.short_path),
        package = 'package="$(readlink -f %s)"\n' % shell.quote(package.short_path) if package else "",
        warpline = 'export FSLA_WARPLINE_PYTHON="$(dirname "$(dirname "$package")")"\n' if package else "",
        built = built,
        experiment = experiment,
    ), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = files + [ctx.file.data]))]

lab_run = rule(
    implementation = _lab_run_impl,
    executable = True,
    attrs = {
        "library": attr.label(doc = "warpline's modules the tool reads (`@warpline//:recorder`, `@warpline//:checker`, or both: `@warpline//:warpline`); none for a tool that reads records only"),
        "data": attr.label(default = "@fsla_data//:root", allow_single_file = True, doc = "the pinned data, by its root's BUILD file (MODULE.bazel's `environment.data`)"),
        "tool": attr.string(default = "tools/lab.py", doc = "the tree's tool that reads compiled code through the checker, or the lab's records"),
        "experiment": attr.label(doc = "an experiment's target (`lab_experiment`): its build, in the runfiles, is what the lab runs and reads"),
        "stamp": attr.label(allow_single_file = True, doc = "the workspace's stamp (`workspace_stamp`), the commit an experiment's run is stored against"),
        "workspace": attr.string(doc = "the repository's module name (`native.module_name()`): an experiment's run stores its samples at `@<workspace>//<package>:<name>`, its own label"),
    },
    doc = "The lab (`tools/lab.py`, or `tool`: the floor model) run by `bazel run` in the checkout, handed the " +
          "pinned data's root as `FSLA_DATA` and the directory of warpline's `warpline` package as `FSLA_WARPLINE_PYTHON`; " +
          "with `experiment`, the experiment's build and the stamp: its arguments follow `--`.",
)

def tree_part(parts = {}):
    """This package's part of the tree (`git ls-files`, @fsla_tree): `tracked`, `tracked_python` and each of `parts`."""
    _tree_part(TRACKED, parts)

#: THE UNIT LINTS (docs/build-graph.md#checks): clang-tidy's device and host profiles and the host warnings, each read
#: from a unit's own compile action -- its argv and its inputs -- so a lint reads what the build compiled

def _one_arch(arch):
    """A configuration of one architecture: its compiles build for it alone, its generated headers are its own."""
    settings = _cuda_settings([arch])
    settings[str(Label("//:archs"))] = [arch]
    return settings

_ARCH_OUTPUTS = [str(Label("//:archs")), str(Label("//bazel:cuda_archs"))]

def _each_built_arch_impl(_settings, _attr):
    return {arch: _one_arch(arch) for arch in BUILT_ARCHS}

#: the lints read each built architecture's compile: an architecture's build set is its own
_each_built_arch = transition(implementation = _each_built_arch_impl, inputs = [], outputs = _ARCH_OUTPUTS)

def _first_built_arch_impl(_settings, _attr):
    return _one_arch(BUILT_ARCHS[0])

#: an editor reads one compile a unit: the first built architecture's
_first_built_arch = transition(implementation = _first_built_arch_impl, inputs = [], outputs = _ARCH_OUTPUTS)

#: clang accepts a CUDA root with a `bin/` (a tree artifact keeps no empty directory, so it holds this note)
_CUDA_ROOT_BIN = "A CUDA root for clang's front end (bazel/defs.bzl's cuda_root): the toolkit's tools are the graph's own.\n"

def _cuda_root_impl(ctx):
    cuda_h = [f for f in ctx.attr.cudart[CcInfo].compilation_context.headers.to_list() if f.path.endswith("/include/cuda.h")]
    root = ctx.actions.declare_directory(ctx.label.name)
    note = ctx.actions.declare_file(ctx.label.name + ".bin.txt")
    ctx.actions.write(note, _CUDA_ROOT_BIN)
    ctx.actions.run_shell(
        command = 'mkdir -p "$1/bin" "$1/include" && cp "$2" "$1/bin/README" && shift 2 && cp "$@" "$OUT/include/"',
        arguments = [root.path, note.path] + [f.path for f in cuda_h + ctx.files.curand],
        env = {"OUT": root.path},
        inputs = cuda_h + ctx.files.curand + [note],
        outputs = [root],
        mnemonic = "FslaCudaRoot",
    )
    return [DefaultInfo(files = depset([root]))]

cuda_root = rule(
    implementation = _cuda_root_impl,
    attrs = {
        "cudart": attr.label(mandatory = True, providers = [CcInfo], doc = "the runtime's headers, `cuda.h` among them"),
        "curand": attr.label(mandatory = True, doc = "cuRAND's headers, which clang's CUDA runtime wrapper includes"),
    },
    doc = "A CUDA root clang's front end accepts (`--cuda-path`): `bin/`, `include/cuda.h` for the version, and the " +
          "cuRAND headers clang's runtime wrapper includes; the rest of the toolkit's headers reach the compile through " +
          "its own include roots.",
)

UnitLintInfo = provider(
    doc = "The unit lints' verdicts and clang commands of every unit at or under a target.",
    fields = {
        "device": "depset: the device profile's clang-tidy verdicts",
        "host": "depset: the host profile's clang-tidy verdicts",
        "warnings": "depset: the host warnings' verdicts",
        "commands": "depset: each unit's clang command (the editor's compile database)",
        "inputs": "depset: what the units' compiles read (the generated headers the database names among them)",
    },
)

#: the compiles the library links: the graph's CUDA units' (bazel/cuda.bzl), and rules_cc's
_COMPILES = ("CudaCompile", "CppCompile")

#: the attributes through which a target carries units: the library's and a unit's deps, the carry arm set's units --
#: so the lints read exactly the units the library links
_CARRIERS = ("deps", "units")

def _unit_lints_impl(target, ctx):
    found = {"device": [], "host": [], "warnings": [], "commands": [], "inputs": []}
    #: the configuration's one architecture (`_each_built_arch`, `_first_built_arch`)
    arch = ctx.attr._archs[BuildSettingInfo].value
    if len(arch) != 1:
        fail("the unit lints read one architecture's compiles; %s's configuration has %s" % (target.label, arch))
    arch = arch[0]
    tool = ctx.file._tool
    for action in target.actions:
        objects = [f for f in action.outputs.to_list() if f.basename.endswith(".pic.o")] if action.mnemonic in _COMPILES else []
        if not objects:
            continue
        unit = objects[0].basename.removesuffix(".pic.o")
        #: what the compile reads: a C++ compile's headers join its inputs only when it runs, so the target's own
        #: headers are named too
        reads = depset(transitive = [action.inputs] + ([target[CcInfo].compilation_context.headers] if CcInfo in target else []))
        stem = "unit_lints/sm_%s/%s/%s" % (arch, target.label.name, unit)
        argv = ctx.actions.declare_file(stem + ".argv.json")
        #: a CUDA unit's action runs the graph's compile script, whose own command line follows its separator
        #: (bazel/cuda.bzl)
        compile_argv = action.argv[action.argv.index("--") + 1:] if action.mnemonic == "CudaCompile" else action.argv
        ctx.actions.write(argv, json.encode(compile_argv))
        command = ctx.actions.declare_file(stem + ".command.json")
        ctx.actions.run(
            executable = ctx.file._python,
            arguments = [tool.path, "command", "--argv", argv.path, "--arch", arch, "--cuda-root", ctx.file._cuda_root.path, "--out", command.path],
            inputs = [argv, tool],
            outputs = [command],
            mnemonic = "FslaUnitCommand",
            progress_message = "The clang command of %s (sm_%s)" % (unit, arch),
        )
        device = action.mnemonic == "CudaCompile"
        config = ctx.file._device_config if device else ctx.file._host_config
        verdict = ctx.actions.declare_file(stem + ".tidy.json")
        ctx.actions.run(
            executable = ctx.file._python,
            arguments = [tool.path, "tidy", "--command", command.path, "--clang-tidy", ctx.file._clang_tidy.path, "--config", config.path, "--package", target.label.package, "--out", verdict.path],
            #: clang reads the C and C++ libraries from the pinned toolchain's whole installation (unit_lints.py's
            #: `toolchain_flags`), which a host compile's inputs carry only in part
            inputs = depset([command, tool, config, ctx.file._clang_tidy, ctx.file._cuda_root] + ctx.files._clang_tidy_resources,
                            transitive = [reads, find_cc_toolchain(ctx).all_files]),
            outputs = [verdict],
            mnemonic = "FslaClangTidy",
            progress_message = "clang-tidy on %s (sm_%s)" % (unit, arch),
        )
        found["device" if device else "host"].append(depset([verdict]))
        found["commands"].append(depset([command]))
        found["inputs"].append(reads)
        if not device:
            warnings = ctx.actions.declare_file(stem + ".warnings.json")
            ctx.actions.run(
                executable = ctx.file._python,
                arguments = [tool.path, "warnings", "--argv", argv.path, "--arch", arch, "--package", target.label.package, "--out", warnings.path],
                inputs = depset([argv, tool], transitive = [reads]),
                outputs = [warnings],
                mnemonic = "FslaHostWarnings",
                progress_message = "The host warnings of %s (sm_%s)" % (unit, arch),
            )
            found["warnings"].append(depset([warnings]))
    #: what carries units: the library's and a unit's deps, the carry arm set's units
    for name in _CARRIERS:
        for dep in getattr(ctx.rule.attr, name, []):
            if UnitLintInfo in dep:
                for field in found:
                    found[field].append(getattr(dep[UnitLintInfo], field))
    return [UnitLintInfo(**{field: depset(transitive = sets) for field, sets in found.items()})]

unit_lints = aspect(
    implementation = _unit_lints_impl,
    attr_aspects = list(_CARRIERS),
    attrs = {
        "_archs": attr.label(default = "//:archs"),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
        "_clang_tidy": attr.label(default = "@clang_tidy//:clang_tidy", allow_single_file = True, cfg = "exec"),
        "_clang_tidy_resources": attr.label(default = "@clang_tidy//:resources", cfg = "exec"),
        "_device_config": attr.label(default = "//tools/lint:clang-tidy-device.yaml", allow_single_file = True),
        "_host_config": attr.label(default = "//:.clang-tidy", allow_single_file = True),
        "_cuda_root": attr.label(default = "//bazel:cuda_root", allow_single_file = True),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
    doc = "Each compile under a target, linted: a clang-tidy verdict a unit, the host warnings' a host unit, and the " +
          "unit's clang command.",
)

_UNIT_LINT_TEST = """\
#!/bin/bash
# the unit lints' verdicts, each unit's for each built architecture (bazel/defs.bzl's unit_lint_test)
exec {python} {tool} verdicts {expect} {verdicts}
"""

def _unit_lint_test_impl(ctx):
    verdicts = depset(transitive = [getattr(unit[UnitLintInfo], ctx.attr.lint) for unit in ctx.attr.units]).to_list()
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    words = [shell.quote(v.short_path) for v in verdicts]
    expect = " ".join(["--expect " + shell.quote(pattern) for pattern in ctx.attr.expect])
    ctx.actions.write(script, _UNIT_LINT_TEST.format(python = shell.quote(ctx.file._python.short_path), tool = shell.quote(ctx.file._tool.short_path), expect = expect, verdicts = " ".join(words)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = verdicts + [ctx.file._python, ctx.file._tool]))]

unit_lint_test = rule(
    implementation = _unit_lint_test_impl,
    test = True,
    attrs = {
        "units": attr.label_list(mandatory = True, cfg = _each_built_arch, aspects = [unit_lints], doc = "the library, or a fixture's unit"),
        "lint": attr.string(mandatory = True, values = ["device", "host", "warnings"], doc = "which verdicts it reads"),
        "expect": attr.string_list(doc = "a fixture's: the findings (patterns) each of its verdicts holds exactly"),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
    },
    doc = "A unit lint over every unit the library links, for each built architecture: red on any unit's finding " +
          "(a fixture's: on any verdict that does not hold exactly the expected findings).",
)

def _compile_database_impl(ctx):
    infos = [unit[UnitLintInfo] for unit in ctx.attr.units]
    commands = depset(transitive = [info.commands for info in infos]).to_list()
    out = ctx.actions.declare_file("compile_commands.json")
    #: the build makes what the commands name (the generated headers and the CUDA root), and the action reads them where
    #: they are: outside the sandbox, so the paths it writes are the editor's -- which is why its output is never cached
    built = depset(ctx.files._cuda_root + [ctx.file._anchor], transitive = [info.inputs for info in infos])
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [ctx.file._tool.path, "database", "--anchor", ctx.file._anchor.path, "--out", out.path] + [c.path for c in commands],
        inputs = depset(commands + [ctx.file._tool], transitive = [built]),
        outputs = [out],
        mnemonic = "FslaCompileDatabase",
        execution_requirements = {"no-sandbox": "1", "no-cache": "1", "no-remote": "1"},
    )
    return [DefaultInfo(files = depset([out]))]

compile_database = rule(
    implementation = _compile_database_impl,
    attrs = {
        "units": attr.label_list(mandatory = True, cfg = _first_built_arch, aspects = [unit_lints]),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
        "_anchor": attr.label(default = "//:MODULE.bazel", allow_single_file = True),
        "_cuda_root": attr.label(default = "//bazel:cuda_root", cfg = _first_built_arch, allow_single_file = True),
    },
    doc = "`bazel build` writes compile_commands.json under .bazel/bin: each unit's clang command, the lints' own, " +
          "every path absolute (the checkout's, the execution root's, the output base's).",
)

#: THE TEST TIERS (docs/testing.md#running-it): each tier one target, as pytest runs it, on the environment with the
#: pinned toolkit the tests compile with (an input, so a new one reruns them), and the anchor that resolves to the
#: checkout, where the front door linked the machine's dev config the tier's locks are read from
_TIER_TOOLKIT = {
    "anchor": Label("//:MODULE.bazel"),
    "nvcc": Label("@cuda_tools//:bin/nvcc"),
    "toolkit": Label("@cuda_tools//:toolkit"),
}

def pytest_tier(name, device = False, **kwargs):
    """A test tier as one target, outside the sandbox: it takes the machine's locks itself (tests/conftest.py); a device
    tier is keyed on the devices' identity and runs alone."""
    _pytest_tier(name = name, venv = VENV, devices = Label("@host_gpu//:identity.txt") if device else None,
                 **(_TIER_TOOLKIT | kwargs))

def device_lane(name, cmd, **kwargs):
    """A device lane: a command over the tree and the library on the machine's device, as a device tier runs, never
    part of a suite a commit runs (`manual`)."""
    _device_lane(name = name, cmd = cmd, venv = VENV, devices = Label("@host_gpu//:identity.txt"),
                 **(_TIER_TOOLKIT | kwargs))

def _full_contract_test_impl(ctx):
    contract = ctx.attr._contract[BuildSettingInfo].value
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    if contract == "full":
        body = "#!/bin/bash\necho 'the gate runs against the full contract'\n"
    else:
        body = "#!/bin/bash\necho 'the gate refuses --//:contract=%s: the full contract is enforced in the gate and in releases (a narrowed contract is a development build)' >&2\nexit 1\n" % contract
    ctx.actions.write(script, body, is_executable = True)
    return [DefaultInfo(executable = script)]

full_contract_test = rule(
    implementation = _full_contract_test_impl,
    test = True,
    attrs = {"_contract": attr.label(default = "//:contract", providers = [BuildSettingInfo])},
    doc = "A test that passes under the full contract and fails, naming the narrowed one, under any other `--//:contract`: the gate's refusal of a narrowed build.",
)

def _contract_transition_impl(_settings, attr):
    return {str(Label("//:contract")): attr.contract}

_contract_transition = transition(
    implementation = _contract_transition_impl,
    inputs = [],
    outputs = [str(Label("//:contract"))],
)

def _with_contract_impl(ctx):
    outs = []
    for f in ctx.attr.target[0][DefaultInfo].files.to_list():
        out = ctx.actions.declare_file("%s/%s" % (ctx.label.name, f.basename))
        ctx.actions.run_shell(inputs = [f], outputs = [out], command = "cp -p \"$1\" \"$2\"", arguments = [f.path, out.path], mnemonic = "FslaWithContract")
        outs.append(out)
    return [DefaultInfo(files = depset(outs), runfiles = ctx.runfiles(files = outs))]

with_contract = rule(
    implementation = _with_contract_impl,
    attrs = {
        "target": attr.label(mandatory = True, cfg = _contract_transition),
        "contract": attr.string(mandatory = True, doc = "the narrowed contract (`hardware/kernels/<kernel>/descriptors/dev/<name>.json`) the target is built under"),
    },
    doc = "A target's files as a build under a narrowed contract (`--//:contract=<contract>`) makes them, beside the default build's: the tests' way to read a narrowed build's tools and arm list on CPU.",
)
