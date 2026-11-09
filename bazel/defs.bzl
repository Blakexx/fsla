"""THE GRAPH'S RULES: the generated headers for a build's parameters, the carry arm units it selects, and the library
-- `rola_cu13/_C.abi3.so` linked by the pinned mold beside its build record, which `bazel run` copies into the checkout
(docs/build-graph.md#compile); and the checks -- over the tree's files, of the commit, and over each unit's own compile
(docs/build-graph.md#checks)."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rola_env//:flags.bzl", "RATIFIED_ARCHS", "VENV")
load("@rola_tree//:tree.bzl", "TRACKED")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(":cuda.bzl", "CudaUnitInfo")

#: the build's parameters (the root package's settings), as every rule here reads them
_SETTINGS = {
    "_archs": "//:archs",
    "_carry_arms": "//:carry_arms",
    "_decode_arms": "//:decode_arms",
    "_carry_parts": "//:carry_parts",
}

def _setting_attrs():
    return {name: attr.label(default = label) for name, label in _SETTINGS.items()}

def _value(ctx, name):
    return getattr(ctx.attr, name)[BuildSettingInfo].value

def _parameters(ctx, args):
    args.add("--archs", ",".join(_value(ctx, "_archs")))
    args.add("--carry-arms", ",".join(_value(ctx, "_carry_arms")))
    args.add("--decode-arms", ",".join(_value(ctx, "_decode_arms")))
    args.add("--carry-parts", _value(ctx, "_carry_parts"))

#: the headers `bazel/headers.py` writes, by `tools/gen_shards.py`'s names
_HEADERS = ["carry_selection.inc", "decode_selection.inc", "carry_parts.inc", "builds.inc"]

def _headers_impl(ctx):
    outs = [ctx.actions.declare_file("generated/" + name) for name in _HEADERS]
    args = ctx.actions.args()
    args.add(ctx.file._script)
    args.add("--out-dir", outs[0].dirname)
    _parameters(ctx, args)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files._sources + [ctx.file._script],
        outputs = outs,
        mnemonic = "RolaHeaders",
        progress_message = "Writing the build's generated headers",
    )
    compilation = cc_common.create_compilation_context(headers = depset(outs), includes = depset([outs[0].dirname]))
    return [DefaultInfo(files = depset(outs)), CcInfo(compilation_context = compilation)]

rola_headers = rule(
    implementation = _headers_impl,
    attrs = _setting_attrs() | {
        "_script": attr.label(default = "//bazel:headers.py", allow_single_file = True),
        "_sources": attr.label(default = "//bazel:generator_sources"),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "The carry and decode arm selections, the carry parts and the carried builds, for the build's parameters.",
)

def _include_root_impl(ctx):
    roots = ["%s/%s" % (ctx.label.package, root) for root in ctx.attr.roots]
    return [CcInfo(compilation_context = cc_common.create_compilation_context(headers = depset(ctx.files.hdrs), includes = depset(roots)))]

include_root = rule(
    implementation = _include_root_impl,
    attrs = {
        "hdrs": attr.label_list(allow_files = True),
        "roots": attr.string_list(doc = "the include roots, relative to the package"),
    },
    doc = "Source headers under `-I` roots, as setup.py's include list has them: searched before any dependency's.",
)

def _arm_set_impl(ctx):
    #: tools/gen_shards.py's family_arms grammar: indices or `all`, empty for every arm, an index out of range refused
    spec = _value(ctx, "_carry_arms")
    want = []
    for token in spec:
        if token == "all":
            want += range(ctx.attr.rows)
        elif not token.isdigit() or int(token) >= ctx.attr.rows:
            fail("--//:carry_arms: %r is not an arm index or 'all'; the list has %d rows (0..%d)" % (token, ctx.attr.rows, ctx.attr.rows - 1))
        else:
            want.append(int(token))
    chosen = [unit for unit, arms in ctx.attr.units.items() if not want or [i for i in json.decode(arms) if i in want]]
    return [
        cc_common.merge_cc_infos(cc_infos = [unit[CcInfo] for unit in chosen]),
        CudaUnitInfo(units = depset(transitive = [unit[CudaUnitInfo].units for unit in chosen])),
    ]

carry_arm_set = rule(
    implementation = _arm_set_impl,
    attrs = {
        "units": attr.label_keyed_string_dict(providers = [CcInfo, CudaUnitInfo], doc = "each arm unit, and the arm indices it carries (JSON)"),
        "rows": attr.int(mandatory = True, doc = "the declared carry arms"),
        "_carry_arms": attr.label(default = "//:carry_arms"),
    },
    doc = "The carry arm units the build's arms name: an arm the build does not want is a unit it never compiles.",
)

#: the build's architectures as rules_cuda's, and its runtime as headers alone: the library links the runtime itself, so
#: no unit carries a dynamic library whose search path the link would write into the binary
def _cuda_settings(archs):
    return {
        #: one spec an architecture, each its own virtual one: `sm_80,sm_86` would compile both from compute_80
        "@rules_cuda//cuda:archs": ";".join(["compute_%s:sm_%s" % (a, a) for a in archs]),
        "@rules_cuda//cuda:runtime": str(Label("//bazel:cuda_headers")),
    }

def _archs_transition_impl(settings, _attr):
    return _cuda_settings(settings[str(Label("//:archs"))])

_archs_transition = transition(
    implementation = _archs_transition_impl,
    inputs = [str(Label("//:archs"))],
    outputs = ["@rules_cuda//cuda:archs", "@rules_cuda//cuda:runtime"],
)

#: the libraries the extension links by path, as setup.py's link does: torch's two, the CUDA runtime and the driver's
#: stub (the real driver at load); each NEEDED by its soname, and no search path written
_CUDA_LINKED = ("libcudart.so.13", "libcuda.so")

_INSTALL = """\
#!/bin/bash
# writes the graph's library into the checkout, and the build's generated headers where the tools that compile the
# tree's units read them (build/generated: tools/life_ranges.py, tools/header_selfcheck.py): copies, never links into
# Bazel's output tree; a header only when its text changed, as the tree's own writer writes one
# (tools/gen_shards.py's write_if_changed: a moved mtime recompiles every unit that includes it), through a temporary
# file renamed into place
set -euo pipefail
dest="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}/rola_cu13"
for f in {files}; do
  install -m 0644 "$f" "$dest/$(basename "$f")"
done
chmod 0755 "$dest/_C.abi3.so"
generated="$BUILD_WORKSPACE_DIRECTORY/build/generated"
mkdir -p "$generated"
for f in {headers}; do
  to="$generated/$(basename "$f")"
  if ! cmp -s "$f" "$to"; then
    install -m 0644 "$f" "$to.graph" && mv -f "$to.graph" "$to"
  fi
done
echo "installed $(basename "$dest")/_C.abi3.so and _build_config.py, and the build's generated headers, from the graph"
"""

def _extension_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    so = ctx.actions.declare_file("rola_cu13/_C.abi3.so")
    ld = [f for f in ctx.files._linker if f.basename == "ld.mold"][0]
    linked = ctx.files.shared_libraries + [f for f in ctx.files._cudart if f.basename in _CUDA_LINKED]
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

    #: THE BUILD GATES AND THE RECORD: rules 1, 3 and 5 and the shards, then the record (bazel/record.py)
    record = ctx.actions.declare_file("rola_cu13/_build_config.py")
    ptxas = [f for f in ctx.files._nvcc if f.basename == "ptxas"][0]
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
    _parameters(ctx, args)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files._record_sources + ctx.files._sources + [ctx.file._record, ptxas, ctx.file._cuobjdump, ctx.file._nvdisasm],
        outputs = [record],
        mnemonic = "RolaRecord",
        progress_message = "Gating the build (rules 1, 3, 5, the shards) and writing its record",
    )

    #: THE BINARY GATES, on the linked library: the SASS gate (KERNEL_STANDARDS section 20) on the graph's
    #: disassemblers, and the arm table with its coverage sweep (the binary asked what it carries)
    sass_verdict = ctx.actions.declare_file("rola_cu13/sass_gate.json")
    sass_args = ctx.actions.args()
    sass_args.add(ctx.file._sass_gate)
    sass_args.add(so)
    sass_args.add("--cuobjdump", ctx.file._cuobjdump)
    sass_args.add("--nvdisasm", ctx.file._nvdisasm)
    sass_args.add("--json", sass_verdict)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [sass_args],
        inputs = ctx.files._gate_sources + [so, ctx.file._sass_gate, ctx.file._cuobjdump, ctx.file._nvdisasm],
        outputs = [sass_verdict],
        mnemonic = "RolaSassGate",
        progress_message = "The SASS gate on %{label}'s library",
    )
    arm_verdict = ctx.actions.declare_file("rola_cu13/arm_gate.txt")
    header = [f for f in ctx.files._headers if f.basename == "carry_selection.inc"][0]
    stub = [f for f in ctx.files._cudart if f.path.endswith("stubs/libcuda.so")][0]
    arm_args = ctx.actions.args()
    arm_args.add(ctx.file._arm_gate)
    arm_args.add("--so", so)
    arm_args.add("--header", header)
    #: the environment's interpreter by its store path, which carries the environment's key
    arm_args.add("--python", VENV + "/bin/python3")
    arm_args.add("--driver-stub", stub)
    arm_args.add("--carry-arms", ",".join(_value(ctx, "_carry_arms")))
    arm_args.add("--out", arm_verdict)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [arm_args],
        inputs = ctx.files._sources + [so, header, stub, ctx.file._arm_gate],
        outputs = [arm_verdict],
        mnemonic = "RolaArmGate",
        progress_message = "The arm table and its coverage sweep on %{label}'s library",
        #: the environment's interpreter reads its packages from the store, which the sandbox does not hold; its store
        #: path on the command line keys the action by the environment
        execution_requirements = {"no-sandbox": "1"},
    )

    verdicts = [sass_verdict, arm_verdict]
    install = ctx.actions.declare_file(ctx.label.name + "_install.sh")
    headers = [f for f in ctx.files._headers if f.extension == "inc"]
    ctx.actions.write(install, _INSTALL.format(files = " ".join([f.short_path for f in (so, record)]),
                                               headers = " ".join([f.short_path for f in headers])), is_executable = True)
    #: the verdicts are runfiles too: `bazel run` installs nothing a gate has not passed
    return [
        DefaultInfo(files = depset([so, record] + verdicts), executable = install,
                    runfiles = ctx.runfiles(files = [so, record] + verdicts + headers)),
        #: the units it links, for ratify's census (`ratify_census`)
        CudaUnitInfo(units = depset(transitive = [dep[CudaUnitInfo].units for dep in ctx.attr.deps if CudaUnitInfo in dep])),
    ]

rola_extension = rule(
    implementation = _extension_impl,
    cfg = _archs_transition,
    executable = True,
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    attrs = _setting_attrs() | {
        "deps": attr.label_list(providers = [CcInfo], doc = "the units and the arm set"),
        "shared_libraries": attr.label_list(allow_files = [".so"], doc = "torch's libraries the extension links"),
        "torch": attr.string(mandatory = True, doc = "the environment's torch version (@rola_env//:flags.bzl)"),
        "torch_cuda": attr.string(mandatory = True, doc = "the CUDA release the environment's torch was built for"),
        "toolchain": attr.string(mandatory = True, doc = "the toolchain's name (its record and manifests)"),
        "cxx11_abi": attr.bool(mandatory = True, doc = "the environment's torch ABI flag"),
        "_linker": attr.label(default = "@mold//:linker"),
        "_nvcc": attr.label(default = "@cuda//:nvcc_all_files"),
        "_cudart": attr.label(default = "@cuda//:cudart_all_files"),
        "_record": attr.label(default = "//bazel:record.py", allow_single_file = True),
        "_record_sources": attr.label(default = "//bazel:record_sources"),
        "_sources": attr.label(default = "//bazel:generator_sources"),
        "_sass_gate": attr.label(default = "//tools:sass_gate.py", allow_single_file = True),
        "_arm_gate": attr.label(default = "//bazel:gates.py", allow_single_file = True),
        "_gate_sources": attr.label(default = "//bazel:gate_sources"),
        "_headers": attr.label(default = "//csrc/rola:generated"),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True, cfg = "exec"),
        "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True, cfg = "exec"),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
    doc = "The extension `rola_cu13/_C.abi3.so` and its build record; `bazel run` copies both into the checkout.",
)

def _census_impl(ctx):
    units = ctx.attr.library[CudaUnitInfo].units.to_list()
    tools = {
        "ptxas": [f for f in ctx.files._nvcc if f.basename == "ptxas"][0],
        "cuobjdump": ctx.file._cuobjdump,
        "nvdisasm": ctx.file._nvdisasm,
        "mold": [f for f in ctx.files._linker if f.basename == "mold"][0],
    }
    harness = ctx.executable.harness_binary
    record = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(record, json.encode_indent({
        "units": [{"source": u.source, "report": u.report.short_path, "object": u.object.short_path} for u in units],
        "tools": {name: f.short_path for name, f in tools.items()},
        "derivation": {"harness": ctx.file.harness.short_path, "binary": harness.short_path},
        "parameters": {
            "archs": ",".join(_value(ctx, "_archs")),
            "carry_arms": ",".join(_value(ctx, "_carry_arms")),
            "decode_arms": ",".join(_value(ctx, "_decode_arms")),
            "carry_parts": _value(ctx, "_carry_parts"),
        },
    }) + "\n")
    files = [record, ctx.file.harness, harness] + tools.values() + [u.report for u in units] + [u.object for u in units]
    return [DefaultInfo(files = depset([record]), runfiles = ctx.runfiles(files = files))]

ratify_census = rule(
    implementation = _census_impl,
    attrs = _setting_attrs() | {
        "library": attr.label(mandatory = True, providers = [CudaUnitInfo], doc = "the library whose units are counted"),
        "harness": attr.label(mandatory = True, allow_single_file = [".cu"], doc = "the derivation harness's source"),
        "harness_binary": attr.label(mandatory = True, executable = True, cfg = "target", doc = "the harness, built"),
        "_nvcc": attr.label(default = "@cuda//:nvcc_all_files"),
        "_linker": attr.label(default = "@mold//:linker"),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True),
        "_nvdisasm": attr.label(default = "@cuda_tools//:bin/nvdisasm", allow_single_file = True),
    },
    doc = "The census record ratify reads in place of its recompile (tools/ratify.py's `--graph`): each unit the " +
          "library links (its source, nvcc's diagnostics, its object), the graph's assembler, disassemblers and " +
          "linker, the derivation harness the graph built, and the build's parameters; every path relative to the " +
          "runfiles' root, which hold them all.",
)

_RATIFIED = """\
#!/bin/bash
# ratify's verdict on the graph's build (bazel/defs.bzl's ratify_test): `tools/ratify.py --graph` in a copy of the tree,
# so a tool that resolves a path reads the copy (rule 14); the census record's paths are relative to the runfiles
set -euo pipefail
tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
unset ROLA_DEV_CONFIG
export PATH={venv}/bin:/usr/bin:/bin
exec python3 tools/ratify.py --graph {census} --runfiles "$TEST_SRCDIR/_main"
"""

def _ratify_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _RATIFIED.format(venv = shell.quote(VENV), census = shell.quote(ctx.file.census.short_path)), is_executable = True)
    runfiles = ctx.runfiles(files = ctx.files.data).merge(ctx.attr.census[DefaultInfo].default_runfiles)
    return [DefaultInfo(executable = script, runfiles = runfiles)]

ratify_test = rule(
    implementation = _ratify_test_impl,
    test = True,
    attrs = {
        "census": attr.label(mandatory = True, allow_single_file = True, doc = "the census record (`ratify_census`)"),
        "data": attr.label_list(allow_files = True, doc = "the tree ratify reads: its sources, manifests and tools"),
    },
    doc = "Ratify's verdict on the graph's build -- every refusal of docs/ratification.md -- cached on the census, " +
          "the tree and the manifests.",
)

_RATIFY = """\
#!/bin/bash
# `tools/ratify.py --graph` on the graph's build, in the checkout (bazel/defs.bzl's ratify_run): `--write` records the
# manifests there, `--census` measures; the census record's paths are relative to the runfiles, where `bazel run` starts
set -euo pipefail
runfiles="$PWD"
cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec {venv}/bin/python3 tools/ratify.py --graph "$runfiles/{census}" --runfiles "$runfiles" "$@"
"""

def _ratify_run_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _RATIFY.format(venv = shell.quote(VENV), census = ctx.file.census.short_path), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.attr.census[DefaultInfo].default_runfiles)]

ratify_run = rule(
    implementation = _ratify_run_impl,
    executable = True,
    attrs = {"census": attr.label(mandatory = True, allow_single_file = True, doc = "the census record (`ratify_census`)")},
    doc = "`bazel run` ratify on the graph's build in the checkout: its arguments are ratify's (`--write`, `--census`).",
)

_TREE_TEST = """\
#!/bin/bash
# a check over the tree's tracked files (bazel/defs.bzl's tree_test): the runfiles copied into a real tree first, so a
# tool that resolves a path -- the runfiles are links into the checkout -- reads the check's inputs and nothing else
set -euo pipefail
tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
#: no machine reaches a check over files: the dev config is the defaults
unset ROLA_DEV_CONFIG
export PATH={venv}/bin:/usr/bin:/bin RUFF_NO_CACHE=true
{git}exec {cmd}
"""

#: the copy as a git index of exactly its files: a check that asks git for the tracked files gets the check's inputs
_TREE_GIT = "git init -q && git add -A\n"

def _tree_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    body = _TREE_TEST.format(venv = shell.quote(VENV), git = _TREE_GIT if ctx.attr.git else "", cmd = " ".join([shell.quote(w) for w in ctx.attr.cmd]))
    ctx.actions.write(script, body, is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = ctx.files.data))]

tree_test = rule(
    implementation = _tree_test_impl,
    test = True,
    attrs = {
        "cmd": attr.string_list(mandatory = True, doc = "the check's command, run from the copy's root"),
        "data": attr.label_list(allow_files = True, doc = "the files it reads: the tree, or the part of it the check reads"),
        "git": attr.bool(default = False, doc = "the copy a git index of its files, for a check that asks git what is tracked"),
    },
    doc = "A check over the tree's tracked files, the environment's tools on PATH: one test, cached on what it reads.",
)

_TOOL_TEST = """\
#!/bin/bash
# a tree tool as a test (bazel/defs.bzl's tool_test): run on the environment's base interpreter, standard library
# only, with its arguments' paths in the runfiles
set -euo pipefail
exec {python} {tool} {args}
"""

def _tool_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    args = [ctx.expand_location(a, ctx.attr.data) for a in ctx.attr.args_]
    ctx.actions.write(script, _TOOL_TEST.format(
        python = shell.quote(ctx.file._python.short_path),
        tool = shell.quote(ctx.file.tool.short_path),
        args = " ".join(args),
    ), is_executable = True)
    runfiles = ctx.runfiles(files = ctx.files.data + [ctx.file.tool, ctx.file._python])
    return [DefaultInfo(executable = script, runfiles = runfiles)]

tool_test = rule(
    implementation = _tool_test_impl,
    test = True,
    attrs = {
        "tool": attr.label(mandatory = True, allow_single_file = [".py"], doc = "the tree's tool, standard library only"),
        "args_": attr.string_list(doc = "its arguments, `$(rootpath ...)` and `$(rootpaths ...)` of `data` expanded"),
        "data": attr.label_list(allow_files = True, doc = "what its arguments name"),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "A check that is one run of a tree tool over files the graph names.",
)

_CHECKOUT_TEST = """\
#!/bin/bash
# a check of the commit, run in the checkout it is made from (bazel/defs.bzl's checkout_test): never cached; the
# machine's dev config is the one the front door names (ROLA_DEV_CONFIG)
set -euo pipefail
{resolve}cd "$(dirname "$(readlink -f "$TEST_SRCDIR/_main/MODULE.bazel")")"
export PATH={venv}/bin:/usr/bin:/bin RUFF_NO_CACHE=true
exec {cmd} "$@"
"""

def _checkout_test_impl(ctx):
    resolve, words = [], [shell.quote(w) for w in ctx.attr.cmd]
    tools = []
    for i, (target, flag) in enumerate(ctx.attr.tools.items()):
        found = target[DefaultInfo].files.to_list()
        if len(found) != 1:
            fail("%s: %s names %d files, not one" % (ctx.label, target.label, len(found)))
        tools += found
        resolve.append('tool%d="$(readlink -f "$TEST_SRCDIR/_main/%s")"\n' % (i, found[0].short_path))
        words += [flag, '"$tool%d"' % i]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _CHECKOUT_TEST.format(resolve = "".join(resolve), venv = shell.quote(VENV), cmd = " ".join(words)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = [ctx.file._anchor] + tools))]

_checkout_test = rule(
    implementation = _checkout_test_impl,
    test = True,
    attrs = {
        "cmd": attr.string_list(mandatory = True, doc = "the check's command, run from the checkout's root"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "graph outputs it reads, each by the flag that names its path"),
        "_anchor": attr.label(default = "//:MODULE.bazel", allow_single_file = True),
    },
)

def checkout_test(name, cmd, **kwargs):
    """A check of the commit rather than of a tree -- what git knows (HEAD, the index, pinned commits): run in the
    checkout, outside the sandbox, never cached, since its input is the repository's state and not a file set."""
    _checkout_test(name = name, cmd = cmd, tags = ["external", "local"], **kwargs)

_DEV_RUN = """\
#!/bin/bash
# a tools/dev.py command as a run target (bazel/defs.bzl's dev_run): in the checkout, on the environment's interpreter,
# the environment's tools named by their resolved paths (the runfiles' links do not outlive the run)
set -euo pipefail
{resolve}cd "${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec {venv}/bin/python3 tools/dev.py {command} "$@"
"""

def _dev_run_impl(ctx):
    resolve, words = [], [shell.quote(w) for w in ctx.attr.command]
    if ctx.attr.checkout:
        words += ["--checkout", '"$BUILD_WORKSPACE_DIRECTORY"']
    if ctx.attr.env:
        words += ["--env", shell.quote(VENV)]
    tools = []
    for i, (target, flag) in enumerate(ctx.attr.tools.items()):
        found = target[DefaultInfo].files.to_list()
        if len(found) != 1:
            fail("%s: %s names %d files, not one" % (ctx.label, target.label, len(found)))
        tools += found
        resolve.append('tool%d="$(readlink -f %s)"\n' % (i, shell.quote(found[0].short_path)))
        words += [flag, '"$tool%d"' % i]
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _DEV_RUN.format(resolve = "".join(resolve), venv = shell.quote(VENV), command = " ".join(words)), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = tools))]

dev_run = rule(
    implementation = _dev_run_impl,
    executable = True,
    attrs = {
        "command": attr.string_list(mandatory = True, doc = "the dev.py command and its fixed arguments"),
        "checkout": attr.bool(doc = "pass the checkout (`--checkout`, $BUILD_WORKSPACE_DIRECTORY)"),
        "env": attr.bool(doc = "pass the graph's environment, its store venv (`--env`)"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "each tool and the flag that names its path"),
    },
    doc = "A tools/dev.py command run by `bazel run` in the checkout (docs/build-graph.md#use): its arguments follow `--`.",
)

_TOOL_RUN = """\
#!/bin/bash
# a tree tool as a run target (bazel/defs.bzl's tool_run): in the checkout, on the checkout's own venv (`rola.venv`,
# which `bazel run //:dev_env` records), so it imports the checkout, its siblings and the environment
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
venv="$(git -C "$checkout" config --get rola.venv || true)"
if [ ! -x "$venv/bin/python" ]; then
  echo "$checkout has no venv of its own: bazel run //:dev_env" >&2
  exit 1
fi
cd "$checkout"
exec "$venv/bin/python" {what} "$@"
"""

def _tool_run_impl(ctx):
    if bool(ctx.attr.tool) == bool(ctx.attr.module):
        fail("%s: a tool run names its script (`tool`) or its module (`module`): exactly one" % ctx.label)
    what = shell.quote(ctx.attr.tool) if ctx.attr.tool else "-m " + shell.quote(ctx.attr.module)
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _TOOL_RUN.format(what = what), is_executable = True)
    return [DefaultInfo(executable = script)]

tool_run = rule(
    implementation = _tool_run_impl,
    executable = True,
    attrs = {
        "tool": attr.string(doc = "the tool's script, from the checkout's root (`tools/lab.py`)"),
        "module": attr.string(doc = "or the module it runs (`rola_devtools.build`)"),
    },
    doc = "A tree tool run by `bazel run` in the checkout on its own venv (docs/build-graph.md#use): its arguments " +
          "follow `--`.",
)

def tree_part(parts = {}):
    """This package's part of the tree (`git ls-files`, @rola_tree): `tracked`, every tracked file it holds;
    `tracked_python`, its Python files; and each of `parts` (name -> a directory prefix), the tracked files under it --
    so a check's data names the files it reads."""
    files = TRACKED[native.package_name()]
    native.filegroup(name = "tracked", srcs = files, visibility = ["//visibility:public"])
    native.filegroup(name = "tracked_python", srcs = [f for f in files if f.endswith(".py")], visibility = ["//visibility:public"])
    for name, prefix in parts.items():
        native.filegroup(name = name, srcs = [f for f in files if f.startswith(prefix)], visibility = ["//visibility:public"])

#: THE UNIT LINTS (docs/build-graph.md#checks): clang-tidy's device and host profiles and the host warnings, each read
#: from a unit's own compile action -- its argv and its inputs -- so a lint reads what the build compiled

def _one_arch(arch):
    """A configuration of one architecture: its compiles build for it alone, its generated headers are its own."""
    settings = _cuda_settings([arch])
    settings[str(Label("//:archs"))] = [arch]
    return settings

_ARCH_OUTPUTS = [str(Label("//:archs")), "@rules_cuda//cuda:archs", "@rules_cuda//cuda:runtime"]

def _each_ratified_arch_impl(_settings, _attr):
    return {arch: _one_arch(arch) for arch in RATIFIED_ARCHS}

#: the lints read each ratified architecture's compile: an architecture's build set is its own
_each_ratified_arch = transition(implementation = _each_ratified_arch_impl, inputs = [], outputs = _ARCH_OUTPUTS)

def _first_ratified_arch_impl(_settings, _attr):
    return _one_arch(RATIFIED_ARCHS[0])

#: an editor reads one compile a unit: the first ratified architecture's
_first_ratified_arch = transition(implementation = _first_ratified_arch_impl, inputs = [], outputs = _ARCH_OUTPUTS)

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
        mnemonic = "RolaCudaRoot",
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
    #: the configuration's one architecture (`_each_ratified_arch`, `_first_ratified_arch`)
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
            mnemonic = "RolaUnitCommand",
            progress_message = "The clang command of %s (sm_%s)" % (unit, arch),
        )
        device = action.mnemonic == "CudaCompile"
        config = ctx.file._device_config if device else ctx.file._host_config
        verdict = ctx.actions.declare_file(stem + ".tidy.json")
        ctx.actions.run(
            executable = ctx.file._python,
            arguments = [tool.path, "tidy", "--command", command.path, "--clang-tidy", ctx.file._clang_tidy.path, "--config", config.path, "--package", target.label.package, "--out", verdict.path],
            inputs = depset([command, tool, config, ctx.file._clang_tidy, ctx.file._cuda_root] + ctx.files._clang_tidy_resources, transitive = [reads]),
            outputs = [verdict],
            mnemonic = "RolaClangTidy",
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
                mnemonic = "RolaHostWarnings",
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
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
        "_clang_tidy": attr.label(default = "@clang_tidy//:clang_tidy", allow_single_file = True, cfg = "exec"),
        "_clang_tidy_resources": attr.label(default = "@clang_tidy//:resources", cfg = "exec"),
        "_device_config": attr.label(default = "//tools/lint:clang-tidy-device.yaml", allow_single_file = True),
        "_host_config": attr.label(default = "//:.clang-tidy", allow_single_file = True),
        "_cuda_root": attr.label(default = "//bazel:cuda_root", allow_single_file = True),
    },
    doc = "Each compile under a target, linted: a clang-tidy verdict a unit, the host warnings' a host unit, and the " +
          "unit's clang command.",
)

_UNIT_LINT_TEST = """\
#!/bin/bash
# the unit lints' verdicts, each unit's for each ratified architecture (bazel/defs.bzl's unit_lint_test)
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
        "units": attr.label_list(mandatory = True, cfg = _each_ratified_arch, aspects = [unit_lints], doc = "the library, or a fixture's unit"),
        "lint": attr.string(mandatory = True, values = ["device", "host", "warnings"], doc = "which verdicts it reads"),
        "expect": attr.string_list(doc = "a fixture's: the findings (patterns) each of its verdicts holds exactly"),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
    },
    doc = "A unit lint over every unit the library links, for each ratified architecture: red on any unit's finding " +
          "(a fixture's: on any verdict that does not hold exactly the expected findings).",
)

_DATABASE = """\
#!/bin/bash
# writes the editor's compile database into the checkout (bazel/defs.bzl's compile_database)
set -euo pipefail
checkout="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}"
exec {python} {tool} database --checkout "$checkout" --out "$checkout/compile_commands.json" {commands}
"""

def _compile_database_impl(ctx):
    infos = [unit[UnitLintInfo] for unit in ctx.attr.units]
    commands = depset(transitive = [info.commands for info in infos]).to_list()
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    words = [shell.quote(c.short_path) for c in commands]
    ctx.actions.write(script, _DATABASE.format(python = shell.quote(ctx.file._python.short_path), tool = shell.quote(ctx.file._tool.short_path), commands = " ".join(words)), is_executable = True)
    #: the build makes what the commands name: the generated headers and the CUDA root
    built = depset(ctx.files._cuda_root, transitive = [info.inputs for info in infos])
    return [DefaultInfo(executable = script, files = built, runfiles = ctx.runfiles(files = commands + [ctx.file._python, ctx.file._tool]))]

compile_database = rule(
    implementation = _compile_database_impl,
    executable = True,
    attrs = {
        "units": attr.label_list(mandatory = True, cfg = _first_ratified_arch, aspects = [unit_lints]),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True),
        "_tool": attr.label(default = "//tools/lint:unit_lints.py", allow_single_file = True),
        "_cuda_root": attr.label(default = "//bazel:cuda_root", cfg = _first_ratified_arch, allow_single_file = True),
    },
    doc = "`bazel run` writes compile_commands.json into the checkout: each unit's clang command, the lints' own.",
)

#: THE TEST TIERS (docs/testing.md#running-it): each tier one target, as pytest runs it

_TIER = """\
#!/bin/bash
# a test tier or device lane (bazel/defs.bzl's pytest_tier, device_lane): its inputs copied into a real tree, the
# tree's packages and the built library importable from it
set -euo pipefail
tree="$TEST_TMPDIR/tree"
cp -rL "$TEST_SRCDIR/_main" "$tree"
rm -f "$tree/$TEST_BINARY"
cd "$tree"
{machine}export PATH={venv}/bin:/usr/bin:/bin PYTHONPATH="$tree" PYTHONDONTWRITEBYTECODE=1
#: the graph's copy of the pinned toolkit, the tree its nvcc runs in (`dev_config.cuda_home`'s handoff)
export ROLA_CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$TEST_SRCDIR/_main/{nvcc}")")")"
{run}"""

#: a pytest tier's run: Bazel's filter (`--test_filter`, pytest's `-k`), its shards (bazel/pytest_shards.py), its
#: arguments (`--test_arg`) and its test report
_PYTEST_RUN = """\
only=()
if [ -n "${{TESTBRIDGE_TEST_ONLY:-}}" ]; then only=(-k "$TESTBRIDGE_TEST_ONLY"); fi
#: Bazel's sharding protocol: this shard's index and count, and the file that says the test honours them
shard=()
if [ -n "${{TEST_TOTAL_SHARDS:-}}" ]; then
  shard=(--bazel-shard "$TEST_SHARD_INDEX/$TEST_TOTAL_SHARDS")
  touch "$TEST_SHARD_STATUS_FILE"
fi
exec {venv}/bin/python -m pytest -p no:cacheprovider -p bazel.pytest_shards -q -rfE --junitxml="$XML_OUTPUT_FILE" \\
  {args} "${{only[@]}}" "${{shard[@]}}" "$@"
"""

#: a tier with no device sees none; every tier reads the machine's dev config (its locks)
_NO_DEVICE = 'export CUDA_VISIBLE_DEVICES=""\n'

def _tier_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    venv = shell.quote(VENV)
    if bool(ctx.attr.cmd) == bool(ctx.attr.tests) or (ctx.attr.cmd and ctx.attr.marks):
        fail("%s: a pytest tier names its paths (`tests`, with `marks`), a lane its command (`cmd`): exactly one" % ctx.label)
    #: a lane reaches a tool by its path alone: a `cmd` word naming it in braces (`{compute-sanitizer}`) is that path
    paths = {}
    for target, binary in ctx.attr.tools.items():
        found = [f for f in target.files.to_list() if f.basename == binary]
        if len(found) != 1:
            fail("%s carries %d files named %s" % (target.label, len(found), binary))
        paths[binary] = found[0]
    if ctx.attr.cmd:
        braced = [w[1:-1] for w in ctx.attr.cmd if len(w) > 2 and w[0] + w[-1] == "{}"]
        unknown = [name for name in braced if name not in paths]
        unused = [name for name in paths if name not in braced]
        if unknown or unused:
            fail("%s: the command's braced words %s and its tools %s must name each other" % (ctx.label, braced, sorted(paths)))
        words = [('"$TEST_SRCDIR/_main/%s"' % paths[w[1:-1]].short_path) if len(w) > 2 and w[0] + w[-1] == "{}" else shell.quote(w) for w in ctx.attr.cmd]
        run = "exec %s \"$@\"\n" % " ".join(words)
    else:
        words = list(ctx.attr.tests) + (["-m", ctx.attr.marks] if ctx.attr.marks else [])
        run = _PYTEST_RUN.format(venv = venv, args = " ".join([shell.quote(w) for w in words]))
    carried = [target.files for target in ctx.attr.tools]
    body = _TIER.format(venv = venv, machine = "" if ctx.attr.devices else _NO_DEVICE, nvcc = ctx.file._nvcc.short_path, run = run)
    ctx.actions.write(script, body, is_executable = True)
    #: a device tier's pass is its devices': their identity is one of its inputs (docs/build-graph.md rule 5)
    identity = [ctx.file.devices] if ctx.attr.devices else []
    files = depset(ctx.files.data + ctx.files._toolkit + [ctx.file._shards] + identity, transitive = carried)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(transitive_files = files))]

_tier_test = rule(
    implementation = _tier_impl,
    test = True,
    attrs = {
        "tests": attr.string_list(doc = "a pytest tier's paths, as pytest takes them"),
        "marks": attr.string(doc = "its marker expression (`-m`); none keeps pyproject's default selection"),
        "cmd": attr.string_list(doc = "a lane's command, in place of pytest, run from the copy's root; a word `{name}` " +
                                      "is the path of the tool binary of that name"),
        "tools": attr.label_keyed_string_dict(allow_files = True, doc = "the command's tools: each label's binary, by its name"),
        "data": attr.label_list(allow_files = True, doc = "what it reads: the tree and the library"),
        "_shards": attr.label(default = "//bazel:pytest_shards.py", allow_single_file = True),
        #: the pinned toolkit the tests compile, assemble and disassemble with: an input, so a new one reruns them
        "_toolkit": attr.label(default = "@cuda_tools//:toolkit"),
        "_nvcc": attr.label(default = "@cuda_tools//:bin/nvcc", allow_single_file = True),
        "devices": attr.label(allow_single_file = True, doc = "a device tier's: the devices' identity (`@host_gpu`), " +
                                                              "an input of its pass; the tier sees the device and takes the GPU lock"),
    },
)

def pytest_tier(name, device = False, tags = [], **kwargs):
    """A test tier as one target, outside the sandbox (`no-sandbox`): it takes the machine's locks itself
    (tests/conftest.py) -- a device tier the GPU lock (`host.gpu_lock`), a sharded tier a host slot a shard
    (`host.lock_dir`) -- and both live in `/tmp` by default, which a sandboxed test sees as its own. A device tier runs
    alone (`exclusive`)."""
    if device:
        kwargs["devices"] = "@host_gpu//:identity.txt"
    _tier_test(name = name, tags = tags + ["no-sandbox"] + (["exclusive"] if device else []), **kwargs)

def device_lane(name, cmd, tags = [], **kwargs):
    """A device lane: a command over the tree and the library on the machine's device, as a device tier runs (alone,
    outside the sandbox, keyed on the devices), never part of a suite a commit runs (`manual`)."""
    _tier_test(name = name, cmd = cmd, devices = "@host_gpu//:identity.txt",
               tags = tags + ["no-sandbox", "exclusive", "manual"], **kwargs)
