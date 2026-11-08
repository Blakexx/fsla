"""THE GRAPH'S RULES FOR THE EXTENSION: the generated headers for a build's parameters, the carry arm units it
selects, and the library -- `rola_cu13/_C.abi3.so` linked by the pinned mold beside its build record, which `bazel
run` copies into the checkout. Docs: docs/build-graph.md#compile."""

load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rola_env//:flags.bzl", "VENV")
load("@rola_tree//:tree.bzl", "TRACKED")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

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
    return [cc_common.merge_cc_infos(cc_infos = [unit[CcInfo] for unit in chosen])]

carry_arm_set = rule(
    implementation = _arm_set_impl,
    attrs = {
        "units": attr.label_keyed_string_dict(providers = [CcInfo], doc = "each arm unit, and the arm indices it carries (JSON)"),
        "rows": attr.int(mandatory = True, doc = "the declared carry arms"),
        "_carry_arms": attr.label(default = "//:carry_arms"),
    },
    doc = "The carry arm units the build's arms name: an arm the build does not want is a unit it never compiles.",
)

#: the build's architectures as rules_cuda's, and its runtime as headers alone: the library links the runtime itself, so
#: no unit carries a dynamic library whose search path the link would write into the binary
def _archs_transition_impl(settings, _attr):
    return {
        #: one spec an architecture, each its own virtual one: `sm_80,sm_86` would compile both from compute_80
        "@rules_cuda//cuda:archs": ";".join(["compute_%s:sm_%s" % (a, a) for a in settings[str(Label("//:archs"))]]),
        "@rules_cuda//cuda:runtime": str(Label("//bazel:cuda_headers")),
    }

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
# writes the graph's library into the checkout: a copy, never a link into Bazel's output tree
set -euo pipefail
dest="${{BUILD_WORKSPACE_DIRECTORY:?run through bazel run}}/rola_cu13"
for f in {files}; do
  install -m 0644 "$f" "$dest/$(basename "$f")"
done
chmod 0755 "$dest/_C.abi3.so"
echo "installed $(basename "$dest")/_C.abi3.so and _build_config.py from the graph"
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
    ctx.actions.write(install, _INSTALL.format(files = " ".join([f.short_path for f in (so, record)])), is_executable = True)
    #: the verdicts are runfiles too: `bazel run` installs nothing a gate has not passed
    return [DefaultInfo(files = depset([so, record] + verdicts), executable = install,
                        runfiles = ctx.runfiles(files = [so, record] + verdicts))]

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

_CHECKOUT_TEST = """\
#!/bin/bash
# a check of the commit, run in the checkout it is made from (bazel/defs.bzl's checkout_test): never cached; the
# machine's dev config is the one the front door names (ROLA_DEV_CONFIG)
set -euo pipefail
cd "$(dirname "$(readlink -f "$TEST_SRCDIR/_main/MODULE.bazel")")"
export PATH={venv}/bin:/usr/bin:/bin RUFF_NO_CACHE=true
exec {cmd}
"""

def _checkout_test_impl(ctx):
    script = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(script, _CHECKOUT_TEST.format(venv = shell.quote(VENV), cmd = " ".join([shell.quote(w) for w in ctx.attr.cmd])), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(files = [ctx.file._anchor]))]

_checkout_test = rule(
    implementation = _checkout_test_impl,
    test = True,
    attrs = {
        "cmd": attr.string_list(mandatory = True, doc = "the check's command, run from the checkout's root"),
        "_anchor": attr.label(default = "//:MODULE.bazel", allow_single_file = True),
    },
)

def checkout_test(name, cmd, **kwargs):
    """A check of the commit rather than of a tree -- what git knows (HEAD, the index, pinned commits): run in the
    checkout, outside the sandbox, never cached, since its input is the repository's state and not a file set."""
    _checkout_test(name = name, cmd = cmd, tags = ["external", "local"], **kwargs)

def tree_part(parts = {}):
    """This package's part of the tree (`git ls-files`, @rola_tree): `tracked`, every tracked file it holds;
    `tracked_python`, its Python files; and each of `parts` (name -> a directory prefix), the tracked files under it --
    so a check's data names the files it reads."""
    files = TRACKED[native.package_name()]
    native.filegroup(name = "tracked", srcs = files, visibility = ["//visibility:public"])
    native.filegroup(name = "tracked_python", srcs = [f for f in files if f.endswith(".py")], visibility = ["//visibility:public"])
    for name, prefix in parts.items():
        native.filegroup(name = name, srcs = [f for f in files if f.startswith(prefix)], visibility = ["//visibility:public"])
