"""THE GRAPH'S RULES FOR THE EXTENSION: the generated headers for a build's parameters, the carry arm units it
selects, and the library -- `rola_cu13/_C.abi3.so` linked by the pinned mold beside its build record, which `bazel
run` copies into the checkout. Docs: docs/build-graph.md#compile."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
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
    want = [int(i) for i in _value(ctx, "_carry_arms")]
    chosen = [unit for unit, arms in ctx.attr.units.items() if not want or [i for i in json.decode(arms) if i in want]]
    return [cc_common.merge_cc_infos(cc_infos = [unit[CcInfo] for unit in chosen])]

carry_arm_set = rule(
    implementation = _arm_set_impl,
    attrs = {
        "units": attr.label_keyed_string_dict(providers = [CcInfo], doc = "each arm unit, and the arm indices it carries (JSON)"),
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

    record = ctx.actions.declare_file("rola_cu13/_build_config.py")
    ptxas = [f for f in ctx.files._nvcc if f.basename == "ptxas"][0]
    args = ctx.actions.args()
    args.add(ctx.file._record)
    args.add("--out", record)
    args.add("--toolchain", ctx.file._toolchain)
    args.add("--ptxas", ptxas)
    args.add("--torch", ctx.attr.torch)
    args.add("--cxx11-abi", str(ctx.attr.cxx11_abi))
    _parameters(ctx, args)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files._record_sources + ctx.files._sources + [ctx.file._record, ctx.file._toolchain, ptxas],
        outputs = [record],
        mnemonic = "RolaRecord",
        progress_message = "Checking the toolchain and writing the build record",
    )

    install = ctx.actions.declare_file(ctx.label.name + "_install.sh")
    ctx.actions.write(install, _INSTALL.format(files = " ".join([f.short_path for f in (so, record)])), is_executable = True)
    return [DefaultInfo(files = depset([so, record]), executable = install, runfiles = ctx.runfiles(files = [so, record]))]

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
        "cxx11_abi": attr.bool(mandatory = True, doc = "the environment's torch ABI flag"),
        "_linker": attr.label(default = "@mold//:linker"),
        "_nvcc": attr.label(default = "@cuda//:nvcc_all_files"),
        "_cudart": attr.label(default = "@cuda//:cudart_all_files"),
        "_record": attr.label(default = "//bazel:record.py", allow_single_file = True),
        "_record_sources": attr.label(default = "//bazel:record_sources"),
        "_sources": attr.label(default = "//bazel:generator_sources"),
        "_toolchain": attr.label(default = "//tools:toolchains/cu13.json", allow_single_file = True),
        "_python": attr.label(default = "@rola_env//:python", allow_single_file = True, cfg = "exec"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
    doc = "The extension `rola_cu13/_C.abi3.so` and its build record; `bazel run` copies both into the checkout.",
)
