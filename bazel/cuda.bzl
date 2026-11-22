"""THE GRAPH'S CUDA COMPILE: nvcc from the pinned toolkit (`@cuda_tools`), its command line written here, run by the graph's own
action, which drops the headers a unit did not read from its cache key and keeps ptxas's report (docs/build-graph.md#compile)."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

CudaUnitInfo = provider(
    doc = "The units a target compiled or links, for ratify's census (bazel/defs.bzl's `ratify_census`), and what is declared beside them: the SASS gate " +
          "reads the notices of their reports and the declarations of their kernels (bazel/defs.bzl).",
    fields = {
        "units": "depset of struct(source, report, object, sass): each unit's source, nvcc's diagnostics, its object, its SASS as text",
        "declarations": "depset of struct(arch, kernel, rule, allow, reason): the declarations stated beside the units (`declaration`), an arm family's among them",
        "arms": "depset of struct(arch, family, arm, kernel): the arms a family declares, each with its exact kernel pattern, which the gate holds the binary to",
    },
)

def declaration(kernel, rule, allow, reason, arch = None):
    """A declaration of a kernel's findings, stated beside the unit that compiles it (`cuda_unit(declarations = [...])`): the checker's form,
    `kernel` a regex over the kernel's demangled name, `rule` a rule of the checker or a notice rule of the project's ledger, `allow` the count
    the kernel may keep (a rule's bound for a bound rule; 0 refuses what the ledger accepts by default), `reason` why.

    The architecture is the attachment's, not the declaration's: `arch` (`sm_90a`) says the build target the declaration attaches to, none
    every architecture the unit is built for; the gate reads each architecture's declarations apart. A declaration the SASS of its
    architecture does not show (a count allowed where none is found) is red.
    """
    if type(kernel) != "string" or not kernel:
        fail("declaration: the kernel regex is empty")
    if type(rule) != "string" or not rule:
        fail("declaration: the rule is empty")
    if type(allow) != "int" or allow < 0:
        fail("declaration of %s on %r: allow is a count, 0 or more (not %r)" % (rule, kernel, allow))
    if type(reason) != "string" or not reason.strip():
        fail("declaration of %s on %r: a reason is required" % (rule, kernel))
    if arch != None and (type(arch) != "string" or not arch.startswith("sm_")):
        fail("declaration of %s on %r: arch is `sm_XX` (not %r)" % (rule, kernel, arch))
    return struct(kernel = kernel, rule = rule, allow = allow, reason = reason, arch = arch)

def regex_escape(text):
    """A regex that matches the text and nothing else."""
    out = ""
    for ch in text.elems():
        if ch in "\\.^$*+?{}[]|()":
            out += "\\"
        out += ch
    return out

def kernel_pattern(text):
    """The exact pattern of a kernel over its demangled name: `text` is the name up to its parameter list (the toolkit's `cu++filt`),
    anchored at the return type and the parameter list, as `arm_kernel_pattern` anchors an arm's."""
    return "^void " + regex_escape(text) + "\\("

#: nvcc's options a unit never states: the compile's mode and the architectures are the rule's (`archs`), never a `copts` entry
_FORBIDDEN_OPTS = ["--cuda", "-cuda", "--preprocess", "-E", "--compile", "-c", "--cubin", "-cubin", "--ptx", "-ptx", "--fatbin", "-fatbin", "--device-link", "-dlink", "--lib", "-lib", "--generate-dependencies", "-M", "--generate-nonsystem-dependencies", "-MM", "--run", "-run"]
_FORBIDDEN_PREFIXES = ["--generate-code", "-gencode", "--gpu-architecture", "-arch", "--gpu-code", "-code", "--relocatable-device-code", "-rdc"]

def _check_opt(opt):
    opt = opt.strip()
    if opt in _FORBIDDEN_OPTS or any([opt.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES]):
        fail("%s is not an option of a unit's `copts`: the compile's mode and architectures are the rule's" % opt)
    return opt

def _gencode(archs):
    """nvcc's `-gencode` flags of the architecture specs (`compute_90a:sm_90a`; `;` between specs, `,` between the cubins of one)."""
    flags = []
    for spec in ";".join(archs).split(";"):
        spec = spec.strip()
        if not spec:
            continue
        virtual, colon, codes = spec.partition(":")
        if not colon or not virtual.startswith("compute_"):
            fail("an architecture is `compute_XX:sm_XX` (a virtual architecture, then the cubins built from it), not %r" % spec)
        for code in codes.split(","):
            if not code.startswith("sm_"):
                fail("the code of %r is a cubin (`sm_XX`): the build carries no PTX and no LTO, not %r" % (spec, code))
            flags += ["-gencode", "arch=%s,code=%s" % (virtual, code)]
    return flags

def _cuda_unit_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    archs = ctx.attr.archs or [ctx.attr._default_archs[BuildSettingInfo].value]
    arch_flags = _gencode(archs)
    if not arch_flags:
        fail("%s: no architecture to build for (`archs`, or `--//bazel:cuda_archs`)" % ctx.label)
    cc_features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    host_compiler = cc_common.get_tool_for_action(feature_configuration = cc_features, action_name = CC_ACTION_NAMES.cpp_compile)

    #: what the units compile against: the deps' include roots, defines and headers, and the toolkit's headers
    merged = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps + [ctx.attr._cuda_headers]])
    context = merged.compilation_context
    root = paths.normalize(ctx.label.workspace_root)
    quote_includes = [root, paths.normalize(paths.join(ctx.bin_dir.path, root))] + context.quote_includes.to_list()
    headers = depset(ctx.files.hdrs, transitive = [context.headers])
    nvcc = ctx.file._nvcc

    #: the prunable inputs: the declared headers, never the host compiler's identity (`host_cc`, its own input) or a tool
    candidates = ctx.actions.declare_file("_objs/%s/headers.txt" % ctx.label.name)
    listed = ctx.actions.args()
    listed.set_param_file_format("multiline")
    listed.add_all(headers)
    ctx.actions.write(candidates, listed)

    objects, reports, listings, units, names = [], [], [], [], {}
    for src in ctx.files.srcs:
        stem = src.basename.removesuffix(".cu")
        if stem in names:
            fail("%s: two units named %s (%s, %s): their objects would collide" % (ctx.label, stem, names[stem], src.path))
        names[stem] = src.path
        obj = ctx.actions.declare_file("_objs/%s/%s.pic.o" % (ctx.label.name, stem))
        report = ctx.actions.declare_file("_objs/%s/%s.ptxas.txt" % (ctx.label.name, stem))
        unused = ctx.actions.declare_file("_objs/%s/%s.unused.txt" % (ctx.label.name, stem))

        #: nvcc's command line, the unit's options alone: the architectures, position-independent host code, the host compiler, the
        #: include roots (quote roots, then the rest), the defines, the options, the source and the object
        command = ["-x", "cu"] + arch_flags + ["-Xcompiler", "-fPIC", "-ccbin", host_compiler]
        for include in quote_includes + context.includes.to_list():
            command += ["-I", include]
        for include in context.system_includes.to_list():
            command += ["-isystem", include]
        command += ["-D" + define for define in context.defines.to_list()]
        command += [_check_opt(opt) for opt in ctx.attr.copts]
        command += ["-c", src.path, "-o", obj.path]
        args = ctx.actions.args()
        args.add(ctx.file._script)
        args.add("--source", src)
        args.add("--candidates", candidates)
        args.add("--unused", unused)
        args.add("--report", report)
        #: the compile's own command line follows the separator: the unit lints read it from there
        args.add("--")
        args.add(nvcc)
        args.add_all(command)
        ctx.actions.run(
            executable = ctx.file._python,
            arguments = [args],
            inputs = depset([src, candidates, ctx.file._script] + ctx.files.host_cc, transitive = [headers, cc_toolchain.all_files, depset(ctx.files._nvcc_files)]),
            outputs = [obj, report, unused],
            unused_inputs_list = unused,
            env = {"PATH": "/usr/bin:/bin:/usr/local/bin"},
            mnemonic = "CudaCompile",
            progress_message = "Compiling %s" % src.path,
        )
        #: the unit's SASS, one text a unit by the graph's own disassembler: read where a SASS diff is wanted (`bazel build :<name>_sass`)
        listing = ctx.actions.declare_file("_objs/%s/%s.sass" % (ctx.label.name, stem))
        ctx.actions.run_shell(
            inputs = [obj],
            tools = [ctx.file._cuobjdump],
            outputs = [listing],
            command = "\"$1\" -sass \"$2\" > \"$3\"",
            arguments = [ctx.file._cuobjdump.path, obj.path, listing.path],
            mnemonic = "CudaSass",
            progress_message = "The SASS of %s" % src.path,
        )
        objects.append(obj)
        listings.append(listing)
        reports.append(report)
        units.append(struct(source = src.path, report = report, object = obj, sass = listing))

    linking, _ = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = cc_features,
        cc_toolchain = cc_toolchain,
        compilation_outputs = cc_common.create_compilation_outputs(pic_objects = depset(objects)),
        alwayslink = True,
        linking_contexts = [merged.linking_context],
        disallow_dynamic_library = True,
    )
    return [
        DefaultInfo(files = depset(objects)),
        OutputGroupInfo(ptxas_reports = depset(reports), sass = depset(listings)),
        CudaUnitInfo(
            units = depset(units),
            declarations = depset([declaration(**json.decode(d)) for d in ctx.attr.declarations]),
            arms = depset(),
        ),
        cc_common.merge_cc_infos(direct_cc_infos = [CcInfo(linking_context = linking)], cc_infos = [merged]),
    ]

_cuda_unit = rule(
    implementation = _cuda_unit_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cu"], doc = "the units, one compile each"),
        "deps": attr.label_list(providers = [CcInfo], doc = "the headers and include roots the units compile against"),
        "copts": attr.string_list(doc = "nvcc's options (tools/build_flags.py's, through @fsla_env//:flags.bzl)"),
        "host_cc": attr.label(mandatory = True, allow_files = True, doc = "the host C++ compiler's identity: an input of every compile, never pruned"),
        "hdrs": attr.label_list(allow_files = True, doc = "headers beside the units (candidates for pruning, like the deps')"),
        "archs": attr.string_list(doc = "the architectures as `compute_XX:sm_XX`; none: the build's (`--//bazel:cuda_archs`, written by the library's transition)"),
        "declarations": attr.string_list(doc = "the declarations of the units' kernels (`declaration`), each as JSON"),
        "_cuda_headers": attr.label(default = "//bazel:cuda_headers", providers = [CcInfo]),
        "_default_archs": attr.label(default = "//bazel:cuda_archs"),
        "_nvcc": attr.label(default = "@cuda_tools//:bin/nvcc", allow_single_file = True, cfg = "exec"),
        "_nvcc_files": attr.label(default = "@cuda_tools//:nvcc_files", cfg = "exec"),
        "_cc_toolchain": attr.label(default = "@rules_cc//cc:current_cc_toolchain"),
        "_script": attr.label(default = "//bazel:cuda_compile.py", allow_single_file = True),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
        "_cuobjdump": attr.label(default = "@cuda_tools//:bin/cuobjdump", allow_single_file = True, cfg = "exec"),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    provides = [CcInfo, CudaUnitInfo],
    doc = "CUDA units compiled by the pinned toolkit's nvcc (a position-independent object, always linked), each by one " +
          "action whose unread headers leave its cache key.",
)

def cuda_unit(name, declarations = [], **kwargs):
    """CUDA units; `declarations` (`declaration`) the
    findings their kernels keep, which the library's SASS gate holds them to. The output group `sass` is the SASS of each of its units as text, by the
    graph's own disassembler (`_objs/<name>/<unit>.sass`, every architecture the build carries), which `unit_sass` (bazel/defs.bzl) names by unit."""
    _cuda_unit(name = name, declarations = [json.encode(d) for d in declarations], **kwargs)

