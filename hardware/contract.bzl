"""THE CONTRACT A BUILD COMPILES AGAINST (docs/internals/hardware/registry.md#narrowed): a kernel's two outputs of the registry that depend on the build's
`--//:contract` -- the contexts `arms_gen` reads and the header the run time holds -- written for the contract the setting names (`full` by default,
or a narrowed contract of one kernel's `descriptors/dev/`, which narrows that kernel alone)."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _contract_output_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.out)
    args = ctx.actions.args()
    args.add(ctx.file._script)
    args.add("--kernel", ctx.attr.kernel)
    args.add("--" + ctx.attr.what, out)
    args.add("--contract", ctx.attr._contract[BuildSettingInfo].value)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [args],
        inputs = ctx.files.registry + [ctx.file._script],
        outputs = [out],
        mnemonic = "FslaContract",
        progress_message = "Writing the %s contract's %s" % (ctx.attr.kernel, ctx.attr.what),
    )
    return [DefaultInfo(files = depset([out]))]

contract_output = rule(
    implementation = _contract_output_impl,
    attrs = {
        "kernel": attr.string(mandatory = True),
        "what": attr.string(mandatory = True, values = ["contexts", "contract-header"]),
        "out": attr.string(mandatory = True),
        "registry": attr.label(mandatory = True),
        "_contract": attr.label(default = "//:contract", providers = [BuildSettingInfo]),
        "_script": attr.label(default = "//hardware:emit.py", allow_single_file = True),
        "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
    },
    doc = "One output of a kernel's registry for the build's contract (`--//:contract`): the contexts file or the run time's contract header.",
)
