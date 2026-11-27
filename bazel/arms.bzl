"""A KERNEL'S ARMS, DECLARED IN STARLARK AND INSTANTIATED BY THE GRAPH (docs/build-graph.md#arms).

A kernel declares its arm family in a `.bzl` beside it: the arm's key (`<FAMILY>_KEY`, the fields that make one
instantiation) and its kernel and declarations. The arms are either literal rows (`arms`, with each field's domain, in canonical order: a
family with per-arm units needs them at load time) or a rows file (`arms_file`) a host tool wrote, the arm list `arms_gen` evaluates
the family's launch function's arm half into over the contract (docs/internals/common/plan/arms_gen.md). `arm_family` makes, for that one list:

- one translation unit per arm and per BUILT architecture (`<name>_arm_<i>_sm_<arch>`) when the family has per-arm units
  (`unit`), whose source is the family's arm header and one `<NAME>_ARM_DEFINE(i, key...)` under the arm's own selection
  guard -- an arm the build does not select is a unit that compiles to nothing, and a unit the build never compiles;
- the selection header of the build (`<name>_selection.inc`): the arms this build carries -- every row of the list, which is the arms the
  build's CONTRACT reaches (`--//:contract`, docs/build.md#narrowed) -- with the arm set the host dispatches over;
- the Python table (`<name>_arms.py`): the key and the rows, which the build's gates and the measurement read;
- when the family states its kernel (`kernel`), its declarations (`<name>_declarations`): each arm's exact kernel-name pattern, generated from the
  arm's template arguments, and the declarations of `declarations` for the arms they select, on each architecture the build carries
  (docs/build-graph.md#declarations).

A family with no rows makes no unit and a selection naming none.

`ARM_FAMILIES` is the registry of the families a build ships: its tables, its gates and its census
parameters are made for each of them, and a family joins by being named there.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(":cuda.bzl", "CudaUnitInfo", "cuda_unit", "declaration", "kernel_pattern")

#: THE FAMILIES A BUILD SHIPS, each `arm_family`'d in `csrc/fsla/BUILD.bazel` with its declaration beside its kernel
#: (`csrc/fsla/src/<family>/arms.bzl`): the build rules read this list and no other
ARM_FAMILIES = ["decode", "factor_forward", "factor_backward", "softmax_forward", "softmax_backward"]

#: the tool every family's generated files come from (bazel/arm_emit.py), and the interpreter that runs it
_EMIT_ATTRS = {
    "_script": attr.label(default = "//bazel:arm_emit.py", allow_single_file = True),
    "_python": attr.label(default = "@fsla_env//:python", allow_single_file = True, cfg = "exec"),
}

def _emit(ctx, what, rows, out, args = [], progress = None):
    """Run the emitter (`what`: selection, table or declarations) over the family's rows file, writing `out`."""
    arguments = ctx.actions.args()
    arguments.add(ctx.file._script)
    arguments.add(what)
    arguments.add("--rows", rows)
    arguments.add("--family", ctx.attr.family)
    arguments.add_all(args)
    arguments.add("--out", out)
    ctx.actions.run(
        executable = ctx.file._python,
        arguments = [arguments],
        inputs = [rows, ctx.file._script],
        outputs = [out],
        mnemonic = "FslaArm" + what.capitalize(),
        progress_message = progress or ("Writing %s's arm " + what),
    )

def _rows_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(out, json.encode({"key": ctx.attr.key, "arms": json.decode(ctx.attr.arms)}) + "\n")
    return [DefaultInfo(files = depset([out]))]

arm_rows = rule(
    implementation = _rows_impl,
    attrs = {
        "key": attr.string_list(mandatory = True),
        "arms": attr.string(mandatory = True, doc = "the declared arms, JSON"),
    },
    doc = "A family's literal rows as the rows file the emitter reads.",
)

def _check(name, key, domains, arms):
    """The declaration, refused field by field: each way it could be wrong is a compile error in a generated file, or a
    silent renumbering of every arm index a header and a unit name."""
    if not key:
        fail("%s: the key names no field" % name)
    if sorted(domains.keys()) != sorted(key):
        fail("%s: the domains name %s, the key's fields are %s" % (name, sorted(domains.keys()), sorted(key)))
    for field in key:
        if not domains[field]:
            fail("%s: the domain of %s is empty" % (name, field))
    for row in arms:
        if len(row) != len(key):
            fail("%s: row %s is not the key %s as %d integers" % (name, list(row), list(key), len(key)))
        for field, value in zip(key, row):
            if value not in domains[field]:
                fail("%s: row %s has %s=%d, which is not in the declared domain %s" % (name, list(row), field, value, domains[field]))
    for i in range(len(arms)):
        if list(arms[i]) in [list(row) for row in arms[:i]]:
            fail("%s: row %s is declared twice; a duplicate would give one kernel two indices" % (name, list(arms[i])))
    if [list(row) for row in arms] != sorted([list(row) for row in arms]):
        fail(("%s: the rows are not in canonical order. An arm's index is named by the selection header and the units, so it " +
              "must be a property of the declaration, not of where a row was typed. Write them sorted: %s") % (name, sorted([list(row) for row in arms])))

def arm_declaration(rule, allow, reason, where = {}, arch = None):
    """A declaration of a family's arms, stated beside its arms (`<FAMILY>_DECLARATIONS`): the checker's form (`declaration` in cuda.bzl) over the arms
    `where` selects (`{field: [values]}`, every arm where none), its kernel pattern generated from each arm's template arguments, not written by hand.

    Args:
      rule: a rule of the checker, or a notice rule of the checker's ledger.
      allow: the count the arm's kernel may keep.
      reason: why; required.
      where: the arms it covers, by the key's fields.
      arch: the build target it attaches to (`sm_90a`), every built architecture where none.
    """
    declaration("^", rule, allow, reason, arch)
    return struct(rule = rule, allow = allow, reason = reason, where = where, arch = arch)

def _arm_declarations_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".json")
    args = ["--kernel", ctx.attr.kernel,
            "--archs", ",".join(ctx.attr.archs_flag[BuildSettingInfo].value)]
    for d in ctx.attr.declarations:
        args += ["--declared", d]
    if ctx.attr.spellings:
        args += ["--spellings", ctx.attr.spellings]
    _emit(ctx, "declarations", ctx.file.rows, out, args, "Writing %s's arm declarations")
    return [
        CudaUnitInfo(units = depset(), declarations = depset(), declaration_files = depset([out])),
        CcInfo(),
    ]

arm_declarations = rule(
    implementation = _arm_declarations_impl,
    attrs = _EMIT_ATTRS | {
        "family": attr.string(mandatory = True),
        "rows": attr.label(mandatory = True, allow_single_file = True, doc = "the family's rows file"),
        "kernel": attr.string(mandatory = True, doc = "the arm's kernel pattern (`cuda.bzl`'s `kernel_pattern`), `@field@` standing for a template argument the emitter fills"),
        "declarations": attr.string_list(doc = "the family's declarations (`arm_declaration`), each as JSON"),
        "spellings": attr.string(doc = "the spelling of each key field whose template argument is a type, JSON: `{field: [the type's name for value 0, for value 1, ...]}`"),
        "archs_flag": attr.label(default = "//:archs", providers = [BuildSettingInfo]),
    },
    provides = [CudaUnitInfo, CcInfo],
    doc = "One arm family's declarations for the build, written by the emitter: the exact kernel pattern of each arm the build carries, for each " +
          "architecture it carries, and the family's declarations on the arms they select. The SASS gate merges the file into what it reads.",
)

def _selection_impl(ctx):
    header = ctx.actions.declare_file("%s/%s_selection.inc" % (ctx.attr.dir, ctx.attr.family))
    args = ["--selected=" + ",".join(ctx.attr.selected)]
    _emit(ctx, "selection", ctx.file.rows, header, ["--includer", ctx.attr.includer] + args, "Writing %s's arm selection")
    return [DefaultInfo(files = depset([header]))]

arm_selection = rule(
    implementation = _selection_impl,
    attrs = _EMIT_ATTRS | {
        "family": attr.string(mandatory = True),
        "rows": attr.label(mandatory = True, allow_single_file = True, doc = "the family's rows file"),
        "includer": attr.string(mandatory = True, doc = "who includes the header, for its own comment"),
        "selected": attr.string_list(default = ["all"], doc = "the arms the header names: every row of the rows file (`all`), which is the arms the build's contract reaches, or a fixed selection of indices (a probe's)"),
        "dir": attr.string(default = "generated", doc = "the directory the header is written to, in the package"),
    },
    doc = "The selection header of one arm family for one build: the arms it carries, and the arm set the host dispatches over.",
)

def _table_impl(ctx):
    out = ctx.actions.declare_file("generated/%s_arms.py" % ctx.attr.family)
    _emit(ctx, "table", ctx.file.rows, out, [], "Writing %s's arm table")
    return [DefaultInfo(files = depset([out]))]

arm_table = rule(
    implementation = _table_impl,
    attrs = _EMIT_ATTRS | {
        "family": attr.string(mandatory = True),
        "rows": attr.label(mandatory = True, allow_single_file = True, doc = "the family's rows file"),
    },
    doc = "The Python table of one arm family: its key and its rows, read by the build's gates and the measurement.",
)

def _arm_set_impl(ctx):
    archs = ctx.attr.archs_flag[BuildSettingInfo].value
    chosen = []
    for unit, where in ctx.attr.units.items():
        at = json.decode(where)
        if at["arch"] in archs:
            chosen.append(unit)
    return [
        DefaultInfo(files = depset(transitive = [unit[DefaultInfo].files for unit in chosen])),
        cc_common.merge_cc_infos(cc_infos = [unit[CcInfo] for unit in chosen]),
        CudaUnitInfo(units = depset(transitive = [unit[CudaUnitInfo].units for unit in chosen]), declarations = depset(), declaration_files = depset()),
    ]

_arm_set = rule(
    implementation = _arm_set_impl,
    attrs = {
        "family": attr.string(mandatory = True),
        "units": attr.label_keyed_string_dict(providers = [CcInfo, CudaUnitInfo], doc = "each unit, and its architecture (JSON)"),
        "archs_flag": attr.label(default = "//:archs", providers = [BuildSettingInfo]),
    },
    doc = "The arm units the build's architectures name: an architecture the build does not want is a unit it never compiles.",
)

def _arm_source(name, i, row, arm_header, define, selection_header):
    native.genrule(
        name = "%s_arm_%d_source" % (name, i),
        outs = ["%s_arm_%d.cu" % (name, i)],
        cmd = "printf '%s' > $@" % (
            "// GENERATED by bazel/arms.bzl -- do not edit: arm %d of the %s family.\\n" % (i, name) +
            "#include \"%s\"\\n#include \"%s\"\\n\\n" % (arm_header, selection_header) +
            "#if defined(FSLA_%s_BUILD_%d)\\n%s(%d, %s)\\n#endif\\n" % (name.upper(), i, define, i, ", ".join([str(v) for v in row]))
        ),
    )

def arm_family(name, key, includer, built_archs, arms = None, domains = None, arms_file = None, unit = None, kernel = None, declarations = [], spellings = {}):
    """One kernel's arm family: its selection header, its Python table and, when `unit` is given, one translation unit
    per arm and built architecture.

    Args:
      name: the family's name (`decode`): `<name>_selection`, `<name>_table` and `<name>_arm_set` are its targets.
      key: the fields of a row, in order.
      includer: who includes the selection header, for the header's comment.
      built_archs: the architectures the units are made for (`BUILT_ARCHS`); a build compiles those its `--//:archs` names.
      arms: the declared rows, in canonical order, with `domains` (each field's admitted values, `{field: [values]}`): a family
        with per-arm units, whose units the graph must know at load time.
      arms_file: a label of the rows file (`{"key", "arms"}`) a host tool wrote (`arms_gen`), in place of `arms`: the arms are then
        known when the build runs, so the family has no per-arm unit.
      unit: None, or a dict(arm_header, define, copts, deps): the header that defines the arm macro, the macro
        (`<NAME>_ARM_DEFINE`) and the CUDA rule's options and dependencies.
      kernel: None, or the arm's kernel name up to its parameter list, as the toolkit demangles it, with `@<field>@` for a template argument
        and `@cc@` for the architecture (`bazel/cuda.bzl`'s `kernel_pattern`): `<name>_declarations` then holds each arm's exact pattern, which the SASS gate
        holds the binary to (every arm carried once), and the `declarations`.
      declarations: the family's declarations (`arm_declaration`), over the arms they select; they need `kernel`.
      spellings: `{field: [spelling of value 0, of value 1, ...]}` for each key field whose template argument is a type (`float`, `__nv_bfloat16`), a code in
        the key: the arm's pattern spells that argument as the type its value indexes, and the host maps the code back to the type in the arm's body.
    """
    if (arms == None) == (arms_file == None):
        fail("%s: the arms are literal rows (`arms`, `domains`) or a rows file (`arms_file`), one of them" % name)
    if arms_file != None and unit != None:
        fail("%s: per-arm units need the rows at load time; a rows file a tool writes has none" % name)
    if declarations and not kernel:
        fail("%s: declarations need the arm's `kernel`, which generates the pattern each is stated over" % name)
    for d in declarations:
        for field, values in d.where.items():
            if field not in key:
                fail("%s: a declaration of %s selects on %r, which is not a field of the key %s" % (name, d.rule, field, list(key)))
            if domains != None:
                for value in values:
                    if value not in domains[field]:
                        fail("%s: a declaration of %s selects %s=%r, which is not in the declared domain %s" % (name, d.rule, field, value, domains[field]))
    if arms != None:
        _check(name, key, domains, arms)
        arm_rows(name = name + "_rows", key = list(key), arms = json.encode([list(row) for row in arms]))
        rows = ":" + name + "_rows"
    else:
        rows = arms_file
    arm_selection(name = name + "_selection", family = name, rows = rows, includer = includer)
    arm_table(name = name + "_table", family = name, rows = rows)
    if kernel:
        arm_declarations(
            name = name + "_declarations",
            family = name,
            rows = rows,
            kernel = kernel_pattern(kernel),
            declarations = [json.encode(d) for d in declarations],
            spellings = json.encode(spellings) if spellings else "",
        )
    if unit == None:
        return
    units = {}
    for i, row in enumerate(arms):
        _arm_source(name, i, row, unit["arm_header"], unit["define"], name + "_selection.inc")
        for arch in built_archs:
            target = "%s_arm_%d_sm_%s" % (name, i, arch)
            cuda_unit(
                name = target,
                srcs = [":%s_arm_%d_source" % (name, i)],
                archs = ["compute_%s:sm_%s" % (arch, arch)],
                copts = unit["copts"],
                deps = unit["deps"],
            )
            units[":" + target] = json.encode({"arch": arch})
    _arm_set(name = name + "_arm_set", family = name, units = units)
