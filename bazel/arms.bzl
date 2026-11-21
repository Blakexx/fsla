"""A KERNEL'S ARMS, DECLARED IN STARLARK AND INSTANTIATED BY THE GRAPH (docs/build-graph.md#arms).

A kernel declares its arm family in a `.bzl` beside it: the arm's key (`<FAMILY>_KEY`, the fields that make one
instantiation), each field's domain (`<FAMILY>_DOMAINS`) and the arms (`<FAMILY>_ARMS`, one row of the key each, in
canonical order). `arm_family` reads that one list and makes, for it:

- one translation unit per arm and per BUILT architecture (`<name>_arm_<i>_sm_<arch>`) when the family has per-arm units
  (`unit`), whose source is the family's arm header and one `<NAME>_ARM_DEFINE(i, key...)` under the arm's own selection
  guard -- an arm the build does not select is a unit that compiles to nothing, and a unit the build never compiles;
- the selection header of the build (`<name>_selection.inc`): which arms this build carries, from `--//:<name>_arms`
  (`flag`), with the arm set the host dispatches over;
- the Python table (`<name>_arms.py`): the key and the rows, which the build's gates and the measurement read;
- when the family states its kernel (`kernel`), its declarations (`<name>_declarations`): each arm's exact kernel-name pattern, generated from the
  arm's template arguments, and the declarations of `declarations` for the arms they select, on each architecture the build carries
  (docs/build-graph.md#declarations).

The Hopper carry fills its list; a family with no rows makes no unit and a selection naming none.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load(":cuda.bzl", "CudaUnitInfo", "cuda_unit", "declaration")

def _check(name, key, domains, arms):
    """The declaration, refused field by field: each way it could be wrong is a compile error in a generated file, or a
    silent renumbering of every arm index a flag, a header and a unit name."""
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
        fail("%s: the rows are not in canonical order. An arm's index is named by --//:%s_arms, the selection header and the units, so it " % (name, name) +
             "must be a property of the declaration, not of where a row was typed. Write them sorted: %s" % sorted([list(row) for row in arms]))

def arm_declaration(rule, allow, reason, where = {}, arch = None):
    """A declaration of a family's arms, stated beside its arms (`<FAMILY>_DECLARATIONS`): the checker's form (`declaration` in cuda.bzl) over the arms
    `where` selects (`{field: [values]}`, every arm where none), its kernel pattern generated from each arm's template arguments, not written by hand.

    Args:
      rule: a rule of the checker, or a notice rule of the project's ledger (`tools/notice_ledger.json`).
      allow: the count the arm's kernel may keep.
      reason: why; required.
      where: the arms it covers, by the key's fields.
      arch: the build target it attaches to (`sm_90a`), every built architecture where none.
    """
    declaration("^", rule, allow, reason, arch)
    return struct(rule = rule, allow = allow, reason = reason, where = where, arch = arch)

def _escape(text):
    """A regex that matches the text and nothing else."""
    out = ""
    for ch in text.elems():
        if ch in "\\.^$*+?{}[]|()":
            out += "\\"
        out += ch
    return out

def arm_kernel_pattern(kernel, key, row, arch):
    """The exact pattern of an arm's kernel, over its demangled name (the toolkit's `cu++filt`): `kernel` is the name's text up to its parameter list,
    with `@<field>@` for a field of the key (its value, as the demangler prints an integer or a bool template argument: the number) and `@cc@` for the
    architecture's compute capability as `__CUDA_ARCH__` reads it (`90a` is 900). The pattern is anchored at the return type and the parameter list."""
    parts = kernel.split("@")
    if len(parts) % 2 == 0:
        fail("%s: an unbalanced `@`" % kernel)
    values = {field: str(value) for field, value in zip(key, row)}
    values["cc"] = str(int(arch.removesuffix("a")) * 10)
    text = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text += _escape(part)
        elif part not in values:
            fail("%s: `@%s@` is no field of the key %s (nor `cc`)" % (kernel, part, list(key)))
        else:
            text += values[part]
    return "^void " + text + "\\("

def _selected(declaration, key, row):
    for field, wanted in declaration["where"].items():
        if row[list(key).index(field)] not in wanted:
            return False
    return True

def _arm_declarations_impl(ctx):
    rows = json.decode(ctx.attr.rows)
    key = ctx.attr.key
    picked = _chosen(ctx.attr.flag[BuildSettingInfo].value, len(rows), "--//:%s_arms" % ctx.attr.family)
    declared = [json.decode(d) for d in ctx.attr.declarations]
    found, arms = [], []
    for arch in ctx.attr.archs_flag[BuildSettingInfo].value:
        on = "sm_" + arch
        for i in picked:
            pattern = arm_kernel_pattern(ctx.attr.kernel, key, rows[i], arch)
            arms.append(struct(arch = on, family = ctx.attr.family, arm = i, kernel = pattern))
            for d in declared:
                if d["arch"] in (None, on) and _selected(d, key, rows[i]):
                    found.append(declaration(pattern, d["rule"], d["allow"], d["reason"], on))
    return [
        CudaUnitInfo(units = depset(), declarations = depset(found), arms = depset(arms)),
        CcInfo(),
    ]

arm_declarations = rule(
    implementation = _arm_declarations_impl,
    attrs = {
        "family": attr.string(mandatory = True),
        "key": attr.string_list(mandatory = True),
        "rows": attr.string(mandatory = True, doc = "the declared arms, JSON"),
        "kernel": attr.string(mandatory = True, doc = "the arm's kernel name with `@field@` for its template arguments (`arm_kernel_pattern`)"),
        "declarations": attr.string_list(doc = "the family's declarations (`arm_declaration`), each as JSON"),
        "flag": attr.label(mandatory = True, providers = [BuildSettingInfo]),
        "archs_flag": attr.label(default = "//:archs", providers = [BuildSettingInfo]),
    },
    provides = [CudaUnitInfo, CcInfo],
    doc = "One arm family's declarations for the build: the exact kernel pattern of each arm the build carries, for each architecture it carries, " +
          "and the family's declarations on the arms they select.",
)

def _chosen(tokens, rows, flag):
    """The arm indices a build's `--//:<family>_arms` names: indices or `all`, empty for every arm, an index out of range
    refused."""
    if not tokens:
        return list(range(rows))
    want = []
    for token in tokens:
        if token == "all":
            want += range(rows)
        elif not token.isdigit() or int(token) >= rows:
            fail("%s: %r is not an arm index or 'all'; the list has %d rows (0..%d)" % (flag, token, rows, rows - 1))
        else:
            want.append(int(token))
    return sorted({i: True for i in want}.keys())

def _selection_impl(ctx):
    rows = json.decode(ctx.attr.rows)
    key = ctx.attr.key
    name = ctx.attr.family
    upper = name.upper()
    if ctx.attr.flag:
        picked = _chosen(ctx.attr.flag[BuildSettingInfo].value, len(rows), "--//:%s_arms" % name)
    else:
        picked = sorted([int(i) for i in ctx.attr.selected])
        for i in picked:
            if i >= len(rows):
                fail("%s: no such arm %d; the list has %d rows" % (ctx.label, i, len(rows)))
    out = ["// GENERATED by bazel/arms.bzl -- do not edit.\n",
           "// the %s ARM SELECTION for this build. NEVER CHECKED IN;\n" % upper,
           "// written by the graph (from --//:%s_arms).\n" % name,
           "// Included only by %s.\n" % ctx.attr.includer,
           "#define ROLA_%s_BUILT 1\n" % upper]
    for i in picked:
        out.append("#define ROLA_%s_BUILD_%d 1\n" % (upper, i))
    out.append("\n// THE ARM SET, for the host's dispatch: one F(i, %s)\n" % ", ".join(key))
    out.append("// per BUILT arm, in index order. See docs/build-graph.md#arms\n")
    out.append("#define ROLA_%s_ARM_COUNT %d\n" % (upper, len(picked)))
    out.append("#define ROLA_%s_ARM_DECLARED %d\n" % (upper, len(rows)))
    if picked:
        body = " \\\n".join(["  F(%d, %s)" % (i, ", ".join([str(v) for v in rows[i]])) for i in picked])
        out.append("#define ROLA_%s_ARM_SET_X(F) \\\n%s\n" % (upper, body))
    else:
        out.append("//: NO ARM IS BUILT: the X-macro expands to nothing, `ROLA_%s_ARM_COUNT` is 0,\n" % upper)
        out.append("//: and the host's dispatch refuses every call by naming the empty set. That is the\n")
        out.append("//: honest state of a family with no kernels, not a hole.\n")
        out.append("#define ROLA_%s_ARM_SET_X(F)\n" % upper)
    header = ctx.actions.declare_file("%s/%s_selection.inc" % (ctx.attr.dir, name))
    ctx.actions.write(header, "".join(out))
    return [DefaultInfo(files = depset([header]))]

arm_selection = rule(
    implementation = _selection_impl,
    attrs = {
        "family": attr.string(mandatory = True),
        "key": attr.string_list(mandatory = True),
        "rows": attr.string(mandatory = True, doc = "the declared arms, JSON"),
        "includer": attr.string(mandatory = True, doc = "who includes the header, for its own comment"),
        "flag": attr.label(providers = [BuildSettingInfo], doc = "the build's `--//:<family>_arms`: indices or `all`, empty for every arm"),
        "selected": attr.string_list(doc = "a fixed selection, in place of the flag (a fixture's)"),
        "dir": attr.string(default = "generated", doc = "the directory the header is written to, in the package"),
    },
    doc = "The selection header of one arm family for one build: the arms it carries, and the arm set the host dispatches over.",
)

def _row(row):
    return "(%s)" % (", ".join([str(v) for v in row]) + ("," if len(row) == 1 else ""))

def _table_impl(ctx):
    rows = json.decode(ctx.attr.rows)
    out = ctx.actions.declare_file("generated/%s_arms.py" % ctx.attr.family)
    text = ['"""GENERATED by bazel/arms.bzl -- do not edit.\n\n',
            "The %s family's declared arms: `KEY` names a row's fields, `ARMS` is the arm table, arm `i` everywhere.\n" % ctx.attr.family,
            '"""\n\n',
            "KEY = (%s)\n" % (", ".join([json.encode(field) for field in ctx.attr.key]) + ("," if len(ctx.attr.key) == 1 else "")),
            "ARMS = (%s)\n" % ("".join(["\n    %s," % _row(row) for row in rows]) + ("\n" if rows else ""))]
    ctx.actions.write(out, "".join(text))
    return [DefaultInfo(files = depset([out]))]

arm_table = rule(
    implementation = _table_impl,
    attrs = {
        "family": attr.string(mandatory = True),
        "key": attr.string_list(mandatory = True),
        "rows": attr.string(mandatory = True, doc = "the declared arms, JSON"),
    },
    doc = "The Python table of one arm family: its key and its rows, read by the build's gates and the measurement.",
)

def _arm_set_impl(ctx):
    rows = ctx.attr.rows
    want = _chosen(ctx.attr.flag[BuildSettingInfo].value, rows, "--//:%s_arms" % ctx.attr.family)
    archs = ctx.attr.archs_flag[BuildSettingInfo].value
    chosen = []
    for unit, where in ctx.attr.units.items():
        at = json.decode(where)
        if at["arm"] in want and at["arch"] in archs:
            chosen.append(unit)
    return [
        DefaultInfo(files = depset(transitive = [unit[DefaultInfo].files for unit in chosen])),
        cc_common.merge_cc_infos(cc_infos = [unit[CcInfo] for unit in chosen]),
        CudaUnitInfo(units = depset(transitive = [unit[CudaUnitInfo].units for unit in chosen]), declarations = depset(), arms = depset()),
    ]

_arm_set = rule(
    implementation = _arm_set_impl,
    attrs = {
        "family": attr.string(mandatory = True),
        "rows": attr.int(mandatory = True, doc = "the declared arms"),
        "units": attr.label_keyed_string_dict(providers = [CcInfo, CudaUnitInfo], doc = "each unit, and its arm and architecture (JSON)"),
        "flag": attr.label(mandatory = True, providers = [BuildSettingInfo]),
        "archs_flag": attr.label(default = "//:archs", providers = [BuildSettingInfo]),
    },
    doc = "The arm units the build's arms and architectures name: an arm or an architecture the build does not want is a unit it never compiles.",
)

def _arm_source(name, i, row, arm_header, define, selection_header):
    native.genrule(
        name = "%s_arm_%d_source" % (name, i),
        outs = ["%s_arm_%d.cu" % (name, i)],
        cmd = "printf '%s' > $@" % (
            "// GENERATED by bazel/arms.bzl -- do not edit: arm %d of the %s family.\\n" % (i, name) +
            "#include \"%s\"\\n#include \"%s\"\\n\\n" % (arm_header, selection_header) +
            "#if defined(ROLA_%s_BUILD_%d)\\n%s(%d, %s)\\n#endif\\n" % (name.upper(), i, define, i, ", ".join([str(v) for v in row]))
        ),
    )

def arm_family(name, key, domains, arms, flag, includer, built_archs, unit = None, kernel = None, declarations = []):
    """One kernel's arm family: its selection header, its Python table and, when `unit` is given, one translation unit
    per arm and built architecture.

    Args:
      name: the family's name (`carry`): `<name>_selection`, `<name>_table` and `<name>_arm_set` are its targets.
      key: the fields of a row, in order.
      domains: each field's admitted values, `{field: [values]}`.
      arms: the declared rows, in canonical order.
      flag: the build setting naming the arms a build carries (`//:carry_arms`).
      includer: who includes the selection header, for the header's comment.
      built_archs: the architectures the units are made for (`BUILT_ARCHS`); a build compiles those its `--//:archs` names.
      unit: None, or a dict(arm_header, define, copts, host_cc, deps): the header that defines the arm macro, the macro
        (`<NAME>_ARM_DEFINE`) and the CUDA rule's options and dependencies.
      kernel: None, or the arm's kernel name up to its parameter list, as the toolkit demangles it, with `@<field>@` for a template argument
        and `@cc@` for the architecture (`arm_kernel_pattern`): `<name>_declarations` then holds each arm's exact pattern, which the SASS gate
        holds the binary to (every arm carried once), and the `declarations`.
      declarations: the family's declarations (`arm_declaration`), over the arms they select; they need `kernel`.
    """
    _check(name, key, domains, arms)
    if declarations and not kernel:
        fail("%s: declarations need the arm's `kernel`, which generates the pattern each is stated over" % name)
    for d in declarations:
        for field, values in d.where.items():
            if field not in key:
                fail("%s: a declaration of %s selects on %r, which is not a field of the key %s" % (name, d.rule, field, list(key)))
            for value in values:
                if value not in domains[field]:
                    fail("%s: a declaration of %s selects %s=%r, which is not in the declared domain %s" % (name, d.rule, field, value, domains[field]))
    rows = json.encode([list(row) for row in arms])
    arm_selection(name = name + "_selection", family = name, key = list(key), rows = rows, includer = includer, flag = flag)
    arm_table(name = name + "_table", family = name, key = list(key), rows = rows)
    if kernel:
        arm_declarations(
            name = name + "_declarations",
            family = name,
            key = list(key),
            rows = rows,
            kernel = kernel,
            declarations = [json.encode(d) for d in declarations],
            flag = flag,
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
                host_cc = unit["host_cc"],
                deps = unit["deps"],
            )
            units[":" + target] = json.encode({"arm": i, "arch": arch})
    _arm_set(name = name + "_arm_set", family = name, rows = len(arms), units = units, flag = flag)
