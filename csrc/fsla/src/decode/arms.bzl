"""THE DECODE FAMILY'S ARM KEY, kernel and declarations (docs/internals/decode/decode.md#decode-arm-0). Its arms are not listed here: they are
what `decode_reach` (`common/plan/decode_launch.h`) reaches over the contract (`hardware/`), which `arms_gen` writes as the rows file `arm_family` reads, an arm's index
its row's in canonical order. The decode step is one hand unit (`decode.cu`) that dispatches over the selection header's arm set, so the
family has no unit of its own (bazel/arms.bzl's `arm_family`, without `unit`).
"""

load("//bazel:arms.bzl", "arm_declaration")

DECODE_KEY = ("d_v", "levels", "decay")

#: the arm's kernel, as the toolkit demangles its name up to the parameter list, with its template arguments the key's fields (`@field@`) and the
#: architecture's compute capability (`@cc@`): bazel/arms.bzl generates each arm's exact pattern from it, which the SASS gate holds the binary to
#: (every arm carried once) and every declaration below is stated over
DECODE_KERNEL = "fsla::decode::detail::decode_step_kernel<denseref::arch::BuildTag<denseref::arch::Build{(int)@cc@}>, (int)@d_v@, (int)@levels@, (bool)@decay@>"

#: the arms' declarations (`arm_declaration`: a rule, the count its kernels may keep, why; over the arms `where` selects): the findings of
#: the two checker rules the gate applies to every function, which every arm predates. Each arm keeps what its SASS shows on sm_90a today, a
#: cap and never more; the rows below are `[d_v, levels, decay, load_in_trip, loop_convergence]`, one arm each
_PREDATES = "predates the rule; fixed at the family's Hopper rewrite: "
_FOUND = {
    "load_in_trip": "global loads whose value the same trip reads, in the step's loops",
    "loop_convergence": "convergence instructions (BSSY, BSYNC, BRA.DIV) inside the step's loops",
}
_FINDINGS = [
    [32, 1, 0, 19, 156],
    [32, 1, 1, 20, 191],
    [32, 2, 0, 19, 210],
    [32, 2, 1, 20, 217],
    [32, 3, 0, 19, 221],
    [32, 3, 1, 20, 219],
    [32, 4, 0, 19, 237],
    [32, 4, 1, 20, 233],
    [64, 1, 0, 27, 184],
    [64, 1, 1, 28, 205],
    [64, 2, 0, 27, 208],
    [64, 2, 1, 28, 229],
    [64, 3, 0, 27, 210],
    [64, 3, 1, 28, 231],
    [64, 4, 0, 27, 224],
    [64, 4, 1, 28, 245],
]

DECODE_DECLARATIONS = [
    arm_declaration(rule, row[3 + i], _PREDATES + _FOUND[rule], where = {"d_v": [row[0]], "levels": [row[1]], "decay": [row[2]]}, arch = "sm_90a")
    for row in _FINDINGS
    for i, rule in enumerate(["load_in_trip", "loop_convergence"])
]
