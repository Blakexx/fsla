"""THE PRODUCER'S SOLVES' ARM FAMILIES, one for each kernel (`entmax.cu`, `factor.cu`; docs/internals/entmax/entmax.md#the-contract). A family's arms are not listed
here: they are what `entmax_reach` (`common/plan/entmax_launch.h`) reaches over the kernel's contract (`hardware/kernels/<kernel>/`), which `arms_gen` writes as the
rows file `arm_family` reads, an arm's index its row's in canonical order. The key is the kernel's template arguments the arm fixes, named as the kernel names them: `lw` (the lane width), `ipt` (the
items a thread), `alpha15` (the four kernels that have an alpha: 1 for alpha 1.5, 0 for 2.0) and each plane's element type as the code `common/plan/entmax_launch.h`
states (0 fp32, 1 bf16; the contracts hold bf16 alone), which `ENTMAX_SPELLINGS` spells as the type in the kernel's name.

Each instantiation keeps what its SASS shows on sm_90a today, a cap and never more: the findings of the two checker rules the gate applies to every function,
which the solves' instantiations predate (docs/internals/tools/sass_gate.md#declarations), fixed at the family's Hopper rewrite -- but for the factor
backward's, whose reason is its own (`_REASON`), and the softmax backward, which has none. A row below is
`[lane, items, alpha15, logit, value, load_in_trip, loop_convergence]` (without `alpha15` for the softmax kernels), one arm each; an arm with no row is clean of both rules.
"""

load("//bazel:arms.bzl", "arm_declaration")

ENTMAX_FAMILIES = ["union_forward", "union_backward", "factor_forward", "factor_backward", "softmax_forward", "softmax_backward"]

#: the file whose host dispatch includes each family's selection header
ENTMAX_INCLUDER = {
    "union_forward": "csrc/fsla/src/entmax/entmax.cu",
    "union_backward": "csrc/fsla/src/entmax/entmax.cu",
    "factor_forward": "csrc/fsla/src/entmax/factor.cu",
    "factor_backward": "csrc/fsla/src/entmax/factor.cu",
    "softmax_forward": "csrc/fsla/src/entmax/factor.cu",
    "softmax_backward": "csrc/fsla/src/entmax/factor.cu",
}

#: the type each dtype code spells in a kernel's name (`arm_family`'s `spellings`): 0 fp32, 1 bf16
ENTMAX_SPELLINGS = {"logit": ["float", "__nv_bfloat16"], "value": ["float", "__nv_bfloat16"]}

UNION_FORWARD_KEY = ("lw", "ipt", "alpha15", "logit", "value")
UNION_BACKWARD_KEY = ("lw", "ipt", "alpha15", "logit", "value")
FACTOR_FORWARD_KEY = ("lw", "ipt", "alpha15", "logit", "value")
FACTOR_BACKWARD_KEY = ("lw", "ipt", "alpha15", "logit", "value")
SOFTMAX_FORWARD_KEY = ("lw", "ipt", "logit", "value")
SOFTMAX_BACKWARD_KEY = ("lw", "ipt", "logit", "value")

ENTMAX_KEYS = {
    "union_forward": UNION_FORWARD_KEY,
    "union_backward": UNION_BACKWARD_KEY,
    "factor_forward": FACTOR_FORWARD_KEY,
    "factor_backward": FACTOR_BACKWARD_KEY,
    "softmax_forward": SOFTMAX_FORWARD_KEY,
    "softmax_backward": SOFTMAX_BACKWARD_KEY,
}

#: each family's kernel, as the toolkit demangles its name up to the parameter list, with its template arguments the key's fields (`@field@`) and the
#: architecture's compute capability (`@cc@`): bazel/arms.bzl generates each arm's exact pattern from it, which the SASS gate holds the binary to
#: (every arm carried once) and every declaration below is stated over
_BUILD = "denseref::arch::BuildTag<denseref::arch::Build{(int)@cc@}>"

ENTMAX_KERNEL = {
    kernel: "fsla::entmax::detail::%s_kernel<%s, (int)@lw@, (int)@ipt@, %s@logit@, @value@>" % (kernel, _BUILD, "(bool)@alpha15@, " if len(ENTMAX_KEYS[kernel]) == 5 else "")
    for kernel in ENTMAX_FAMILIES
}

_PREDATES = "predates the rule; fixed at the family's Hopper rewrite: "
#: A FAMILY WHOSE DECLARATIONS HAVE THEIR OWN REASON: the entmax factor backward's trip loop holds convergence instructions whose cause is open
#: (docs/internals/entmax/factor.md#backward-stages); every other family's are `_PREDATES`
_REASON = {
    "factor_backward": "cause open, the counts fit the slow paths of the trip's division and of each item's square root: ",
}
_FOUND = {
    "load_in_trip": "global loads whose value the same trip reads, in the solves' loops",
    "loop_convergence": "convergence instructions (BSSY, BSYNC, BRA.DIV) inside the solves' loops",
}
_FINDINGS = {
    "union_forward": [
        [8, 1, 0, 1, 1, 2, 108],
        [8, 1, 1, 1, 1, 2, 109],
        [16, 1, 0, 1, 1, 2, 132],
        [16, 1, 1, 1, 1, 2, 133],
        [32, 1, 0, 1, 1, 2, 35],
        [32, 1, 1, 1, 1, 2, 34],
        [32, 2, 0, 1, 1, 4, 55],
        [32, 2, 1, 1, 1, 4, 56],
        [32, 4, 0, 1, 1, 8, 95],
        [32, 4, 1, 1, 1, 8, 100],
        [32, 8, 0, 1, 1, 0, 185],
        [32, 8, 1, 1, 1, 0, 198],
    ],
    "union_backward": [
        [8, 1, 0, 1, 1, 10, 48],
        [8, 1, 1, 1, 1, 11, 56],
        [16, 1, 0, 1, 1, 10, 54],
        [16, 1, 1, 1, 1, 11, 62],
        [32, 1, 0, 1, 1, 10, 24],
        [32, 1, 1, 1, 1, 11, 32],
        [32, 2, 0, 1, 1, 20, 34],
        [32, 2, 1, 1, 1, 22, 50],
        [32, 4, 0, 1, 1, 40, 62],
        [32, 4, 1, 1, 1, 44, 94],
        [32, 8, 0, 1, 1, 80, 150],
        [32, 8, 1, 1, 1, 88, 214],
    ],
    "factor_forward": [
        [8, 1, 0, 1, 1, 1, 74],
        [8, 1, 1, 1, 1, 1, 75],
        [16, 1, 0, 1, 1, 1, 94],
        [16, 1, 1, 1, 1, 1, 95],
        [32, 1, 0, 1, 1, 1, 15],
        [32, 1, 1, 1, 1, 1, 14],
        [32, 2, 0, 1, 1, 2, 19],
        [32, 2, 1, 1, 1, 2, 20],
        [32, 4, 0, 1, 1, 4, 31],
        [32, 4, 1, 1, 1, 4, 36],
        [32, 8, 0, 1, 1, 0, 65],
        [32, 8, 1, 1, 1, 0, 78],
    ],
    "factor_backward": [
        [16, 1, 0, 1, 1, 0, 4],
        [16, 1, 1, 1, 1, 0, 8],
        [32, 1, 0, 1, 1, 0, 4],
        [32, 1, 1, 1, 1, 0, 8],
        [32, 2, 0, 1, 1, 0, 2],
        [32, 2, 1, 1, 1, 0, 10],
        [32, 4, 0, 1, 1, 0, 2],
        [32, 4, 1, 1, 1, 0, 18],
        [32, 8, 0, 1, 1, 0, 2],
        [32, 8, 1, 1, 1, 0, 34],
    ],
    "softmax_forward": [
        [16, 1, 1, 1, 1, 30],
        [32, 1, 1, 1, 1, 7],
        [32, 2, 1, 1, 2, 24],
        [32, 4, 1, 1, 4, 44],
        [32, 8, 1, 1, 8, 84],
    ],
    "softmax_backward": [],
}

def _declarations(kernel):
    key = ENTMAX_KEYS[kernel]
    return [
        arm_declaration(rule, row[len(key) + i], _REASON.get(kernel, _PREDATES) + _FOUND[rule], where = {field: [row[j]] for j, field in enumerate(key)}, arch = "sm_90a")
        for row in _FINDINGS[kernel]
        for i, rule in enumerate(["load_in_trip", "loop_convergence"])
        if row[len(key) + i]
    ]

ENTMAX_DECLARATIONS = {kernel: _declarations(kernel) for kernel in ENTMAX_FAMILIES}
