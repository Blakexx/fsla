"""THE DECODE FAMILY'S ARMS, declared: d_v x levels x decay, every one built and shipped
(docs/internals/decode/decode.md#decode-arm-0). Rows in canonical order: an arm's index is its row's. The decode step
is one hand unit (`decode.cu`) that dispatches over the selection header's arm set, so the family has no unit of its own
(bazel/arms.bzl's `arm_family`, without `unit`).
"""

DECODE_KEY = ("d_v", "levels", "decay")

DECODE_DOMAINS = {
    "d_v": [32, 64],
    "levels": [1, 2, 3, 4],
    "decay": [0, 1],
}

#: the arm's kernel, as the toolkit demangles its name up to the parameter list, with its template arguments the key's fields (`@field@`) and the
#: architecture's compute capability (`@cc@`): bazel/arms.bzl generates each arm's exact pattern from it, which the SASS gate holds the binary to
#: (every arm carried once) and every declaration below is stated over
DECODE_KERNEL = "rola::decode::detail::decode_step_kernel<denseref::arch::BuildTag<denseref::arch::Build{(int)@cc@, (int)0}>, (int)@d_v@, (int)@levels@, (bool)@decay@>"

#: the arms' declarations (`arm_declaration`: a rule, the count its kernels may keep, why; over the arms `where` selects): none. Every decode
#: arm is clean of what the SASS gate refuses and of the toolkit's notices (ptxas prints only its per-function summaries for them), so none
#: keeps a finding and none is declared
DECODE_DECLARATIONS = []

DECODE_ARMS = [
    [32, 1, 0],
    [32, 1, 1],
    [32, 2, 0],
    [32, 2, 1],
    [32, 3, 0],
    [32, 3, 1],
    [32, 4, 0],
    [32, 4, 1],
    [64, 1, 0],
    [64, 1, 1],
    [64, 2, 0],
    [64, 2, 1],
    [64, 3, 0],
    [64, 3, 1],
    [64, 4, 0],
    [64, 4, 1],
]
