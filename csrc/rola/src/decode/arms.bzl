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
