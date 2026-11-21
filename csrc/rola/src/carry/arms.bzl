"""THE CARRY FAMILY'S ARMS, declared: EMPTY. No carry kernel is written on this line (the ampere family's schedules and
their arms are the branch `ampere`'s), so no arm is declared, none is built, and the carry op refuses by name. The key
and its domains stand as the carve's operation-plan pair (docs/internals/engine/rules.md#carve) until a hopper carry's
design re-keys them; the carry fills `CARRY_ARMS` and `bazel/arms.bzl`'s `arm_family` makes a unit of each, per built
architecture, with the selection header and the Python table (docs/build-graph.md#arms).
"""

#: a row's fields: the fold's dense rows a warp and its operand's factor count, the readout's the same, the box's
#: channel extent, and whether both sides list the same level
CARRY_KEY = ("fold_rows", "fold_factors", "read_rows", "read_factors", "channels", "same_split")

CARRY_DOMAINS = {
    "fold_rows": [16, 32, 64, 128],
    "fold_factors": [1, 2],
    "read_rows": [16, 32, 64, 128],
    "read_factors": [1, 2],
    "channels": [32, 64],
    "same_split": [0, 1],
}

#: the arms, one row of the key each, in canonical order: an arm's index is its row's
CARRY_ARMS = []
