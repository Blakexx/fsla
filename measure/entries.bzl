"""THIS TREE'S TIMING ENTRIES AND DIFF SIDES, generated from the central cells' fields (`@fsla_cells//:cells.bzl`'s
`CELLS`, docs/measurement.md#entries): every subject of `measure/subjects.py` on every cell of its kind it takes, and
each diff surface of `tests/oracle/sides.py` on the cells it runs on -- selected by the cells' facts, never by a name's
spelling, and each a label, so a name this tree does not have fails the analysis of whatever names it."""

load("@fsla_cells//:cells.bzl", "CELLS")
load("//bazel:defs.bzl", "diff_side", "timing_entry")

#: THE SUBJECTS (`measure.subjects.SUBJECTS`): each one's cell kind, and whether it takes only a cell with decode steps
#: (a test holds this table to the roster)
SUBJECTS = {
    "prefill": ("carry", False),
    "entmax_solve": ("layer", False),
    "layer_prefill": ("layer", False),
    "layer_op": ("layer", False),
    "decode_step": ("layer", True),
}

#: THE DIFF SURFACES (`tests/oracle/sides.py`): the side executor, the cells it runs on (a cell kind and, where it has
#: one, a tier), and whether it runs on the device. The oracle runs `naive_fsla` forward AND backward in fp64, and
#: autograd retains the recurrence's state at every token: MEASURED (2026-09-16) a peak above nine times
#: `B * H * T * N * (DV + 1) * 8` bytes, so it admits only the cells whose retained state at B 1, H 2 is under
#: `RETAINED_STATE_BUDGET`, and the excluded cells are a consequence a reader can compute.
SURFACES = {
    "oracle": ("tests.oracle.sides:oracle", "carry", "oracle", False),
    "producer": ("tests.oracle.sides:producer", "producer", None, False),
    "carry-kernel": ("tests.oracle.sides:carry_kernel", "carry", "oracle", True),
}
RETAINED_STATE_BUDGET = 512 * 1024 * 1024
BACKWARD_SURFACES = ["oracle"]

def _states(cell):
    n = 1
    for width in cell["routing"]["topology"]:
        n *= width
    return n

def _retained_state_bytes(cell, batch = 1, heads = 2):
    return batch * heads * cell["tokens"] * _states(cell) * (cell["dv"] + 1) * 8

def surface_cells(surface):
    """The cells one surface runs on: its kind, at its tier where it has one, under the backward's budget where it runs
    a backward."""
    _executor, kind, tier, _device = SURFACES[surface]
    return sorted([name for name, cell in CELLS.items() if cell["kind"] == kind and
                   (tier == None or cell["tier"] == tier) and
                   (surface not in BACKWARD_SURFACES or _retained_state_bytes(cell) < RETAINED_STATE_BUDGET)])

def entry_name(arm, cell):
    """A timing entry's name: its arm and its cell."""
    return "%s_%s" % (arm, cell)

def timing_entries():
    """One timing entry an arm and a cell (`measure.provider:timed`, the arm its `params` name); returns the names."""
    names = []
    for arm, (kind, decodes) in SUBJECTS.items():
        for cell, fields in sorted(CELLS.items()):
            if fields["kind"] != kind or (decodes and fields["decode_steps"] == 0):
                continue
            timing_entry(
                name = entry_name(arm, cell),
                cells = ["@fsla_cells//:" + cell],
                executor = "measure.provider:timed",
                params = {"arm": arm},
            )
            names.append(entry_name(arm, cell))
    return names

def diff_sides():
    """One diff side a surface (`side_<surface>`), over the cells it runs on."""
    for surface, (executor, _kind, _tier, device) in SURFACES.items():
        diff_side(
            name = "side_" + surface,
            cells = ["@fsla_cells//:" + c for c in surface_cells(surface)],
            device = device,
            executor = executor,
        )
