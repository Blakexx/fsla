"""THE FLOOR GATE (KERNEL_STANDARDS section 28, CONSTITUTION's performance targets; docs/internals/tools/floor.md): a kernel's work
declaration on its perf cell, priced by warpline's floor at the card's rates, the bound written by a build action and judged by
warpline's checker over the kernel's stamped run manifests. `kernel_floor` is the whole gate of one kernel on one cell: the
bound (`<name>_bound`, warpline's `warpline_floor_bound`) and the test (`<name>_floor`, warpline's `warpline_compare_test`).
`tools/coverage.json`'s `floor` obligation points at the test, or says why a kernel has none."""

load("@fsla_cells//:cells.bzl", "CELLS")
load("@warpline//:checker.bzl", "warpline_compare_test")
load("@warpline//:floor.bzl", "warpline_floor_bound")

#: THE LINE (KERNEL_STANDARDS section 28): a kernel's time at most 1.15x the largest of its floors
FACTOR = "1.15"

#: the run manifest's figure the bound is on: the grid's span, every CTA's launch at once (the work declared is the launch's)
FIGURE = "launch/grid_ns"

def cell_parameters(cell):
    """The parameters a work declaration's expressions read on a central cell: the cell's numeric fields (`tokens`, `B`, `H`,
    `dv`, ...), each as a number's text; a cell no central registry has fails."""
    if cell not in CELLS:
        fail("kernel_floor: no central cell %r (@fsla_cells//:cells.bzl)" % cell)
    return {k: str(v) for k, v in CELLS[cell].items() if type(v) in ("int", "float")}

def kernel_floor(name, work, cell, rates, runs, **kwargs):
    """One kernel's floor gate on one perf cell.

    Args:
      name: the stem: `<name>_bound` (the declarations, and the floor's text in its output group `floor`) and `<name>_floor`
        (the test).
      work: a label of the kernel's work declaration (`warpline-work/1`), its expressions over the cell's numeric fields.
      cell: the central cell's name (`@fsla_cells//:cells.bzl`'s `CELLS`): its numeric fields are the parameters.
      rates: a label of the card's rates table (`warpline-rates/1`, written by warpline's probes on the card).
      runs: labels of the kernel's stamped run manifests on that cell (`warpline run --stamps`), judged as one.
      **kwargs: passed to the test (`size`, `tags`).
    """
    warpline_floor_bound(
        name = name + "_bound",
        work = work,
        rates = rates,
        params = cell_parameters(cell),
        factor = FACTOR,
        figure = FIGURE,
        reason = "the perf cell %s" % cell,
    )
    warpline_compare_test(
        name = name + "_floor",
        manifests = runs,
        declare = ":" + name + "_bound",
        **kwargs
    )
