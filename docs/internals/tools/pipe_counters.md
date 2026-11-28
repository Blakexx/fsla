# `tools/pipe_counters.py` — the standard counters

KERNEL_STANDARDS §22 and §29 item 14: the profiler's counters the lab reads for every experiment kernel, and their
reading. The module imports nothing outside the standard library, because the lab's graph checks import it.

<a id="standard"></a>

## The standard counters

`pipe_counters.py` holds the ONE list of the profiler's counters, each metric once: `METRICS` (a short name for each of
the instructions by pipe, the shared-memory pipe's traffic by kind, L2 sectors, elapsed cycles and the profiler's own
busy shares of issue, the integer and the FMA pipes) and `STALLS` (the stall reasons, `stall_metric`); `metric_names()`
is the whole set, which `tools/lab.py run` asks `ncu` for on every experiment kernel (its first launch). `short` turns
`ncu`'s values by metric name into a launch's counters by short name, an absent one 0, with `stalls` by reason; a metric
the card or the profiler does not collect reads 0 and is named in the launch's `uncollected`, in the report and in the
record. `sms` is the launch's SM count (elapsed cycles summed over their average).

The lab prints what the profiler measured: each pipe's busy share of the elapsed cycles (issue, integer, FMA), L2
sectors read and written a unit, and the three largest stall reasons. The counters are priced against no rate here: a count
times a rate bounds perfect overlap and full use (the floor gate's floors, [`floor.md`](floor.md):
`docs/KERNEL_STANDARDS.md` §28 and §29 item 14), and what a composed kernel reaches is read from its timeline (`docs/KERNEL_STANDARDS.md#stepped`). The profiler's tensor-pipe ratio is never read as utilization (§22). A `wgmma` is
counted by a counter of its own (`sm__inst_executed_pipe_tensor_op_gmma`), which the `hmma` counters do not include.
