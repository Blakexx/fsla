# `tools/pipe_counters.py` — the profiler's pipe and resource counters for one launch

KERNEL_STANDARDS §22. One untimed launch of a cell of a kernel family (`--kernel`, required, one family of `tools/kernel_targets.py`'s table, which is empty on this line) through fsla's runner (`measure/provider.py::oneshot_argv`, the arm a comparison times), under
`ncu --metrics` with the counters in `COUNTERS`:
- HMMAs and instructions executed;
- scheduler cycles, elapsed and active;
- issue activity, warps active and warps eligible;
- shared loads and stores, ldsm, global reductions and global loads;
- shared-memory wavefronts and bank conflicts on loads and stores;
- LSU writeback activity, global reduction requests and sectors;
- DRAM bytes read and written.
- the standard set (`METRICS` and `STALLS`, below), the one the lab reads for every experiment kernel.

    bazel run //tools:pipe_counters -- flagship-dense --json counters.json

The JSON holds every counter's LAUNCH value (`launch`), the counters the profiler did not report (`missing`), and the
launch's CTA-windows. Nothing derived is stored. A reader divides a sum by `cta_windows` for a unit, and computes
tensor utilization as HMMA count × the calibrated cycles an HMMA holds the pipe, over SMs × schedulers × active cycles
(§22: never the profiler's pipe-active ratio). The build ledger, the composer and the suite read this JSON; before
2026-09-12 each carried its own copy of the capture.

<a id="standard"></a>

## The standard counters

`pipe_counters.py` holds the ONE list of the profiler's counters, each metric once: `METRICS` (a short name for each of
the instructions by pipe, the shared-memory pipe's traffic by kind, L2 sectors, elapsed cycles and the profiler's own
busy shares of issue, the integer and the FMA pipes), `STALLS` (the stall reasons, `stall_metric`), and `_OWN` (what a
production launch reads beside them); `COUNTERS` is the whole list a capture asks `ncu` for, `metric_names()` the
standard set alone, which `tools/lab.py run` asks for on every experiment kernel (its first launch). `short` turns
`ncu`'s values by metric name into a launch's counters by short name, an absent one 0, with `stalls` by reason; a metric
the card or the profiler does not collect reads 0 and is named in the launch's `uncollected`, in the report and in the
record. `sms` is the launch's SM count (elapsed cycles summed over their average). The module imports nothing outside
the standard library, because the lab's graph checks import it; the capture's own imports are made when it runs.

The lab prints what the profiler measured: each pipe's busy share of the elapsed cycles (issue, integer, FMA), L2
sectors read and written a unit, and the three largest stall reasons. The counters are priced against no rate here: a count
times a rate bounds perfect overlap and full use (the overlap gate's floors, `docs/KERNEL_STANDARDS.md` §28 and §29
item 14), and what a composed kernel reaches is read from its timeline (`docs/KERNEL_STANDARDS.md#stepped`). The profiler's tensor-pipe ratio is never read as utilization (§22). A `wgmma` is
counted by a counter of its own (`sm__inst_executed_pipe_tensor_op_gmma`), which the `hmma` counters do not include.
