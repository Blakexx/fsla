#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PIPE COUNTERS (KERNEL_STANDARDS §22): the profiler's pipe and resource counters for one launch of a kernel
family on a cell as launch totals, raw.

    bazel run //tools:pipe_counters -- nl64k-alt-k4 --kernel FAMILY --json counters.json

One oneshot launch through the probe worker under `ncu --metrics` (the toolchain's `ncu`): HMMAs, wgmmas and instructions
executed, scheduler cycles elapsed and active, issue activity, warps active and eligible, shared loads and stores, ldsm
and global reductions and loads, shared-memory wavefronts and bank conflicts, LSU writeback, global reduction requests
and sectors, DRAM bytes, and the standard set `METRICS` and `STALLS` below, which the lab reads for every experiment
kernel. The JSON carries every counter's launch value and the launch's units (`tools/kernel_targets.py`), so a reader
divides a sum into a unit and derives utilization; nothing derived is stored. Stdlib only at import: the lab imports the
counter list from here, and the capture's own imports are made where it runs.
Docs: docs/internals/tools/pipe_counters.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: THE STANDARD COUNTERS the lab reads for every experiment kernel, by short name (ncu's metric names): instructions by
#: pipe, the shared-memory pipe's traffic by kind, L2 sectors, elapsed cycles and the measured busy shares
METRICS = {
    "hmma": "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "imma": "sm__inst_executed_pipe_tensor_op_imma.sum",
    "inst": "smsp__inst_executed.sum",
    "alu": "sm__inst_executed_pipe_alu.sum",
    "fma": "sm__inst_executed_pipe_fma.sum",
    "fmaheavy": "sm__inst_executed_pipe_fmaheavy.sum",
    "xu": "sm__inst_executed_pipe_xu.sum",
    "wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "load_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "stores": "sm__sass_inst_executed_op_shared_st.sum",
    "store_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "atomics": "sm__sass_inst_executed_op_shared_atom.sum",
    "atomic_wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
    "copy_bypass_bytes": "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass.sum",
    "copy_access_bytes": "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access.sum",
    "global_reads": "l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read.sum",
    "global_fill_bytes": "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
    "l2_read": "lts__t_sectors_srcunit_tex_op_read.sum",
    "red_requests": "lts__t_requests_srcunit_tex_op_red.sum",
    "l2_write": "lts__t_sectors_srcunit_tex_op_write.sum",
    "elapsed": "sm__cycles_elapsed.avg",
    "elapsed_sum": "sm__cycles_elapsed.sum",
    "busy_issue": "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
    "busy_alu": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "busy_fma": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "busy_fmaheavy": "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
}
STALLS = ("long_scoreboard", "short_scoreboard", "mio_throttle", "math_pipe_throttle", "barrier", "wait",
          "not_selected", "lg_throttle", "dispatch_stall", "branch_resolving", "no_instruction", "membar", "drain",
          "sleeping", "tex_throttle", "imc_miss", "misc")


def stall_metric(reason: str) -> str:
    return f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct"


def metric_names() -> list[str]:
    """Every ncu metric the set reads."""
    return [*METRICS.values(), *map(stall_metric, STALLS)]


def short(values: dict[str, float]) -> dict:
    """A launch's counters by short name (`METRICS`), absent ones 0, with `stalls` by reason, from ncu's values by
    metric name."""
    c: dict = {name: values.get(metric, 0.0) for name, metric in METRICS.items()}
    c["stalls"] = {r: values.get(stall_metric(r), 0.0) for r in STALLS}
    return c


def sms(c: dict) -> float:
    return c["elapsed_sum"] / c["elapsed"] if c["elapsed"] else 1.0


#: what a production launch reads beside the standard set
_OWN = (
    "gpu__time_duration.sum",
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "sm__inst_executed_pipe_tensor_op_gmma.sum", "smsp__inst_executed.sum",
    "sm__cycles_elapsed.avg", "smsp__cycles_active.avg",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.per_cycle_active", "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__sass_inst_executed_op_shared_ld.sum", "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_ldsm.sum", "smsp__sass_inst_executed_op_global_red.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "lts__t_sectors_op_red.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
)
COUNTERS = (*_OWN, *(m for m in metric_names() if m not in _OWN))


def measure(cell: str, family: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(ROOT / "measure"))
    import dev_config
    import kernel_targets
    from fsla_devtools import process

    from measure.cells import by_name
    from measure.provider import arm_name, oneshot_argv

    t = kernel_targets.target(family)
    spec = by_name(cell)
    cmd = [dev_config.ncu(), "--target-processes", "all", "-k",
           f"regex:{t.symbol}", "-c", "1", "--metrics", ",".join(COUNTERS), "--csv",
           *oneshot_argv(cell, arm_name(t.arm))]
    done = process.run(cmd, cwd=ROOT, timeout=3600)
    values = {}
    for line in done.stdout.splitlines():
        parts = [x.strip().strip('"') for x in line.split('","')]
        if len(parts) > 12 and parts[0].isdigit() and parts[-3] in COUNTERS:
            with contextlib.suppress(ValueError):
                values[parts[-3]] = float(parts[-1].replace(",", ""))
    if not values:
        raise SystemExit(f"pipe_counters: no counters read (ncu rc {done.returncode}):\n{(done.stdout + done.stderr)[-1500:]}")
    return {"cell": cell, "kernel": family, "unit": t.unit, "units": kernel_targets.units(t, spec), "launch": values,
            "missing": sorted(set(COUNTERS) - set(values))}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kernel_targets

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--kernel", required=True, choices=sorted(kernel_targets.TARGETS),
                    help="the kernel family launched (tools/kernel_targets.py)")
    a = ap.parse_args()
    doc = measure(a.cell, a.kernel)
    a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    print(f"{a.cell}: {len(doc['launch'])} counters, HMMAs {doc['launch'].get('sm__inst_executed_pipe_tensor_op_hmma.sum')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
