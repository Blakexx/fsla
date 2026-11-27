# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE STANDARD COUNTERS (KERNEL_STANDARDS §22, §29 item 14): the ONE list of the profiler's counters the lab reads for
every experiment kernel (`tools/lab.py run`), each metric once, and their reading by short name. Standard library only:
the lab's graph checks import it. Docs: docs/internals/tools/pipe_counters.md.
"""
from __future__ import annotations

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
