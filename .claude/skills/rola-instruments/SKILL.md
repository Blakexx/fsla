---
name: rola-instruments
description: Load before attributing a shipped kernel's cost (ncu, region_ledger, phase_ledger, stall_census, pipe_counters, life_ranges, pipe_timeline, overlap_gate): which instrument answers which question, and how a reading is taken and read.
---
# Attributing a shipped kernel

The suite and how a reading is read: `docs/KERNEL_STANDARDS.md#s22`; the protocol: `docs/KERNEL_STANDARDS.md#s21`;
the map of the tools: `docs/internals/tools/README.md`.

1. Every instrument is a target of `declare.py`; take readings through it (`rola-bench`) so they are stored. A tool
   run by hand prints or writes `--json` and stores nothing.
2. Which instrument: cycles per phase per warp, `tools/phase_ledger.py` (`docs/internals/tools/phase_ledger.md`);
   instructions and stalls per component against the budget, `tools/region_ledger.py`
   (`docs/internals/tools/budgets.md#checking-it`); stall reasons by line, `tools/stall_census.py`
   (`docs/internals/tools/stall_census.md`); pipe and resource counters, `tools/pipe_counters.py`
   (`docs/internals/tools/pipe_counters.md`); live registers, `tools/life_ranges.py`
   (`docs/internals/tools/life_ranges.md`); the tensor pipe over time, `tools/pipe_timeline.py`
   (`docs/internals/tools/pipe_timeline.md`); time against the floors, `tools/overlap_gate.py`
   (`docs/internals/tools/overlap_gate.md`).
3. A reading obeys §21 (a quiet GPU, the baseline's own schedule, counts beside time) and the GPU lock
   (`docs/testing.md#running-it`).
4. Classify every ledger entry (a)/(b)/(c) (`docs/internals/tools/budgets.md#checking-it`) before any verdict.
