---
name: rola-budget
description: Load when stating, checking or correcting an instruction budget (a lab's Bench::budget, tools/budgets/*.json): build it from prices and trips, find a miss by phase, correct the price.
---
# Budgets

The method and its data: `docs/internals/tools/budgets.md`. The law: `docs/KERNEL_STANDARDS.md#s29` (items 11-15).

1. The phases are the kernel's `//:` headings and its components the definitions the region ledger finds, a class
   over a plan's members by their own names (`docs/internals/tools/budgets.md#building-a-budget`).
2. Trips per cell come from the experiment's mirror (its `.inputs.py`); a lane-divergent loop is priced at the warp's
   longest lane (`docs/KERNEL_STANDARDS.md#s29` item 12).
3. Prices come from `docs/internals/tools/budgets.md#prices`. A primitive without one: compile it alone, read
   `bazel run //tools:lab -- sass <experiment>`, and add its row. Each line names its form from
   `docs/internals/tools/budgets.md#forms` or why it departs; a value re-derived to save a register is its own line
   (`docs/KERNEL_STANDARDS.md#s32` item 3).
4. State it before the code: `Bench::budget`, with `Bench::units` for a kernel without MMAs
   (`docs/internals/tools/lab.md#the-kit-measurelablabcuh`); a production kernel's in `tools/budgets/<kernel>.json`,
   read by `tools/region_ledger.py --budget` (`docs/internals/tools/budgets.md`).
5. Check it: `bazel run //tools:lab -- run <experiment>` refuses a kernel over its budget, and one whose loops hold
   convergence instructions it did not declare (`LOOP_CONVERGENCE`). It prints every pipe's cycles per card, the
   binding one named (`docs/internals/tools/pipes.md`): state the binding pipe's cycles beside the instructions, and a
   kernel over 1.3x its binding pipe names the stall that holds it (`docs/KERNEL_STANDARDS.md#s29` item 14).
6. Find a miss: `bazel run //tools:lab -- ledger <experiment> --kernel <regex>` gives instructions by phase;
   classify each (`docs/internals/tools/budgets.md#checking-it`), then fix the code to its floor or correct the price.
7. Restate a budget only with the corrected price, in the same commit, and add the lab's row to
   `docs/internals/tools/budgets.md#ledger`.
8. A production kernel's budget also states each component's live registers and each phase's predicted cycles
   (`docs/internals/tools/budgets.md#building-a-budget` item 5, `docs/KERNEL_STANDARDS.md#s32`); check them with
   `bazel run //tools:life_ranges -- --arm 0 --kernel <regex> --source <kernel file> --source <component files>...
   --budget tools/budgets/<kernel>.json` (`docs/internals/tools/life_ranges.md#budget`). A phase's cycles are read
   by its family's phase ledger, which lands with the family's first schedule; a phase over 1.3x its prediction stops
   work until the wrong price or model is named.
