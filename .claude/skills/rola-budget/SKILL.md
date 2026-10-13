---
name: rola-budget
description: Load when stating, checking or correcting an instruction budget (a lab's Bench::budget, tools/budgets/*.json): build it from prices and trips, find a miss by phase, correct the price.
---
# Budgets

The method and its data: `docs/internals/tools/budgets.md`. The law: `docs/KERNEL_STANDARDS.md#s29` (items 11-13).

1. The phases are the kernel's `//:` headings (`docs/internals/tools/budgets.md#building-a-budget`).
2. Trips per cell come from the experiment's mirror (its `.inputs.py`); a lane-divergent loop is priced at the warp's
   longest lane (`docs/KERNEL_STANDARDS.md#s29` item 12).
3. Prices come from `docs/internals/tools/budgets.md#prices`. A primitive without one: compile it alone, read
   `python tools/lab.py sass <experiment>`, and add its row.
4. State it before the code: `Bench::budget`, with `Bench::units` for a kernel without MMAs
   (`docs/internals/tools/lab.md#the-kit-measurelablabcuh`); a production kernel's in `tools/budgets/<kernel>.json`,
   read by `tools/region_ledger.py --budget` (`docs/internals/tools/budgets.md`).
5. Check it: `python tools/lab.py run <experiment>` refuses a kernel over its budget, and one whose loops hold
   convergence instructions it did not declare (`LOOP_CONVERGENCE`).
6. Find a miss: `python tools/lab.py ledger <experiment> --kernel <regex>` gives instructions by phase; classify each
   (`docs/internals/tools/budgets.md#checking-it`), then fix the code to its floor or correct the price.
7. Restate a budget only with the corrected price, in the same commit, and add the lab's row to
   `docs/internals/tools/budgets.md#ledger`.
