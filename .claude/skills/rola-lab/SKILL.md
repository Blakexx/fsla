---
name: rola-lab
description: Load when creating, changing or running a lab experiment (measure/lab/experiments/): its question, inputs and mirror, checks, budgets, record, doc section and card block.
---
# Lab experiments

The method: `docs/internals/tools/lab.md#the-loop`. The law: `docs/KERNEL_STANDARDS.md#s26`.

1. One question per experiment, stated in its header.
2. Data comes from `<name>.inputs.py`'s `write(directory)`, with a mirror of the kernel in Python that defines its
   exact outputs and counts the trips its budget uses.
3. Every variant checks its outputs (`Bench::check`: exact against the mirror, or invariants where the kernel races)
   or says why not (`Bench::unchecked`): `docs/KERNEL_STANDARDS.md#s29` item 8.
4. Every kernel is budgeted before its code (load `rola-budget`), its SASS shape declared with `Bench::expect`.
5. Read the SASS (`python tools/lab.py sass <name>`); the first run is under a kill timeout
   (`docs/KERNEL_STANDARDS.md#first-run`); then `python tools/lab.py run <name>`, and over budget
   `python tools/lab.py ledger <name>`.
6. Record with `python tools/lab.py run <name> --record`.
7. Write its section in `docs/internals/tools/lab.md` (the question, the form, the measured table, what it decides) and
   the card's MEASURED block (the user skill `rola-card`); commit with `rola-commit`.
