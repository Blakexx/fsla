---
name: rola-lab
description: Load when creating, changing or running a lab experiment (measure/lab/experiments/): its question, inputs and mirror, checks, budgets, record, doc section and card block.
---
# Lab experiments

The method: `docs/internals/tools/lab.md#the-loop`. The law: `docs/KERNEL_STANDARDS.md#s26`.

1. `bazel run //tools:lab -- list` first: the question may be asked already. One question per experiment, stated in its
   header, with its `TIMELINE:` written before the kernel (`docs/internals/tools/lab.md#hazards`).
2. Data comes from `<name>.inputs.py`'s `write(directory)`, with a mirror of the kernel in Python that defines its
   exact outputs and counts the trips its budget uses.
3. Every variant checks its outputs (`Bench::check`: exact against the mirror, or invariants where the kernel races)
   or says why not (`Bench::unchecked`): `docs/KERNEL_STANDARDS.md#s29` item 8; an MMA kernel without a budget states
   its exemption (`Bench::exempt`).
4. Every kernel is budgeted before its code (load `rola-budget`), its SASS shape declared with `Bench::expect`.
5. Read the SASS (`bazel run //tools:lab -- sass <name>`); the first run is under a kill timeout
   (`docs/KERNEL_STANDARDS.md#first-run`); then `bazel run //tools:lab -- run <name>`, and over budget
   `bazel run //tools:lab -- ledger <name>`; a traffic question is `bazel run //tools:lab -- ncu <name> --metric M`
   (`docs/internals/tools/lab.md#the-loop`). Read its pipes report: the binding pipe per card, and on this card how
   far the kernel runs over it and which stalls hold it (`docs/internals/tools/pipes.md`).
6. Record with `bazel run //tools:lab -- run <name> --record`. For a card elsewhere (a rented box has no toolkit):
   `pack` it here, run it there with `carried --no-counters HOST`, naming the host that grants no counters
   (`docs/internals/tools/lab.md#pack`).
7. Write its page `docs/lab/<name>.md` (the question, the form, the measured table, what it decides), its row in
   `docs/lab/README.md` and, where a design card tracks it, its MEASURED block (the site's own skill); commit with
   `rola-commit`.
