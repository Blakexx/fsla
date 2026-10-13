---
name: rola-kernel-code
description: Load before writing or changing CUDA in this repo (csrc/ or measure/lab/): the order of work and the gates every kernel change passes. The rules are in the docs it cites; read them, never work from memory.
---
# Writing kernel code

1. The design first: the component's page under `docs/internals/` (it mirrors `csrc/rola/src/`) and
   `docs/ARCHITECTURE.md`. A new component starts as a skeleton of stubs (`docs/KERNEL_STANDARDS.md#s19`).
2. The budget before the code: load `rola-budget` (`docs/KERNEL_STANDARDS.md#s29`).
3. Each shared buffer's access table before the code: load `rola-shared-layout` (`docs/KERNEL_STANDARDS.md#s30`).
4. Write in the forms ptxas cannot undo: `docs/KERNEL_STANDARDS.md#s29` item 9, a step as one block
   (`docs/KERNEL_STANDARDS.md#s27`), the signatures the SASS gate reads (`docs/KERNEL_STANDARDS.md#s20`), and a warp
   collective never under a short-circuit or a lane-dependent branch (`docs/KERNEL_STANDARDS.md#first-run`).
5. Name the phases: each phase of a kernel body opens with a `//:` heading, which the ledger attributes by
   (`docs/internals/tools/budgets.md#checking-it`). Comments are declaration blocks; prose goes to `docs/internals/`
   (`docs/KERNEL_STANDARDS.md#comment-density`).
6. Read the SASS before anything runs: `python tools/lab.py sass <experiment>` in the lab, `python tools/sass_gate.py`
   on an extension build.
7. Run new device code first under a kill timeout (`docs/KERNEL_STANDARDS.md#first-run`).
8. Prove it: correctness with `rola-test`; cost with `rola-lab` (an experiment) or `rola-bench` (the shipped kernel).
   Commit with `rola-commit`.
