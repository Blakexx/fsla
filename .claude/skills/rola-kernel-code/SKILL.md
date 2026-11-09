---
name: rola-kernel-code
description: Load before writing or changing CUDA in this repo (csrc/ or measure/lab/): the order of work and the gates every kernel change passes. The rules are in the docs it cites; read them, never work from memory.
---
# Writing kernel code

0. The steady state first: one warp's window (or item) as a timeline -- each phase's stream, every load a stage
   ahead of its use, each phase's live registers -- before any component, and components split at those stages
   (`docs/KERNEL_STANDARDS.md#s32`). Read the concern's atom (`csrc/rola/src/common/atoms/`) for a primitive before writing one.
1. The design first: the component's page under `docs/internals/` (it mirrors `csrc/rola/src/`; a csrc file without
   its page, or a page naming an identifier the tree lacks, fails the commit:
   `docs/internals/tools/lint.md#the-gating-set`) and `docs/ARCHITECTURE.md`. A new component starts as a skeleton of
   stubs (`docs/KERNEL_STANDARDS.md#s19`).
2. Declare a new kernel in `tools/coverage.json` -- every obligation a pointer, an exemption or an owed line
   (`docs/internals/tools/coverage.md#declaration`); an undeclared kernel fails the commit.
3. The budget before the code (`docs/KERNEL_STANDARDS.md#s29`) and each shared buffer's access table before the code
   (`docs/KERNEL_STANDARDS.md#s30`): the skills `rola-budget` and `rola-shared-layout` arrive with those edits.
4. Write in the forms ptxas cannot undo: `docs/KERNEL_STANDARDS.md#s29` item 9, a step as one block
   (`docs/KERNEL_STANDARDS.md#s27`), the signatures the SASS gate reads (`docs/KERNEL_STANDARDS.md#s20`), a warp
   collective never under a short-circuit or a lane-dependent branch (`docs/KERNEL_STANDARDS.md#first-run`).
5. Name the phases: each phase opens with a `//:` heading, which the ledger attributes by
   (`docs/internals/tools/budgets.md#building-a-budget`). Comments are declaration blocks; prose goes to the mirror
   page (`docs/KERNEL_STANDARDS.md#comment-density`).
6. Read the SASS before anything runs: `bazel run //tools:lab -- sass <experiment>` in the lab,
   `bazel run //tools:sass_gate -- <.so>` on an extension build (it runs inside every build too); its step cover reads
   every MMA loop's pipe share with its warps together, and the target is the line -- a debt's floor is only its
   ratchet (`docs/KERNEL_STANDARDS.md#s32` item 4).
7. Run new device code first under `timeout -s KILL` (`docs/KERNEL_STANDARDS.md#first-run`).
8. Prove it: correctness with `rola-test`; cost with `rola-lab` (an experiment), `rola-instruments` (a shipped
   kernel's attribution) or `rola-bench` (the declared measurement). Commit with `rola-commit`.
