---
name: rola-commit
description: Load before committing or pushing in this repo: the gate a commit runs, how it is run by hand, and how to answer each hook.
---
# Committing

1. The gate is the tracked hook `tools/git-hooks/pre-commit`: `bazel test //:gate` on the commit as git will write it
   (`docs/build-graph.md#checks`); a refusal naming no `bazel` means the checkout is unwired:
   `bazel run //:dev_env` (`docs/internals/tools/dev.md`). Never `--no-verify` (`docs/KERNEL_STANDARDS.md#no-verify`).
2. Run it before committing: `bazel test //:gate` (`docs/internals/tools/lint.md#running-the-roster-by-hand`).
3. `docrefs`: a new or moved citation fails until it is read against its target; then
   `python3 tools/docrefs.py accept <doc>` and stage the lock(s) it wrote
   (`docs/internals/tools/docrefs.md#the-gate-and-the-pass`). A skill's references are held strictly: it changes in
   the commit that changes what it cites (`docs/KERNEL_STANDARDS.md#s31`).
4. A lint finding is fixed at its site; a ratchet's baseline only shrinks (`--shrink`, same commit)
   (`docs/internals/tools/lint.md#ratchets`). A changed dev environment needs the container check (`rola-dev`).
5. The subject states what the commit makes true, in words, never a work code.
6. After the commit, its review lands detached (`tools/git-hooks/post-commit`; `python tools/review_gate.py --show 1`,
   `docs/internals/tools/review_gate.md`): read it before the next commit, and fix a BLOCK finding or say in words why
   it is wrong.
