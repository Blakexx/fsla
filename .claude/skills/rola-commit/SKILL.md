---
name: rola-commit
description: Load before committing or pushing in this repo: the gate a commit runs, how it is run by hand, and how to answer each hook.
---
# Committing

1. The gate is the tracked hook `tools/git-hooks/pre-commit`: `bazel test //:gate` on the commit as git will write it
   (`docs/build-graph.md#checks`); its refusal names the fix: no `bazel` on PATH, `python3 tools/bootstrap.py`; the
   checkout's gate not wired, `bazel run //:dev_env` (`docs/internals/tools/dev.md`). Never `--no-verify` (`docs/KERNEL_STANDARDS.md#no-verify`).
2. Run it before committing: `bazel test //:gate` (`docs/internals/tools/lint.md#running-the-roster-by-hand`).
3. `docrefs`: a new or moved citation fails until it is read against its target; then
   `bazel run //:docrefs -- accept <doc>` and stage the lock(s) it wrote
   (`docs/internals/tools/docrefs.md#the-gate-and-the-pass`). A skill's references are held strictly: it changes in
   the commit that changes what it cites (`docs/KERNEL_STANDARDS.md#s31`).
4. A lint finding is fixed at its site; a ratchet's baseline only shrinks (`--shrink`, same commit)
   (`docs/internals/tools/lint.md#ratchets`). A change to the devbox image needs the container check (`rola-dev`).
5. The subject states what the commit makes true, in words, never a work code.
6. After the commit, its review lands detached (`tools/git-hooks/post-commit`;
   `bazel run //tools:review_gate -- --show 1`): never wait on it. Read it when it lands, and fix its findings in the
   next commit after it (`docs/internals/tools/review_gate.md#pipelined`); a BLOCK is answered in a commit's message
   by its trailer, `Review-Fix: <sha>:<n> <words>` or `Review-Override: <sha>:<n> <words>`, without which
   `tools/git-hooks/commit-msg` refuses the commit (`docs/internals/tools/review_gate.md#blocking`).
7. A push to the default branch publishes (CI mirrors it): before it, every review of its commits has landed and been
   read (`docs/internals/tools/review_gate.md#pipelined`). A push of any other branch to the private remote waits on
   nothing.
