---
name: rola-dev
description: Load before setting up or checking the dev environment, creating a worktree or venv, or when a tool cannot import rola_devtools: one command sets up, checks and describes a host or container.
---
# The dev environment

The law: `docs/KERNEL_STANDARDS.md#s23`. The command: `docs/internals/tools/dev.md`; the walkthrough:
`docs/setup.md`.

1. `bazel run //:check` first; every red names its fix. A tool that cannot import `rola_devtools` runs outside
   the checkout's venv (`docs/setup.md`, the checkout's venv): `bazel run //:dev_env` writes it.
2. A new worktree: `bazel run //:worktree -- <name> <sha>` -- its venv and commit gate (`bazel run //:dev_env`) and
   its extension (`bazel run //:library`) in one step; never a bare `git worktree add`.
3. A fresh machine: `python3 tools/bootstrap.py` (the pinned bazelisk), `bazel run //:init`, `bazel run //:clock --
   --mhz N` once, then `bazel run //:check`. The image runs its own init when it is built; the container is
   proven by `container check` (`docs/setup.md`). A change to the environment's inputs carries a passing container
   check before it commits (`docs/KERNEL_STANDARDS.md#s23`).
4. The settings are the dev config (`docs/internals/tools/dev_config.md`): `bazel run //:show`, `bazel run //:get --
   <key>`.
