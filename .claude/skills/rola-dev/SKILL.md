---
name: rola-dev
description: Load before setting up or checking the dev environment, creating a worktree or venv, or when a tool cannot import rola_devtools: one command sets up, checks and describes a host or container.
---
# The dev environment

The law: `docs/KERNEL_STANDARDS.md#s23`. The command: `docs/internals/tools/dev.md`; the walkthrough:
`docs/setup.md`.

1. `python tools/dev.py check` first; every red names its fix. A tool that cannot import `rola_devtools` runs outside
   the checkout's venv (`docs/setup.md`, the checkout's venv): `bazel run //:dev_env` writes it.
2. A new worktree: `python tools/dev.py worktree <name> <sha>` -- its venv and commit gate (`bazel run //:dev_env`) and
   its extension (`bazel run //:library`) in one step; never a bare `git worktree add`.
3. A fresh machine or container: `python tools/dev.py init`, `clock --mhz` once, then `check`; the container is
   proven by `container check` (`docs/setup.md`). A change to the environment's inputs carries a passing container
   check before it commits (`docs/KERNEL_STANDARDS.md#s23`).
4. The settings are the dev config (`docs/internals/tools/dev_config.md`): `python tools/dev.py show|get <key>`.
