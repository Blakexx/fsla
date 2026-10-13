# rola-standalone — session rules

The operating contract for all agent work in this repo lives in the coordinator's
development queue (its own `CLAUDE.md`, outside this repo) — read it first; every
discipline block there is binding here.

The rules most often violated by newcomers, restated:

- CODE IS STATE. The repo is one atomic self-consistent state. A deletion
  deletes: no tombstone comments ("X was removed", "used to..."), no
  commented-out code, no names or comments referencing removed mechanisms.
  Reasoning for removals lives in docs/internals/DELETIONS.md and git —
  never at the site.
- COMMENTS: code is self-documenting and authoritative for HOW. A comment states only a
  constraint or invariant the code cannot express, in a line or three; reasons and
  connections live in docs/ (docs/internals/ mirrors csrc). The rule in full:
  docs/KERNEL_STANDARDS.md#comment-density.
- DOCS ARE STATE: current truth written fresh; past states appear only as
  design rationale ("why it is shaped this way"), never narrative. The map of
  the docs: docs/README.md.
- The fp64 oracle (`rola/ops/naive.py`) is the executable spec — crown jewels;
  changes go through the dual-run protocol, no exceptions.
- Never run unscoped `pytest tests/`. Every GPU entry point takes
  `gpu_lock()` itself (`rola_devtools.locks.gpu`) — invoke GPU commands BARE,
  never wrapped in an external `flock` on the same lock path (self-deadlocks).

## Skills: procedures that ship with the repo (KERNEL_STANDARDS §31)

The repo's skills are in `.claude/skills/`; each is a procedure whose rules and facts are references into `docs/`,
held current by `tools/docrefs.py`. A PreToolUse router puts the skill that covers an action -- from the checkout the
action touches, so every worktree uses its own branch's skills -- into the acting agent's context the first time it
takes the action (`.claude/skill-routes.json`). A session started here is routed by this repo's hook; sessions started
anywhere else by the user-level bootstrap that `python tools/dev.py init` installs (`check` verifies it is current).

| action | skill |
|---|---|
| writing or changing CUDA (`csrc/`, `measure/lab/*.cu`) | `rola-kernel-code` |
| a lab experiment (`measure/lab/`, `tools/lab.py`) | `rola-lab` |
| an instruction budget | `rola-budget` |
| attributing a shipped kernel's cost (ncu, the ledgers) | `rola-instruments` |
| a shared-memory layout, the bank model | `rola-shared-layout` |
| tests (`tests/`, `pytest`) | `rola-test` |
| measuring the shipped kernels | `rola-bench` |
| a build | `rola-build` |
| ratification | `rola-ratify` |
| the dev environment, a worktree | `rola-dev` |
| docs, skills, routes, this file | `rola-docs` |
| a commit or a push | `rola-commit` |

