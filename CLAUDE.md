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
- COMMENTS: code is self-documenting; the code IS the authoritative,
  most-current statement of HOW — if you want how, read the code. A comment
  may state only a constraint or invariant the code cannot express, in a
  line or three. Mechanism narration, derivations, design essays, "what
  this block does" — all of it lives in docs/ (docs/internals/ mirrors
  csrc), which explains WHY and how pieces CONNECT, never how a body works.
  A long docstring is the same defect as a long comment block. If you are
  about to write a paragraph above code, you are writing a docs/ section in
  the wrong file.
- DOCS ARE STATE: current truth written fresh; past states appear only as
  design rationale ("why it is shaped this way"), never narrative.
- The fp64 oracle (rola/reference) is the executable spec — crown jewels;
  changes go through the dual-run protocol, no exceptions.
- Never run unscoped `pytest tests/`. Every GPU entry point takes
  `gpu_lock()` itself (`rola_devtools.locks.gpu`, K46) — invoke GPU commands BARE,
  never wrapped in an external `flock` on the same lock path (self-deadlocks).
  Builds: MAX_JOBS=4, nice.

## Skills: procedures that ship with the repo (KERNEL_STANDARDS §31)

The repo's skills are in `.claude/skills/`; each is a procedure whose rules and facts are references into `docs/`,
held current by `tools/docrefs.py`. Start a session in this checkout or worktree, or add it with `/add-dir`, so its
own branch's skills load. A PreToolUse hook names the skill that covers an action the first time a session takes it
(`.claude/skill-routes.json`):

| action | skill |
|---|---|
| writing or changing CUDA (`csrc/`, `measure/lab/*.cu`) | `rola-kernel-code` |
| a lab experiment (`measure/lab/`, `tools/lab.py`) | `rola-lab` |
| an instruction budget, a ledger | `rola-budget` |
| a shared-memory layout, the bank model | `rola-shared-layout` |
| tests (`tests/`, `pytest`) | `rola-test` |
| measuring the shipped kernels | `rola-bench` |
| a build | `rola-build` |
| ratification | `rola-ratify` |
| a commit or a push | `rola-commit` |

