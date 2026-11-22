# fsla-standalone — session rules

THE REPO IS ONE ATOMIC DESIGN. Its docs are the canonical current state, whole and self-contained:
`docs/CONSTITUTION.md` (Blake's standing rulings), `docs/KERNEL_STANDARDS.md` (the laws), `docs/ARCHITECTURE.md` and
`docs/architecture/` (the design) and `docs/internals/` (each source file's why and how it connects). A rule or a
design is real only once it is written here, and nothing here points outside the repo for one -- another project's
design may be cited only as the reason ours differs. Read the repo's doc before acting on anything remembered. Before
asserting a kernel fact, grep the source or contract doc in the current worktree; if unchecked say `unverified`; cite
a searchable string, never a line number. `docs/ESSENTIALS.md` maps what exists and where, by action.

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
- The fp64 oracle (`fsla/ops/naive.py`) is the executable spec — crown jewels;
  changes go through the dual-run protocol, no exceptions.
- Never run unscoped `pytest tests/`. Every GPU entry point takes
  `gpu_lock()` itself (`fsla_devtools.locks.gpu`) — invoke GPU commands BARE,
  never wrapped in an external `flock` on the lock path (the lock is the files beside it, so that serializes
  against nothing); a binary with no `gpu_lock()` runs under `python -m fsla_devtools.locks.gpu -- <command>`.
- A commit subject states in words what the commit makes true, never a work code.

