---
name: rola-docs
description: Load before writing or changing anything under docs/, a skill, a route or CLAUDE.md: docs as current truth, the mirror contract, the deletion ledger, the doc references, and what a skill may contain.
---
# Docs, skills and routes

1. Docs are current truth: no history narrative; a removal gets a row in `docs/internals/DELETIONS.md` and nothing at
   the site (`CLAUDE.md`).
2. `docs/internals/` mirrors `csrc/rola/src/` page for page and names only identifiers the tree carries
   (`docs/internals/tools/lint.md#the-gating-set`); a page explains why and how pieces connect, never how a body
   works (`docs/KERNEL_STANDARDS.md#comment-density`).
3. Every citation is pinned: after writing, `bazel run //:docrefs -- pending` for what moved, `accept <doc>` for what
   was read (`docs/internals/tools/docrefs.md#the-gate-and-the-pass`).
4. A skill is procedure over references, never a copy of a rule (`docs/KERNEL_STANDARDS.md#s31`); it is routed by an
   entry in `.claude/skill-routes.json` (its `about` field states the matching convention) and a row in `CLAUDE.md`'s
   table; it names no machine path (`docs/internals/tools/lint.md#portable-agent-files`).
