---
name: rola-commit
description: Load before committing or pushing in this repo: the hooks a commit runs and how to answer each.
---
# Committing

1. The hooks run `python3`: commit with the worktree's environment first on `PATH`. Never `--no-verify`
   (`docs/KERNEL_STANDARDS.md#no-verify`).
2. A new or moved citation fails `docrefs` until it is read against its target: then
   `python3 tools/docrefs.py accept <doc>` and stage the lock it wrote (`docs/internals/tools/docrefs.md`). A skill
   follows what it cites in the same commit (`docs/KERNEL_STANDARDS.md#s31`).
3. The lints (`docs/internals/tools/lint.md`): fix a finding at its site; a ratchet's baseline only shrinks.
4. The subject says what the commit makes true; no work codes.
