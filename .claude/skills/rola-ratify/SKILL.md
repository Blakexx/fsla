---
name: rola-ratify
description: Load before running tools/ratify.py: milestone-only ratification of the closed-world codegen, what precedes it, what it proves and refuses.
---
# Ratifying

1. Milestones only, after the last edit of any kind (`docs/KERNEL_STANDARDS.md#s16`); what it proves and what it
   refuses: `docs/ratification.md#what-the-gate-refuses`.
2. Before `--write`: the gate build with `ROLA_STRICT_MANIFEST=1` (`docs/build.md#building`, the table), nothing
   edited since. A re-ratification is a measurement, not an edit
   (`docs/ratification.md#re-ratification-is-a-measurement-not-an-edit`): a second `--write` on the same tree changes
   nothing.
3. Its flags and modes: `python tools/ratify.py --help` (`docs/internals/tools/ratify.md`).
