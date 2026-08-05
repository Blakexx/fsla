# `docs/internals/` — the source mirror

One markdown file here per source file under `csrc/rola/`, at the same relative
path (`csrc/rola/src/consumer_kernel.cuh` → `docs/internals/consumer_kernel.md`;
`csrc/rola/src/paging/vmm_owner.cu` → `docs/internals/paging/vmm_owner.md`).

## Why this directory exists

The kernel was written with every measured constraint recorded at the line it
constrained. That solved a real problem — a forgotten constraint is a rung
failure — and it inverted the files: `consumer_kernel.cuh` reached 61% comment
lines, a design journal with code interleaved. The 2026-08-04 mandate splits the
two artifacts. **Code carries structure; this tree carries reasons.**

What stays in the source:

- a file-header block: what the file is, what it owns, where its reasons live;
- a block before an important declaration: what it is and what it means;
- a **hazard stub** at a site where a mechanical edit silently changes results.

A hazard stub is one line, no prose, no numbers beyond its anchor id:

```cpp
//: HAZARD fp32-prefix -- docs/internals/intra_panel.md#fp32-prefix
```

Stubs are for **measured** hazards only — a place where moving, reordering or
retyping the code changes the numbers and the compiler will not say so. They are
never used for general design notes; those are here, in full.

## How to read a doc in this tree

Each file opens with what the source file is and the invariants it must keep,
then a numbered section per anchored site. Anchor ids are stable: they are cited
from the source and from `tests/unit/test_docs_mirror.py`, which fails if a
source file has no mirror or a stub cites an anchor no doc defines.

`DELETIONS.md` is the ledger of code removed from this tree — what, when, and
why it was safe.

## Where the other docs live

| Directory | Holds |
|---|---|
| `docs/internals/` | per-source-file reasons (this tree) |
| `docs/architecture/` | the permanent decision record: the choices that shaped the design and why they were made |
| `docs/` (top level) | operator-facing: build, bringup, testing, ratification, provenance |
