# `docs/internals/` — the source mirror

One markdown file here per source file under `csrc/rola/`, at the same relative
path (`csrc/rola/src/decode.cuh` → `docs/internals/decode.md`;
`csrc/rola/src/paging/vmm_owner.cu` → `docs/internals/paging/vmm_owner.md`).

The mirror rule runs SOURCE → DOC, so a deleted source leaves an orphan doc that
no test catches. Deleting the doc with the source is therefore part of deleting
the source, and the removal is recorded in `DELETIONS.md` like any other.

## Why this directory exists

Every measured constraint has to be recorded where a reader meets it before
editing — a forgotten constraint is a rung failure — and a kernel body is not
that place: reasons written at the line they constrain invert the file, and a
kernel body carrying them runs to 61% comment lines. The two artifacts are
therefore separate. **Code carries structure; this tree carries reasons.**

What stays in the source:

- a file-header block: what the file is, what it owns, where its reasons live;
- a block before an important declaration: what it is and what it means;
- a **hazard stub** at a site where a mechanical edit silently changes results.

A hazard stub is one line, no prose, no numbers beyond its anchor id:

```cpp
//: HAZARD update-then-read -- docs/internals/decode.md#update-then-read
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

**A bare commit hash cited anywhere in this tree resolves in the source
repository named by [`docs/provenance.md`](../provenance.md)
(`flash-linear-attention-rola`, branch `rola-v3`), not in this one, whose history
begins at the extraction.**

## Where the other docs live

| Directory | Holds |
|---|---|
| `docs/internals/` | per-source-file reasons (this tree) |
| `docs/architecture/` | the permanent decision record: the choices that shaped the design and why they were made |
| `docs/` (top level) | operator-facing: build, bringup, testing, ratification, provenance |
