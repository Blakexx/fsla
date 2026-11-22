# `docs/internals/` — the source mirror

**Two mirror roots.** One markdown file here per source file under `csrc/fsla/`, at the same relative path minus the
`src/` component the mirror does not repeat (`csrc/fsla/src/decode/decode.cuh` → `docs/internals/decode/decode.md`;
`csrc/fsla/fsla_api.cpp` → `docs/internals/fsla_api.md`); a `.cu` and its `.cuh` share one doc, since they are one
component. And one page a Python module group the kernels' host side is built from: `engine/` mirrors `fsla/engine/`
(the package, the rule layer, the two DAGs), `state.md` mirrors `fsla/_state.py`, `paging/paging.md`
mirrors `fsla/ops/paging.py` and `padding.md` mirrors `fsla/ops/padding.py`. `tools/` mirrors `tools/`, one page a
script with something to say ([the tools map](tools/README.md)); `hardware/` mirrors `hardware/` (the contract registry: [its page](hardware/registry.md)).

**No exemptions.** The per-arm units are the graph's, generated into its output tree from each family's `arms.bzl`
([`docs/build-graph.md`](../build-graph.md#arms)); no source in `csrc/fsla/` is generated, so every source file has its page.

The mirror rule runs SOURCE → DOC, so a deleted source leaves an orphan doc unless the doc is deleted with it: deleting
the doc is part of deleting the source, and the removal is recorded in `DELETIONS.md` like any other.

## What stays in the source

Code carries structure; this tree carries reasons (the comment rule:
[`KERNEL_STANDARDS.md#comment-density`](../KERNEL_STANDARDS.md#comment-density)). What stays in the source is a
file-header block, a block before an important declaration, and a **hazard stub** at a site where a mechanical edit
silently changes results.

A hazard stub is one line, no prose, no numbers beyond its anchor id:

```cpp
//: HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read
```

Stubs are for **measured** hazards only — a place where moving, reordering or
retyping the code changes the numbers and the compiler will not say so. They are
never used for general design notes; those are here, in full.

## How to read a doc in this tree

Each file opens with what the source file is and the invariants it must keep,
then a numbered section per anchored site. Anchor ids are stable: they are cited
from the source and from `tools/lint/lint_standards.py`, which fails if a
source file has no mirror or a stub cites an anchor no doc defines.

`DELETIONS.md` is the ledger of code removed from this tree — what, when, and
why it was safe.

## Naming something that is not there

A page may only name identifiers its source still carries. `lint_standards.py` reads
every name a page writes in code font and fails on one that exists nowhere in the tree:
a mirror page can otherwise outlive the mechanism it documents, and every earlier check
was about the page's SHAPE — that it exists, sits at the right path and resolves its
anchors — rather than about whether its subject was still there.

Naming what is GONE is often the fact worth keeping: an entry that was unbound, a
mechanism that retired, another project's file the convention was borrowed from. Declare
those names on one line beside the prose that explains them,

    <!-- ABSENT: fsla_consumer_forward fsla_page_admit_mask -->

and the check reads the declaration in BOTH directions — a declared-absent name the tree
actually carries fails too, because the passage is then telling a reader the opposite of
the truth.

**A commit hash cited in this tree resolves in this repository, unless it predates the
extraction (2026-08-01): those name commits of the retired pre-extraction fork and
resolve in no published repository ([`docs/provenance.md`](../provenance.md)).**

The map of every other document is [`docs/README.md`](../README.md).
