# `tools/docrefs.py` — the doc references

Mirrors `tools/docrefs.py`. Every citation a doc makes of the tree is pinned to the content of what it cites, so a
change to that content marks the citation for rereading instead of leaving it to rot. A commit that changes code or a
doc another doc cites re-reads those citations and re-pins them in the same commit (Blake, 2026-11-22): the gate fails
on a reference that is pending, unread or broken, so a change cannot land with a sentence left standing over a target it
no longer describes.

## What a reference is

`references` reads every tracked markdown file except the deletion ledger (a ledger row describes the tree at its own
commit), outside fenced code blocks, and takes five forms:

- a possessive, a path in code font followed by `'s` and a name in code font: the name's primitive in that file;
- a dotted name in code font whose first component is one of `PACKAGES`: the longest prefix that is a module in the
  tree, and the rest as the primitive in it;
- a path in code font with a known suffix or a trailing `/`: the file or directory;
- a path in code font that starts `data:` (`data:lab/<experiment>/<arch>.json`): that file or directory of THE PINNED
  DATA, the data repository at the one commit `MODULE.bazel` pins (`@fsla_data`), which the check is handed by its
  BUILD file (`--data`, `@fsla_data//:root`) and resolves the path in; checked only to exist there, since the pin fixes
  its content ([`measurement.md`](../../measurement.md#the-data));
- a markdown link: the file, or the section its `#anchor` names.

`move OLD NEW` is a target that moved -- a file, a directory or every path under a prefix, into the tree or into the
pinned data: each pin of a target under `OLD` is re-pinned under `NEW`, read as it was read, and each doc's text is
rewritten from `OLD` to `NEW`, refused with nothing written unless every moved target resolves at its new path to the
digest its pin holds (the lab's records moved to `data:lab` this way, every pin's content the same).

`locate` finds a target in a `Version` of the tree (the working tree, or a commit). A Python primitive is a function,
class, method or constant, found through `ast` and followed through a `from x import name` re-export; it is hashed as
`ast.unparse` of the node with docstrings dropped, so a comment, a docstring or a reformat moves nothing. A device or
host C++ primitive is a `//: @ref NAME` ... `//: @end` region, else the first brace-matched definition of the name,
hashed with comments dropped and whitespace collapsed; a name the file carries without either is checked only to
exist. A markdown section runs from its heading (or `<a id>`) to the next heading at its level or above. A file or a
directory is checked only to exist, and a build output the tree ignores (`_generated`) counts as existing.

## The locks

`docs/references.lock.json` holds one pin per (doc, target): the digest, the commit it was pinned at, and whether a
docs pass has read it. A doc declared `private` in `.github/mirror/declarations.json` keeps its pins in `.claude/references.lock.json`
(`_visibility` reads the declarations), so the public lock never names it. `init` pinned the tree's
references unread, the broken ones with a null digest.

Each lock records the Python version its digests were taken under (`PYTHON`): `ast.unparse`'s output moves between
versions, so a lock read under another one would show drift where nothing moved, and the tool refuses to give a verdict
there. It runs from the checkout's venv, as the commit gate does.

## The gate and the re-read

`verdict` compares the survey to the locks. The gate (`bazel test //:docrefs`, run by the commit gate and a CI step)
fails on any reference that is not clean:

- NEW: a citation no pin holds, or one that resolves to nothing (a reference that resolves to nothing cannot be accepted);
- PENDING: a pinned reference whose target's digest moved or vanished since its pin;
- UNREAD: a pin no docs pass has read (`init` pinned the tree's references unread);
- BROKEN: a reference that resolved to nothing when it was pinned and still does;
- STALE: a pin whose reference no doc makes any more (`prune`).

Its message names each citing `doc:line` and target, and the one command that re-pins after the reading. The reading is
the work: for each pending reference `pending` prints the target's own source as a diff from its pin (`change`), the
citing sentence is read against it and fixed where it no longer holds, and only then is the doc accepted. Accepting a doc
without reading each of its pending references is the failure the gate exists to prevent.

`accept DOC` pins the doc's new and changed references (a pending one is a changed one), read; `accept --whole DOC` also
marks every other reference in it read, for a pass that read the whole page. When the target of a pinned reference is
gone, `accept` drops its pin: a bare name on a mirror page then stops being a citation, and a reference of any other form
shows again as one that resolves to nothing, which stays red until the sentence names something that exists. `accept`
writes `docs/references.lock.json` and/or `.claude/references.lock.json`: stage them.

`tests/unit/test_docrefs.py` holds the gate to a fixture tree: an unread reference is red until its doc is accepted whole,
a changed target is red naming the citing `doc:line` and the accept command and green once accepted, a comment in the
target moves nothing, and a vanished target is red until its sentence is fixed.
