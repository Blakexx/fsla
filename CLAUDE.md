# rola-standalone — session rules

The operating contract for all agent work in this repo lives at
`~/rola-scratch/development/queue/CLAUDE.md` — read it first; every
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
- Never run unscoped `pytest tests/`. GPU commands under
  `flock /tmp/rola_gpu.lock`. Builds: MAX_JOBS=4, nice.

## Kernel standards (read before writing or reviewing any kernel code)
`docs/KERNEL_STANDARDS.md` — craft rules grounded in FlashAttention/CUTLASS/ThunderKittens (fragment-grain loops, copy atoms and swizzles, pipelines, unroll and instantiation hygiene, compile budgets) plus the RoLA-specific patterns. Design authority stays with the queue card; the standards govern HOW a designed piece is realized. Every brief cites it; every review applies it line by line.
