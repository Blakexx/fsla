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
- MINIMAL COMMENTS: file-header/declaration blocks state constraints the
  code cannot show; everything granular lives in docs/ (docs/internals/
  mirrors csrc). A comment describing what code no longer does is a defect.
- DOCS ARE STATE: current truth written fresh; past states appear only as
  design rationale ("why it is shaped this way"), never narrative.
- The fp64 oracle (rola/reference) is the executable spec — crown jewels;
  changes go through the dual-run protocol, no exceptions.
- Never run unscoped `pytest tests/`. GPU commands under
  `flock /tmp/rola_gpu.lock` (rola-probe bare). Builds: MAX_JOBS=4, nice.
