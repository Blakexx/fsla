# `facts_pipeline`

Mirrors `measure/lab/experiments/facts_pipeline.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The two-stage facts pass end to end in a warp: a warp streams an item (the vote; on the write list level also the
lists' exponents and the packed stash), keeps its bits in registers, issues its next item's first block, then builds
its four 32-digit groups' lists from them (the walk build: each digit's entries dealt evenly, a walk totalling each
(digit, pass) by residue, a segmented scan of ranks, a walk placing each entry; a write entry's byte from the stash).
Seven warps an SM (a 2 x 4 KB ring and ~4.8 KB of build scratch each). Bytes a chip cycle:

| k of 256 | voted only | stage 1 alone | two stages |
|---|---|---|---|
| 4 | 430.1 | 352.4 | 182.1 |
| 16 | 429.9 | 335.3 | 148.0 |

The build in the stream's warps serializes them: a warp builds instead of streaming, and seven warps an SM cannot keep
the stream's bytes in flight around it; the build's re-read of its bits from global (instead of registers held across
the item) also misses L2 at this footprint. The form the facts pass takes instead: `facts_block`. Record:
`measure/lab/results/facts_pipeline/sm_86.json`.
