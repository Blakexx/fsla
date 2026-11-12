# `fold_stream`

Mirrors `measure/lab/experiments/fold_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A box as wide as the read side wants is written by every token, sparsely: 4 of its 256 leaves. Folding it list by
list needs each token's value copied into list order, which [`carry_floor`](../internals/tools/carry_floor.md)
prices as the form's largest term. Can the fold instead be fed by ONE PASS over the tokens -- no list, no rank, no
look back -- and what does a token then cost a CTA, the producer's work beside the fold's?

THE FORM. A CTA an SM: producer warps and one fold warpgroup that holds the box's state, 256 leaves × 64 channels,
as accumulators of WIDE leaves. Both operands are M|N-major with bricks 17 lines apart, so a token's value is one
4-byte store a lane on 32 banks, and so is a row of 64 coefficients.
- STEPS OF 64 LEAVES, four groups, A PRODUCER A GROUP: a producer looks at every token, and where a token's leaves
  fall in its group it loads the value (128 bytes, coalesced), stores it at its own running count, and stores the
  token's row of 64 coefficients beside it. The count is the producer's own register.
- STEPS OF ALL 256 LEAVES, one group, ONE PRODUCER OR FOUR: a token's slot is its index, so there is no count. A
  producer takes the tokens of its residue: it stores the value, clears the 4 words the slot's last token set in the
  row, and sets its own 4. The step's zeros do the placing.
- A PRODUCER'S LOADS RUN A BATCH AHEAD: it holds two batches of 8 tokens' values in registers, the next loaded
  before this one is stored, so no store waits on its own load.
- THE FOLD, a chunk of 64 tokens behind: the chunk's 4 steps a group in one commit group, with no branch. The
  producers hand a chunk over by an arrival each at a barrier, after a fence; the fold frees its tiles by another.
The lab's tokens put one leaf in each quarter of the box, so every group of 64 takes every token and every tile
fills exactly: arrivals that vary need a group's own back-pressure and steps on empty tiles, and NEITHER IS IN THIS
EXPERIMENT -- a group here holds two chunks of tiles and no more. Variants: the producers alone, the fold alone,
both (the state checked exactly, after 4,096 tokens). The values come from a table every CTA reads in step, of
65,536 tokens (8 MB) or 1,048,576 (128 MB): a line is fetched from memory once and read by the other CTAs from the
cache, so the larger table is a stream the cache cannot keep, not a cold load a CTA.

A FINDING OF THE COMPILER: a step under a branch on a loaded count compiles (ptxas C7520) to SERIALIZED wgmmas.

UNMEASURED IN THIS FORM. The record is the experiment's earlier form at 342fc132 -- 32 groups of 8 leaves, one
producer of about 110 instructions a token with its fill counts in shared memory (H100 SXM, vast.ai instance
60284495, the state exact; SM-cycles a token a CTA): the fold alone 9.1, 36 cycles a chunk's step; the producer
alone 550.9; together 557.8 (599.5 from the larger table). That form's sweep was one total: it does not separate a
narrow step's own time from what else a sweep of 32 steps holds.

Record: `measure/lab/results/fold_stream/sm_90a.json` (the experiment at 342fc132).
