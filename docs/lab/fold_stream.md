# `fold_stream`

Mirrors `measure/lab/experiments/fold_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A box as wide as the read side wants is written by every token, sparsely: 4 of its 256 leaves. Folding it list by
list needs each token's value copied into list order. Can the fold instead be fed by ONE PASS over the tokens -- no
list, no rank, no look back -- and what does a token then cost a CTA, by the step's width and the producers' count,
the producers' work beside the fold's? [`carry_floor`](../internals/tools/carry_floor.md) prices the fold's tensor
time from this experiment.

THE FORM. A CTA an SM: producer warps and one fold warpgroup that holds the box's state, 256 leaves × 64 channels,
as accumulators of WIDE leaves. Both operands are M|N-major with bricks 17 lines apart, so a token's value is one
4-byte store a lane on 32 banks, and so is a row of 64 coefficients.
- STEPS OF 64 LEAVES, four groups, A PRODUCER A GROUP: a producer looks at every token, and where a token's leaves
  fall in its group it stores the value at its own running count and the token's row of 64 coefficients beside it; it
  loads every token's value (128 bytes, coalesced), a batch ahead of the test. The count is the producer's own register.
- STEPS OF ALL 256 LEAVES, one group, ONE PRODUCER OR FOUR: a token's slot is its index, so there is no count. A
  producer takes the tokens of its residue: it stores the value, clears the 4 words the slot's last token set in the
  row, and sets its own 4. The step's zeros do the placing.
- A PRODUCER'S LOADS RUN TWO BATCHES AHEAD (the header's TIMELINE): it holds two batches of 8 tokens' values at
  fixed registers; a trip stores a batch and then loads those registers for the batch two on, so no store waits on
  its own load and no register is copied. A token that is not a group producer's is stored to a scratch word, with
  no branch; that producer's hit test still compiles to 6 divergent instructions a trip, declared.
- THE FOLD, a chunk of 64 tokens behind: the chunk's 4 steps a group in one commit group, with no branch. The
  producers hand a chunk over by an arrival each at a barrier, after a fence; the fold frees its tiles by another.
The lab's tokens put one leaf in each quarter of the box, so every group of 64 takes every token and every tile
fills exactly: arrivals that vary need a group's own back-pressure and steps on empty tiles, and NEITHER IS IN THIS
EXPERIMENT -- a group here holds two chunks of tiles and no more. Variants: the producers alone, the fold alone,
both (the state checked exactly, after 4,096 tokens). The values come from a table every CTA reads in step, of
65,536 tokens (8 MB) or 1,048,576 (128 MB): a line is fetched from memory once and read by the other CTAs from the
cache, so the larger table is a stream the cache cannot keep, not a cold load a CTA.

A FINDING OF THE COMPILER: a step under a branch on a loaded count compiles (ptxas C7520) to SERIALIZED wgmmas.

UNMEASURED IN THIS FORM: no card has run it.

Record: `measure/lab/results/fold_stream/sm_90a.json` (the experiment at 9eed68e9: its two batches rotated by copies,
its group producer under a branch).
