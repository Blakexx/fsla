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
  fall in its group it stores the value at its own running count and the token's row of 64 coefficients beside it; it
  loads every token's value (128 bytes, coalesced), a batch ahead of the test. The count is the producer's own register.
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

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60286161, driver 580.173.02; 2026-11-11, the experiment at 9eed68e9;
7 repetitions, the state exact after 4,096 tokens in every `together` form; SM-cycles a token a CTA, the median, the
spread under 0.5%):

| the form | the fold alone | the producers alone | together | together, the larger table |
|---|---|---|---|---|
| steps of 64 leaves, 4 producers (one a group) | 11.8 | 205.6 | 211.8 | 211.8 |
| steps of 256 leaves, 1 producer | 11.5 | 83.7 | 93.6 | 116.0 |
| steps of 256 leaves, 4 producers | 11.5 | 25.9 | 32.0 | 36.1 |

WHAT IT SHOWS.
- THE ONE-PASS FILL IS CORRECT at both widths, and the transposed operands are laid out as CUTLASS's descriptors
  say.
- A STEP'S TIME IS ITS OWN, NOT ITS COLUMNS'. A chunk's steps cost 47 cycles each at 64 columns (16 a chunk) and
  185 at 256 (4 a chunk): the same 11.5 to 11.8 a token either way. The experiment's first form read 36 a step at
  8 columns. So a token in a list of a box costs a sixteenth of a step whatever the step's width up to 64, and a
  wider step only spends the same time on more columns.
- LOADS A BATCH AHEAD ARE WORTH A FACTOR OF FOUR: the one-producer form was 318 cycles a token with each value
  loaded in the trip that stored it, and is 84.
- FOUR PRODUCERS BY RESIDUE SCALE (84 to 26); A PRODUCER A GROUP DOES NOT PAY (206): each of the four looks at every
  token, loads every value and stores a row of 64 coefficients, where the one-group form stores four words.
- THE CHEAPEST FORM IS 32 CYCLES A TOKEN A CTA, the producers' 26 and the fold's 11.5 overlapping to 32. Lanes 4 to
  31 of a row's store land on a scratch word that shares banks with the row: the producer's 26 holds that conflict.
- A box every token writes is visited by every token: on a card of 128 such CTAs, 32 cycles a visit is 4,100
  SM-cycles a token, and the fold's 11.5 alone 1,470.

Record: `measure/lab/results/fold_stream/sm_90a.json`.
