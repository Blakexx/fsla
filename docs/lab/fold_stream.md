# `fold_stream`

Mirrors `measure/lab/experiments/fold_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A box as wide as the read side wants is written by every token, sparsely: 4 of its 256 leaves. Folding it list by
list needs each token's value copied into list order, which [`carry_floor`](../internals/tools/carry_floor.md)
prices as the form's largest term. Can the fold instead be fed by ONE PASS over the tokens -- no list, no rank, no
look back -- and what does a token then cost a CTA, the producer's work beside the fold's?

THE FORM. A CTA an SM: one producer warp and one fold warpgroup that holds the box's state, 256 leaves × 64
channels, as 32 accumulators of 8 leaves (`m64n8k16`).
- THE PRODUCER walks the tokens in order, 8 at a time into registers by one coalesced 128-byte load a token. For
  each of a token's 4 leaves it stores the value into the leaf's group's tile at the group's fill count, and the
  token's row of 8 coefficients (one of them its own) beside it. Both operands are M|N-major, so a value is one
  4-byte store a lane on 32 banks and a row is one 16-byte line.
- THE FOLD, a chunk of 128 tokens behind: it sweeps the 32 groups with a step each and no branch -- on the group's
  tile if it filled, else on a tile of zeros. A step under a branch on a loaded count compiles (ptxas C7520) to
  SERIALIZED wgmmas; the select of an address does not.
- A group holds two tiles, so the producer fills one while the fold reads the other.
The lab's tokens give every group exactly a step's 16 arrivals a chunk; arrivals that vary would run some steps on
the zeros and are not in this experiment. Variants: the producer alone, the fold alone, both (the state checked
exactly, after 4,096 tokens). The tokens' values come from a table every CTA reads, one the cache holds (65,536
tokens, 8 MB) or does not (1,048,576 tokens, 128 MB).

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): the producer's 8 tokens are 8 `LDG`, no stack memory;
a token's 4 leaves are 4 value stores, 4 row stores and 4 fill counts; the sweep is 32 `HGMMA` in one commit group.

UNMEASURED: no card has run it. The transposed operand layout is read from CUTLASS's descriptors and proven only
by this experiment's check.
