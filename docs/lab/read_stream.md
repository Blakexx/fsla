# `read_stream`

Mirrors `measure/lab/experiments/read_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the readout's rows are streamed a digit at a time into each warp's ring (merge width 1) instead of gathered from a
staged pool, what does the stream cost beyond the readout, how much of its copies hides, and at what ring depth?

The step is the readout's tokens-on-N form ([`read_image`](read_image.md)): a tile is 8 tokens, 4 bf16 MMAs against
the digit's state from the image, each output pair exact and added into its token's panel row; every digit's first
tile runs in one straight-line block, then each digit's rest. The inputs (`read_stream.inputs.py`) are nl64k-alt-k4's
read lists over 64 windows, colored and placed as the facts pass emits them; each digit steps at least one tile (an
empty digit's all pads, as the readout steps every digit's first). The 80 CTAs are boxes 0 .. 79 of one head, channel
tile fastest; a box's read row for a token is its level-0 slice group's 64 bytes, fetched by token from global memory,
so L2 serves what one box fetched to the others. A CTA-window holds 45.0 tiles, the busiest warp's x 8 46.5.

STREAMED: each warp's ring of `D` stages, a stage a tile's 8 read rows and its 8 entry words (576 bytes); the producer
walks its window's stage record (every digit's first tile, then each digit's rest; each entry its token and its word),
copied two windows ahead, writes each stage's words beside its rows, and issues one stage a tile, its addresses running
pointers. Each CTA's panel is checked word for word.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | cycles |
|---|---|
| streamed, ring 4 (the first tiles' block holds the ring, nothing ahead) | 3,733 |
| streamed, ring 5 | 3,376 |
| streamed, ring 6 | 3,366 |
| streamed, ring 7 | 3,382 |
| bound: the readout alone (the producer, no row copies) | 3,092 |
| bound: the copies alone | 2,003 |
| bound: the readout alone, no producer (entries from the record) | 2,370 |

What it settles:

- The stream costs the readout 1.42x (streamed at ring 6 against the readout with no producer): 996 cycles a
  CTA-window, of which the producer is 722 and the exposed copies 274 -- 86% of the copies hide.
- A ring of 5 stages is enough; 4 (the first tiles' block, nothing ahead) costs 11%.
- The readout with no producer runs near `read_image`'s free-running tokens-on-N rate (1,896 on its 32 windows).
- Its second point is owed: a cell of longer read lists (alt-k16) needs more than this lab's one 256-slot panel pass.

Record: `measure/lab/results/read_stream/sm_86.json`.
