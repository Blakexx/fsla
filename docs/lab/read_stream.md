# `read_stream`

Mirrors `measure/lab/experiments/read_stream.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

When the readout's rows are streamed a digit at a time into each warp's ring (merge width 1) instead of gathered from a
staged pool, what does the stream cost beyond the readout, at what ring depth, and in what form?

The step is the readout's tokens-on-N form ([`read_image`](read_image.md)): a tile is 8 tokens, 4 bf16 MMAs against
the digit's state from the image, each output pair exact and added into its token's panel row. A box-window's colored
slots run in passes of 256 (a panel's rows): each pass steps every digit's first tile in one straight-line block, then
each digit's rest, then drains its panel (`red.global.add`) between two CTA barriers. The inputs
(`read_stream.inputs.py`) are three cells' read lists, colored and placed per pass as the facts pass emits them:
nl64k-alt-k4 over 64 windows, flagship-alt-k16-L16384 and flagship-cohort-k16-L16384 over 32. Each digit steps at least
one tile a pass (an empty digit's all pads, as the readout steps every digit's first). The 80 CTAs are boxes 0 .. 79 of
one head, channel tile fastest; a box's read row for a token is its level-0 slice group's 64 bytes, fetched by token
from global memory, so L2 serves what one box fetched to the others. Each CTA's drained panel is checked word for word.

| a CTA-window | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| tiles: mean; the busiest warp's a pass, x 8 | 45.0; 55.8 | 155.8; 180.8 | 175.7; 355.9 |
| passes | 1.00 | 2.00 | 1.98 |

STREAMED: each warp's ring of `D` stages, a stage a tile's 8 read rows and its 8 entry words (576 bytes); the producer
walks its window's token record (each stage's 8 tokens, a pad the tile's first live token), copied a window ahead with
the stage group it is issued in, and copies each stage's rows and its words (two lanes, from the window's word record
in global memory), one stage a tile, its addresses running pointers.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a CTA-window:

| form | alt-k4 | alt-k16 | cohort-k16 |
|---|---|---|---|
| streamed, ring 4 (the first tiles' block holds the ring, nothing ahead) | 5,750 | | |
| streamed, ring 5 | 5,486 | 13,717 | 24,542 |
| streamed, ring 6 | 5,793 | | |
| streamed, ring 5, the consumer's words from global memory a tile ahead | 6,474 | 17,366 | 33,733 |
| streamed, ring 5, the next stage's token a stage ahead | 5,583 | 14,020 | 25,072 |
| streamed, two tiles a stage, ring 3 (6 tiles) | 5,379 | 13,660 | 25,028 |
| bound: the readout with the producer, no row copies | 5,249 | 13,692 | 23,089 |
| bound: the same, two tiles a stage | 4,896 | | |
| bound: the same, the consumer's words a tile ahead | 5,096 | | |
| bound: the same, the token a stage ahead | 5,250 | | |
| bound: the stream with no waits on its stages | 5,517 | | |
| bound: the copies alone | 2,695 | | |
| bound: the readout alone, no producer (its words from global memory a tile ahead) | 4,057 | 10,361 | 16,838 |

Instructions (ncu, a stage): the producer 35.2 at alt-k4, 26.8 at alt-k16, 28.0 at cohort-k16 (its records, cursor,
waits and the entry words' copy); the row copy 3.8-3.9.

What it settles:

- The stream costs the readout 1.35x at alt-k4, 1.32x at alt-k16 and 1.46x at cohort-k16 (streamed at ring 5 against
  the readout with no producer, drains included): 1,429, 3,356 and 7,704 cycles a CTA-window. As a factor of the bare
  readout it predicts its third cell from the other two within 8%; a fixed part and a part a stage does not (-21% with
  the busiest warp's stages a pass, cohort-k16's passes twice as uneven as its mean).
- The producer is nearly all of it, its own instructions between tiles of 4 MMAs, not its waits: with no waits the
  stream runs the same (5,517 against 5,486), and without its row copies it costs 1,192 of the 1,429 at alt-k4.
- None of three producer forms helps: the consumer's words from global memory (+18% to +38%, its loads beside the row
  copies), the token a stage ahead (+2%), two tiles a stage (-2% to +2%: the producer's own cost falls 30%, 1,192 to
  839 at alt-k4, and the copies' interleaving takes it back).
- A ring of 5 stages: 4 costs 5%; 6 costs 6%, its records only a window ahead (a window holds 4 stages at least, fewer
  than the 5 groups ring 6 waits on, so its entry drains the ring).

Record: `measure/lab/results/read_stream/sm_86.json`.
