# `intra_chain`

Mirrors `measure/lab/experiments/intra_chain.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The intra is the term of a window's own pairs ([`ARCHITECTURE.md`](../ARCHITECTURE.md#64-the-intra--the-within-window-term)):
each level's pair products are integer tensor operations of the rows' bytes, the levels multiply in fp32, and the
value product follows in bf16. No hopper intra is designed. What does that chain cost a tile pair -- 64 readers by
64 writers -- alone on this card, and which of its parts is the cost? The answer is the first figure a hopper intra's
schedule is written from: whether it can sit beside the carry's roles or needs CTAs of its own.

THE FORM. One warpgroup a CTA, a CTA an SM; a trip is a tile pair.
- THE LEVELS' PAIR PRODUCTS: for each of two levels, S steps of `m64n64k32`, u8 by u8 into s32, both operands
  shared memory -- S is the level's digits over 32, 8 at a level of 256 digits and 2 at one of 64 -- into the
  level's accumulator of 32 registers a thread, the two levels' steps one commit group.
- THE PAIR'S WEIGHT: the two accumulators made fp32 and multiplied; where the row asks, nothing where the writer is
  not before the reader (a thread's 32 pairs a mask word); the weight rounded to bf16, a pair a word, IN THE
  REGISTERS IT LIES IN -- an accumulator of 64 columns is four writer groups of 16, each group's 4 words the A
  operand a value step reads.
- THE VALUE STEPS: 4 steps of `m64n64k16`, A those registers and B a tile of values, into the output's
  accumulator, 64 readers by 64 channels.
A product is made fp32 on the float pipe -- an integer under 2^24 as two halves of 12 bits, each the low bits of a
float, joined by one multiply-add -- the form the standards drew from another family's conversion unit; one row
converts by a cast, which the toolkit compiles as `I2F`, to price the two on this card. Every operand tile is constant, so every accumulator is checked
exactly; a level's steps read their own tiles (two levels on the same tiles compile as one level's steps and a copy).
The output's line and its drain are the carry's, measured there, and not in a trip.

ROWS, at 256 and at 64 digits a level: the chain with the causal mask and without; the integer steps alone; the
arithmetic alone; the value steps with the arithmetic and no integer step; the chain with the conversion unit.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60540284, driver 580.173.02; 2026-11-13, the experiment at
c063e24d; 7 repetitions, every accumulator exact; the median, cycles a tile pair):

| the row | 256 digits a level | 64 digits a level |
|---|---|---|
| the chain, the causal mask on | 1,336 | 952 |
| the chain, no mask | 1,272 | 887 |
| the integer steps alone | 573 | 189 |
| the arithmetic alone, the mask on | 40 | 40 |
| the value steps and the arithmetic, no integer step | 206 | 206 |
| the chain, the causal mask on, the products by the conversion unit | 985 | 601 |

WHAT IT SHOWS.
- A TILE PAIR IS 985 CYCLES at 256 digits a level and 601 at 64, its products converted by the conversion unit:
  64 readers by 64 writers, 0.24 cycles a pair of tokens. A window of 1,024 tokens is 136 tile pairs with its
  diagonal, so the chain alone is some 131 SM-cycles a token at 256 digits a level with no pair skipped
  (ESTIMATED from the tile pair: no window was run), beside the carry's composed window
  ([`carry_window`](carry_window.md)).
- THE PRODUCTS' CONVERSION ON THE FLOAT PIPE LOSES ON THIS CARD: 1,336 cycles against 985 by `I2F` -- 64
  conversions a thread a tile pair as two adds and a multiply-add each are some six instructions where the
  cast is one. Which unit of the card runs the cast is not read here; the standards' rule, drawn from another
  family, is amended for hopper on this row (KERNEL_STANDARDS 29, item 6).
- THE INTEGER STEPS ARE 36 CYCLES EACH in a group of 16 (573 a tile pair) and 47 in a group of 4 (189): the
  levels' products are about three fifths of the chain at 256 digits and a third at 64.
- FOUR VALUE STEPS FROM REGISTERS ARE 206 CYCLES: in that row the words of A are constants the toolkit
  formed before the trip, so it is the steps' own time.
- THE ARITHMETIC'S OWN ROWS DO NOT ISOLATE IT: with no integer step the products are constants and the toolkit
  makes them floats once, outside the trip. Its cost is the chain less the two step rows: 557 cycles on the
  float pipe and 206 by the conversion unit.
- THE CHAIN'S LIVE VALUES ARE 112 REGISTERS A THREAD by its timeline -- two integer accumulators of 32, the
  output's 32 and 16 words of A -- and the toolkit assigned its kernels 144 to 172 (its allocation, not a count of
  live ranges). The carry's CTA has no register left for a role of 112 ([`carry_window`](carry_window.md)), so a
  hopper intra runs in CTAs of its own: after the carry first (Blake, 2026-11-13), its share of the call measured
  before any overlap is built.

Record: `measure/lab/results/intra_chain/sm_90a.json`.
