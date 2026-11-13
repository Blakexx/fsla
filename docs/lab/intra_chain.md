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
float, joined by one multiply-add -- since the standards keep the conversion unit out of a step loop; one row
converts by `I2F` to price that choice on this card. Every operand tile is constant, so every accumulator is checked
exactly; a level's steps read their own tiles (two levels on the same tiles compile as one level's steps and a copy).
The output's line and its drain are the carry's, measured there, and not in a trip.

ROWS, at 256 and at 64 digits a level: the chain with the causal mask and without; the integer steps alone; the
arithmetic alone; the value steps with the arithmetic and no integer step; the chain with the conversion unit.

MEASURED: not yet run.
