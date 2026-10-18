# `handoff`

Mirrors `measure/lab/experiments/handoff.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the carry's hand-off cost a window on sm_86 -- each digit's exponent from the fold's state (its largest value
over the box's 32 lists, each list at its own unit) and the state stored into the image, f16 at each digit's power of
two -- in the carry's form and in a lean form, the image whole (sm_80's form) or in two halves with a barrier between
(sm_86's), and how near its floor does it run?

8 warps hold a box's state in their fold accumulators (lists `w + 8 j`, 32 digits, 32 channels; the inputs
`handoff.inputs.py`: units a list, one list never live, values that span octaves by list and digit, digits zero in some
lists and one zero in all), made opaque each window so nothing is hoisted. A window is the maxima, a CTA barrier, the
exponents, the stores and a barrier; a sample is a CTA's span over 64 windows, 80 CTAs. Every form's image and
exponents are checked byte for byte against the mirror's, which follows `csrc/rola/src/carry/parts.cuh`'s rules.

- THE CARRY'S FORM (`parts.cuh`'s `post_maxima`, `digit_exp`, `store_batches`): a list's largest value a digit row as
  `ceil_exp` plus its unit, two shuffles a row, posted int16; after the barrier lane `d` the largest of the 8 warps'
  posts; a scale a (digit row, list) from its clamped exponent, each channel pair scaled, packed and stored 4 bytes at
  an address formed for it.
- LEAN: a value compared across lists as an integer KEY -- its fp32 bits plus (its unit less the box's top unit) x 2^23,
  so the exponent field moves and the mantissa stays; a zero INT_MIN -- one integer max a list, the digit row's 4 lanes
  joined by two shuffles, one shared atomic max a lane at its digit's slot (`4 g + 2 M + H`); after the barrier each
  lane's 4 keys in one 16-byte load and its exponents `(key >> 23) - 127 + (mantissa != 0)` plus the top unit, exact
  while the live units span under 255; the stores at 4 held bases (a channel tile's) and compile-time offsets.

sm_86 (RTX 3080 Ti), SM clock locked at 1.665 GHz, cycles a window (a CTA's):

| form | the image whole | two halves |
|---|---|---|
| the carry's | 3,059 | 3,184 |
| lean | 2,248 | 2,266 |
| bound: lean, the maxima alone | 582 | |
| bound: lean, the stores alone | 1,416 | |

Instructions a CTA-window (ncu): the carry's 5,894, lean 4,688 -- of which on the ALU pipe 3,250 and 2,619 (the
maxima's float max, the packs, the integer work), the FMA pipe 1,650 and 1,371, the load/store pipe 833 and 657.
Issue active 46% and 49%; the stalls a warp's: the shared queue full 15% and 17% (the stores alone 31%), a dependent
result 16% and 12%, the math pipe busy 10% and 11%, the barriers 6-8%.

What it settles:

- The lean form forms the carry's image and exponents exactly in 0.73x its cycles (0.71x in halves), 0.80x its
  instructions.
- Its floor is the shared pipe and the ALU pipe, not issue. The image is 512 stores of 4 bytes a lane a window at 2
  cycles each ([`smem_rates`](smem_rates.md)), 1,024 cycles; the maxima's float max, the packs and the integer work run
  on sm_86's half-rate ALU pipe, ~1,310 cycles a scheduler a window.
- In halves the hand-off alone costs 1% more (the carry's 4%): sm_86's halves cost their readout between them, not
  their barrier.
- The stores' cost is the image's layout: a lane's 8 values of a (digit, list) are 4 channel pairs 8 channels apart, 4
  bytes in each of 4 of today's 16-byte units. An image whose units hold a lane's channels (2t, 2t + 1, 8 + 2t, 9 + 2t,
  ...) stores them in one 16-byte store (128 a window, ~585 cycles at 4.6 a store) and pairs the readout's outputs
  (c, c + 2) where today it pairs (c, c + 8), which the panel and the drain carry. Not built: a layout decision, its
  stores' bank check first.
- The hand-off alone in the carry's form runs 3.06K cycles, against 3.7K for the maxima and the stores in the composed
  window ([`carry_window`](carry_window.md)'s form 6): the rest is the maxima's barrier waiting on the slowest fold warp
  and the next window's staging, which the store phase holds.

Record: `measure/lab/results/handoff/sm_86.json`.
