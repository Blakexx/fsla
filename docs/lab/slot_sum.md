# `slot_sum`

Mirrors `measure/lab/experiments/slot_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

An atomic add to global memory goes out at one line about every 8.5 to 12 cycles an SM whatever the warps and the
add's type ([`add_rate`](add_rate.md)), and the carry adds a line a token a pair of boxes
([`carry_window`](carry_window.md)). Were every CTA to STORE a token's partial line where no other CTA writes, and
a later pass to add a token's partials, what does a line cost stored and what does a token cost that pass -- into
and out of memory nothing has touched, and a ring the cache holds because the pass comes soon enough?

THE FORM. A CTA an SM.
- STORE: 1 or 4 warps write 16,384 lines of 128 bytes, each its own words, into the CTA's own 64 MB of global
  memory. FRESH: each line once, PACKED or ONE EVERY 4 KB (as a slot sits where a token's 32 are kept together);
  a call's 264 MB of lines are five times the card's cache. RING: the CTA's first 1,024 lines, 128 KB, over and
  over.
- SUM: 4, 8 or 16 warps. A token is 32 partials of 128 bytes, f16; 8 lanes take a token, a lane 16 bytes of a
  partial an instruction -- 8 f16 turned to fp32 and added into the lane's 8 sums -- then the sums times the
  token's reciprocal, rounded to bf16 and stored, a 128-byte line a token. A partial's loads run two batches of 8
  ahead of their use. PACKED: a token's partials lie together, 4 KB a token. GATHERED: partial `s` lies in stream
  `s` of the CTA's 32, at a row scattered over the stream, as it would among the lines a CTA copied out by itself.
  FRESH: a CTA's 1,024 tokens, 4 MB, 528 MB a call. RING: its first 32 tokens' 128 KB over and over (by 4 and 8
  warps).
The SUM is a pass of this experiment's own shape, not the op's dividing pass: it scales by a reciprocal it loads
and writes bf16, where that pass divides by a summed channel and writes fp32.
Every stored word and every output value is checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60338744, driver 595.91.07; 2026-11-12, the experiment at
fbdaa4cb; 7 repetitions, every stored word and output value exact; the median, the spread under 9%).
A line stored, cycles a line a CTA:

| where | 1 warp | 4 warps |
|---|---|---|
| memory nothing has touched, lines packed | 8.2 | 7.8 |
| the same, one line every 4 KB | 18.2 | 8.6 |
| a ring of 128 KB, over and over | 14.3 | 6.3 |

A token's 32 partials summed, scaled and stored, cycles a token a CTA:

| the partials | 4 warps | 8 warps | 16 warps |
|---|---|---|---|
| packed a token, a CTA's 4 MB | 328.8 | 311.4 | 304.7 |
| gathered from 32 streams, a CTA's 4 MB | 363.8 | 302.4 | 285.6 |
| packed, 32 tokens' 128 KB over and over | 66.0 | 53.2 | |
| gathered, 32 tokens' 128 KB over and over | 108.8 | 55.9 | |

WHAT IT SHOWS.
- A LINE THAT GOES OUT TO MEMORY COSTS AN SM ABOUT 8 CYCLES, STORED OR READ, AND WARPS DO NOT BUY IT BACK: 7.8 to
  8.2 a line stored packed, and a token's 32 lines read and summed some 300, 9.5 a line, at 4 warps or 16. Every
  SM at once, that is the order of the card's memory bandwidth.
- A STORE INTO A RING THE CACHE HOLDS IS LITTLE CHEAPER: 6.3 cycles a line by four warps where the untouched
  lines read 7.8. [`add_rate`](add_rate.md)'s 1.8 was of 16 lines; 128 KB is not held that near.
- A SUM OF PARTIALS THE CACHE HOLDS IS A SIXTH OF ONE FROM MEMORY: 53 cycles a token by 8 warps against 311, the
  partials packed; gathered from 32 streams the same by 8 warps (56) and twice that by 4 (109).
- GATHERED AND PACKED READ ALIKE from memory at 8 warps and more (302 and 286 against 311 and 305).
- BESIDE AN ATOMIC ADD ([`add_rate`](add_rate.md): 8.5 to 12 cycles a line by the card, of lines the cache holds): a token of
  29 partials is some 290 to 350 by adds; by stores to memory and a sum from it, some 225 and 300; by stores into
  a ring and a sum from it, some 180 and 55. These are this page's rates times a count, not a composed
  measurement.

Record: `data:records/fsla/measure/lab/slot_sum.run/20261112T095959Z-730bab71/output.json`.
