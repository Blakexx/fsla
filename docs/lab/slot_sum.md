# `slot_sum`

Mirrors `measure/lab/experiments/slot_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

An atomic add to global memory goes out at one line about every 10 to 12 cycles an SM whatever the warps and the
add's type ([`add_rate`](add_rate.md)), and in the composed carry the adds are what the other roles wait on
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

MEASURED: not yet run.
