# `slot_sum`

Mirrors `measure/lab/experiments/slot_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

An atomic add to global memory is taken at one line about every 10 to 12 cycles an SM whatever the warps and the
add's type ([`add_rate`](add_rate.md)), and in the composed carry the adds are what the other roles wait on
([`carry_window`](carry_window.md)). Were every CTA to STORE a token's partial line where no other CTA writes, and
the pass that divides the sums to add a token's partials first, what does a line cost stored to memory nothing has
touched, and what does a token cost that pass -- its partials lying packed, or gathered from the CTAs' own streams?

THE FORM. A CTA an SM.
- STORE: 1 or 4 warps write 16,384 lines of 128 bytes from registers into the CTA's own 64 MB of global memory,
  each line once: PACKED, or ONE EVERY 4 KB, as a slot of a layout that keeps a token's 32 slots together sits. A
  call's 264 MB of lines are five times the card's cache.
- SUM: 4, 8 or 16 warps. A token is 32 partials of 128 bytes, f16; 8 lanes take a token, a lane 16 bytes of a
  partial an instruction -- 8 f16 turned to fp32 and added into the lane's 8 sums -- then the sums times the
  token's reciprocal, rounded to bf16 and stored, a 128-byte line a token. A partial's loads run two batches of 8
  ahead of their use. PACKED: a token's partials lie together, 4 KB a token. GATHERED: partial `s` lies in stream
  `s` of the CTA's 32, at a row scattered over the stream, as it would among the lines a CTA copied out by itself.
  540 MB of partials, a CTA its own 4 MB.
Every stored word and every output value is checked exactly.

MEASURED: not yet run.
