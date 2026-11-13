# `call_pair`

Mirrors `measure/lab/experiments/call_pair.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The hopper intra is a kernel of its own beside the carry
([`ARCHITECTURE.md`](../ARCHITECTURE.md#64-the-intra--the-within-window-term)). What is a call's time with both,
and does launching the two TOGETHER on two streams buy anything over one after the other? The carry's CTA takes its
multiprocessor's whole register file for the call ([`carry_window`](carry_window.md)), so an intra CTA runs only on
a multiprocessor the carry left free or has left.

THE FORM. The two labs' own kernels, unchanged, in one binary: [`carry_window`](carry_window.md)'s winner's form
with the den and no clock inside a role, on the cell's 128 CTAs of the card's 132 multiprocessors, and
[`intra_window`](intra_window.md)'s form with the levels tile-major and the values token-major, 16 CTAs a window.
Each runs 16 calls of 65,536 tokens, so a launch's own time is small beside its work. Rows, by wall time a call:
the carry alone; the intra alone; the two on one stream; the two on two streams, either launched first. The
kernels' outputs are their own experiments' to check: this one adds no device code.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60545556, driver 595.91.07; 2026-11-13, the experiment at
89ce5851; 7 repetitions of 16 calls, by wall time; the median):

| the row | milliseconds a call of 65,536 tokens | least to most |
|---|---|---|
| the carry alone | 0.285 | 0.284 to 0.302 |
| the intra alone | 0.072 | 0.071 to 0.076 |
| the carry, then the intra: one stream | 0.401 | 0.400 to 0.405 |
| the carry and the intra together: two streams | 0.392 | 0.389 to 0.393 |
| the intra and the carry together, the intra launched first | 0.400 | 0.398 to 0.401 |

WHAT IT SHOWS.
- A CALL WITH THE CARRY, ITS DEN AND THE INTRA IS 0.39 TO 0.40 MILLISECONDS a head at 65,536 tokens on this card:
  the carry alone 0.285, the intra alone 0.072 -- a quarter of the carry's time. No facts pass and no producer
  is in it and no tile pair of the intra is skipped. THE TABLES ARE CACHED, so this is the upside and not the price
  (R22.2): the call with tables the cache does not hold is unmeasured.
- LAUNCHING THE TWO TOGETHER BUYS 2%: 0.392 on two streams against 0.401 on one, and nothing with the intra
  launched first (0.400). The carry's 128 CTAs hold their multiprocessors for the call, so the intra's CTAs have
  the card's other 4 until the carry leaves. Nothing closer than the two streams is built.
- THE TWO ONE AFTER THE OTHER ARE 44 MICROSECONDS A CALL OVER THEIR OWN TIMES' SUM (0.357). What the
  second kernel waits for is NOT ISOLATED; it is a sixteenth of what a launch of 16 calls lost, so a launch of one
  call may lose more.

Record: `measure/lab/results/call_pair/sm_90a.json`.
