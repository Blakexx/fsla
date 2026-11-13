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

MEASURED: not yet run.
