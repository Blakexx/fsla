# `wgmma_width`

Mirrors `measure/lab/experiments/wgmma_width.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`wgmma_rate`](wgmma_rate.md) read an `m64n64k16` step's own time, 32 cycles, as the slope of a commit group's
cycles over its steps. Other experiments read steps of 16 and of 256 columns only as a sweep's total over its
count. What is a step's own time BY ITS OUTPUT WIDTH, and does a step narrower than 64 columns cost less?
[`carry_floor`](../internals/tools/carry_floor.md) prices the fold's steps on the answer.

THE FORM. One warpgroup a CTA, a CTA an SM. A trip issues K steps into one accumulator in one commit group and
waits for it, K = 1, 2, 4 and 8, at 8, 16, 32, 64, 128 and 256 columns (64 rows, K 16); both operands
shared-memory tiles of ones, every descriptor held in a register. The slope of a trip's cycles over K is a step's
own time and the intercept the group's round trip. Every accumulator is checked: 16 a step in every value.

UNMEASURED: no card has run it.
