# `wgmma_width`

Mirrors `measure/lab/experiments/wgmma_width.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`wgmma_rate`](wgmma_rate.md) read an `m64n64k16` step's own time, 32 cycles, as the slope of a commit group's
cycles over its steps. Other experiments read steps of 16 and of 256 columns only as a sweep's total over its
count. What is a step's own time BY ITS OUTPUT WIDTH, and does a step narrower than 64 columns cost less?
[`carry_floor`](../internals/tools/carry_floor.md) prices the fold's steps from this record.

THE FORM. One warpgroup a CTA, a CTA an SM. A trip issues K steps into one accumulator in one commit group and
waits for it, K = 1, 2, 4 and 8, at 8, 16, 32, 64, 128 and 256 columns (64 rows, K 16), and K = 32 at 8 columns;
both operands shared-memory tiles of ones, every descriptor held in a register. A step's own time is the increment
from 4 steps a group to 8, and the rest of a trip its group's round trip. Every accumulator is checked: 16 a step in every value.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60311139, driver 595.91.07; 2026-11-12, the experiment at
1eba273a; 7 repetitions, every accumulator exact; cycles a commit group, the median, the spread under 0.1%;
THE STEP is a step more once a group holds several, from 4 steps to 8 -- a group's first steps cost less each,
so the slope from 1 is under it -- and THE GROUP what the 8-step trip holds beside eight of them):

| columns | 1 step | 2 | 4 | 8 | the step | the group |
|---|---|---|---|---|---|---|
| 8 | 77 | 90 | 121 | 193 | 18.0 | 49 |
| 16 | 81 | 96 | 131 | 211 | 20.0 | 51 |
| 32 | 89 | 108 | 151 | 247 | 24.0 | 55 |
| 64 | 105 | 132 | 191 | 319 | 32.0 | 63 |
| 128 | 137 | 191 | 319 | 575 | 64.0 | 63 |
| 256 | 201 | 319 | 575 | 1087 | 128.0 | 63 |

WHAT IT SHOWS.
- A NARROWER STEP COSTS LESS, AND NOT IN PROPORTION: 18 cycles at 8 columns, 20 at 16, 24 at 32, 32 at 64 -- 16
  a step whatever its width and a quarter cycle a column. A step of 8 columns is over half a step of 64.
- PAST 64 COLUMNS A STEP IS WHOLE 64-COLUMN STEPS: 64 at 128 and 128 at 256. The 64-column step is the 32
  [`wgmma_rate`](wgmma_rate.md) read.
- THE COMMIT GROUP'S ROUND TRIP is 50 to 65 cycles beside its steps.
- So a step's work is cheapest a column at 64 columns, and the same there as at any multiple of 64.
- A GROUP OF 32 EIGHT-COLUMN STEPS is 625 cycles: a box of 256 leaves folded 8 at a step, 19.5 a step with its
  round trip.

Record: `data:lab/wgmma_width/sm_90a.json`.
