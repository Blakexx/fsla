# `wgmma_width`

Mirrors `measure/lab/experiments/wgmma_width.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`wgmma_rate`](wgmma_rate.md) read an `m64n64k16` step's own time, 32 cycles, as the slope of a commit group's
cycles over its steps. Other experiments read steps of 16 and of 256 columns only as a sweep's total over its
count. What is a step's own time BY ITS OUTPUT WIDTH, and does a step narrower than 64 columns cost less?
[`carry_floor`](../internals/tools/carry_floor.md) prices the fold's steps from this record.

THE FORM. One warpgroup a CTA, a CTA an SM. A trip issues K steps into one accumulator in one commit group and
waits for it, K = 1, 2, 4 and 8, at 8, 16, 32, 64, 128 and 256 columns (64 rows, K 16); both operands
shared-memory tiles of ones, every descriptor held in a register. The slope of a trip's cycles over K is a step's
own time and the intercept the group's round trip. Every accumulator is checked: 16 a step in every value.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60309361, driver 595.91.07; 2026-11-12, the experiment at
f0311efc; 7 repetitions, every accumulator exact; cycles a commit group, the median, the spread under 0.1%;
THE STEP is the slope from 1 step to 8, THE GROUP what is left of the 1-step trip):

| columns | 1 step | 2 | 4 | 8 | the step | the group |
|---|---|---|---|---|---|---|
| 8 | 77 | 90 | 121 | 193 | 16.6 | 60 |
| 16 | 81 | 96 | 131 | 211 | 18.6 | 62 |
| 32 | 89 | 108 | 151 | 247 | 22.6 | 66 |
| 64 | 105 | 132 | 191 | 319 | 30.6 | 74 |
| 128 | 137 | 191 | 319 | 575 | 62.6 | 74 |
| 256 | 201 | 319 | 575 | 1087 | 126.6 | 74 |

WHAT IT SHOWS.
- A NARROWER STEP COSTS LESS, AND NOT IN PROPORTION: 16.6 cycles at 8 columns, 18.6 at 16, 22.6 at 32, 30.6 at 64 --
  about 14.6 a step whatever its width and a quarter cycle a column. A step of 8 columns is over half a step of 64.
- PAST 64 COLUMNS A STEP IS WHOLE 64-COLUMN STEPS: 62.6 at 128 and 126.6 at 256, two and four of them.
- THE COMMIT GROUP'S ROUND TRIP is 60 to 75 cycles beside its steps.
- So a step's work is cheapest a column at 64 columns, and the same there as at any multiple of 64.

Record: `measure/lab/results/wgmma_width/sm_90a.json`.
