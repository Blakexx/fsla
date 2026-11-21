# `step_rolled`

Mirrors `measure/lab/experiments/step_rolled.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a tensor step cost its warpgroup when it is issued from a rolled loop with a run-time trip count, against the
same steps written as a straight block? The checker rule `step_rolled` refuses a tensor step in a loop that holds no wait
on it ([`KERNEL_STANDARDS.md` §27](../KERNEL_STANDARDS.md#s27); [`LEARNINGS.md`](../internals/LEARNINGS.md#step-rolled)),
and this page is its evidence.

THE FORM. A CTA an SM, one warpgroup of 128 threads. A trip is 16 `m64n16k16` steps into 16 accumulators (128 registers
a thread), both operands one tile of ones in shared memory, their descriptors held in registers; 4,096 trips are timed,
each ending in one commit and one wait. Every sum is checked exactly: 16 a step in every value.
- STRAIGHT: the 16 steps one block behind one fence.
- FENCED: the same with a fence before every step.
- ROLLED: for each of the 16 accumulators a loop of a run-time count (one) around its step, a fence before every step --
  the form of a list's steps issued by its count ([`carry_token`](carry_token.md)'s fold).
The row's cost is a CTA's span over the timed trips' steps, in cycles a step. It is recorded at one CTA and at the full
grid of 132.

WHAT IT WILL DECIDE: whether a step in a rolled loop of a run-time count costs its warpgroup more than the same step in a
straight block, by how much, and whether the fence a step the rolled form needs is part of that cost (FENCED against
STRAIGHT); that is, whether the rule `step_rolled` states a measured cost.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM r06xatq1, driver 580.196.07; 2026-11-21 (the one-CTA and the full-grid records), the experiment
at bf48ca16; 7 repetitions; the 3 checked rows at one CTA and the 3 at 132 pass, every sum exact; the median, the widest
spread of a row 0.0%). Cycles a step:

| form | 1 CTA | 132 CTAs |
|---|---:|---:|
| STRAIGHT: a straight block behind one fence | 23.19 | 23.19 |
| FENCED: a straight block, a fence a step | 23.19 | 23.19 |
| ROLLED: each step in a loop of a run-time count of one, a fence a step | 37.63 | 37.63 |

WHAT IT SHOWS.
- AT 1 CTA, THE STRAIGHT BLOCK: 23.19 cycles a step.
- AT 1 CTA, A FENCE A STEP: 23.19 cycles a step, the same as the 23.19 of STRAIGHT: a fence a step is free.
- AT 1 CTA, EACH STEP IN A ROLLED LOOP OF A RUN-TIME COUNT: 37.63 cycles a step, 1.62 times the 23.19 of STRAIGHT (14.44 cycles a step above).
- AT 132 CTAs, THE STRAIGHT BLOCK: 23.19 cycles a step.
- AT 132 CTAs, A FENCE A STEP: 23.19 cycles a step, the same as the 23.19 of STRAIGHT: a fence a step is free.
- AT 132 CTAs, EACH STEP IN A ROLLED LOOP OF A RUN-TIME COUNT: 37.63 cycles a step, 1.62 times the 23.19 of STRAIGHT (14.44 cycles a step above).

Record: `measure/lab/results/step_rolled/sm_90a.blocks1.json` and `sm_90a.blocks132.json`.
