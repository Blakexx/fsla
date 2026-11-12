# `load_rate`

Mirrors `measure/lab/experiments/load_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A warp's own load of a scattered 128-byte row costs it about 33 cycles, and four warps a quarter of that
([`value_gather`](value_gather.md)). What is that cost a property of: the instruction, its bytes, the loads in
flight, or the warps issuing?

THE FORM. A CTA an SM of 1, 4, 8 or 16 warps. Each warp fetches its own scattered rows of a table of 65,536 rows
(8 MB, one the cache holds) and folds every word into a chain a lane, nothing stored: the chain's order keeps a
word's use out of the trip that loaded it.
- NARROW: a lane loads 4 bytes, an instruction a row.
- WIDE: a lane loads 16 bytes, an instruction FOUR rows, 8 lanes a row.
A ring of 4, 16 or 64 instructions is in flight, in two halves at their own registers. In the narrow form the
toolkit copies two of a trip's loaded words at the trip's end, whatever the source's form: those two rows' loads
run half a trip ahead of their use. Every lane's chain is checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60313590, driver 595.91.07; 2026-11-12, the experiment at
3699007c; 7 repetitions, every chain exact; cycles a 128-byte row a CTA, the median, the spread under 1.5%).
Sixteen instructions in flight a warp:

| the warps | 4 bytes a lane, a row an instruction | 16 bytes a lane, four rows an instruction |
|---|---|---|
| 1 | 33.00 | 10.55 |
| 4 | 8.46 | 3.49 |
| 8 | 4.71 | 2.15 |
| 16 | 3.33 | 1.46 |

Four warps, 4 bytes a lane:

| instructions in flight a warp | cycles a row a CTA | a warp's cycles a row |
|---|---|---|
| 4 | 22.84 | 91.4 |
| 16 | 8.46 | 33.8 |
| 64 | 4.95 | 19.8 |

WHAT IT SHOWS.
- A WARP'S SCATTERED LOAD IS BOUND BY ITS LATENCY, NOT BY ITS ISSUE. With 4 instructions in flight a warp takes a
  row every 91 cycles; with 16, every 34; with 64, every 20. A load lands some 350 to 550 cycles after its issue,
  and a warp's rate is the rows it has in flight over that.
- SO A ROW COSTS WHAT ITS WARP KEEPS IN FLIGHT, and three things multiply that: the ring's depth, the lane's width
  (16 bytes a lane puts four rows in an instruction: 10.6 against 33.0 for one warp) and the warps.
- THE WARPS SCALE TO FOUR AND FLATTEN BEYOND: one row an instruction, 33.0, 8.5, 4.7 and 3.3 at 1, 4, 8 and 16
  warps; four rows an instruction, 10.6, 3.5, 2.2 and 1.5.
- SIXTEEN WARPS WITH FOUR ROWS AN INSTRUCTION FETCH A SCATTERED ROW IN 1.5 CYCLES A CTA, under a third of the row's
  share of the bulk engine's packed copy (33 cycles a KB, 4.2 a row: [`state_move`](state_move.md)).
- [`value_gather`](value_gather.md)'s 33.4 and 8.5 are this experiment's narrow rows at 16 in flight (33.0, 8.5):
  the store into shared memory added nothing, and the number was the ring's depth.

Record: `measure/lab/results/load_rate/sm_90a.json`.
