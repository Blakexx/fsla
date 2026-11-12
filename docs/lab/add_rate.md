# `add_rate`

Mirrors `measure/lab/experiments/add_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

In the composed carry the warps that add each token's line to the output are the role the others wait on
([`carry_window`](carry_window.md)), and four warps adding lines from registers read 9.8 cycles a line a CTA
([`drain_role`](drain_role.md)). What holds an add to global memory: a limit a warp, the lanes an instruction
carries, the lines it touches, the add's type, or its being atomic?

THE FORM. A CTA an SM of 1, 2, 4 or 8 warps. Each warp issues 16 instructions a trip to its own rows of global
memory, from a register that never changes and to addresses that are constants off the warp's base: nothing waits
on anything.
- ONE LINE: an instruction's lanes add f16 pairs into one 128-byte line -- by 32 lanes, by 16 and by 8 (the others
  issue nothing).
- SPREAD: its 32 lanes each add into a different line.
- STORE: the one-line form as a plain 4-byte store a lane, no add.
- INTEGER AND fp32: the one-line form by a 32-bit integer add and by an fp32 add; and 64-bit integer adds, 8 bytes
  a lane, two lines an instruction.
Every row's words are checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60337303, driver 595.91.07; 2026-11-12, the experiment at
78cbf76f; 7 repetitions, every word exact; the cycles in which each warp of the CTA issues one instruction, the
median; the spread under 3.2%, but the spread form at four warps 5.4%, which the record marks unstable):

| the form | 1 warp | 2 warps | 4 warps | 8 warps |
|---|---|---|---|---|
| f16 adds into one line, 32 lanes | 12.6 | 21.9 | 41.3 | 80.6 |
| the same, 16 lanes | 8.2 | | | |
| the same, 8 lanes | 7.8 | | | |
| f16 adds, a lane a line of its own | 86.5 | | 376.6 | |
| 32-bit integer adds into one line | 11.9 | | 40.6 | |
| fp32 adds into one line | 12.2 | | 39.9 | |
| 64-bit integer adds, two lines an instruction | 22.2 | | 78.7 | |
| plain stores into one line, 32 lanes | 1.8 | | 1.8 | |

WHAT IT SHOWS.
- THE ADD IS NOT HELD BY THE WARP. One warp issues an add of a whole line every 12.6 cycles; two warps
  each every 21.9, four every 41.3, eight every 80.6. An SM gets one line's add out about every 10 to 12
  cycles whichever of its warps issues it, and more warps adding buy nothing. The rows are of lines a CTA owns and
  rewrites, which stay in its cache.
- IT IS THE ATOMIC ADD, WHATEVER ITS TYPE: a line by 32-bit integers is 11.9 and 40.6 at one and four warps, by
  fp32 12.2 and 39.9, and 64-bit integers, two lines an instruction, twice that. No integer form is faster.
- AN INSTRUCTION COSTS MOSTLY ITSELF, THEN ITS LINES: 32 lanes into one line 12.6, into 32 lines 86.5 -- some 10
  an instruction and 2.4 a line it touches. Fewer lanes into the one line cost less, 8.2 at 16 and 7.8 at 8;
  those two rows also carry a lane's predicate around each add, so what of their cost is the lanes' is not
  separated.
- A PLAIN STORE OF THE SAME LINE IS 1.8 CYCLES AND DOES NOT SLOW WITH THE WARPS -- of lines it rewrites, which
  its cache holds. That is no store's cost to a line nothing has touched; [`slot_sum`](slot_sum.md) asks that.
- WHERE THE LIMIT SITS IS OPEN: every SM adds at once here, so a limit an SM and one shared by the card and split
  among its SMs read the same.

Record: `measure/lab/results/add_rate/sm_90a.json`.
