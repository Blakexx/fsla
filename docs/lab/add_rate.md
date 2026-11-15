# `add_rate`

Mirrors `measure/lab/experiments/add_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's output is a line added to global memory a token a pair of boxes
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
- VECTOR: one add of four f16 pairs a lane, 16 bytes -- 32 lanes an instruction are FOUR lines, 8 lanes one: is an add's
  price its instruction's or its lines'?
- ON A FEW SMs: the one-line form, 4 and 8 warps, with CTAs on 8 and on 32 of the card's SMs: is the limit an SM's own,
  or the card's shared among its SMs? NOT YET MEASURED.
Every row's words are checked exactly.

MEASURED (H100 PCIe, 114 SMs, a Hyperstack spot VM, driver 580.196.07; 2026-11-15, the experiment at d2b3017d; 7
repetitions, every word exact; the cycles in which each warp of the CTA issues one instruction, the median; this
card's cycles are shorter than the SXM's this page held before, 12.6 a line there):

| the form | 1 warp | 2 warps | 4 warps | 8 warps |
|---|---|---|---|---|
| f16 adds into one line, 32 lanes | 10.2 | 18.9 | 35.9 | 68.3 |
| the same, 16 lanes | 7.8 | | | |
| the same, 8 lanes | 7.8 | | | |
| f16 adds, a lane a line of its own | 69.2 | | 294.2 | |
| 32-bit integer adds into one line | 10.2 | | 36.3 | |
| fp32 adds into one line | 10.2 | | 35.9 | |
| 64-bit integer adds, two lines an instruction | 17.7 | | 65.9 | |
| plain stores into one line, 32 lanes | 1.8 | | 1.8 | |
| VECTOR f16 adds, 16 bytes a lane: four lines an instruction | 34.6 | | 132.0 | 265.2 |
| the same, 8 lanes: one line an instruction | 13.9 | | 33.1 | |

WHAT IT SHOWS.
- THE ADD IS NOT HELD BY THE WARP. One warp issues an add of a whole line every 10.2 cycles; two warps
  each every 18.9, four every 35.9, eight every 68.3. An SM gets one line's add out about every 8.5 to 10
  cycles whichever of its warps issues it, and more warps adding buy nothing. The rows are of lines a CTA owns and
  rewrites, which stay in its cache.
- IT IS THE ATOMIC ADD, WHATEVER ITS TYPE: a line by 32-bit integers is 10.2 and 36.3 at one and four warps, by
  fp32 10.2 and 35.9, and 64-bit integers, two lines an instruction, 17.7 and 65.9. No integer form is faster.
- A VECTOR ADD IS PRICED BY ITS LINES, NOT BY ITS INSTRUCTION: four lines an instruction cost 34.6 cycles at one
  warp and 132.0 and 265.2 at four and eight -- 8.3 to 8.7 a line, where the scalar form's line is 8.5 to 10.2.
  One line by 8 lanes of 16 bytes is 13.9 at one warp, over the scalar line's 10.2, and 8.3 a line at four. So the
  path takes a line's bytes at its own rate however few instructions carry them: the vector form saves the
  instructions (a warp's one where the scalar's are four) and at most a seventh of the cycles. The 64-bit rows
  said the same: two lines an instruction, 8.2 to 8.9 a line.
- A LINE'S ADDS COST LESS TOGETHER THAN APART: 32 lanes into one line 10.2, into 32 lines 69.2 -- 2.2 a line touched
  for 4 bytes of it. Fewer lanes into the one line cost less, 7.8 at 16 and at 8; those two rows also carry a
  lane's predicate around each add, so what of their cost is the lanes' is not separated.
- A PLAIN STORE OF THE SAME LINE IS 1.8 CYCLES AND DOES NOT SLOW WITH THE WARPS -- of lines it rewrites, which
  its cache holds. That is no store's cost to a line nothing has touched; [`slot_sum`](slot_sum.md) asks that.
- WHERE THE LIMIT SITS IS OPEN: every SM adds at once here, so a limit an SM and one shared by the card and split
  among its SMs read the same.

Record: `measure/lab/results/add_rate/sm_90a.json`.
