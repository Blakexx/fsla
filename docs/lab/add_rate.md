# `add_rate`

Mirrors `measure/lab/experiments/add_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A warp adding f16 lines to global memory from registers, with nothing between its adds, gets one out about every 39
cycles, and the rate scales with the warps up to four ([`drain_role`](drain_role.md),
[`drain_rate`](drain_rate.md)); in the composed carry the warps that add are the role the others wait on
([`carry_window`](carry_window.md)). What holds a warp to that: a limit a warp, the lanes an instruction carries,
the lines it touches, or the add being atomic?

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

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60336666, driver 595.91.07; 2026-11-12, the experiment at
f89cc6b3; 7 repetitions, every word exact; the cycles in which each warp of the CTA issues one instruction, the
median, the spread under 0.5% but the spread form at four warps, 12%):

| the form | 1 warp | 2 warps | 4 warps | 8 warps |
|---|---|---|---|---|
| f16 adds into one line, 32 lanes | 12.2 | 21.7 | 41.1 | 79.6 |
| the same, 16 lanes | 8.2 | | | |
| the same, 8 lanes | 7.8 | | | |
| f16 adds, a lane a line of its own | 86.3 | | 377.9 | |
| plain stores into one line, 32 lanes | 1.8 | | 1.8 | |

WHAT IT SHOWS.
- THE ADD IS HELD BY THE SM, NOT BY THE WARP. One warp issues an add of a whole line every 12.2 cycles; two warps
  each every 21.7, four every 41.1, eight every 79.6. The SM takes one such instruction about every 10 to 12
  cycles whoever issues it, and more warps adding buy nothing.
- THE COST IS THE ADD BEING ATOMIC. The same line as a plain store is 1.8 cycles an instruction, and four warps
  each store one in the same 1.8: a store is 7 times cheaper for one warp, 23 times for four, and it scales.
- AN INSTRUCTION COSTS MOSTLY ITSELF, THEN ITS LINES: 32 lanes into one line 12.2, into 32 lines 86.3 -- some 9.8
  an instruction and 2.4 a line it touches. Fewer lanes into the one line cost less, 8.2 at 16 and 7.8 at 8;
  those two rows also carry a lane's predicate around each add, so what of their cost is the lanes' is not
  separated.
- [`drain_role`](drain_role.md)'s line from registers, 9.8 cycles a CTA by four warps, is this experiment's
  41.1 over four: the SM's rate, which one warp nearly reaches alone.

Record: `measure/lab/results/add_rate/sm_90a.json`.
