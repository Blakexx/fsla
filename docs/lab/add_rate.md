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
Every row's words are checked exactly.

MEASURED: not yet run.
