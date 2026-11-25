# `common/atoms/tensor.cuh` — the hopper tensor step and a warpgroup's registers

Mirrors `csrc/fsla/src/common/atoms/tensor.cuh`, a concern of the baseline operations ([`../ops.md`](../ops.md)): one
PTX instruction of sm_90a each. A kernel that calls one compiles for sm_90a alone. No shipped kernel calls them: they
are the carry's ([`../../../architecture/hopper-carry.md`](../../../architecture/hopper-carry.md)), which is designed
and not built; the forms are the lab's, measured there ([`../../../lab/carry_token.md`](../../../lab/carry_token.md)), and the
lab's harness (`measure/lab/lab.cuh`) calls these atoms by its own names, so an instruction has one definition.
BLACKWELL'S FORM IS OWED: its tensor step is another instruction with its accumulators in tensor memory, and no
blackwell carry is designed ([`../../../architecture/hopper-carry.md`](../../../architecture/hopper-carry.md#blackwell)); the atom of that family is
added with that design, selected at the atom as R8 asks.

- <a id="operand"></a>`step_operand(a, lbo, sbo)`, `step_operand_lines(line, lbo, sbo)`: a step's shared-memory
  operand, unswizzled, as the instruction's 64-bit descriptor: the tile's first 16-byte line, and two offsets in
  lines. K-MAJOR (a line is one row's or column's K 0 to 7, a brick 8 such lines, its K 8 to 15 `lbo` lines on, the
  next brick `sbo` lines on) or M|N-MAJOR (a line is one K's 8 rows or columns, a brick's K 0 to 7 8 lines, its K 8
  to 15 `lbo` lines on, the next brick `sbo` lines on), as the step's form says.

- <a id="edges"></a>`steps_fence()`, `steps_commit()`, `steps_wait<InFlight>()`, `step_hold(r)`: a group of steps is
  issued behind a fence (the registers the group reads are written before it), committed, and waited for; a register
  a step wrote is read only behind the wait, and `step_hold` states that dependence where the compiler cannot see it
  (an accumulator carried across a loop of groups).

- <a id="steps"></a>`step_n16(d, a, b, add)`: `wgmma` m64n16k16, bf16 by bf16 into fp32, both operands K-major shared
  memory: `d` += A [64 x 16] B [16 x 16], or with `add` false `d` = A B, an accumulator's first step, which needs no
  register cleared. `step_n64(d0, d1, d2, d3, a, b, add)`: m64n64k16 with both operands M|N-major, its accumulator
  four 16-column sets end to end (set `s` is columns `16 s` on: the registers a 16-column step of that set writes).

- <a id="registers"></a>`registers_down<Count>()`, `registers_up<Count>()`: `setmaxnreg`, a warpgroup's registers a
  thread, 24 to 256 in eights; a raise waits for registers another warpgroup of the CTA released. Honoured only in a
  kernel whose `__launch_bounds__` give the count at entry, each role's whole body in its own branch
  ([`../../../lab/reg_trade.md`](../../../lab/reg_trade.md)).
