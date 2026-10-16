# `carry/carry.cu` and `carry/carry.cuh` — the carry's constants, operand block and host seam

Mirrors `csrc/rola/src/carry/carry.cuh` (the constants and the operand block every component reads) and
`csrc/rola/src/carry/carry.cu` (the kernel boundary's refusals, the block's derivation from the carve, the dispatch
over the built arms). The body has its own page, `carry_kernel.md`; the components `parts.md` and `fold.md`; an arm's
plan `plan.md`.

The carry is the inter-window term: each CTA owns a BOX of leaves (the carve's, `docs/internals/engine/rules.md#carve`,
`docs/internals/engine/carve.md`) and walks a call's windows in order. Every window it reads the state before the
window's writes out to the window's readers (the readout), then folds the window's writers into it (the fold). The
state never leaves the registers inside a call; the readout reads an f16 image of it.

<a id="constants"></a>
## The constants

- `kWarps = 8`, `kThreads = 256`: one CTA an SM, its register file holding the box's state.
- `kWindow = 512`: the window the intra and the facts pass share, so the inter/intra split is one recurrence.
- THE BOX, every card's: `kBoxLists = 32` lists (the write side's list level) x `kBoxDigits = 32` digits (the read
  side's) x `kBoxChannels = 32` channels, `kStateValues` = 32,768 values = 256 lanes x 128 fp32 registers.
- `kListBound = 64`, `kListAt = 8`: a list's boundaries, its entries before every 64th union slot (the last its
  length), so the carry chunks a union at any multiple of 64 the facts pass never sees, out in prefix form over the
  group's lists modulo 2^16 (`kPreRows = 33`: a row a list, then the group's) beside the group's exact totals;
  `kListCap = kWindow` entries a list.
- `kPlanWords = 16`, `kMaxSteps = 30`: a window's plan record for a box class, its plan word and a 16-bit step word a
  step (`schedule.md#plan`).
- `kSliceBytes = 64`: a level's slice in a pool row, the box's 32 digits at f16.
- `kMaxLevels = 4`: the depth the operand block addresses.

<a id="sides"></a>
## A side of the carve

`Side` is one side's plan as the kernel addresses it. A side with a sparse level lists it (`list`) and forms its rows
from its other levels (`operand`, in level order). A side with no sparse level (`list` -1) names two dense levels in
`operand`: its row level, then its coefficient level, whose digit is each slot's coefficient for a list or a digit
(`docs/internals/engine/carve.md#sides`). Each operand level carries its plane and element bytes. A level's slice is the
box's digits of it from the box's first digit: `origin_of` decomposes the CTA's box index in level order, the channel
group innermost.

<a id="lists"></a>
## The lists

`SideLists` is the facts pass's output at a side's list level in its final form, per (batch-head, window)
([`facts.md`](../facts/facts.md#lists-out)):

- `entries`: a list's 8-byte entries in slot order, `kListCap` a list, no pads. A read entry is its slot's pool row
  offset in the union in 16-byte chunks with its swizzle folded in (the first word's low half), its token (its high
  half), the coefficient (f16, the second word's low half) and the panel row's offset from the dummy rows, its swizzle
  folded in (its high half): a chunk from a multiple of 64 takes its pool and panel back by its first slot's offsets
  (`parts.md#readout`). A write entry is the slot, then `c_v` and `c_m`. A step's lanes past a chunk's entries are
  masked by the carry (a read lane's coefficient 0 on a dummy panel row, a write lane's row 0 with c = 0).
- `bounds` (int16, a row a list) and `totals` (int32, the group's): `chunk_runs` turns a chunk's slots into each list's
  run and its place, `side_counts` reads the totals.
- `union_tok` and `union_count` (int32): per 32-digit group each slot's token, window-relative, and the slots' extent;
  a GAP slot holds `kGap` (token 0 with bit 15 set), so a fill masks the token to the window and zero-fills the row.

The addresses both kernels compute live here, one copy: `kGap`, the panel's 64-byte row `kPanelRow` past 8 dummy rows
(`kPadBytes`, a pad's adds landing on its place's), `panel_row` (a slot pair a 128-byte line, each slot's halves at
the segments its pair's residue bits XOR) and `pool_chunk` (a pool row's 16-byte chunk swizzled by the slot's residue,
or its high bits within each four for a row of an odd number of 64-byte quarters); `parts.cuh`'s `chunk_at` is
`pool_chunk` at a compiled row.

<a id="params"></a>
## The operand block

`CarryParams` reaches the kernel as a grid constant. The box is the carve's: `extents` a level, `channels`, and
`stride`, a level's step in the box index (the channel groups innermost). Every pointer is the bundle's own memory:
the rows and values at their strides (the row alignment law, `docs/internals/facts/facts.md`), the facts pass's
lists, the den pass's units (`unit`, an exponent byte a token, a window wide so a window's copy is aligned). The
outputs are `num` (packed s16 pairs at each token's unit, added as u32: `parts.md#drain`) and `check` (the final
state's sum). `plans` are the plan pass's records (`schedule.md#pass`), a window's for each box class.
`readout_f16` is the readout's accumulation (`carry_api.md#carry-readout-accumulate`); `forced` a plan form every
window takes where it fits, -1 for none (`schedule.md#planner`). A bound capture copies the block and its grid out before the
launch (`CarryCapture`, `carry_api.md#carry-capture-bind`).

<a id="refusals"></a>
## The kernel boundary refuses

`carry_forward` takes the strict form and refuses everything else, naming the field:

- depth 1 to 4; every width a power of two from 16, every extent a power of two dividing it; the box's leaves times
  its channels exactly a CTA's state; the channel extent 32, dividing `d_v`.
- a side lists at most one level and forms its operand from all the others; a list level spans the box's 32 lists or
  digits; a side with no list level names exactly a row level and a coefficient level; an operand slice is a 4-byte
  copy at least and `kSliceBytes` at most.
- every tensor's dtype, contiguity and exact shape (the lists' five, `unit` `[BH, windows x W]`); the arm key's channels
  and factor counts agreeing with the carve; the readout's accumulation fp32 (0) or fp16 (1); a forced form -1 or a
  mode (0 to 2) with two chunk fields in 14 bits; a window's token stride under 2^31 / `kWindow` (a window's offsets
  are 32-bit).

The closed-world arch refusal (`check_arch_table`) is the first statement of every entry.

## Dispatch

The key is `rola.ops.carry.arm_key(carve)`; `arm_switch<CarryArmSet>` resolves it to the generated arm's launch or
refuses naming the field no built arm carries (`docs/internals/common/arm_switch.md`). The grid is
`(boxes, batch-heads)`. The phase ledger, when bound by `carry_ledger_bind`, must cover the grid.
`carry_build_stamp` reads four facts back off a kernel this binary compiled (`kWindow`, the capability digest,
`kStateValues`, `kListBound`), the device-side check a stale binary cannot pass.
