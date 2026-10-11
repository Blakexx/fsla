# `carry/carry.cu` and `carry/carry.cuh` — the carry's constants, operand block and host seam

Mirrors `csrc/rola/src/carry/carry.cuh` (the constants and the operand block every component reads) and
`csrc/rola/src/carry/carry.cu` (the kernel boundary's refusals, the block's derivation from the carve, the dispatch
over the built arms). The body has its own page, `carry_kernel.md`; the components `parts.md`; an arm's plan
`plan.md`.

The carry is the inter-window term: each CTA owns a BOX of leaves (the carve's, `docs/internals/engine/rules.md#carve`,
`docs/internals/engine/carve.md`) and walks a call's windows in order. Every window it reads its state out to the
window's readers (the readout), then folds the window's writers into it (the fold). The state never leaves the
registers inside a call.

<a id="constants"></a>
## The constants

- `kWarps = 8`, `kThreads = 256`: one CTA an SM, its register file holding the box's state (`kStateValues` = 256 lanes
  x 128 fp32 registers = 32,768 values; at 32 channels a box is 1,024 leaves).
- `kWindow = 512`: the window the intra and the facts pass share, so the inter/intra split is one recurrence.
- `kPassSlots = 128`: a pass is 128 union slots, the pools' row count; `kListAt = 8` pass starts a list (four passes'
  starts, then the end repeated); `kListCap = kWindow` entries a list.
- `kSliceBytes = 64`: a level's slice in a pool row, its box extent's digits at f16. The producer caps an extent at
  32 digits (`docs/internals/engine/carve.md`), so 64 bytes holds any slice; a u8 slice uses its first half.
- `kMaxLevels = 4`: the depth the operand block addresses.

<a id="sides"></a>
## A side of the carve

`Side` is one side's plan as the kernel addresses it. `list` is the level whose digits the facts pass lists (-1 for a
side with none: then every window token is a slot and every warp folds or reads every slot). `operand` holds the
levels whose slices form the MMA operand, in level order, each with its plane and element bytes (u8 for a sparse
level's rows, f16 for a dense write level's). A level's slice is its box extent's digits from the box's first digit:
`origin_of` decomposes the CTA's box index in level order, the channel group innermost.

<a id="lists"></a>
## The lists

`SideLists` is the facts pass's output at a side's list level (`docs/internals/facts/facts.md#lists-out`): per
(batch-head, window) and 32-digit group, the union's tokens (window-relative) and count; per digit, its entries
(write: 8-byte records, the slot then `c_v` and `c_m`; read: the slot) and its pass starts. A box's list digits lie
inside one 32-digit group, because the box's extent there is at most 32 and its first digit a multiple of it.

<a id="params"></a>
## The operand block

`CarryParams` reaches the kernel as a grid constant. The box is the carve's: `extents` a level, `channels`, and
`stride`, a level's step in the box index (the channel groups innermost). Every pointer is the bundle's own memory:
the rows and values at their strides (the row alignment law, `docs/internals/facts/facts.md`), the facts pass's
lists, the den pass's units (`unit`, an exponent byte a token). The outputs are `num` (packed s16 pairs at each
token's unit, added as u32: `docs/internals/carry/parts.md#drain`) and `check` (the skeleton's sum of the final
state, so the stubs' work is not dead code).

<a id="refusals"></a>
## The kernel boundary refuses

`carry_forward` takes the strict form and refuses everything else, naming the field:

- depth 1 to 4; every width a power of two from 16, every extent a power of two dividing it; the box's leaves times
  its channels exactly a CTA's state; the channel extent 32 or 64, dividing `d_v`.
- each side lists ONE level and forms its operand from all the others. A side with two sparse levels needs the facts
  pass's box list groups (the fused unit kernel, ruled 2026-10-10), so until those land the carry refuses it
  rather than drop a level's weights.
- a list level's box extent at most 32 (one union group); an operand slice a 4-byte copy at least and
  `kSliceBytes` at most.
- every tensor's dtype, contiguity and exact shape; the arm key's channels and factor counts agreeing with the carve.

The closed-world arch refusal (`check_arch_table`) is the first statement of every entry.

## Dispatch

The key is `rola.ops.carry.arm_key(carve)`; `arm_switch<CarryArmSet>` resolves it to the generated arm's launch or
refuses naming the field no built arm carries (`docs/internals/common/arm_switch.md`). The grid is
`(boxes, batch-heads)`. The phase ledger, when bound by `carry_ledger_bind`, must cover the grid.
`carry_build_stamp` reads four facts back off a kernel this binary compiled (`kWindow`, the capability digest,
`kStateValues`, `kPassSlots`), the device-side check a stale binary cannot pass.
