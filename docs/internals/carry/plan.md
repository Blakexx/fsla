# `carry/plan.cuh` — one arm's plan

Mirrors `csrc/rola/src/carry/plan.cuh`.

<a id="plan"></a>
## The plan

An arm's key (`tools/gen_shards.py`'s `ARM_KEY`) fixes what compiles: each side's warp tile of dense rows and its
operand's level count, the channel extent, the publish route. The plan derives the rest:

- the box's leaves (`kStateValues` over the channels: 1,024 at 32) and a warp's share (`kWarpLeaves`, 128);
- each side's tile, `kFoldLists x kFoldRows` and `kReadLists x kReadRows` (lists the list level's digits, rows the
  operand's leaves);
- the registers: `kStateRegs` fp32 for the fold, `kReadRegs` packed f16 pairs for the readout;
- the pool rows: a write row is the fold's operand slices (`kSliceBytes` a level) then the values at the CTA's
  channels, a read row the readout's slices; `pitch_of` rounds a row to an odd multiple of 32 bytes so eight
  consecutive rows' 16-byte chunk meets eight bank groups.

<a id="units"></a>
## The fold's units and its two slots

The fold steps over UNITS of facts passes (`docs/internals/carry/fold.md#unit`). `unit_passes` picks the largest U of
4, 2, 1 whose two slots fit the phase region with slot 0 clear of the readout's pool and panel: U = 2 at the even box
(two 256 x 160 B slots, 80 KB, slot 0 past the readout's 32 KB), U = 1 at the dense-dense plan (224-byte rows). A
list's pads then cost at most one partial step a unit. Measured on nl64k-alt-k4, the 128-slot passes cut a list into
1.95 steps a window against 1.00 at U = 2.

## The shared memory

Persistent tenants first: both sides' union tokens (1 KB each) and counts, each side's box lists' pass starts, the
window's units, the warps' clocks and the write slots' barriers. Then ONE phase region the publish, the readout and
the fold take in turn (sm_86, Blake 2026-10-10): the publish's f16 state (64 KB at the even box, none when it stays in
registers), the readout's two pool slots and its panel (`kPassSlots` x `kPanelWords` u32), the fold's two unit
slots and their heads. The fold's share binds at both shipped plans: 88 KB at the even box (`[SD, DS]`, 32 x 32: two
40 KB unit slots, slot 0 past the readout's 32 KB, and the heads), 78 KB at the dense-dense plan (two f16 operand
slices; slot 0 past the readout's 48 KB). Every size is fixed by the plan, never by the data (R22).
