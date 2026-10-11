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

## The shared memory

Persistent tenants first: both sides' union tokens (1 KB each) and counts, each side's box lists' pass starts, the
window's units, the warps' clocks and the write slots' barriers. Then ONE phase region the publish, the readout and
the fold take in turn (sm_86, Blake 2026-10-10): the publish's f16 state (64 KB at the even box, none when it stays in
registers), the readout's two pool slots and its panel (`kPassSlots` x `kPanelWords` u32), the fold's three pool
slots. At the even box (`[SD, DS]`, 32 x 32) the fold's 60 KB binds; at the dense-dense plan (two f16 operand
slices) the fold's 84 KB. Every size is fixed by the plan, never by the data (R22).
