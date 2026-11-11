# `bulk_drain`

Mirrors `measure/lab/experiments/bulk_drain.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A CTA's fp32 output panel leaves shared memory for a global sum that other CTAs add into too. What does a token cost
by one asynchronous bulk reduction a panel (`cp.reduce.async.bulk`, fp32 add, shared to global), by one a token's
span, and by the warps' own reductions a value straight from registers?

THE FORM. A CTA an SM, 4 warps. A round writes a panel of 64 tokens × 64 channels fp32 (16 KB) and drains it: one
bulk add of the panel, waited before the next round; the same over two panels in turn with one drain in flight; one
bulk add a token's 256 bytes, each of 64 threads issuing its own; or no panel and a `red.global.add.f32` a value.
The panel's stores alone, with the round's barriers, are the base (one neighbour's word read a round keeps its stores
live). Every CTA adds into ONE global panel, the contended sum, or into its own; every sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a bulk add is one `UBLKRED.G.S.ADD.F32.RN` whatever its
size, issued by one thread; the panel is 8 `STS.128` a thread; the warps' own drain is 32
`REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` a thread a round.

UNMEASURED: no hopper card has run it. Its cycles a token by form, contended and not, are owed to its first run
(`bazel run //tools:lab -- pack bulk_drain --arch sm_90a`, then `carried` on the card, which runs it under its kill
timeout and writes its sm_90a record beside the other experiments').
