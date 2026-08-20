# `csrc/rola/src/facts/facts.cu` — the facts layer's dispatch and torch entries

Adopted verbatim from the deleted chunk consumer's dispatch TU (`chunk.cu`, K31
deletion batch; baseline = tag `baseline/pre-k31`). Three entries cross the
extension boundary — `facts` (the union table and, on a stateful call, the
atom-grain write bitmap: one pass, both row sets), `block_bits` (the liveness
epilogue's reduction over a table this call already built) and `arms` (the
built `(C, BC, D, B)` dispatch vocabulary with the level plan's quantities).

<a id="closed-world"></a>**`check_arch_table` — the closed-world R4 runtime
half.** Its compile-time half is `arch_caps.cuh`'s
`static_assert(tabulated(kArch))`; this is the refusal at first launch on a
device the table has no row for, or a row the driver disagrees with. Every
launching entry runs it first — `tests/unit/test_closed_world_arch_gate.py`
gates that per entry.

<a id="l-bound"></a>**The bound on `L`** is retained from the consumer era
(`kMaxGatherL`): the widest gathered row the family ever addressed was the fp32
output at `d_v * 4` bytes, and the facts tables were built for exactly the
launches that bound admitted. K31 R2 restates it from its own geometry.
