# `carry/params.cuh` — the parameter block

Mirrors `csrc/rola/src/carry/params.cuh`.

One `struct` by value: the addressing block the host derived, the operands, the two runtime
dials the body reads today, and the two problem scalars. It reaches the kernel as a GRID
CONSTANT, which is what lets every inlined stage read its fields through a reference without
forcing a per-thread local copy — see [`carry.md#launch`](carry.md#launch).

Each field's class in the primitive ledger is tabulated in
[`carry_kernel.md#ledger`](carry_kernel.md#ledger). The rule that table enforces is that a
class-4 or class-5 quantity — an architecture's constant, or a quotient of others — never
appears here as a knob.

The two grains, `readout_grain_bits` and `fold_grain_bits`, are the per-side dials: each
selects the compiled body its phase runs. Where the switch on them sits, and why that
placement is additive rather than multiplicative, is
[`carry_kernel.md#forms`](carry_kernel.md#forms). The form dial that once stood beside
them is gone with the union body (`DELETIONS.md`).
