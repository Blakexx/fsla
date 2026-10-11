# `carry/parts.cuh` — the carry's components

Mirrors `csrc/rola/src/carry/parts.cuh`. Each component is at its real signature, tenants and copies; the MMAs and
conversions are stubs (`carry_kernel.md#kernel`).

<a id="place"></a>
## The CTA's place

`place_of` is the CTA's batch-head (`blockIdx.y`) and its channel group (the box index's innermost digit);
`origin_of` a level's first box digit. `slices_of` forms each operand level's slice base once: the plane at the CTA's
batch and head, displaced to the box's first digit. `window_of` is a side's window: its union group (the 32-digit
group holding the box's list digits) and the box's first list digit, or every token a slot.

<a id="lists-in"></a>
## Lists in

A side's union tokens are copied whole (512 slots, 1 KB) with `cp.async`, its box lists' pass starts one 16-byte copy
a list, its count by one thread. Copying the whole union is a fixed 1 KB a side a window; the count-bounded copy (the
count a window ahead) is this component's budget work.

<a id="fill"></a>
## The fill

A pass's slots, each its token's operand slices into its pool row: a slice's 16-byte runs (8 or 4 bytes for a slice
under 16) over the CTA's threads, then on the write side the values' runs. The token is the union's slot (or the
slot itself where the side has no list level). The pool row's swizzle and the quad-a-token copy pattern
(`docs/internals/owner/owner_fill.md`) are this component's budget work.

<a id="publish"></a>
## The publish

The fold's fp32 state to f16 pairs, the readout's operand. When the two plans' warps own different leaves (the even
box: the fold's warp owns 4 write-list digits x 32 rows, the readout's 4 read-list digits x 32 rows) the pairs go
through shared memory; when they own the same leaves (the dense-dense plan), they stay in registers. The stub's
shared layout is lane-contiguous; the real one is the readout's fragment order.

<a id="readout-pass"></a>
## A readout pass

A warp walks its lists' entries in the pass (`[at[q], at[q + 1])`, slots residue-ordered by the facts pass) or, with
no read list, every slot. The real step forms the read operand from the slot's slices times the entry's read
coefficient and multiplies the f16 state, its outputs times the token's unit to packed s16 panel adds. The stub reads
one word of the row and adds one panel word an entry.

<a id="drain"></a>
## The drain

A pass's panel to `num`: each slot's `kPanelWords` words added as u32 atomics at its token (packed s16 pairs at the
token's unit, decoded once each final fits s16: the den pass ruling, 2026-10-06), the panel cleared for the
next pass. CTAs join only here (R22: independent units, joined by atomics).

<a id="fold-pass"></a>
## A fold pass

A warp walks its lists' records in the pass or, with no write list, every slot. The real step forms the write
operand from the slot's slices times the record's coefficients (`c_v`, `c_m`) and accumulates the values into the
fp32 state. The stub reads the record and one word of the row's values and adds their product into the state.
