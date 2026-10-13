# `carry/carry_api.cuh` — the carry's host entries

Mirrors `csrc/rola/src/carry/carry_api.cuh`, the declarations the operator registration unit declares on the stable
ABI (`csrc/rola/rola_api.cpp`); the Python caller is `rola/ops/carry.py`.

<a id="carry-forward"></a>
## `carry_forward`

One launch over a call's windows. In: both sides' level rows and the values (the bundle, read where it lies), both
sides' lists in the facts pass's final form (`(entries, at, heads, union, count)` each, empty tensors for a side with no
list level, `carry.md#lists`), the den pass's unit bytes a window wide, and the carve flattened: widths, extents,
channels, each side's list level (-1 for none) and operand levels, and the arm key. Out: `num` and `check`. The
refusals are `carry.md#refusals`.

<a id="carry"></a>
## `rola.ops.carry.carry`

The Python call launches over the bundle's rows and values, the facts pass's lists and exponents and the den pass's
units, the facts pass run in this arm's `form` (its readout pass, its fold unit, its read pool row, read off the
binary's census for the device's card), so the lists' passes and offsets are the arm's. `arm_key` is the compiled body
a carve selects: each side's dense rows a warp and its operand's level count, the channel extent and the publish
field.

`carry_ledger_bind`, `carry_build_stamp`, `carry_census` and `carry_arms` are the instruments: the phase ledger
(`carry_kernel.md#phases`), the device stamp, and the binary's own arm list and census.
