# `carry/carry_api.cuh` — the carry's host entries

Mirrors `csrc/rola/src/carry/carry_api.cuh`, the declarations the pybind translation unit binds
(`csrc/rola/rola_api.cpp`); the Python caller is `rola/ops/carry.py`.

<a id="carry-forward"></a>
## `carry_forward`

One launch over a call's windows. In: both sides' level rows and the values (the bundle, read where it lies), both
sides' lists (`(entries, at, union, count)` each, empty tensors for a side with no list level), the den pass's unit
bytes, and the carve flattened: widths, extents, channels, each side's list level (-1 for none) and operand levels,
and the arm key. Out: `num` and `check`. The refusals are `carry.md#refusals`.

<a id="carry"></a>
## `rola.ops.carry.carry`

The Python call builds the outputs and flattens the carve. The write side's records are the facts pass's 8-byte
`(slot, 0, c_v, c_m)` entries; the facts op hands them out as their slot column, so the call views the four columns
back. `arm_key` is the compiled body a carve selects: each side's warp tile of dense rows and its operand's level
count, the channel extent, and whether the publish stays in registers.

`carry_ledger_bind`, `carry_build_stamp`, `carry_census` and `carry_arms` are the instruments: the phase ledger
(`carry_kernel.md#phases`), the device stamp, and the binary's own arm list and census.
