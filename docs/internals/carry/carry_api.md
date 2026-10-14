# `carry/carry_api.cuh` — the carry's host entries

Mirrors `csrc/rola/src/carry/carry_api.cuh`, the declarations the operator registration unit declares on the stable
ABI (`csrc/rola/rola_api.cpp`); the Python caller is `rola/ops/carry.py`.

<a id="carry-forward"></a>
## `carry_forward`

One launch over a call's windows. In: both sides' level rows and the values (the bundle, read where it lies), both
sides' lists in the facts pass's final form (`(entries, at, union, count)` each, empty tensors for a side with no
list level, `carry.md#lists`), the den pass's unit bytes a window wide, and the carve flattened: widths, extents,
channels, each side's list level (-1 for none) and operand levels, the arm key, the readout's accumulation (0
fp32, 1 fp16), the readout candidate every window's plan takes first where it fits (-1 none, else 0 to 4 in the
plan's order, `schedule.md#planner`), and optionally the caller's plans tensor (`[BH, windows, box classes, 48]` u8,
else the call's own). Out: `num`, `check` and the plans. The refusals are `carry.md#refusals`.

<a id="carry-readout-accumulate"></a>
## `carry_readout_accumulate`

The readout's accumulation on a device: fp16 (1) where its fp32-accumulate MMA is slower than its fp16-accumulate one by
its rates -- its product's row by its name, else its architecture's
([`../common/arch_runtime.md#mma-quanta`](../common/arch_runtime.md#mma-quanta)) -- else fp32 (0): a GeForce card's fp32
accumulate runs at half its fp16 rate, a professional or datacenter part's at its full rate (ruled 2026-10-07). The two
forms' numerics differ, GeForce's the coarser; both are gated per slot (`tests/oracle/test_carry_readout.py`).


<a id="carry"></a>
## `rola.ops.carry.carry`

The Python call launches over the bundle's rows and values, the facts pass's lists and exponents and the den pass's
units, its readout in the device's accumulation (`readout_accumulate`) or the one a gate names, each window's plan its
own unless a gate or a measurement names a form (`prefer`, one of `CANDIDATES`), the facts pass run with this arm's
read pool row (`form`, read off the binary's census for the device's card). `plans_out` allocates the plans a gate
reads (`PLAN_BYTES` a window, `tests/oracle/test_carry_plan.py`). `arm_key` is the compiled body
a carve selects: each side's dense rows a warp and its operand's level count, the channel extent and the publish
field.

`carry_ledger_bind`, `carry_build_stamp`, `carry_census` and `carry_arms` are the instruments: the phase ledger
(`carry_kernel.md#phases`), the device stamp, and the binary's own arm list and census.
