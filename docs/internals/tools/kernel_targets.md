# `tools/kernel_targets.py` — the instruments' kernels

What a per-launch instrument needs to read one kernel family on a central cell, in one table (`TARGETS`): the
profiler's kernel filter (`symbol`, which `ncu -k regex:` matches against the plain function name), the provider arm
whose one untimed launch it captures (`arm`, `measure.provider.oneshot_argv`), the cubin member and the sources the
region ledger attributes SASS lines against (`member`, `sources`), the family's budget (`tools/budgets/<family>.json`)
and the budget's unit.

| family | arm | unit |
|---|---|---|
| `carry` | `carry_forward` | CTA-windows (the carve's boxes x windows) |
| `den` | `den_pass` | CTA-windows (the tiles x windows) |
| `intra` | `intra_forward` | warp tile pairs (a warp's 16 readers against a writer tile at or before theirs) |
| `facts` | `facts_pass` | windows |

`units(target, cell)` is the launch's count of the unit from the cell's record alone (one batch row and write head, as
the central cells draw); `function(target, cell)` the instantiation the cell launches, as the region ledger's regex
over the cubin's mangled names: the den's form by the cell's declared sparsity, the intra's depth and the card's
accumulation (`rola.ops.carry.readout_accumulate`).

`tools/stall_census.py`, `tools/pipe_counters.py` and `tools/phase_ledger.py` take `--kernel FAMILY` from this table,
the carry by default; `declare.py` declares the den's and the intra's on the cells of 16K tokens and more
(`census-den`, `counters-den`, `phases-den`, `census-intra`, `counters-intra`). `python tools/kernel_targets.py CELL`
prints every family's target on a cell. The pipe timeline reads the carry only (its calibration plateau is the carry's).
