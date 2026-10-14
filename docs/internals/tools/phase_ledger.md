# `tools/phase_ledger.py` — the kernel's phase clock, read on a cell

One carry launch (by default three, averaged, after one warm-up) on a registry cell with `PhaseClock` bound
(`rola.ops.carry.ledger_bind`; `docs/internals/carry/carry_kernel.md#phase-ledger`), printed as cycles a
warp a window per phase (`rola.ops.carry.LEDGER_PHASES`, the kernel's order: `carry_kernel.md#phases`), averaged over
the CTAs and warps, then the readout's and the fold's cycles per warp so an owner imbalance shows.

The operands are `measure.subjects.carry_operands`, the same the timing session's `carry_forward` entry times. The
ledger has one row a CTA, a CTA a box of the cell's carve (`ctas(cell)`).

Measured work: the GPU lock exclusive (`rola_devtools.locks.gpu`). Under `ncu` it is the one launch to
profile with `--launches 1`.

`--per-warp` names the phases whose per-warp rows are printed (default `readout,fold`). A single launch's totals move
by up to ~5% run to run; time is the probe's. `--warmup` (default 1) runs launches before the ledger is bound: a fresh
binary's first launch is never a reading. `--prefer` names the plan's form each window takes where it builds
(`rola.ops.carry.CANDIDATES`, `docs/internals/carry/schedule.md#planner`), so forms are read side by side.

`--json PATH` also writes the reading: the cell, launches, warm-up, the launch's CTA-windows, every phase's cycles a
warp a window and the total, and every phase's per-warp row. `cta_windows(cell)` (CTAs times windows) is the one divisor
every per-unit number in the instruments uses (`tools/pipe_counters.py`, `tools/stall_census.py`).
