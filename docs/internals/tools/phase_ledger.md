# `tools/phase_ledger.py` — the kernel's phase clock, read on a cell

One carry launch (by default three, averaged, after a 500 ms warm-up) on a registry cell with `PhaseClock` bound
(`rola.ops.carry.ledger_bind`; `docs/internals/carry/carry_kernel.md#phase-ledger`), printed as cycles a
warp a window per phase (`rola.ops.carry.LEDGER_PHASES`, the kernel's order: `carry_kernel.md#phases`), averaged over
the CTAs and warps, then the readout's and the fold's cycles per warp so an owner imbalance shows.

The operands are `measure.subjects.carry_operands`, the same the timing session's `carry_forward` entry times. The
ledger has one row a CTA, a CTA a box of the cell's carve (`ctas(cell)`).

`--kernel den` runs the den pass instead (`rola.ops.den.ledger_bind`; `docs/internals/den/den_kernel.md#phases`) over
the facts pass's output on the cell's bundle, a CTA a tile of the head: each ROLE's phases printed on their own line
(`rola.ops.den.LEDGER_ROLES`: the write warps' and the den warps', each booking only its own phases), the per-warp rows
of the den's and the writes' phases by default.

Measured work: the GPU lock exclusive (`rola_devtools.locks.gpu`). Under `ncu` it is the one launch to
profile with `--launches 1`.

`--per-warp` names the phases whose per-warp rows are printed (default `readout,fold`). A single launch's totals move
by up to ~5% run to run; time is the probe's.

The clock is not free: bound, the carry runs ~16% longer (nl64k-alt-k4's first 32 windows on sm_86: 902 us against
1,045, `docs/lab/carry_window.md`), since each lap's clock read is a point ptxas does not schedule across. The ledger
attributes a launch's time to phases; the time itself is read with it unbound. Launches run before the ledger is bound, at least `--warmup` (default 1)
and for at least `--warmup-ms` (default 500): a fresh binary's first launch is never a reading, and a GPU just locked
reads ~11% slow for its first launches (nl64k-alt-k4 25,460 cycles a warp-window against 22,657, 2026-10-19). `--forced` names the plan's form each window takes where it fits
(`rola.ops.carry.forced_word`, `docs/internals/carry/schedule.md#planner`), so forms are read side by side.

<a id="budget"></a>
`--budget tools/budgets/<kernel>.json` judges each phase against the cycles a warp a window the design predicted for it
(the cell's `phase_cycles` in the budget, KERNEL_STANDARDS §32.2): red over `MISS` = 1.3 times its prediction, or with
none. The carry's predictions are owed by the window's design, so its check is red until they exist.

`--json PATH` also writes the reading: the cell, launches, warm-up, the launch's CTA-windows, every phase's cycles a
warp a window and the total, and every phase's per-warp row. `cta_windows(cell)` (CTAs times windows) is the one divisor
every per-unit number in the instruments uses (`tools/pipe_counters.py`, `tools/stall_census.py`).
