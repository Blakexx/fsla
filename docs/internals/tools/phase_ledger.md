# `tools/phase_ledger.py` — the kernel's phase clock, read on a cell

One carry launch (by default three, averaged) on a registry cell with `PhaseClock` bound
(`carry_ledger_bind`; `docs/internals/carry/carry_kernel.md#phase-ledger`), printed as cycles a
warp a window per phase — head, readout, fold, snapshot, edges, sweep — averaged over the CTAs and
warps, then the readout's and the fold's cycles per warp so an owner imbalance shows.

The launch carries a state plane out (the `fresh` arm, what `tools/probe_cells.py` measures)
unless `--state-arm null`. The first ledger runs passed no plane, so the kernel skipped its exit
sweep and the ledger never saw the phase that was a third of a sparse kernel's time (the scattered
2-byte stores, `carry_kernel.md#state-io`). An instrument that measures a different launch than
the A/B is not an instrument.

Measured work: the GPU lock exclusive (`tools/gpu_lock.py`). Under `ncu` it is the one launch to
profile with `--launches 1`.
