# `carry/carry_kernel.cuh` — the carry kernel

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`: the window loop over the components (`parts.md`) for one arm's plan
(`plan.md`). The design and its lines were ruled 2026-10-10.

<a id="kernel"></a>
## The kernel

One CTA a box and batch-head, 8 warps, one CTA an SM. The fold's fp32 state (`kStateRegs` = 128 registers a lane) is
the C fragments of the fold's MMAs and stays in registers for the whole call; the readout's f16 copy of it
(`kReadRegs` = 64) is the A operand of the readout's.

A window, in sm_86's phase order (Blake, 2026-10-10: shared memory time-shared by phase; sm_80 runs the readout and
the fold side by side):

1. **Lists in**: both sides' union tokens and counts, and each side's box lists' pass starts (`parts.md#lists-in`).
2. **Units in**: the window's tokens' unit bytes, the den pass's (`docs/internals/carry/parts.md#drain`).
3. **Publish**: the state to the readout's operand, through shared memory when the two plans' warps own different
   leaves (`parts.md#publish`).
4. **Readout passes**: per 128 read-union slots, the pool filled with each slot's read operand slices
   (`parts.md#fill`), the warps' read steps adding into the panel (`parts.md#readout-pass`), the panel drained to
   `num` (`parts.md#drain`). The pool is two slots by `cp.async` groups: pass q + 1 fills while pass q steps.
5. **Fold units**: the write side's slots in units of U passes (`plan.md#units`), each unit's slot filled with its
   slots' write operand slices and values, the warps' fold steps into the state (`fold.md#unit`). Two unit slots ring,
   each landed on its `mbarrier` (one `cp.async.mbarrier.arrive.noinc` a thread a fill); a slot's parity is its use
   count across the call's windows, so the ring runs on through window boundaries.

Each component's state, as built: the fold steps a listed side with one operand level (`fold.md`), and `fold_stub`
stands in for its other paths (`fold.md#stub`); the readout pass is a stub that reads what the real step reads and adds
one panel word an entry (`parts.md#readout-pass`); the publish converts in its real arithmetic through a stub shared
layout (`parts.md#publish`). The state's final sum goes to `check`.

<a id="phases"></a>
## The phase ledger's phases

`CarryPhase` is the ledger's column order: `lists`, `units`, `publish`, `readout` (a pass's fill wait and steps),
`drain`, `fill` (the fold's units, its fills' issue and a unit's landing wait), `fold` (a unit's steps and barrier). A lap books the cycles since the last into its phase, so the columns sum to
the warp's whole run. `rola.ops.carry.LEDGER_PHASES` names the same order.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` (`docs/internals/common/phase_clock.md`) keeps each warp's accumulators in shared memory, lane 0 laps,
and `publish` adds them into the bound ledger `[ctas][warps][phases]` at the end. Unbound, the clock is gated off
and its laps issue nothing.
