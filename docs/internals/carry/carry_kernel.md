# `carry/carry_kernel.cuh` — the carry kernel

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`: the window loop over the components (`parts.md`, `fold.md`) for one
arm's plan (`plan.md`). The form was ruled 2026-10-12 (the hand-off form): readout-first windows, the hand-off
by barriers, the image in parts on a card that cannot hold it whole.

<a id="kernel"></a>
## The kernel

One CTA a box and batch-head, 8 warps, one CTA an SM. The fold's fp32 state (`kStateRegs` = 128 registers a lane) is
the C fragments of the fold's MMAs and stays in registers for the whole call. A window:

1. **The maxima**: each warp posts its per-digit maxima of the state before the window's writes (`parts.md#maxima`).
2. **The passes**, per readout pass (one at sparse cells): the next fill into the readout ring's other slot (the next
   pass, or after the last the fold's first unit), the pass's slot waited; then per part of the image a barrier, every
   warp's stores of that part (`parts.md#store`), a barrier, the readout of the part's digits (`parts.md#readout`);
   then a barrier, after the last pass the fold ring's image units issued, and the drain (`parts.md#drain`).
3. **The next window's inputs** staged (`#staging`).
4. **The fold**, unit by unit (`fold.md#unit`): unit 0 in the readout ring, the rest in the fold ring over the image
   and the panel; each unit waited, folded, and closed by a barrier that frees its slot. Unit 0's barrier also frees
   the panel (the drain is done) and the readout ring's slot, so the panel's units and the next window's first pass
   are issued there. Where a unit lay in the panel, the panel is zeroed before the next window's adds.

Barriers a window on sm_86's even box at one pass: 2 a part, 1 before the drain, 1 a unit: 5 + U.

Each part as built: the hand-off, the fills, the staging, the ring and the drain are their components'. The readout's
tiles and the fold's steps issue their MMAs in their real counts; what each component still owes is named in its
section (`parts.md#readout`, `fold.md#unit`, `parts.md#maxima`). The census on sm_86: 254 registers, no local memory;
the SASS gate is clean on both arms and both cards.

<a id="staging"></a>
## The staging

Every async copy lands on an mbarrier, none on a commit group: `cp.async.mbarrier.arrive.noinc` makes a barrier track
all of a thread's earlier copies, so each barrier waits only on what it needs. The barriers: the readout ring's two
(`f`, the next fill to consume, picks the slot; a bit of `ring_par` each slot's phase), the fold ring's (`fold_par`),
the staging barrier (`stage_window`: both sides' lists, the facts, the warp's read heads; one phase a window) and the
fold heads' (one phase a window). A window's inputs are staged after the window before it is read out, into the other
parity, so nothing staged overwrites what is still read.

<a id="phases"></a>
## The phase ledger's phases

`CarryPhase` is the ledger's column order: `landed` (a pass's wait), `maxima`, `store`, `readout`, `drain`, `fill` (a
fold unit's wait) and `fold` (a unit's steps and barrier). A lap books the cycles since the last into its phase, so the
columns sum to the warp's whole run. `rola.ops.carry.LEDGER_PHASES` names the same order.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` (`docs/internals/common/phase_clock.md`) keeps each warp's accumulators in shared memory, lane 0 laps,
and `publish` adds them into the bound ledger `[ctas][warps][phases]` at the end. Unbound, the clock is gated off
and its laps issue nothing.
