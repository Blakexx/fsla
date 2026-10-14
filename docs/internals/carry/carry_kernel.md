# `carry/carry_kernel.cuh` — the carry kernel

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`: the window loop over the components (`parts.md`, `fold.md`) under each
window's plan (`schedule.md`) for one arm's plan (`plan.md`). The form was ruled 2026-10-14: the carry is one planned
pipeline -- shared memory holds what is active and what lands next, sized by each window's actual counts, the planner
choosing chunk sizes the card's area holds.

<a id="kernel"></a>
## The kernel

One CTA a box and batch-head, 8 warps, one CTA an SM, after the plan pass (`schedule.md#pass`). The fold's fp32 state
(`kStateRegs` = 128 registers a lane) is the C fragments of the fold's MMAs and stays in registers for the whole call.
Window 0's inputs and plan are staged and its first read chunk filled before the loop. A window:

1. **The maxima**: each warp posts its per-digit maxima of the state before the window's writes
   (`parts.md#maxima`), and waits for every warp's -- so every warp is past the window before, its buffers and the
   other parity's tenants free. Then the next window's inputs are staged, and the fold's first buffer filled where it
   lands beside the readout.
2. **The readout** in the plan's form (`schedule.md#forms`): IMAGE -- every batch stored, one barrier, then each chunk:
   the chunk before drained once every warp has read it, the next chunk into its buffer once that drain is everywhere
   done, the chunk's tiles over every batch. UNION -- batches 0 and 1 stored into the ring, then each batch: its stores
   waited, its tiles over the resident union, its slot freed and, two batches on, stored again; then the drain. TILED
   -- the image form's chunk loop with the ring's batches stored and read for each chunk (on one read buffer: the chunk
   read, drained, then the next filled). Then the readout's release.
3. **The next window's plan**, landed with its inputs, and its first read chunk filled now where its buffer ends below
   every fold buffer (after the release).
4. **The fold**, chunk by chunk (`fold.md#chunk`): a buffer's fill waits for its last readers, and over the readout's
   zone for the release; each chunk's landing waited, its steps, its buffer freed. A warp folds as soon as its own
   stores are done -- its registers are the state -- so warps flow from their reads into their folds. Then the next
   window's first read chunk, if it waited for the fold.

<a id="staging"></a>
## The staging

Every async copy lands on an mbarrier, none on a commit group: `cp.async.mbarrier.arrive.noinc` makes a barrier track
all of a thread's earlier copies. A window's inputs (`stage_window`: both sides' lists, the facts and the window's plan
for the CTA's box class) are staged into the other parity right after the maxima's barrier, a window ahead.

<a id="barriers"></a>
## The barriers

`Arrivals` keeps a bit a barrier (`plan.md#barriers`), flipped at each of the thread's arrivals. Every thread arrives
at every phase of every barrier, and every wait is for the phase of the thread's own last arrival, so the bit names its
parity; a barrier never arrived at passes (its phase before the first). Waiting on one's own last arrival never falls
two phases behind: the next phase needs the waiter's own arrival. A buffer's refill waits for the phase its last
readers complete (a read buffer's drain, a fold buffer's chunk, an image slot's batch), so a region is rewritten only
once every warp is done with it, with no barrier over the whole CTA except the maxima's and a form's stores.

<a id="budget"></a>
## The budget

`tools/budgets/carry.json` (KERNEL_STANDARDS §29): each component's instructions a CTA-window on the gating cell
nl64k-alt-k4, its price a trip times its trips, the trips at each warp's maximum from the facts pass's lists and the
plans' model (`tests/oracle/carry_plan.py`). A warp-window there: a union of ~216 slots and ~256 entries a side, the
union form, a resident fold; 7 readout tiles over 4 digits, 4 eight-token fold steps over 4 lists.

| component | a trip's price (sm_86 SASS) | trips a warp-window | design a warp-window |
|---|---|---|---|
| readout (`read_chunk`) | a tile ~40 (4 MMAs, 3 shared loads, 4 shuffles, 4 shared adds, ~12 float, ~10 address); a digit ~16; an item ~10 | 7 tiles, 4 digits, 4 items | 384 |
| fold (`fold_chunk`) | an 8-token step ~30 (8 MMAs, 3 loads, 4 shuffles, 4 HMUL2); a list ~12 and its rescale 32 FMUL | 4 steps, 4 lists | 296 |
| units (`fold_units`) | once a window | 1 | 30 |
| stores (`store_batches`) | a batch: 4 lists x (4 x 5 for two values' scale, pack, store, address + 4) + 5 | 4 batches | 400 |
| maxima (`post_maxima`) | a (list, half-tile): 8 FMNMX with abs, an exponent ~4, its max ~2; 16 of them + 20 | 1 | 245 |
| drain | a panel word ~6 (load, token, address, predicated add); two slots' loop ~6 | 13.5 words a thread | 120 |
| read fill (`read_fill`) | a 16-byte copy ~13; an entry run ~10 | 6.9 copies a thread, 4 runs | 130 |
| fold fill (`fold_fill`) | the same | 6.75 copies a thread, 4 runs | 128 |
| staging (`stage_window`) | its ~240 copies over 256 threads, and its tests | 1 | 40 |
| barriers (`Arrivals`) | an arrival ~3, a satisfied wait ~3 | ~27 | 81 |
| glue (the kernel's body) | the plan's fields and the windows' statics read once, loop control | 1 | 150 |

The design is ~2,020 instructions a warp-window: ~4.0K cycles of issue a window over 8 warps on 4 schedulers, ~0.55 ms a
call at one head. Measured (2026-10-14, carry-k5 839dadb5) 3,884 (31,069 a CTA-window), every component over its line
but the stores (1.09x): the fills 2.3-3.0x (each copy re-derives its token, its source and its swizzle; each run its
list's row), the glue 2.6x and the barriers' own instructions 2.8x (fields re-read at each use), the drain 2.3x, the
readout 1.7x, the fold 1.4x, the maxima 1.3x -- each class (b), the mechanism right and its form above its floor.
Beside them the barriers' spinning (~280 a warp-window, the waits for other warps) and ~24.4K cycles a warp-window
against the instructions' ~7.8K: the warps' latency-bound chains.

<a id="phases"></a>
## The phase ledger's phases

`CarryPhase` is the ledger's column order: `landed` (a read chunk's wait), `maxima`, `store`, `readout`, `drain`,
`fill` (a fold chunk's wait), `fold` and `plan`. A lap books the cycles since the last into its phase, so the columns
sum to the warp's whole run. `rola.ops.carry.LEDGER_PHASES` names the same order.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` (`docs/internals/common/phase_clock.md`) keeps each warp's accumulators in shared memory, lane 0 laps,
and `publish` adds them into the bound ledger `[ctas][warps][phases]` at the end. Unbound, the clock is gated off
and its laps issue nothing.
