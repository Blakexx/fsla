# `carry/carry_kernel.cuh` — the carry kernel

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`: the window loop over the components (`parts.md`, `fold.md`) under each
window's plan record (`schedule.md`) for one arm's plan (`plan.md`). The form was ruled 2026-10-14: the carry is one planned
pipeline -- shared memory holds what is active and what lands next, sized by each window's actual counts, the planner
choosing chunk sizes the card's area holds.

<a id="kernel"></a>
## The kernel

One CTA a box and batch-head, 8 warps, one CTA an SM, after the plan pass (`schedule.md#pass`). The fold's fp32 state
(`kStateRegs` = 128 registers a lane) is the C fragments of the fold's MMAs and stays in registers for the whole call.
Window 0's inputs and record are staged and its first entry filled before the loop. A window:

1. **The maxima**: each warp's digit maxima posted (`parts.md#maxima`), every warp's joined, each digit's exponent,
   recorded by every warp (the same values) for its stores and the readers.
2. **The store**: the next window's inputs staged; the image stored whole where the window's mode is whole.
3. **Its steps** (`schedule.md#rule`), each by its step word (`schedule.md#steps`): the rendezvous where the retire is
   followed by a drain or a fill; the retired read chunk's panel drained to `num`, and the rendezvous where a fill
   reuses its place; the fills -- entry `t` where it was not filled ahead, then the next entry where the word says (the
   next window's first once that window's inputs have landed); the units stepped at the window's first fold chunk, or
   at its last step where it folds nothing; then the work: a READ chunk's tiles over every digit from the whole image,
   or from the halves it stores itself (a rendezvous before each half's reuse); a FOLD chunk's steps (`fold.md#chunk`).

The work is an exclusive chain over the entry's side and the image mode (R9's case marker, a trapping last arm), not
`uniform_switch`: the primitive's independent tests made a body 14% slower at nl64k-alt-k4 (31.7K against 27.9K
cycles a warp-window, 2026-10-14). A warp moves on as soon as its own work is done: a FOLD needs only the warp's own
stores (its registers are the state), so warps flow from their reads into their folds; only a rendezvous waits for the
other warps.

THE REGISTERS (255 a thread, the state 128 of them): what lives across a window's steps is the window, the flow word
(`#barriers`) and the step's place; everything a step needs is re-read at its use -- the record's words, the counts, a
side's statics (volatile), and the offsets formed from the window (`parts.md#window`: its first token and parity
opaque, so a window-invariant address is formed at each use rather than hoisted and held across the readout, where the
peak is).

<a id="staging"></a>
## The staging

Every async copy lands on an mbarrier, none on a commit group: `cp.async.mbarrier.arrive.noinc` makes a barrier track
all of a thread's earlier copies. A window's inputs (`stage_window`: both sides' lists, the facts and the window's plan
record for the CTA's box class) are staged into the other parity at the window's store, after the maxima retire the
window before, a window ahead.

<a id="barriers"></a>
## The barriers

Every CTA-wide rendezvous -- a retire, a drain's reuse, the maxima, the store before a whole image's first read, the
halves' stores and reuses -- is the hardware barrier (`ops::rendezvous`): every warp runs the same steps, so a barrier
where the consumer starts means every warp is done with what came before. On sm_80 and sm_86 an mbarrier is software
-- an arrive fences the warp's outstanding atomics, the waiters poll -- about twice the hardware barrier's cost a round
(`docs/lab/image_ring.md`). mbarriers remain only where async copies land: the staging and each slot's landed
(`plan.md#barriers`). `Flow` keeps a bit a barrier, flipped at each of the thread's arrivals; a wait is for the phase of
the thread's own last arrival, so the bit names its parity (a barrier never arrived at passes: its phase before the
first), which holds only while a barrier has one phase in flight -- a slot is refilled only after its entry has been
waited for and retired. The same word carries the stream's cursor: the entry's slot and the window's step.

<a id="budget"></a>
## The budget

`tools/budgets/carry.json` (KERNEL_STANDARDS §29): each component's instructions a CTA-window on the gating cell
nl64k-alt-k4, its price a trip times its trips, the trips at each warp's maximum from the facts pass's lists and the
plans' model (`tests/oracle/carry_plan.py`). A warp-window there: a union of ~216 slots and ~256 entries a side, the
image in halves, one read chunk and one fold chunk; 7 readout tiles over 4 digits, 4 eight-token fold steps over 4
lists.

| component | a trip's price (sm_86 SASS) | trips a warp-window | design a warp-window |
|---|---|---|---|
| readout (`read_chunk`) | a tile ~40 (4 MMAs, 3 shared loads, 4 shuffles, 4 shared adds, ~12 float, ~10 address); a digit ~16; a chunk ~10 | 7 tiles, 4 digits, 4 chunks | 384 |
| fold (`fold_chunk`) | an 8-token step ~30 (8 MMAs, 3 loads, 4 shuffles, 4 HMUL2); a list ~12 and its rescale 32 FMUL | 4 steps, 4 lists | 296 |
| units (`fold_units`) | once a window | 1 | 30 |
| stores (`store_batches`) | a batch: 4 lists x (4 x 5 for two values' scale, pack, store, address + 4) + 5 | 4 batches | 400 |
| maxima (`post_maxima`) | a (list, half-tile): 8 FMNMX with abs, an exponent ~4, its max ~2; 16 of them + 20 | 1 | 245 |
| drain | a panel word ~6 (load, token, address, predicated add); two slots' loop ~6 | 13.5 words a thread | 120 |
| read fill (`read_fill`) | a 16-byte copy ~13; an entry run ~10 | 6.9 copies a thread, 4 runs | 130 |
| fold fill (`fold_fill`) | the same | 6.75 copies a thread, 4 runs | 128 |
| staging (`stage_window`) | its ~240 copies over 256 threads, and its tests | 1 | 40 |
| barriers (`Flow`) | an arrival ~3, a satisfied wait ~3 | ~27 | 81 |
| glue (the kernel's body) | the plan's fields and the windows' statics read once, loop control | 1 | 150 |

The design is ~2,020 instructions a warp-window: ~4.0K cycles of issue a window over 8 warps on 4 schedulers, ~0.55 ms a
call at one head. Measured (2026-10-14, carry-k5 839dadb5) 3,884 (31,069 a CTA-window), every component over its line
but the stores (1.09x): the fills 2.3-3.0x (each copy re-derives its token, its source and its swizzle; each run its
list's row), the glue 2.6x and the barriers' own instructions 2.8x (fields re-read at each use), the drain 2.3x, the
readout 1.7x, the fold 1.4x, the maxima 1.3x -- each class (b), the mechanism right and its form above its floor.
Beside them the barriers' spinning (~280 a warp-window, the waits for other warps) and ~24.4K cycles a warp-window
against the instructions' ~7.8K: the warps' latency-bound chains.

<a id="check"></a>
## The check

The state's fp32 sum, the skeleton's `check` (`carry_api.md#carry-forward`): each lane's sum of its state, the warp's
lanes' joined by five shuffles, one atomic add a warp. One add a thread put 32,768 adds a call on one address; they
serialized at its L2 slice for ~47 us a call on sm_86 (`docs/lab/carry_window.md`), most of the call's fixed cost.

<a id="phases"></a>
## The phase ledger's phases

`CarryPhase` is the ledger's column order: `landed` (an entry's copies waited), `maxima`, `store`, `readout`, `drain`,
`fill` (a fill's copies issued), `fold` (with the units) and `wait` (a step's rendezvous and the staging, every
warp's). A lap books the cycles since the last into its phase, so the columns sum to the warp's whole run.
`rola.ops.carry.LEDGER_PHASES` names the same order.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` (`docs/internals/common/phase_clock.md`) keeps each warp's accumulators in shared memory, lane 0 laps,
and `publish` adds them into the bound ledger `[ctas][warps][phases]` at the end. Unbound, the clock is gated off
and its laps issue nothing.
