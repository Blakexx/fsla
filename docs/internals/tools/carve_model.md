# `tools/carve_model.py` — the carry's mechanics for any carve

KERNEL_STANDARDS §32.2: a schedule or composition change is predicted before it is built. This model predicts the
carry's work for a carve -- a box partitioned among warps, each phase's warp tile, streamed or resident rows, the
hand-off route -- on a card, over a real cell's lists, and ranks nothing until it reproduces the measured forms.

<a id="statistics"></a>
## The cells' statistics (`stats`)

Every box-window of a registry cell as the carry reads it: each side's union and each list's entries by the union's
64-slot segments, from the facts pass's prefix rows (`pre[l][j]` the boundaries of lists below `l`; a list's entries in
a segment are differences of differences, modulo 2^16), and the window's plan word from the plan pass's model
(`tests/oracle/carry_plan.py`). A dense side's lists are implied: every slot in every list.

<a id="events"></a>
## The events (`events`)

Today's carve's work a CTA-window from the plan words and the lists: readout tile-steps (a warp's digit pair `w + 8i`,
two at a time, as many tiles as the pair's longer run; a dense side a digit at a time over every slot), fold steps (a
list's 16-token steps while more than 8 entries remain, then an 8-token tail), images stored (a half counts a half),
chunks, rows filled, entries, panel slots drained, the window's steps. The MMAs these imply equal the measured MMAs by
shape on every cell measured (nl64k alt-k4, dense, g4; flagship alt-k4, alt-k16, cohort-k4, dense).

<a id="measured"></a>
## The measured side (`measured`)

One launch's `ncu --section SourceCounters` dump: each SASS line's executed instructions and shared wavefronts,
attributed to the innermost MMA loop holding it (or none) and its innermost carry function (the extension's line
information), a CTA-window's worth. Attributing by component (the region ledger's outermost frame) split the fold's and
the readout's instructions wrongly where their code is the same; by innermost function it does not.

<a id="prices"></a>
## The prices

Each function has one driving event (`DRIVERS`: a readout tile-step, a digit, a fold step, an image stored, a row, an
entry, a chunk, a drained slot; a function not listed is the window's bookkeeping, per step). A price is a function's
instructions and wavefronts per event on a calibration cell: nl64k-alt-k4 for the listed side's code, nl64k-dense for
the dense side's (its readout and fold run other code paths).

<a id="validation"></a>
## Validation (2026-10-17, carry-k5 at 1b059f42, sm_86)

Instructions and wavefronts a CTA-window, predicted from the calibration cells' prices, against held-out cells:

| cell | instructions | wavefronts |
|---|---|---|
| flagship-alt-k4 | -4.5% | -2.0% |
| flagship-alt-k16 | -0.1% | -8.3% |
| flagship-dense | -0.5% | -0.1% |
| flagship-cohort-k4 | +1.3% | +17% |
| nl64k-alt-k4-g4 | -12.1% | -31% |

The readout and the hand-off within 2% on every cell; the fold within 3% (g 4: -13%). The g 4 miss is mostly its
`pre_at` reads (3,789 instructions, 10,152 wavefronts a CTA-window: bank conflicts in the grouped prefix rows, a
defect) and tables that grow with the query heads, which the events do not carry yet. The fills and the bookkeeping
miss by up to 30% a phase, in opposite directions.

What it shows of today's carry at nl64k-alt-k4 (a CTA-window): 34.7K warp instructions, 71% outside the fold and the
readout (bookkeeping 9.3K, about 4,000 a step of the plan; fills 6.8K; hand-off 6.4K, 3,660 an image stored; drain
3.4K, 16 a drained slot); each phase 2-3x its own binding floor, the phases in turn.

<a id="cycles"></a>
## Cycles (`validate --ledgers DIR`)

A phase's cycles a CTA-window are its binding floor -- the largest of its instructions over the four schedulers, its
shared wavefronts at one a clock, its MMAs' pipe time over the schedulers -- times its EFFICIENCY, the measured cycles
over that floor on the calibration cell of its side (the phase ledger, clock locked, the GPU quiet). Each model phase
is ledger columns: the readout, the fold, the hand-off (maxima and store), the fills (fill and landed: the wait for
their copies), the drain, the bookkeeping (wait). A phase ends at a rendezvous, so the warps other than its slowest
IDLE through the slowest warp's extra work, booked in the ledger's wait: the phase's cycles times the share of its work
that differs between warps (its tile-steps or fold steps, not its per-digit or per-list work) times its slowest warp's
work over its mean warp's, less one. Measured per warp, a phase's own time differs little between warps (at most 1.15x
at cohort-k4, whose per-window work differs 3.2x): the difference is in the wait.

Held-out window cycles (2026-10-17, clock locked at 1.665 GHz): flagship-dense -0.3%, cohort-k4 +7.6%, flagship
alt-k4 -9.2%, alt-k16 +8.7%, nl64k g 4 -17.1% (its readout's instructions and wavefronts predicted exactly, at 3.0x its
floor against alt-k4's 2.4x; its fills carry the `pre_at` conflicts). The flagship cells' bookkeeping misses by -24 to
-50%: two windows a CTA, so a call's first window's waits weigh on it.

What it shows of today's carry, a CTA-window: every phase 2-3x its own binding floor, the phases in turn; the dense fold
at 1.10x its tensor floor.

<a id="forms"></a>
## Forms (`forms`), the window (`window`) and streamed rows

A FORCED FORM is planned by the plan model's `forced_plan` and priced the same way. Against measured launches at
nl64k-alt-k4 (ncu, the rule 5.473M cycles): the whole image with 128-slot read chunks predicted 1.155x, measured 1.156x;
halves with 128-slot read chunks 1.271x against 1.371x; halves with 128-slot fold chunks 1.121x against 1.233x -- an
extra chunk's cost read ~7-9% low.

THE WINDOW AT k x 512 TOKENS (`merged`): consecutive windows of a box merged exactly -- unions added, each list's
entries carried over (a segment's entries spread over its slots and re-binned at the merged union's 64-slot boundaries,
since the prefix rows hold no finer place) -- planned by the same rule in the area the larger window tables leave
(7.2 KB less at g 1). STREAMED ROWS (`streamed`, the read ring's depth): a listed side's chunks hold no rows or entries
(a read chunk is its panel), every entry fetches its row into its warp's ring, and each warp's entries arrive as its
window's stage record (a pre-pass's plan, copied ahead with the stage group it is issued in); the read ring (576 bytes a
stage, 8 rows and their entry words) sits beside the image and the fold's takes the image's place (a window's read
chunks all run before its fold); a dense side stays resident. A streamed side costs its MMA phase times the labs'
measured stream factor -- the fold 1.62 ([`row_stream`](../../lab/row_stream.md)), the readout with its drain 1.34
([`read_stream`](../../lab/read_stream.md)), each the mean of three cells, each third cell predicted from the other two
within 7% and 3% -- the factor's excess applied to the model's phase after rescaling at nl64k-alt-k4 (lab bare / model
phase: fold 0.604, readout and drain 0.648), booked in `fills`; a 4-stage read ring adds 264 cycles a CTA-window.
With every phase at its floor the stream's cost scales with its phase (an assumption: the producer's own instructions
need not shrink with it).

## The forms ranked (`rank`)

Every form at a cell -- resident, or streamed at each read ring -- at W 512 and 1024, best first, against today at the
calibration cells' efficiencies and with every phase at its floor (the phases still in turn; their overlap is the
kernel's next work). On sm_86, a 512 tokens (2026-10-17; W = 512 ruled, W 1024 shown for its revisit on the A100):

| cell | resident W 1024 | streamed W 512 | streamed W 1024 | streamed W 512, phases at their floors |
|---|---|---|---|---|
| nl64k-alt-k4 | 1.12x | 0.83x (read ring 5) | 0.65x (4) | 0.39x |
| nl64k-alt-k4-g4 | 1.53x | 0.85x (5) | 1.32x (5) | 0.38x |
| flagship-alt-k16 | 1.11x | 0.63x (4) | 0.87x (5) | 0.29x |
| flagship-cohort-k4 | 0.95x | 0.86x (5) | 0.73x (4) | 0.40x |
| nl64k-dense | 0.99x | 1.00x (resident) | 0.99x | 0.60x |

Resident, a 1,024-token window's union no longer fits beside the image, the read splits into chunks and each re-stores
the halves: W 1024 loses. Streamed, the entries' staging and the chunks' fill machinery go with the rows, and where the
read chunk -- its panel, 64 bytes a slot -- fits whole at W 1024 beside the image and a 4-stage read ring, the hand-off,
the readout's per-digit work and the bookkeeping fall by half a token; at g 4 (a ~1,700-slot read union) and alt-k16 it
still splits. In every box-window of these forms the fold's ring fits beside the last read chunk's panel
(`ring_beside_panel`), so that panel can drain while the fold runs. Only the leaf carve's forms are ranked: the value
carve and the 64-channel box were dropped on measured grounds (their fold shared-memory bound, a 16-channel tile's
warps each reloading the shared operand; the box's halved lists doubling the readout's partials), and pricing them here
needs their readout and hand-off, which no lab has measured.

<a id="cards"></a>
## The forms on each card, the global path in (`rank-cards`)

Every form at a cell at its floors on sm_86 and sm_80, a 512 tokens: `in turn` the phases' binding floors summed, the
global path a floor beside them; `overlapped` the largest of the window's totals -- issue, shared wavefronts, the
tensor pipe at the card's MMA rates, the SM's global path ([`pipes.md`](pipes.md): L2 lines and atomic requests at
`common/hardware.json`'s prices, every SM busy). A form's global path: a resident side its union's rows once and its
staged entries; a streamed listed side a row an entry (pads not copied) and its records (64 bytes an 8-entry read
stage, 96 a fold step); the drain a request a slot. A streamed side's producer (20 warp instructions a read stage, 27
a fold step: the staged labs) and every copy's write (0.5 shared-pipe cycles a 128 bytes) join the prices. sm_80's L2
price is announced from sm_86, so it is ranked at that and at half (`L2_BRACKET`).

At today's prices (2026-10-18; the carry 2-3x its instruction floor) issue binds every sparse form and the ranking
stands: nl64k-alt-k4 a CTA-window streamed W 512 7,424 against resident 8,674 on both cards, the global path 4,165
(430 lines, 216 atomics) and 3,527 (340 lines); flagship-alt-k16 streamed W 512 ring 4 global-bound at 14,187 (1,663
lines against the resident form's 751) against resident 19,633 (issue); flagship-cohort-k4 streamed W 1024 5,744
against resident 8,062; nl64k-dense tensor-bound on sm_86, issue on sm_80, alike in every form. The global path
binds where the stream refetches most -- a token's row once for each digit or list that reads it (alt-k16's read side
~2.7 a union token) -- and at the design floor (the staged labs' forms, the 16-byte hand-off, the drain's best form,
a 231-instruction warp-window of glue: at alt-k4 issue ~3.3K, shared ~3.9K) it is the streamed form's largest pipe
on both cards. Its levers: fewer bytes a union token -- groups of digits fetching a token's row once (the grouped
readout's unified form fetched ~30% fewer L2 lines at m 2 and 4 than at m 1), the stream or the union staged chosen by
the planner per side and window from exact counts -- and sm_80's own L2 rate.

OWED: the alternative carves' prices, sm_80's L2 rate (the rental's first measurement), (each step's primitives by its warp tile, checked against the readout and fold
labs), registers, sm_80's calibration (the A100), and labs for the prices it lacks.
