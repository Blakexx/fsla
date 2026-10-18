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

OWED before it ranks a carve: cycles per phase (each phase's efficiency against its floors, calibrated on a quiet GPU's
phase ledger -- a Windows-side GPU user inflated it on 2026-10-17), the alternative carves' prices (each step's
primitives by its warp tile), streamed rows, W = 1024 (consecutive windows' lists merged, exactly), and the labs for the
prices it lacks.
