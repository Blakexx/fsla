# Instruction budgets: the method, the prices, the ledger

KERNEL_STANDARDS [§29](../../KERNEL_STANDARDS.md#s29) asks for every kernel's instructions to be budgeted by opcode
class before code. This page is how a budget is built, checked and corrected, and holds the two pieces of data the
method runs on: the PRICES of the primitives kernels are written in, and the LEDGER of stated budgets against measured
counts. A lab states its budget with `Bench::budget` (per MMA, or per declared unit with `Bench::units`); a production
kernel in `tools/budgets/<kernel>.json` ([phase_ledger.md](phase_ledger.md) reads those).

## Building a budget

1. Name the kernel's PHASES as its source does: each phase is a `//:` heading in the kernel body (a device function is
   its own phase). The ledger attributes instructions by these headings, so the budget and the code share one outline.
2. For each phase, its TRIPS per unit of work on each cell, from the experiment's mirror (its `.inputs.py`) or the
   facts it stands for -- a lane-divergent loop at the warp's longest lane, never the mean (§29.12).
3. For each trip, the PRICE: the primitives the phase is written in, each from the table below for the architecture.
   A primitive missing from the table is compiled alone and its SASS read (`tools/lab.py sass`) before it is priced;
   it then enters the table with the experiment it was read from.
4. The line item is trips x price, by class; the budget is the sum, stated before the code (`Bench::budget`).

## Checking it

`tools/lab.py run <experiment>` gates the executed counts by class (ncu's per-opcode counts of the first launch). When
a kernel is over, `tools/lab.py ledger <experiment> [--kernel REGEX]` attributes the executed instructions to its
phases, so the miss is a phase and a price rather than a total. The attribution is ncu's SourceCounters of one launch
mapped to source lines (the lab builds carry `-lineinfo`); an inlined helper's instructions land on the helper's own
lines, so a helper is a phase of its own.

Every ledger entry is classified (the method `tools/region_ledger.py` applies to production kernels):
- (a) inherent: the design's own cost, near its floor;
- (b) the mechanism is right and its implementation above its floor -- fix to the floor;
- (c) work the cell does not need.

Instruction counts never stand alone: time, IPC and stall reasons go beside them (§21).

## Correcting it

A restated budget names the price that was wrong and corrects the table in the same commit; a restatement that
corrects no price is a guess replaced by a measurement, and teaches the next budget nothing (§29.13). The ledger below
records each lab's stated line against its measured count.

<a id="prices"></a>
## Prices

SASS instructions per use, sm_86 (ptxas of the pinned toolkit), with where each was read. A price is the instructions
the warp issues; a rate (cycles) belongs to `common/hardware.json` and its labs.

| primitive | price | read in |
|---|---|---|
| a warp reduction (`__reduce_*_sync`) used by every lane | 2: `REDUX` + a move from its uniform register | `color_union` |
| a warp collective inside a loop ptxas cannot prove convergent | +1 `BRA.DIV` per collective | `color_union` |
| a conditional single-lane or selected-value store (`if (lane == 0)`, `in ? a : b`) | +3-4: `BSSY`, `BRA`, `BSYNC` | `color_union` |
| a lane-dependent select between two expressions (`lane & s ? f(x) : g(x)`) | both expressions, predicated | `color_union` |
| a counter bumped by a uniform index into k registers | k x (compare + select + add) | `color_union` |
| a 32 x 32 bit transpose by shuffles | 5 x (rotate + mask + `SHFL` + merge) = 20 | `color_union` |
| a shared-memory address through `tile_at`'s XOR swizzle with a runtime row | 3 | `color_union` |
| a lane-divergent loop | its body x the warp's longest lane | `color_union` |
| a shared-atomic claim on a digit, withdrawn on failure | ~10 per digit per pass (claim, compare, withdrawal, full mark) | `color_union` |
| ptxas's additions around `cp.async`: three never-executed `LDS` before each `LDGSTS`, the loop's exit `CALL`, alignment `NOP`s, `UIADD3` fillers between dependent MMAs | as counted | `read_image` |

<a id="ledger"></a>
## Ledger

Stated lines against measured counts, by lab. A miss over 1.3x means the table was wrong somewhere; its correction is
the entry's last column.

| lab | stated | measured | correction |
|---|---|---|---|
| `read_image` readout (instructions an MMA) | ~9 | 18.7 -> 8.2 | the §29.9 source patterns; ptxas's additions priced above |
| `color_union` serial (a box-window) | ~45 a token + 1.15K | ~64 a token + 2.6K | `REDUX` moves, uniform-index counters, `BRA.DIV`; the fixed part itemized |
| `color_union` batched (a box-window) | 1.8K / 3.7K / 1.1K | 3.8K / 7.8K / 1.6K | the warp's longest lane, swizzled addresses, both-sided selects, conditional stores |
