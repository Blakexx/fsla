# Instruction budgets: the method, the prices, the ledger

KERNEL_STANDARDS [§29](../../KERNEL_STANDARDS.md#s29) asks for every kernel's instructions to be budgeted by opcode
class before code. This page is how a budget is built, checked and corrected, and holds the two pieces of data the
method runs on: the PRICES of the primitives kernels are written in, and the LEDGER of stated budgets against measured
counts. A lab states its budget with `Bench::budget` (per MMA, or per declared unit with `Bench::units`); a production
kernel in `tools/budgets/<kernel>.json`.

## Building a budget

1. Name the kernel's PHASES as its source does: each phase is a `//:` heading in the kernel body (a device function is
   its own phase). The ledger attributes instructions by these headings, so the budget and the code share one outline.
2. For each phase, its TRIPS per unit of work on each cell, from the experiment's mirror (its `.inputs.py`) or the
   facts it stands for -- a lane-divergent loop at the warp's longest lane, never the mean (§29.12).
3. For each trip, the PRICE: the primitives the phase is written in, each from the table below for the architecture.
   A primitive missing from the table is compiled alone and its SASS read (`tools/lab.py sass`) before it is priced;
   it then enters the table with the experiment it was read from.
4. The line item is trips x price, by class; the budget is the sum, stated before the code (`Bench::budget`).
5. A production kernel's budget also states, before the code, each component's peak live registers (`registers`,
   gated by `tools/life_ranges.py --budget`, [`life_ranges.md#budget`](life_ranges.md#budget)) and, per cell, each
   ledger phase's predicted cycles a warp a window (`phase_cycles`, to be gated by the family's phase ledger, which is owed: none exists yet):
   KERNEL_STANDARDS §32.

A COMPONENT is a definition the region ledger's function map finds (`tools/region_ledger.py`): a function, a struct or a struct's method, nested structs included; a kernel whose device code is a class over its plan (the intra's `IntraAt<Walk<B>>`) keeps its components by their own names.

## Checking it

`tools/lab.py run <experiment>` gates the executed counts by class (ncu's per-opcode counts of the first launch). When
a kernel is over, `tools/lab.py ledger <experiment> [--kernel REGEX]` attributes the executed instructions to its
phases, so the miss is a phase and a price rather than a total. The attribution is ncu's SourceCounters of one launch
mapped to source lines (the lab builds carry `-lineinfo`); an inlined helper's instructions land on the helper's own
lines, so a helper is a phase of its own.

`tools/region_ledger.py` maps a production launch's samples the same way, over the one function `--kernel` names: a
member holding several (a carry arm holds the carry and its plan pass) is refused without it, since offsets restart in
every function and two functions' frames would overwrite each other's.

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
the warp issues; a rate (cycles) belongs to `common/hardware.json` and its labs -- each pipe's, the shared-memory
pipe's by kind of traffic ([hardware.md#shared](../common/hardware.md#shared)) -- and a lab prints every pipe's cycles
from both ([pipes.md](pipes.md)).

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
| a 4-lane reduce-scatter of 4 values (lane `t` ends with value `t`'s max over its quad) | 12: 3 `SHFL` + 6 `SEL` (which value each lane keeps and sends) + 3 `IMNMX` | `post_keys` (production, 2026-11-01) |
| a per-slot fact handed to the slot's lanes (a lane forms it, a step reads it) | 1 `SHFL` a fact a step + 1 `IADD3` (the step's source lane) | `drain` (production, 2026-11-02) |
| a 64-bit address from a 32-bit word offset (`word_at`) | 2: `LEA` + `LEA.HI.X` (ptxas turns `mad.wide` by an immediate 4 into the shift-adds) | `drain` (production, 2026-11-02) |

<a id="forms"></a>
## Forms

The cheapest known form for each recurring job, what it costs and where that was measured. A budget line names the
form it uses, or why it departs (KERNEL_STANDARDS §29.15); a form found cheaper replaces its row in the commit that
measures it. Pipe cycles are an SM's, from `common/hardware.json` ([pipes.md](pipes.md) prices them).

| job | the form | its cost | measured in |
|---|---|---|---|
| a maximum over values held in different units (scaled exponents) | integer keys: the fp32 bits plus the unit's shift in the exponent field (zero the least key, the shift clamped), compared as integers | an integer add and max a value, no conversion | `handoff` |
| a reduction across warps into a slot an item | one shared atomic (max, add) a lane at the item's slot, then one wide load of a lane's slots | 2 pipe cycles an atomic instruction; one `LDS.128` for four slots | `handoff`, `smem_rates` |
| an address at every access of a level | a base held for the level plus compile-time offsets (§29.3); a lane's constants held (`lab::held`) | 0-1 instructions an access | `read_image`, `handoff` |
| a lane's several words into shared memory | the widest store the layout allows | a 4-byte store 2 pipe cycles a wavefront, 8 or 16 bytes ~1.0-1.14 | `smem_rates` |
| contiguous global rows into shared memory, sm_80/86 | `cp.async.cg` | 0.5 pipe cycles a 128 B (`.ca` 5.4; a load into registers 1.6-1.7 and its registers in flight) | `smem_rates` |
| a stream's per-step scalars (coefficients, tokens, entries) | in the record riding the stage copy, read lane-ready | one 8-byte load a lane a step | `row_stream`, `read_stream` |
| a pad in a stream | not copied: its zero coefficient cancels the stale row | nothing | `row_stream` |
| fp32 to integer | one FFMA adding 1.5 x 2^23, a pair packed by one PRMT (§29.6) | 1.5 instructions a value, the FMA pipe | `read_image` |
| an accumulation into shared memory from many lanes, sm_80/86 | the 32-bit integer `red.shared.add` (their only native shared atomic add) | 2 pipe cycles an instruction | `read_image`, `smem_rates` |
| a branch on loaded data around collectives | a vote (`__any_sync`), a loop bottom-tested on it (§29.9) | no convergence barrier | `read_image` |
| a value every lane needs from a warp reduction | `__reduce_*_sync` | 2 instructions (its uniform-register move) | `color_union` |

HOW IT IS WRITTEN (sm_90a, ptxas of the pinned toolkit): what the compiler does with a way of writing a kernel,
each read off a build of the experiment named -- its SASS, its notices, or the lab's refusal.

| the job | written this way | written the other way, the compiler | read in |
|---|---|---|---|
| several asynchronous steps with register operands | every step's operand in its own registers, all written before the group's first step | a register written between two steps that read it makes it wait out the earlier step (the steps serialized); with too few registers for the sets it serializes them all (C7512) | `carry_window` (the readout at 80 and at 72 registers a thread) |
| an accumulator's first step of a trip | the step writes the accumulator whole (its add flag off); nothing is cleared | a cleared accumulator inside a role's branch serializes the steps that follow (C7520) | `readout_role`, `carry_window` |
| a loop of global accesses at distinct addresses (a batch's loads, a chunk's adds) | rolled over its batches (`#pragma unroll 1`), a batch's few accesses unrolled inside | unrolled whole, every access's 64-bit address is formed ahead and held: local memory at the role's count (64 adds: a 144-byte frame) | `carry_window` (the drain of one warp, the producer's batches, the 40-register roles) |
| batches alternating two register halves | the outer loop rolled in steps of two, the half an unrolled inner index | a half chosen by a rolled loop's index is a runtime index into registers: local memory | `carry_window` (the producer's two batches) |
| a choice between two loads into the same registers | one load whose arguments are selected (`last ? next : cur`) | a branch with a load in each arm: both arms' address arithmetic, and a convergence barrier | `carry_window` (the producer's parts) |
| a value a lane derives once and uses every trip | held (`lab::held`): an empty asm hides its derivation | rematerialized from `S2R` and its arithmetic at every use | `read_image`, `carry_window` |
| a role's clock in a role short of registers | the start stored as it is read (`lab::WarpMark`) | a held 64-bit start costs two registers for the role's whole span: the last spill of a 40-register producer and drain | `carry_window` (the five-warpgroup row) |
| values published for another warpgroup's step | the tile's major chosen so a 16-byte line is one thread's own values, packed by `PRMT` | a line that is eight lanes' values is eight 2-byte stores a line: 32 store instructions a thread where 4 do | `carry_window` (the image) |
| a role's wait for work | at its barrier, the poll SUSPENDING (`mbarrier.try_wait`) | a plain `test_wait` loop spins a test and a branch a trip and loses 2 to 3% of the composed window (the lab's page holds the session's cycles) | `carry_window` |
| a readiness test a warp branches on | one vote of the warp | each lane's own read of the barrier under a collective is a lane-divergent branch the SASS cannot rule out | `carry_window` (the serve warps, removed) |
| registers given to a warpgroup's roles | one `setmaxnreg` a whole warpgroup, first in the role's branch | two counts inside one warpgroup compile clean and hang the CTA on the device | `carry_window` (the per-warp row, removed) |
| a role's order around a global load | the loads issued, then the role's arithmetic and stores, then the loaded values' use | the use straight after the issue holds the warp for the load's whole latency (`LOAD_IN_TRIP`) | `carry_window` (the producer), `load_rate` |

THE HOPPER CARRY'S FORMS (sm_90, an H100), each beside the form it was measured against:

| job | the form | its cost | measured in |
|---|---|---|---|
| scattered global rows into shared memory, each its own request | warps' own loads, 16 bytes a lane and four rows an instruction, then stores; a producing warp holds TWO batches of 32 entries where the lean readout frees it the registers, one otherwise | 3.5 cycles a row from four warps, 10.6 from one (a row a lane's 4 bytes: 8.5 and 33.0); a warp's rate rises with the entries it holds up to about a batch | [`load_rate`](../../lab/load_rate.md), [`value_gather`](../../lab/value_gather.md), [`carry_window`](../../lab/carry_window.md) |
| an output row summed into global memory | one atomic add of the row's whole 128-byte line, from warps that do nothing else | 12.2 cycles a line a warp alone on the SXM, 10.2 on the PCIe card, and with every SM adding some 10 an SM: the card's
limit, some 13 lines a cycle, a vector add's lines the same; the issuing warp is held until the SM passes the add | [`add_rate`](../../lab/add_rate.md), [`carry_window`](../../lab/carry_window.md) |
| a register state published as a wgmma's shared-memory operand | the major whose 16-byte line is one thread's own values (K-major for a fold's accumulators) | 4 stores a thread a tile group; a few percent of a window (the page's row with the image's stores out) | [`carry_window`](../../lab/carry_window.md) |
| registers moved between a CTA's roles | `setmaxnreg`, one count a whole warpgroup, each role's body in its own branch; the counts sum to what the kernel enters with, which goes by whole warpgroups (128 a thread at 512 threads, 96 at 640) | about 200 cycles a warp, once | [`reg_trade`](../../lab/reg_trade.md), [`reg_five`](../../lab/reg_five.md) |
| the register A operands of several asynchronous steps | TWO sets reused (depth 2, KERNEL_STANDARDS 4's two stages), a digit's steps a commit group, the freed 16 registers a thread to the producers, which then hold two batches; with suspended waits the fastest composed form | 4 words a thread a set held through its group's wait; the extra commit groups' round trips, paid back by the producers' second batch: the two are NOT ISOLATED from each other | [`carry_window`](../../lab/carry_window.md) |

FORMS THAT LOST there or bought nothing, with what each was measured against (a row stays until a later measurement
overturns it; one that says NOT ISOLATED or OPEN is open). The cycles are on each lab's page, which a session
re-measures whole; a row here states the size of the difference, not the session's figures:

| the form | what it lost to | by how much | measured in |
|---|---|---|---|
| a scattered row by the bulk copy engine, one request a row | four warps' own loads | several times the cycles a row | [`state_move`](../../lab/state_move.md), [`value_gather`](../../lab/value_gather.md) |
| four producers each holding one value instruction's four entries (40 registers a thread) | two producers each holding a batch | slower on the fold's side alone, with twice the warps | [`carry_window`](../../lab/carry_window.md) |
| the readout adding its own rows after its steps | two drain warps beside it | the window 4 to 7% longer across records: its adds hold it, and the window becomes the readout's | [`carry_window`](../../lab/carry_window.md) |
| every step's A in its own registers, one commit group of 8 (depth 8) | two sets reused with the registers to the producers | 1 to 4% composed with suspended waits across records (NOT ISOLATED from the producers' second batch); depth 8 stands only where no one can use the 16 registers | [`carry_window`](../../lab/carry_window.md) |
| the den's fold as a step a list, A a tile of ones in the list's row (16 steps a window, a 17th accumulator set of 8) | the coefficients as A -- 8 lists' bricks on 8 digits by a descriptor a pool tile apart -- and B a column of ones: four 8-column steps a window into 4 registers a thread | an eighth of the composed window: the 17th set does not fit a 160-register fold, which took 168 from the producers | [`carry_window`](../../lab/carry_window.md) (the losing form's session is the experiment's at an earlier commit; its record was replaced) |
| an image tile pitch of an even count of lines | an odd count | 5% of the composed window: a warp's lanes store to four tiles two apart, and two of them share banks | [`carry_window`](../../lab/carry_window.md) |
| a fifth warpgroup (640 threads) of serve warps | four warpgroups (512) | the window a tenth longer: 61,440 registers leave the serve warps 40 a thread | [`carry_window`](../../lab/carry_window.md), [`reg_five`](../../lab/reg_five.md) |
| a role's long wait as a plain polled `test_wait` loop | the suspending poll (`mbarrier.try_wait`), or a sleep at a named barrier | 2 to 3% of the composed window; the suspending poll is the faster replacement (the sleep buys under 1%) | [`carry_window`](../../lab/carry_window.md) |
| the readout's coefficient bytes as 4-byte loads, a lane the word it needs (8 instructions a chunk of 64 tokens) | one wide load a warp a chunk, a lane a row's 16 bytes, relaid through the warp's scratch | the wide load is the read side alone without those loads; composed the two are within a session's noise | [`carry_window`](../../lab/carry_window.md) |

<a id="ledger"></a>
## Ledger

Stated lines against measured counts, by lab. A miss over 1.3x means the table was wrong somewhere; its correction is
the entry's last column.

| lab | stated | measured | correction |
|---|---|---|---|
| `read_image` readout (instructions an MMA) | ~9 | 18.7 -> 8.2 | the §29.9 source patterns; ptxas's additions priced above |
| `color_union` serial (a box-window) | ~45 a token + 1.15K | ~64 a token + 2.6K | `REDUX` moves, uniform-index counters, `BRA.DIV`; the fixed part itemized |
| `color_union` batched (a box-window) | 1.8K / 3.7K / 1.1K | 3.8K / 7.8K / 1.6K | the warp's longest lane, swizzled addresses, both-sided selects, conditional stores |
| `handoff` 16-byte form, integer/address (a CTA-window) | 960 | 1,085 -> budget 1,088 | each scale's float bits are one instruction (16 a warp), priced first as address forms a held base removes |
| the carry's hand-off at nl64k-alt-k4 (a CTA-window: keys, exponents, stores) | 4,360 | 4,431 | the exponents 10 -> 14 a warp: the two parities' addresses, the warp's sync |
