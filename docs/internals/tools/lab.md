# `tools/lab.py`, `measure/lab/` — the lab: small CUDA experiments that become kernel rules

The kernels are written to rules about the hardware and the compiler — what a pipe costs beside the tensor pipe, what
ptxas does to a pattern, when two warps interfere. Those rules come from measurements, and a measurement taken in a
one-off probe is easy to get wrong and impossible to repeat. The lab makes an experiment cheap to write and safe to
trust, and keys every record by the architecture, so a rule says where it was measured and a new card or toolkit
re-runs the experiments its rules cite (`docs/KERNEL_STANDARDS.md` §26).

## The loop

1. A QUESTION about the machine, asked as one experiment, `measure/lab/experiments/<name>.cu`: its header states the
   QUESTION and the METHOD, its kernels are template instantiations (one per variant), and its `main` declares the SASS
   each kernel must contain and times the variants.
2. `python tools/lab.py run <name>` compiles it with the pinned toolkit for the running card, reads the SASS and
   REFUSES to measure unless every declared expectation holds, no kernel touches local memory and no kernel holds a
   convergence call nobody declared (the SASS gate's classification, `tools/sass_gate.py`: ptxas's loop-exit
   `CALL.REL.NOINC` and guarded fallbacks are not ones); checks the GPU is quiet; runs the repetitions; prints median, min and max and flags a result
   whose samples spread past 5%.
3. `--record` writes `measure/lab/results/<name>/<sm_XX>.json` — the device, driver, toolkit, commit, the results with
   their samples, and each kernel's registers and leading opcodes. A rule cites this file.
4. The finding becomes a law in `docs/KERNEL_STANDARDS.md` (§26), naming the experiment, the record and the
   architectures it holds on.

`python tools/lab.py list` names the experiments by their QUESTION; `python tools/lab.py sass <name> [REGEX] [--full]`
shows each kernel's registers and leading opcodes (or the whole disassembly). `python tools/lab.py ncu <name> --metric M [--metric ...]
[--record]` reads hardware counters per kernel (the median over its launches) into `<sm_XX>.counters.json`, for a
question about traffic rather than time.

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
* `lab::WarpSpan` — a warp's start and end `clock64`; `Bench::span(warps)` is the mean block's span (last end less
  first start), the throughput when warps finish apart.
* The operand pieces the kernels use: `ldsm` and `ldsm_t` (`ldmatrix.x4`, `.trans`), `hmul2`, `pack_half2`, `smem`;
  the mbarrier as `ops.cuh` spells it: `mbar_init`, `mbar_arrive`, `mbar_wait` (polled), `mbar_test` (once).
* `lab::sink(out, v)` — a fold of everything a thread computed, stored unconditionally.
* The atoms as one PTX instruction each: `mma_f16_f32` (and `_fresh`), `mma_f16_f16`, `mma_bf16_f32`, `mma_u8s8`.
* `lab::Bench` — the host runner: `expect(kernel, op, cmp, count)` declares SASS (printed under `--expect`, run by the
  driver before anything is timed), `variant(launch, measure)` warms up and repeats a launch, `record(name, unit,
  samples)` prints a result, `cycles(warps)` and `mean(...)` read the warps' clocks.
* `lab::kernel("tax", {3, 17})` — a kernel's demangled name as an expectation's pattern.

## What goes wrong, and what guards it

Every item below cost a wrong number before the lab existed; each is now a guard or a habit.

* **Dead work.** ptxas deletes an instruction whose result is dead — including a result only tested against a value it
  can fold (a probe's MMAs vanished behind `acc == x && m == y` with `m` constant). Guard: `lab::sink` stores
  unconditionally, and the expectations refuse a kernel missing what it means to time.
* **Rewritten arithmetic.** `asm volatile` binds the front end, not ptxas. ptxas folds a repeated `x += k`, rewrites a
  whole integer-add recurrence (adds are exactly associative), merges two-input adds into the three-input `IADD3`, and
  issues adds as `IMAD.IADD` on the FMA pipe at will. Habit: an instruction's operands are loop-variant values from
  other chains; integer adds are timed paired with a non-associative op; an expectation may name alternatives
  (`IADD3|IMAD.IADD`).
* **Collectives under role-dependent control flow.** A warp collective under a branch on a thread-derived value — even
  a warp-uniform role like `threadIdx.x >> 5`, and even when the role only sets a loop's trip count — compiles into a
  CALL into a convergence subroutine. Guard: the driver refuses an undeclared CALL; `collective_branch` records the
  behaviour.
* **A sink that cancels.** XOR-folding a value twice folds to zero, and ptxas deleted a whole kernel's MMAs behind
  it. Guard: the expectations.
* **The mean warp undercounts.** When warps finish apart, the mean warp's cycles read below the tensor floor; the
  block's span is the throughput. Habit: `WarpSpan` wherever warps can drift.
* **Light register pressure.** A step that runs at its floor beside 150 registers may spill beside the kernel's 255.
  Habit: an experiment about a kernel's step holds the kernel's live accumulators (`fold_step`).
* **Runtime choices inside a timed loop.** A `kind` argument switched at run time inside the loop measured branches, not
  MMAs. Habit: every variant is a template parameter.
* **The operand pattern.** An instruction's cost beside the tensor pipe grows with the registers it reads: packed f16
  math with a constant operand measured ~4x cheaper than with two registers. Habit: write the operand pattern the
  kernels use.
* **Phase lock.** Two warps a scheduler can fall into a slow relative phase on some launches (up to 22% for a pure HMMA
  stream). Guard: repetitions; the driver flags spread.
* **A busy GPU.** Another consumer time-slices every kernel. Guard: the driver refuses above 5% utilization before a
  run (`--force` to override).

<a id="tensor-tax"></a>

## `tensor_tax`

Warps 0-3 (one a scheduler) run 8 MMAs a loop into 8 accumulators; warps 4-7 issue one instruction class in 8 chains,
each instruction's operands other chains' values; the class is sized to land inside the chain; the tax is the chain's
slowdown over the class's count. sm_86, CUDA 13.0, cycles an instruction beside an fp16-accumulate HMMA (16 cycles) /
an fp32-accumulate HMMA (32 cycles): FP32 `FMUL`/`FFMA`/`FADD` 1.0 / 0.51; packed `HMUL2`/`HFMA2`/`HADD2`/bf16 FMA
0.48 / 0.21; `HMNMX2` 0.44 / 0.16; an add and xor pair 1.2 / 0.71; `IMAD` 0.19 / 0.10; `LOP3`, `PRMT`, `FMNMX`, the
conversions 0.13-0.22 / 0.05-0.08; `LDS` (with its address math) 0.85 / 0.53. Beside an int8 IMMA (16 cycles) the
numbers match the fp16-accumulate column. Record: `measure/lab/results/tensor_tax/sm_86.json`.

<a id="mma-pairing"></a>

## `mma_pairing`

Every warp of a 256-thread block runs a fold-like step of 32 fp16-accumulate HMMAs (8 lists x 4 n-tiles), plus one atom
of a kind. The median launch sits on the floor (1024 cycles a step a warp at two warps a scheduler); some launches
phase-lock up to 1.22x. One fp32-accumulate atom among them costs 32 cycles a step pair, not its 64. Record:
`measure/lab/results/mma_pairing/sm_86.json`.

<a id="collective-branch"></a>

## `collective_branch`

Eight shuffle chains in a loop, one shuffling warp a scheduler: with no role-dependent control flow the shuffle is
inline (8.3 cycles a shuffle), and so under a branch on the block index or a kernel argument (8.3); under a branch on
the warp id (13.6) or with the role setting the trip count (13.9) it is a CALL into a convergence subroutine, ~1.65x.
Record: `measure/lab/results/collective_branch/sm_86.json`.

<a id="fold-step"></a>

## `fold_step`

The fp16 fold's dense step transplanted from `owner_fold.cuh` (two warps a scheduler, the arm's accumulators live --
254 registers on the fp16 arm, as in the owner part), over shared rows, one variant a removal or a stage. sm_86, cycles
a step a warp over the floor (1056 fp16 arm, 2112 fp32 arm): as built 1.074 / 1.022; the MMAs alone 1.033 / 1.004; no
coefficient 1.055 / 1.016; no operand loads 1.067 / 1.026; a CTA barrier every pass of 8 steps 1.171 / 1.026. A class
replaced by a `LOP3` reads slower than the class (shuffles 1.117, HMUL2 1.090): the replacement is not free, so those
two rows bound nothing. Staged forms on the fp32 arm (all, loads alone, coefficients alone) 1.024-1.027, no gain; on
the fp16 arm every staged form spills (12-36 LDL/STL) and is not run. The step alone is near its floor and the barrier
realigns the warps. The timed region holds the steps only: the fp16 arm's partial join (every 256 tokens) falls between
regions, and with it the arm runs 1.25x free-running (`pass_pipeline`). Record:
`measure/lab/results/fold_step/sm_86.json`.

<a id="mma-overlap"></a>

## `mma_overlap`

A step of 512 tensor cycles a warp (fp16-acc, fp32-acc or int8 MMAs) plus the fold's operand chain, 1-4 warps a
scheduler, inline or staged, free-running, a CTA barrier every pass of 8 steps, or a two-stage mbarrier (arrive at a
pass's end, wait for the pass before) polled or sleeping 64 ns between polls; the block's span a step over the floor.
sm_86 (fp16-acc / fp32-acc / int8):

| warps a scheduler | inline, free | staged, free | inline, barrier | inline, two-stage polled | inline, two-stage sleeping |
|---|---|---|---|---|---|
| 1 | 1.42 / 1.23 / 1.48 | 1.15 / 1.14 / 1.17 | 1.44 / 1.28 / 1.49 | 1.47 / 1.28 / 1.52 | 1.47 / 1.27 / 1.52 |
| 2 | 1.05 / 1.03 / 1.07 | 1.04 / 1.05 / 1.06 | 1.11 / 1.08 / 1.12 | 1.05 / 1.04 / 1.09 | 1.06 / 1.05 / 1.08 |
| 3 | 1.05 / 1.02 / 1.06 | 1.04 / 1.04 / 1.05 | 1.08 / 1.06 / 1.09 | 1.04 / 1.06 / 1.07 | 1.04 / 1.04 / 1.06 |
| 4 | 1.04 / 1.04 / 1.06 | 1.04 / 1.04 / 1.06 | 1.08 / 1.07 / 1.09 | 1.08 / 1.07 / 1.10 | 1.05 / 1.05 / 1.07 |

One warp a scheduler cannot hide its own chain; a stage recovers most of it. Two or more free-running warps hide each
other's, and a stage adds nothing. A barrier a pass costs 3-6%; the two-stage mbarrier recovers it, polled at two warps
a scheduler, sleeping at three or four (a polling warp takes issue slots from the working ones). Record:
`measure/lab/results/mma_overlap/sm_86.json`.

<a id="pass-pipeline"></a>

## `pass_pipeline`

A whole pass: the owner part's fill (a quad a token, cp.async from a 33 MB source), its restage (two threads a row, the
coefficient run included, since the pool slot refills two passes ahead) and the fold's step at the arm's registers, two
warps a scheduler. Serial = fill, barrier, restage, barrier, steps, barrier (the owner part). Pipelined = two stages
under mbarriers: fill two passes ahead, restage one ahead -- in a burst at the pass's start, by warp w after its step w,
or by every warp after its step 3 (mid-pass) -- each restage waiting for its tile slot's readers just before it. The
fp16 arm's partial join every two passes: a burst, one list a step (a switch on the step: unrolling the step loop
spills the arm), or the burst staggered (a scheduler's two warps on alternate passes). sm_86, the block's span a pass
over the steps' floor (2112 / 1056 a step):

| pass | fp32 arm | fp16, join burst | fp16, join a list a step | fp16, join staggered |
|---|---|---|---|---|
| serial | 1.20 | 1.54 | 1.65 | 1.55 |
| pipelined, restage burst | 1.11 | 1.34 | 1.48 | 1.36 |
| pipelined, restage a warp a step | 1.11 | 1.35 | 1.45 | 1.35 |
| pipelined, restage mid-pass | 1.05 | 1.27 | 1.32 | 1.25 |
| steps alone | 1.03 | 1.25 | 1.32 | 1.16 |

Mid-pass is the pipeline's best: half a pass of slack each way between the restage's wait (the slot's last readers)
and the next pass's wait (the restage). A burst at the pass's start makes every warp wait at the boundary for the
slowest one's previous pass -- a barrier by another name. The join in a burst costs the fp16 arm 12 points, because both
warps of a scheduler leave the tensor pipe at once and the pair falls into lockstep; staggered it costs ~3; spread a
list a step it costs more (the switch, and FP32 adds beside every step's MMAs). Record:
`measure/lab/results/pass_pipeline/sm_86.json`.

<a id="pool-layout"></a>

## `pool_layout`

The owner part's fill into 160-byte pool rows plain, with the 16-byte halves of each 32-byte pair swapped on rows with
bit 2 set, or 176 bytes apart; and the steps' in-place reads of a row (two `ldmatrix.x4.trans` over the values, four
coefficient bytes a lane). sm_86: L2 sectors (`ncu`) 7 a token plain and swapped, 9 at 176 bytes (odd rows 16 bytes
off phase); the fill a pass from DRAM ~3.6K cycles in every layout, from L2 1.70K / 1.71K / 2.28K; the reads 192 cycles
a step plain (2-way bank conflicts: 160 bytes puts every row's chunk on the same parity of 16-byte bank group), 119
swapped, 118 at 176 bytes. Swapping halves inside an aligned 32-byte pair keeps cp.async's sector requests and removes
the conflict. Records: `measure/lab/results/pool_layout/sm_86.json`, `sm_86.counters.json`.

<a id="sparse-pass"></a>

## `sparse_pass`

The pass with each warp's live tokens drawn at a density (the CTA union every token, four passes a window): the
owner part's fill through a token permutation, the restage, the fold step at the arm's registers, the fp16 join
staggered, the window's lists two passes ahead. Forms: serial (today's owner part); a value tile with two pool slots;
values in place from swizzled pool rows with two or three pool slots; the fold side's rows listed (entries built ahead,
free here) or derived in the step from the warp's union words; and LOAD-AHEAD (fp32 arm only, 232-236 registers): a
warp takes up to three steps' operands into registers at a time and signals its slots drained after its last loads,
not its last MMA, its restage share after the loads of the chunk holding its mid-pass step. The later forms take the fixed
work apart (fp32 arm): the factor products precomputed (an 8-byte plane read in place), those and the dense digits
precomputed as f16 (read in place by `ldmatrix` from 192-byte pool rows, chunk `c ^ ((r >> 1) & 3)`: no restage, no
tile, no staged signal), and the pass's rows built once at its start (word-parallel over its four words) in place of a
step at a time. The block's span a pass over the busiest scheduler's
paired steps at tensor time (busiest-scheduler steps a pass: 16.0 / 5.6 / 2.0):

| pass (fp32 arm / fp16 arm) | all live | 1 in 4 | 1 in 16 |
|---|---|---|---|
| serial, value tile, listed | 1.26 / 1.66 | 1.60 / 2.39 | 2.54 / 4.27 |
| pipelined, value tile, derived | 1.13 / 2.06 | 1.63 / 2.78 | 2.48 / 4.47 |
| pipelined, in place, 2 pool slots, derived | 1.33 / 2.05 | 1.85 / 3.05 | 2.88 / 5.09 |
| pipelined, in place, 3 pool slots, derived | 1.09 / 2.09 | 1.60 / 2.69 | 2.26 / 4.13 |
| pipelined, in place, 3 pool slots, listed | 1.21 / 1.97 | 1.55 / 2.35 | 2.05 / 3.53 |
| load-ahead, in place, 3 pool slots, derived | 1.13 / - | 1.30 / - | 2.06 / - |
| load-ahead, in place, 3 pool slots, listed | 1.14 / - | 1.26 / - | 1.88 / - |
| load-ahead, derived, factor products precomputed | 1.13 / - | 1.28 / - | 1.96 / - |
| load-ahead, derived, products and f16 dense precomputed (no restage) | 1.13 / - | 1.25 / - | 1.94 / - |
| no restage, rows built a pass at its start | 1.14 / - | 1.20 / - | 1.91 / - |

Three pool slots are what in-place reads need: with two, a slot refills only after its pass drains, so the fill and
the restage land on the pass boundary and the warps couple (1.33x dense fp32). Dense, three slots derived is the best
form (fp32 1.09x). Sparse, no pipeline beats serial at 1 in 4 (all 1.58-1.84x), and at 1 in 16 three slots listed is
best (2.06x against 2.53x): a warp's few steps leave the mid-pass point near its pass's end, so the half-pass slack
shrinks to nothing, and a warp alone on its scheduler cannot hide its own chain. Load-ahead is what sparse needed: a slot frees when every warp
has loaded from it, so the fill and the restage stop waiting on the slowest warp's MMAs and the warps drift -- 1 in 4
1.27-1.30x against 1.55-1.60x, 1 in 16 1.87-2.07x against 2.05-2.26x, at ~3% dense (its chunked loads; 1.13x against
1.09x). Derived rows against listed: better dense, 3-10% worse sparse (a step scans up to four sparse words); the
listed rows' build is free here and is not in the owner part.

The per-warp phase ledger (`LEDGER` 1, a lap clock at each phase) says where a sparse pass goes:

| form, density (K cycles a warp a pass) | start waits | rows | ldmatrix | coefficient | mid-pass | MMA issue | end |
|---|---|---|---|---|---|---|---|
| load-ahead, derived, 1 in 4 | 0.52 | 1.88 | 0.13 | 0.35 | 0.89 | 4.07 | 0.04 |
| load-ahead, derived, 1 in 16 | 0.26 | 1.22 | 0.05 | 0.14 | 0.97 | 1.69 | 0.04 |
| load-ahead, listed (free entries), 1 in 4 | 0.60 | 0.70 | 0.13 | 0.75 | 1.03 | 4.13 | 0.07 |
| load-ahead, listed (free entries), 1 in 16 | 0.38 | 0.40 | 0.05 | 0.31 | 1.13 | 1.72 | 0.07 |
| no restage, derived, 1 in 4 | 0.25 | 1.87 | 0.14 | 0.40 | 0.78 | 4.07 | 0.05 |
| no restage, derived, 1 in 16 | 0.36 | 1.21 | 0.06 | 0.16 | 0.64 | 1.78 | 0.05 |
| no restage, rows a pass, 1 in 4 | 0.34 | 0.96 | 0.16 | 0.62 | 0.91 | 4.16 | 0.05 |
| no restage, rows a pass, 1 in 16 | 0.49 | 0.75 | 0.08 | 0.25 | 0.79 | 1.74 | 0.05 |

Rows derived a step at a time are the largest non-MMA piece (a word loop, warp syncs and scratch round trips each
step: ~650-940 cycles a step in sparse passes); built once a pass they cost about half, 1 in 4 1.25x -> 1.20x. The
precomputed forms take the restage out of the mid-pass work (-0.1 to -0.3K a warp-pass; 1 in 4 1.30x -> 1.25x, 1 in 16
2.06x -> 1.94x, dense unchanged). At 1 in 16 a warp has about one step a pass, and its fixed work a pass (rows ~0.75K,
mid-pass ~0.8K, waits ~0.5K) still outweighs that step: the 128-row pass is too small a unit there. The fp16 arm reads 1.95-2.09x dense: every pipelined form sits at 251-254 registers, and the bookkeeping
(rows, cursor, lists, waits) leaves its steps no room -- the pipeline needs register headroom on that arm; its rows
also move between runs (1 in 4, three slots derived: 4.42x in an earlier 3-repetition run, 2.69x here). Record:
`measure/lab/results/sparse_pass/sm_86.json`.

<a id="state-publish"></a>

## `state_publish`

The composed owner kernel's hand-off each window: the fold's warps own 8 R-lists x 16 S-digits (128 fp32 registers),
the readout's 2 S-digits x 64 R-leaves (64 f16 registers), so the publish transposes 1024 leaves x 32 channels across
warps: f16 stores at `[S][R][channel]` (swizzled; a padded variant gives each S-digit's block 16 more bytes), a barrier
or an mbarrier, each warp's slice pulled by `ldmatrix.x4.trans`. sm_86: the publish alone 1572 cycles a window,
padded or not -- the padding removes the stores' 2-way bank conflicts (ncu) and changes nothing, since 64 KB written and
64 KB read (~1K cycles at 128 bytes a clock) plus two syncs bind it. In windows of fold and readout MMAs (a dense window
32 fold steps and 64 read tiles a warp, a sparse one 8 and 16), the block's span a window:

| window | dense | sparse |
|---|---|---|
| fold + readout | 98.1K | 24.3K |
| + publish (barrier) | 99.7K | 25.7K |
| + boundary work (barriers) | 100.0K | 26.2K |
| + boundary, then publish | 102.1K | 28.1K |
| + boundary interleaved with publish | 101.9K | 28.0K |
| + publish by mbarrier over the read start-up (padded) | 101.0K | 27.2K |

After barriers (the stand-in lists phase), the warps reach the publish together and its whole cost shows; behind an
mbarrier, with the readout's start-up between a warp's stores and its pull, about half hides (+1.0K dense, ~1%; +0.95K
sparse, ~4%). Registers peak at 222-224 (the fold state and the pulled slice), no spills. Records:
`measure/lab/results/state_publish/sm_86.json`, `sm_86.counters.json`.

<a id="pass-fixed"></a>

## `pass_fixed`

The fold's fixed work alone, one block a SM: the fill (a pass of 128 scattered rows, waited each pass), the restage (a
pass's 256 items) and the window's entries (8 warps' 16 union words). sm_86: the fill a pass, DRAM / L2 -- as built
2.35K / 1.37K cycles (47 instructions a row: four factor planes, the u8 dense chunk), the factor products precomputed
2.22K / 1.29K (39 a row), with f16 dense digits too 2.47K / 1.29K (more bytes from DRAM), 32-bit offsets from window
bases no better; the restage 294 cycles a pass, 224 with the products precomputed; the entries a window, 1 in 4 / 1 in
16 -- a serial loop over set bits 2.10K / 1.28K, word-parallel by popc 2.65K / 2.62K (sixteen word iterations whatever
the density). Inside the load-ahead pipeline the fill and restage instructions are ~4-7% of a sparse pass; what costs
there is in `sparse_pass`'s ledger. Record: `measure/lab/results/pass_fixed/sm_86.json`.

<a id="list-fold"></a>

## `list_fold`

The fold's step forms for the carve re-price, 8 warps (two a scheduler), fp32 accumulate, the kernel's 128 accumulator
registers live, 256 rows resident (160 bytes: level S's 32 f16 digits, 32 f16 values, level R's 32 u8 digits; the
halves of each 32-byte pair swapped on rows with bit 2 set). PER LIST: a warp steps each list over its own tokens'
rows (gathered by hash, or the pass's own contiguous 16), a row past the list's live count reading row 0 with c = 0;
MERGED: a warp steps the union, every list's MMAs on every token, the operands loaded once. Two steps' operands loaded
ahead of their MMAs; the block's span a pass against its tensor floor (the busiest scheduler's two warps' MMAs at 32
cycles). sm_86:

| form (a warp's lists x level-S digits x channels) | steps | live | rows | x floor |
|---|---|---|---|---|
| merged, 8 x 16 x 32 | 8 | 16 | contiguous | 1.055 |
| merged, 4 x 32 x 32 | 8 | 16 | contiguous | 1.081 |
| per list, 4 x 32 x 32 (each list reloading the same rows) | 8 a list | 16 | contiguous | 1.262 |
| the same, staged (step s + 1's loads before step s's MMAs) | 8 a list | 16 | contiguous | 1.349 |
| per list, 8 x 16 x 32 | 8 a list | 16 | contiguous | 1.977 |
| per list, 4 x 32 x 32 | 2 a list | 16 | gathered | 1.424 |
| per list, 4 x 32 x 32 | 2 a list | 8 | gathered | 1.167 |
| per list, 4 x 32 x 32 | 1 a list | 8 | gathered | 1.166 |
| per list, 8 x 16 x 32 | 2 a list | 16 | gathered | 2.160 |
| per list, 8 x 16 x 32 | 1 a list | 8 | gathered | 1.706 |

At dense, per list (all four lists reloading the same rows) is 25.8K cycles a pass against the merged form's 18.8K at
the same tile: 1.37x, from four times the operand loads and a mass atom a list step. Per list at today's 16 x 32 tile
is operand-bound (1.7-2.2x); at 32 x 32 a half-full step (alt-k4's ~8 tokens a list a window) runs 1.17x its floor
even with gathered rows, a full gathered step 1.42x (16 random rows share bank groups where 8 pads on row 0
broadcast). The staged form does not interleave: ptxas keeps the next step's loads in a block ahead of the MMAs, each
entry's pad check a branch that fences the scheduler (its SASS), and it is slower. Record:
`measure/lab/results/list_fold/sm_86.json`.

<a id="read-step"></a>

## `read_step`

The readout's step on the even carve: a warp owns 4 level-S read digits x 32 level-R digits x 32 channels, its slice
as f16 B fragments (K = level R, permuted so a lane's four k rows are one token's four consecutive digits) beside the
fold's 128 fp32 accumulators (registers 240-254); a step's A = 16 tokens' level-R bytes by one `ldmatrix.x4` (96-byte
read rows) and a PRMT + HFMA2 a register at the token's level-S coefficient; 8 fp16-accumulate MMAs and 2 for den; the
outputs to s16 pairs (two channels a word) added by `red.shared.add` (ATOMS) into a 128-slot panel of 17-word rows; the
drain two slots a warp request by `red.global.add.u32`, each word cleared. PER DIGIT steps each digit over its own
gathered tokens and adds after each; MERGED sums the warp's 4 digits over the union in registers and adds once. The
block's span a pass against its tensor floor (the busiest scheduler's MMAs at 16 cycles). sm_86, cycles a pass:

| form | steps | steps alone | + panel adds | + drain burst | + drain spread | + burst, aligned | + spread, aligned |
|---|---|---|---|---|---|---|---|
| merged | 8, 16 live | 12.5K | 11.8K | 13.7K | 14.7K | 13.0K | 13.9K |
| per digit | 8 a digit, 16 live | 17.4K | 32.0K | 33.5K | 35.6K | 32.6K | 35.8K |
| per digit | 2 a digit, 16 live | 4.46K | 8.28K | 9.91K | 10.7K | 9.31K | 9.84K |
| per digit | 1 a digit, 8 live | 1.66K | 3.32K | 5.58K | 5.79K | 4.31K | 4.72K |

"Aligned" drains the channel words alone to 64-byte target rows (den carried elsewhere); the other drains write 17-word
rows (channels and den) to a 17-word pitch, den a request of its own: ncu counts 316 L2 reduction requests a CTA a pass
for their 128 RED instructions -- a two-token request spans three or four lines, den two. The panel adds cost about as
much as a per-digit step itself (~5 cycles a warp's ATOMS across the SM); merged adds once a token for its four
digits. The drain costs ~1.0K cycles a 128-slot pass aligned (2.3K misaligned with den), and spreading it through the
next pass's steps under a double panel and mbarriers costs 0.4-1K more than the burst, not less: the drain's
instructions and waits land in the steps. Record: `measure/lab/results/read_step/sm_86.json`.

<a id="publish-rounds"></a>

## `publish_rounds`

The even carve's publish: the fold's warps own 4 level-R digits x 32 level-S x 32 channels (fp32), the readout's 4
level-S x 32 level-R x 32 channels (f16 B fragments), a transpose of 32K values a window through `[S][R][channel]` (an
S block 16 bytes past 2 KB, channel chunks swizzled by `(R >> 1) & 3`), whole (64 KB) or in four rounds of 16 KB (a
level-S quarter, two readout warps' slices each) under mbarriers. In windows of fold steps (32 fp32-accumulate MMAs a
step) and readout tiles (32 fp16-accumulate MMAs a tile), the block's span a window, sm_86:

| publish | dense window | sparse window |
|---|---|---|
| none | 96.6K | 24.4K |
| whole, barriers (64 KB) | 99.5K | 25.8K |
| whole, mbarrier (64 KB) | 99.5K | 26.0K |
| four rounds, one 16 KB buffer | 100.6K | 26.8K |
| four rounds, two buffers (32 KB) | 100.0K | 26.3K |

Rounds double-buffered cost 0.4-0.6K a window over the whole publish, single-buffered 1.0-1.1K: the rounds'
handshakes, not their bytes. Record: `measure/lab/results/publish_rounds/sm_86.json`.

<a id="den-sweep"></a>

## `den_sweep`

The carry's window-aligned den as a pass of its own (den depends only on routing: the read leaf product against the
write mass of the windows before -- `tests/oracle/reference.py`), so the readout could add at a common unit known
beforehand. A CTA owns 16 level-S rows x all 256 level-R digits of a head's mass, a thread a digit's 16 values in fp32
registers. Each window the mass is published f16 at logical columns (the R order that makes a lane's operand bytes one
contiguous run); the den is MMAs over the window's tokens read-live on any of the CTA's rows (A = the token's u8
level-R read row from global, PRMT to f16; B = the published mass by `ldmatrix`; N = the 16 rows), each row times the
token's level-S read coefficient, summed, one `red.global.add.f32` a token; the writes, each thread over its digit's
list (digit-major; token and u8 value) against the f16 level-S write coefficients and the gains, 16 FFMAs an entry with
three entries in flight. Half the warps write first. Each warp finds its tiles' tokens from the union of the rows'
facts words by binary search (no shared list); lists, coefficients and gains staged a window ahead by `cp.async`, the
words two, a window's first tile's read rows during the window before. A routing pool per head (five heads, 22 MB,
streamed from DRAM); every den within 1% of a double reference before timing. sm_86 at base clock, cycles a
CTA-window (16 CTAs a head; a head at 64K tokens is 16 x cycles x 128 windows over 80 SMs at 1.36 GHz):

| write k, read k | full | den alone | writes alone | skeleton | DRAM a CTA-window | a head |
|---|---|---|---|---|---|---|
| 4, 4 | 9.83K | 6.98K | 5.78K | 2.72K | 35.9 KB | 0.185 ms |
| 16, 16 | 25.1K | | | | 33.9 KB | 0.47 ms |
| 4, 256 | 19.6K | | | | 33.7 KB | 0.37 ms |

The skeleton (the staging, the publish, the barriers) runs at 89% of DRAM peak (793 GB/s): re-reading the routing is
the pass's floor, 36 KB a CTA-window or ~4.9K cycles at alt-k4, and the full pass is 2.0x it. The den's MMAs compute
all 16 rows for each read-live token, and at read k 4 a token reads about one of them: its 249 MMAs a CTA-window (2K
cycles a scheduler) are mostly waste, so a sparse read level wants a per-pair FFMA form (8 lanes a pair over the 256
digits); a dense read level keeps the MMAs (1024 a CTA-window, 8K cycles a scheduler, 19.6K with the writes). The
writes are divergent (a warp runs its longest digit, about twice the mean at k 4) and their coefficient load waits on
the entry's token, hence the token two entries ahead; f16 coefficient rows are 32 bytes, so random tokens spread over
four bank groups (eight with the swap on bit 2), where fp32 rows' two quadrupled the conflicts. A dense write level
would take the carry fold's MMA form at one channel (1024 MMAs a CTA-window); not built here. Record:
`measure/lab/results/den_sweep/sm_86.json`, counters `sm_86.counters.json`.
