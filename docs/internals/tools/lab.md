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

An experiment that reads data -- a cell's routing, say -- has a `<name>.inputs.py` beside it whose `write(directory)`
writes the files and returns their names. `run` and `ncu` call it first (in the scratch space), pass the binary
`--inputs <directory>`, and the record carries each file's digest under `inputs`.

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
* `lab::WarpSpan` — a warp's start and end `clock64`; `Bench::span(warps)` is the mean block's span (last end less
  first start), the throughput when warps finish apart.
* The operand pieces the kernels use: `ldsm` and `ldsm_t` (`ldmatrix.x4`, `.trans`), `hmul2`, `pack_half2`, `smem`;
  the mbarrier as `ops.cuh` spells it: `mbar_init`, `mbar_arrive`, `mbar_wait` (polled), `mbar_test` (once).
* `lab::sink(out, v)` — a fold of everything a thread computed, stored unconditionally.
* The atoms as one PTX instruction each: `mma_f16_f32` (and `_fresh`), `mma_f16_f16`, `mma_bf16_f32`, `mma_u8u8`,
  `mma_u8s8`.
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

<a id="mma-queue"></a>

## `mma_queue`

Whether a warp can issue MMAs ahead of the tensor pipe -- fill a queue, then go on to other work while it drains.
One warp a scheduler; each repetition reads the SM clock, issues K independent MMAs into fresh accumulators (their A
operand tied to the clock read by a run-time zero, so none issues before it; the SASS keeps clock, MMAs, clock in
order), reads the clock again -- in-order issue: after the K-th MMA has issued -- then waits for the outputs, so the
next repetition starts on an empty pipe. sm_86, cycles from clock to clock:

| K back to back | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| fp16-accumulate HMMA | 60.0 | 76.0 | 92.0 | 108.0 | 172.0 | 236.0 | 300.0 |
| fp32-accumulate HMMA | 60.0 | 92.8 | 124.0 | 155.5 | 282.2 | 411.5 | 539.8 |

Linear from the second MMA: each issues one MMA's pipe time after the one before (16.0 and 32.1 cycles), so there
is no queue -- the pipe takes the next MMA only when it has taken the last, and a warp issuing a burst of N is held
at it for N times that. (The 60 is the clock read's latency, the operand's and the first MMA's.) An MMA therefore
needs an issue slot at the cycle the pipe frees, and any other warp issuing then costs the pipe that cycle; Hopper's
`wgmma` is asynchronous, which is what lets a warpgroup issue its MMAs and leave. Record:
`measure/lab/results/mma_queue/sm_86.json`.

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
| per list, 4 x 32 x 32 | 2 a list | 16 | gathered, residue-ordered | 1.263 |
| per list, 4 x 32 x 32 | 2 a list | 8 | gathered, residue-ordered | 1.107 |
| per list in one block, c precomputed, 4 x 32 x 32 | 8 a list | 16 | contiguous | 1.015 |
| the same | 2 a list | 16 | gathered, residue-ordered | 1.017 |
| the same | 2 a list | 8 | gathered, residue-ordered | 1.017 |
| the same | 1 a list | 8 | gathered, residue-ordered | 1.030 |

At dense, per list (all four lists reloading the same rows) is 25.8K cycles a pass against the merged form's 18.8K at
the same tile: 1.37x, from four times the operand loads and a mass atom a list step. Per list at today's 16 x 32 tile
is operand-bound (1.7-2.2x); at 32 x 32 a half-full step (alt-k4's ~8 tokens a list a window) runs 1.17x its floor
even with gathered rows, a full gathered step 1.42x (16 random rows share bank groups where 8 pads on row 0
broadcast). The staged form does not interleave: ptxas keeps the next step's loads in a block ahead of the MMAs, each
entry's pad check a branch that fences the scheduler (its SASS), and it is slower.

THE PER-LIST STEP AT ITS FLOOR (form 5, 2026-10-07). KERNEL_STANDARDS 27's shape -- a pass one block (every list's
steps unrolled, step i + 1's operands loaded before step i's MMAs into a two-slot ring at fixed registers), entries
as byte offsets (`at(row, 0)`: a lane's chunk bit is lane-fixed, so its row is an XOR and the m-tile and n-chunk
offsets immediates), pads masked (row 0, c = 0), each token pair's c precomputed as f16 `{c_v, c_m}` pairs (what the
facts pass would emit, so the step only multiplies A by it), the mass accumulated in place (no fresh fragment and its
adds) -- with gathered rows residue-ordered (each 8 of a step distinct mod 8: the rows' bit-2 swap then lands an
`ldmatrix`'s 8 rows on 8 bank groups). Non-MMA instructions an MMA in the step loop, the same for sm_80 and sm_86:
2.5 / 3.4 / 4.7 at 8 / 2 / 1 steps a list, against 13.4-17.0 for form 1 and 5.6 for the merged form -- under sm_80's 7
issue slots an 8-cycle MMA at every density. Timed, 1.015-1.030x. Residue order alone takes form 1's full gathered
step from 1.374x to 1.263x (ncu: 58M bank conflicts in 145M wavefronts hashed, 0.06M ordered); the shape and the
precomputed c take the rest. Tensor active 49% of the fp16 peak, the fp32-accumulate pipe's whole rate on this card.
Records: `measure/lab/results/list_fold/sm_86.json`, `sm_86.counters.json`.

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
| merged | 8, 16 live | 12.0K | 11.8K | 13.6K | 14.6K | 13.0K | 13.8K |
| per digit | 8 a digit, 16 live | 14.6K | 31.2K | 32.0K | 36.5K | 31.3K | 35.9K |
| per digit | 2 a digit, 16 live | 3.75K | 8.11K | 9.54K | 10.6K | 8.93K | 9.92K |
| per digit | 1 a digit, 8 live | 1.54K | 3.27K | 5.61K | 5.77K | 4.23K | 4.68K |

"Aligned" drains the channel words alone to 64-byte target rows (den carried elsewhere); the other drains write 17-word
rows (channels and den) to a 17-word pitch, den a request of its own: ncu counts 316 L2 reduction requests a CTA a pass
for their 128 RED instructions -- a two-token request spans three or four lines, den two. The panel adds above are each
a branch on its token (`if (in) red.shared`), issued right after their own step's MMAs; merged adds once a token for
its four digits. The drain costs ~1.0K cycles a 128-slot pass aligned (2.3K misaligned with den), and spreading it
through the next pass's steps under a double panel and mbarriers costs 0.4-1K more than the burst, not less: the
drain's instructions and waits land in the steps. The coefficient loads are unconditional (a pad's slot is a real row,
its value masked), which took the steps alone from 17.1K / 4.39K / 1.66K.

THE DEN PASS'S STEP (8 MMAs, den out), its adds issued after the next step's MMAs:

| per digit | 8 a digit | 2 a digit | 1 a digit, 8 live |
|---|---|---|---|
| MMA floor | 8.19K | 2.05K | 1.02K |
| adds a step behind, each a branch | 26.4K (3.2x) | 6.75K (3.3x) | 2.60K (2.5x) |
| + operands a step ahead, a switch on the digit and a guarded prefetch | 34.9K | 8.92K | 3.79K |
| + no branch: loads unconditional, adds a value masked to zero | 16.9K (2.1x) | 4.60K (2.2x) | 2.32K (2.3x) |
| + the sequence unrolled: rings at fixed registers | 12.8K (1.56x) | 3.42K (1.67x) | 2.03K (1.99x) |
| + the coefficient after the MMAs, in the conversion | 12.2K (1.49x) | 3.31K (1.62x) | 2.11K (2.06x) |
| the same, converted by F2I (an f16 multiply, then the conversion unit) | 12.7K | 3.49K | 2.47K |
| the coefficient after the MMAs, entries as byte offsets | 12.4K (1.52x) | 3.25K (1.59x) | 1.94K (1.89x) |
| + the conflict-free panel (20-word rows, pads to their own rows) | 12.4K (1.52x) | 3.26K (1.59x) | 1.67K (1.64x) |
| isolation: without the conversion (values wrong) | 9.60K (1.17x) | 2.79K (1.36x) | 1.46K (1.43x) |
| isolation: without the adds (values wrong) | 8.89K (1.09x) | 2.34K (1.14x) | 1.22K (1.19x) |
| isolation: the conversion independent of the outputs | 11.7K (1.43x) | 3.14K (1.54x) | 1.89K (1.85x) |
| isolation: half the values converted | 11.4K (1.39x) | 2.96K (1.44x) | 1.54K (1.51x) |
| isolation: the conversion replaced by as many dependent LOP3/PRMT | 13.6K (1.66x) | 3.78K (1.85x) | 1.76K (1.72x) |
| isolation: those integer operations independent of the outputs | 14.4K (1.76x) | 3.69K (1.80x) | 1.95K (1.90x) |
| 12, MMA j reading output word j - 1 masked to zero | 12.2K (1.49x) | 3.15K (1.54x) | 1.75K (1.71x) |
| the same, each word also reading an MMA's output | 12.7K (1.55x) | 3.24K (1.58x) | 1.77K (1.73x) |
| 12, a scheduler's two warps taking turns at the MMAs (named barriers) | 14.2K (1.73x) | 3.60K (1.76x) | 1.89K (1.85x) |
| 12 with the level-R side native f16 (A loaded as it is, no unpack) | 10.7K (1.31x) | 2.89K (1.41x) | 1.57K (1.53x) |
| the same, accumulating in fp32 (half rate on this card) | 17.6K (2.15x) | 4.48K (2.19x) | 2.31K (2.25x) |

Merged, adds a step behind: 10.5K (1.28x). The aligned burst drain adds ~1K a pass. The last form converts on the FMA
pipe (each half to fp32, x 64 + 1.5 x 2^23, the low halves packed: round-to-nearest, exhaustively checked over every
in-range half) in place of `F2I.S16.F16` on the quarter-rate conversion unit; it moved nothing at 8 steps a digit
(16.9K either way) -- issue binds, not that unit. ncu, the first branchy form: issue 40%, tensor 31%, 163 instructions
a digit-step, stalls on fixed-latency and short-scoreboard waits and branch resolution; the unrolled form issues 132
a digit-step, against 128 cycles of MMAs a warp-step: two warps a scheduler leave it at the edge of issue-bound.
Applying each token's level-S coefficient after the MMAs (it factors out of the sum over level-R digits: A is then
the raw bytes, unpacked by PRMT less 1024, and the coefficient times the unit rides in the conversion's FFMA, a pad's
scale zero, so nothing is masked) issues 118 a digit-step; the rest is the conversion (40: each half to fp32, the
FFMA, the pack, for 16 values a step), the unpack (16), the MMAs and adds (16) and addresses. A shared f16x2 atomic
(`red.shared.add.noftz.f16x2`) compiles on sm_86 to a compare-and-swap loop (`ATOMS.CAST.SPIN`), not an add, so an
f16 panel is no way around the conversion here. Converting by F2I instead (24 instructions a step against the FMA path's 40) is slower at every
density: the quarter-rate conversion unit binds before issue does. Entries carrying their byte offsets (a token's row with the swizzle folded in,
its panel row and a pad bit, as the facts pass could emit them) bring the step to 105 instructions a digit-step:
issue fits under the MMAs (two warps 210 against 256 cycles). What remains is the measured tax (KERNEL_STANDARDS 26.3
a): beside fp16-accumulate MMAs an FP32 instruction costs ~1.0 cycle of the chain, so the conversion's 16 FFMAs and
the unpack's packed adds and permutes add ~30 cycles to a 128-cycle step -- a tax-inclusive floor of ~1.23x against
the 1.5x measured at 8 steps a digit.

THE GAP TAKEN APART. The panel's 17-word rows put the 8 tokens of an atomic add on overlapping banks (~3 wavefronts an
add, 33 a step against 9 without adds); 20-word rows land slots of distinct residue mod 8 on disjoint 4-bank groups (the
tokens of each step's aligned 8 already hold distinct residues -- the facts pass's residue order), and pads adding zero
each to slot 0 serialized on one word, so they add to 8 dummy rows: 0 atomic bank conflicts, 17 wavefronts a step,
-14% at 8 live tokens, nothing at full steps -- shared memory was not what bound them. Removing the conversion (the
words added raw) or the adds as well isolates the rest: at 8 steps a digit the operands and MMAs run 1.09x, the adds
take it to 1.17x and the conversion to 1.52x -- 34 of the 52 points; at 2 steps a digit 14 / 22 / 23, at 1 step 19 / 24
/ 21.

THE CONVERSION TAKEN APART (8 steps a digit; a scheduler's floor is two warps' 8 MMAs, 257 cycles a step). Its 40
instructions a warp-step (16 `HADD2.F32` to fp32, 16 `FFMA`, 8 `PRMT`) cost 0.345 of the floor over the raw adds.
Computed on values independent of the MMA outputs (each word XORed with it, so the same instructions issue, none on
the output-to-atomic chain), they cost 0.261; half of them on the chain 0.215. So: the chain's latency 0.08, the
instructions 0.26, linear in their count (0.08 + 0.13 at half). The tensor_tax rates beside fp16-accumulate MMAs
price the instructions at 16 x 1.0 + 16 x 0.13 + 8 x 0.22 = 20 cycles a warp-step, 0.155 at two warps; the other
~0.1 is issue pressure -- the off-chain form issues 94% of the floor's issue capacity (ncu: issue 65.5% at 1.43x;
`not_selected` 0.46 an issue against 0.30 without the conversion). The tax itself is not issue: in `tensor_tax` one
filler warp beside one MMA warp a scheduler leaves 15 free slots an MMA, where collisions would cost ~1/16 cycle an
instruction, and classes of one issue cost tax 1.0 (FP32), 0.48 (packed half), 0.13-0.22 (conversions, LOP3, PRMT).
The integer class is no escape on sm_86: LOP3 and PRMT run at half rate (16 lanes), so 40 of them a step load the ALU
pipe past the floor (ncu: 67.7% of its peak at 1.76x, 119% of the floor's time) -- in the chain 1.66x, off it 1.76x.
At 2 steps a digit the same split: chain 0.055, instructions 0.17.

THE INTERLEAVE IS PTXAS'S. Mode 12's SASS bursts 24 MMAs (about 4 instructions apart), then runs ~200 instructions
without one. Written MMA by MMA in the source (each MMA, then one of the last step's output words), the SASS came out
identical to 12's. Per-instruction samples (ncu source counters) put a warp at an MMA 49% of the time -- 28% waiting
on the busy tensor pipe, 7% ready but not selected while the other warp issued (~13% of a scheduler's cycles with
the pipe free) -- and the conversion mostly issuing: no instruction class favors a ready MMA. A zero-valued
dependency places the MMAs (MMA j's accumulator takes output word j - 1 ANDed with a run-time zero, `passes >> 31`):
the longest MMA-free run in the step falls from 208 to 83 instructions, not-selected MMAs to 3.8%, and the step to
1.49x. Tied both ways (each word also reads an MMA's output) it falls to 54, but the MMA latency joins the chain:
1.55x. Turn-taking -- the two warps of a scheduler alternating at a step's MMAs under named barriers (`bar.sync`,
MMAs, `bar.arrive`), each converting while the other's MMAs run -- does not survive ptxas: the barriers fence memory
only, the MMAs move outside them, and the barriers cost 1.73x. `tools/sass_gate.py` reports the MMA gap (the longest
run in a step loop's body, and around it) and `lab.py sass` prints it.

INSTRUCTIONS AN MMA, NOT THEIR ORDER. The step loop's non-MMA instructions an MMA, the same compiled for sm_80 and
sm_86 (`lab.py sass read_step --arch sm_80`): 12.2 for mode 12; 9.6 with the level-R side native f16 (rows of 32
halves, 64 bytes, chunks swizzled by `(slot >> 1) & 3`; two `ldmatrix` a step and no PRMT/HADD2 unpack); 7.6 with
that and fp32 accumulation (no `HADD2.F32`; outputs converted by the FFMA and packed). Removing the unpack's ~21
instructions a step took the step from 1.52x to 1.31x at 8 steps a digit (ncu: 693M instructions against 864M,
tensor 75% against 65%) -- more than the tax rates price the unpack at (~0.05), since near issue saturation each
instruction also costs contention. The fp32-accumulate form runs at half rate here (32-cycle MMAs, 2.15x the fp16
floor, 1.07x its own); a full-rate card (A100, A6000) is where it would run, and its time there is unmeasured. sm_80's
shadow is 8 cycles an MMA, 7 issue slots: 7.6 is the first form near it.

A per-digit step's gathered slots are distinct (a list never holds a token twice). The gate: `read_step`'s branchy
forms declare their branches (KERNEL_STANDARDS 27). Records: `measure/lab/results/read_step/sm_86.json`,
`sm_86.counters.json`, `sm_86.source.json` (the per-instruction samples, by class and stall reason).

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

<a id="gang-ring"></a>

## `gang_ring`

Whether sidecar CTAs can build each window's per-list slot lists from the bitlists into a ring of R windows fast
enough for the carry CTAs consuming them, in ONE PERSISTENT COOPERATIVE GRID (a CTA an SM, every CTA resident by
`cudaLaunchCooperativeKernel`, so spin-waits cannot starve a consumer). A synthetic head: 64 windows, each token live
in k of 256 level-R digits; carry CTA c owns group c % 8 (32 digits, 4 lists a warp) and steps them in `list_fold`
form 5's shape over a static pool (runtime list lengths, two operand slots, the odd step peeled), each warp staging
its lists a window ahead by cp.async and freeing the slot; a sidecar builds a (window, group)'s 32 lists -- the
union's ranks, each digit's entries round-robin by rank & 7, steps of 16, each entry its pool row offset and f16
coefficient pair (the level-R byte, staged from the rows, times the token's (F, m)) -- after the gang freed the slot
(window w - R staged by every carry warp). Spans are each CTA's own (SMs' clocks do not compare), the slowest CTA's
taken; floor = the busiest scheduler's paired warps' MMAs at 32 cycles. sm_86, cycles a window (the build as four
teams of two warps a sidecar, each team an item under its own named barrier; still issue-heavy, ~6K warp-instructions
an item at k 4):

| k | lists given (76 carry CTAs) | build alone, 8 sidecars (a sidecar) | ring of 3, 8 sidecars | ring of 3, 4 sidecars | ring of 2 / 4, 4 sidecars |
|---|---|---|---|---|---|
| 4 | 4.00K (1.47x) | 4.29K (32.1K) | 4.92K (1.80x) | 9.50K (3.48x) | 9.54K / 9.51K |
| 16 | 7.43K (1.04x) | 9.54K (71.9K) | 10.4K (1.46x) | 20.9K (2.91x) | 21.1K / 21.3K |

The ring works -- no deadlock, depth 2-4 alike, the consumers waiting only on the build -- and the build binds: a
sidecar's throughput 4.0K (k 4) / 9.0K (k 16) cycles an item. The first cut (a CTA an item, serial thread-0 prefixes,
scattered 2-byte global stores) ran 17.6K / 33K; the second (a lane a digit, a warp a word pair, the union a warp OR,
prefixes by shuffles, lists assembled in shared memory and copied out whole) 7.7K / 11.3K; the third (four teams of
two warps, the coefficients gathered four entries a thread) above. The ncu source profile of the third: fixed-latency
chains 28%, the gathers 17%, the per-entry round-robin position (~35 instructions) and a serial pad loop. The comparison here is the fold's steps alone; a carry window (fold, readout, fill) is ~20K (alt-k4 target) to
~61K (alt-k16). Record: `measure/lab/results/gang_ring/sm_86.json`.

<a id="smem-rates"></a>

## `smem_rates`

What an SM's shared-memory pipe sustains, for the hardware configuration (`common/hardware.json`): one CTA an SM, 8 or 16 warps, each issuing
independent operations at addresses that change every iteration (volatile asm; an address that repeats each iteration
lets ptxas hoist a plain `ld.shared` out of the loop -- the first build did, and timed a loop without its loads). The
block's span over the wavefronts the SM issued, the wavefronts confirmed by ncu
(`l1tex__data_pipe_lsu_wavefronts_mem_shared`). sm_86, cycles a wavefront (8 warps; 16 the same):

| operation | wavefronts an instruction | cycles a wavefront | cycles an instruction |
|---|---|---|---|
| `ld.shared.v4` (LDS.128) | 4 | 1.01 | 4.0 |
| `ldmatrix.x4` (LDSM) | 4 | 1.00 | 4.0 |
| `red.shared.add.u32` (ATOMS.ADD), a lane a bank | 1 | 2.00 | 2.0 |
| the same, two lanes a bank | 2 | 1.00 | 2.0 |
| the same, four lanes a bank | 4 | 1.00 | 4.0 |

Loads run at the documented bank rate, a wavefront a cycle. A shared atomic takes the larger of two cycles an
instruction and a cycle a wavefront: a two-way conflict between the warp's halves is free. Records:
`measure/lab/results/smem_rates/sm_86.json`, `sm_86.counters.json`.

<a id="den-sweep"></a>

## `den_sweep`

The carry's window-aligned den as a pass beside the carry (den depends only on routing: the read leaf product against
the write mass of the windows before -- `tests/oracle/reference.py`), so the readout could add at a common unit known
beforehand, with a second channel beside it: the same product against the mass of F = m x the writing token's value
scale (its largest value channel's magnitude) is num, and num / den, a weighted average of the read tokens' value
scales, bounds every readout channel of the token, so its unit is known before the readout too. At 8 CTAs a head (64
level-S rows x 128 level-R digits each, 16 warps), on the cells' own routing: `den_sweep.inputs.py` writes each probe
cell at the flagship widths with 32 windows (`flagship-{alt-k4,alt-k16,cohort-k16,dense}-L16384`) as the facts pass
hands it to the carry, `fm`'s (F, m) the gains. A sparse side's MMA takes its N from the sparse level as 4 of its
digits x the two channels, an entry live in two of N's 8 columns; a dense side takes an n-tile a channel.

The list form (a cell whose level R writes and level S reads are sparse). Writes (f16 x f16 -> fp32): a warp owns 8
level-R digits, two MMAs' N; a 4-digit tile's list entries are K, arranged so each aligned 8 holds distinct token & 7
(as the facts pass could emit them); the 64 rows M; the mass the accumulators, held at the window's gain scales
(powers of two the facts pass would emit, the gains pre-scaled, the accumulators rescaled when they change); A = the
tokens' staged f16 coefficients by `ldmatrix.trans` (a token's 16-byte chunks XORed with token & 7), B = an entry's
coefficients (its gain x value a channel) in its columns. The den (u8 x u8 -> s32): a warp owns 4 level-S rows (N) and
its band's tiles of 16 read-list entries (token, digit, u8 value; M); K = the CTA's 128 digits; A = an entry's u8 read
row as loaded (a lane's contiguous quarter, the K order the published mass follows); B = the mass as 16-bit fixed
point (a column scaled by the power of two putting its largest value in [2^14, 2^15)) in high and low byte planes, two
integer MMAs a 32 digits; an entry's results are its row's two columns (high x 256 + low) x 2^-e x its value, each
added by the lane holding it (den and num side by side a token). The read lists stage once the den is done with the
window's (one buffer of 4096 entries, cohort's lumpiest CTA-window holding 3065), the next window's read rows
prefetching to L2.

The dense form (both sides dense). Writes: K = the window's tokens in chunks of 64, the coefficients and the values
staged double-buffered, B = the values by `ldmatrix.trans` times each token's gains (f16 at the window's scales), an
MMA a channel. The den: every token against every row, a warp's 8 rows (an n-tile a channel) over its half of the
16-token tiles, one tile in flight (two spill at 128 registers), each tile's integer results times the token's level-S
read coefficients, summed over the quad, added once a token a channel a warp.

Every den and num within 1% of a double reference before timing. One head's CTAs; sm_86 at base clock, cycles a
CTA-window, the one-channel pass in brackets (commit 790dfac6):

| cell | full | den alone | writes alone |
|---|---|---|---|
| `flagship-alt-k4-L16384` | 12.7K (11.5K) | 8.45K (7.56K) | 8.83K (7.64K) |
| `flagship-alt-k16-L16384` | 29.4K (26.7K) | 16.4K (15.4K) | 18.1K (15.5K) |
| `flagship-cohort-k16-L16384` | 35.3K (30.8K) | 19.9K (16.9K) | 22.7K (18.9K) |
| `flagship-dense-L16384` (dense form) | 68.9K (39.9K) | 23.9K (13.5K) | 45.8K (26.2K) |

The list form pays for F in structure, not MMAs: a write tile holds half the entries (more padding a tile), a den band
is one warp's 4 rows (the two warps of an 8-row band no longer share its tiles), the publish doubles (the skeleton alone
4.42K on alt-k4, from 4.17K). The dense form pays in MMAs: its writes' HMMAs double to 4096 a CTA-window, 32.8K cycles
at sm_86's 32 a scheduler for fp32 accumulate, and its den's IMMAs double. Earlier revisions on synthetic pools: 16
warps a CTA lifted issue from ~26% to ~38% (commit 5b26266f); the integer den and the residue order (79d058ef); the f16
den (ece3d4d6); FFMA forms, digit-major (0db7bf1e) and token-major (673c2375). Record:
`measure/lab/results/den_sweep/sm_86.json` (the inputs' digests in it), counters `sm_86.counters.json` (a kernel's
median over the cells it ran).
