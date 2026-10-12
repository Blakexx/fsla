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

THE COEFFICIENT FORMED IN THE FOLD (forms 6 and 7, 2026-10-07): what the fold pays when the facts pass emits 2-byte
entries and the warp forms each token's c itself -- the list digit's byte from the token's pool row (f16 through the
mantissa: `0x6400 | b`, less 1024), times the token's `(F, m)` f16 pair, two HMUL2s a token pair. The row now holds
level R's 28 digits and that pair unswizzled after the swizzled level-S and value chunks (form 5 unchanged: 1.015 /
1.018 / 1.030). Form 6's entries are slots (the row's offset and the gains' index computed from each); form 7's are
the rows' offsets, the gains read from the row. Against form 5 at 8 / 2 / 1 steps a list:

| form | sm_86 x floor | non-MMA instructions an MMA (sm_80 = sm_86 SASS) |
|---|---|---|
| 5, c in the entry | 1.015 / 1.018 / 1.030 | 2.46 / 3.44 / 4.70 |
| 6, slot entries | 1.039 / 1.048 / 1.062 | 6.11 / 7.05 / 8.30 |
| 7, offset entries, gains in the row | 1.032 / 1.042 / 1.051 | 5.31 / 6.25 / 7.50 |

Forming c costs ~2.9 instructions an MMA at best (the loads, the byte's conversion, the pairs' PRMTs and HMUL2s);
slots add ~0.8 of address arithmetic. sm_86's fp32-accumulate MMA (32 cycles) hides it, +1.6-2.4%; sm_80's 8-cycle MMA
leaves 7 issue slots, which form 7 fills at two steps a list and overruns at one. The numerics differ too: c from
f16 gains at the window's scale, not fp32 gains at the list's own, so a list of light tokens loses its scaling.
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

<a id="row-span"></a>

## `row_span`

How fast the facts pass's row stream reads DRAM when a work item owns a wider span of a level's digits. A level's
rows are `[T, H, 256]` u8 (a token's head's row contiguous, heads side by side); the facts pass's units own 32 digits,
one 32-byte piece of each token's row. The facts pass's ring (four 1 KB stages a warp, three ahead, two quanta a lane
a stage, eight lanes reading one token's run), items taken in order with the span innermost (as the pass deals its
units), every item 64 KB; read only. sm_86 (RTX 3080 Ti), bytes a chip cycle against a flat stream of the same bytes:

| span | 2 x 8 warps an SM | 1 x 16 | 1 x 8 |
|---|---|---|---|
| flat stream | 541.7 (1.00) | 539.8 (1.00) | 525.0 (1.00) |
| 32 digits (32 B a token) | 268.0 (0.50) | 243.0 (0.45) | |
| 64 digits | 436.3 (0.81) | 438.0 (0.81) | |
| 128 digits | 528.0 (0.98) | 524.3 (0.97) | 533.2 (1.02) |
| 256 digits (the whole row) | 538.9 (0.99) | 537.3 (1.00) | 524.7 (1.00) |

The flat stream is the card's DRAM peak (~900 GB/s). A 32-byte piece of a row is half a DRAM burst: the stream reads
at half rate unless the row's neighbouring piece is read by the same warp at the same time -- the facts pass's vote
items (four consecutive units of a level, their warps reading together) recover it, its 64-digit list items do not.
From 128 digits on the stream is at the peak at any block shape: neither the second block an SM nor sixteen warps are
needed to stream. SPANS APART (the span outermost in the item order, so a row's other spans are read a whole sweep
later, as a facts pass whose blocks drift apart reads them): 128 digits 527.8 (0.98), the same as innermost; 64 digits
319.8 (0.59) against 0.81 innermost -- a 64-byte run is a whole burst only when its row's neighbour is read about the
same time, a 128-byte run always. Record: `measure/lab/results/row_span/sm_86.json`.

<a id="vote-stash"></a>

## `vote_stash`

What the facts pass's first stage pays to keep each digit's live bytes (a STASH, one byte an entry, which a second
stage reads to form the lists' coefficients), beside the vote. Rows `[level-side, T, H, 256]`, each token's head live in
k random digits; one block of eight warps an SM, each warp taking (window, level-side, head, 128-digit span) items in
order and streaming each as 16 blocks of 32 tokens x 128 bytes through a ring of three 4 KB stages (two ahead, issued
on into the next item, eight lanes a token's run); each block's four 32-digit sub-words voted as the facts pass votes a
word and stored word-major. A STASH A DIGIT: each lane's digit's live bytes stored at the digit's own region (512 bytes
a (window, digit)) at its running count. PACKED: a sub-word's live bytes appended to the item's one stash, lane by
lane after a scan of the lanes' counts, each lane's in token order. Bytes a chip cycle (the flat stream 541):

| k of 256 | vote | + stash a digit | + stash packed |
|---|---|---|---|
| 4 | 454.4 | 190.2 | 440.6 (-3.0%) |
| 16 | 454.7 | 111.9 | 419.7 (-7.7%) |
| 64 | 455.2 | 80.9 | 272.4 (-40%) |

A stash a digit amplifies its writes: at k 16, 93 MB of DRAM writes for 17 MB of stash, each byte store dirtying a
sector of one of 128 regions that leaves L2 before its neighbours are written. Packed, the writes are the stash's
bytes. What remains is the lanes' serial walk, ~2.4 warp instructions an entry at k 16: a warp walks as long as its
busiest lane, so at 25% density the stage halves; a lane-parallel extraction is owed for dense list levels. The vote's
own 84% of the flat stream is this lab's one block of eight warps (issue 10%, DRAM 93% at k 16 under ncu). TILED (each
warp's item bits in an 8 KB shared tile, its rows out as 64-byte runs at the item's end: the on-chip footprint a
second stage needs), packed stash: 6 warps an SM 424.0 (k 4) / 398.1 (k 16), 4 warps with three stages 407.8 / 297.5,
against 8 warps untiled 441.6 at k 4 -- the tile's footprint costs warps, and four warps cannot hide the walk at k 16.
Record: `measure/lab/results/vote_stash/sm_86.json`.

<a id="facts-pipeline"></a>

## `facts_pipeline`

The two-stage facts pass end to end in a warp: a warp streams an item (the vote; on the write list level also the
lists' exponents and the packed stash), keeps its bits in registers, issues its next item's first block, then builds
its four 32-digit groups' lists from them (the walk build: each digit's entries dealt evenly, a walk totalling each
(digit, pass) by residue, a segmented scan of ranks, a walk placing each entry; a write entry's byte from the stash).
Seven warps an SM (a 2 x 4 KB ring and ~4.8 KB of build scratch each). Bytes a chip cycle:

| k of 256 | voted only | stage 1 alone | two stages |
|---|---|---|---|
| 4 | 430.1 | 352.4 | 182.1 |
| 16 | 429.9 | 335.3 | 148.0 |

The build in the stream's warps serializes them: a warp builds instead of streaming, and seven warps an SM cannot keep
the stream's bytes in flight around it; the build's re-read of its bits from global (instead of registers held across
the item) also misses L2 at this footprint. The form the facts pass takes instead: `facts_block`. Record:
`measure/lab/results/facts_pipeline/sm_86.json`.

<a id="facts-block"></a>

## `facts_block`

What shape lets the facts pass's list build hide under its row stream, at or under 1.15x its DRAM floor
(KERNEL_STANDARDS §28). Rows `[level-side, T, H, 256]` u8, each token's head live in k random digits; level-side 0
the write list level (8-byte records whose coefficient needs the byte, the token's `(F, m)` pair and the window's
maxima of b x gains), 1 the read list level (2-byte slot lists from its bits), 2-3 voted only. One block of sixteen
warps an SM, warp w the window's w-th 32 tokens; the build is the facts pass's walk build (four warps a 32-digit
group). Time against the DRAM floor -- the rows, the gains, the bits and union words, the lists and their pass starts
-- at a flat read's rate measured in the same run (539 bytes a chip cycle); each build's lists checked against the
live count and, by checksum, bit for bit against the first form's (they agree). sm_86, x the DRAM floor:

| form | vote only | k 4 | k 16 | resident k 4 / k 16 |
|---|---|---|---|---|
| BULK: 64-digit items over two whole-window buffers, three barriers an item, bit words stored from the vote | 1.18 | 1.29 | 1.60 | 1.11 / 1.40 |
| TILED: the same, bits through a shared tile as 64-byte runs, the maxima in the build's first walk | 1.10 | 1.28 | 1.27 | 0.72 / 1.08 |
| SPLIT: the write level two 64-digit halves in one held window, the others 128-digit items through a per-warp ring of 8-token stages, one barrier an item | 1.07 | 1.16 | 1.20 | 0.54 / 0.88 |

What each step found (each measured in this lab on the way; the dropped forms are in its history):
- THE VOTE'S BIT STORES WERE ITS WHOLE COST: each warp storing one 4-byte word to each of 32 rows took ~4K cycles an
  item; through a tile and out as 64-byte runs after the barrier the vote runs at 4x the floor's rate (resident 0.23x).
- THE MAXIMA OVER LIVE ENTRIES: the write level's per-digit maxima of b x gains over every byte (64 f16 products a lane
  a group in the vote) move to the build's first walk, which visits only the live entries.
- RESIDENT IS NOT ENOUGH: TILED's build computes at 0.72x the floor at k 4 yet streams at 1.28x -- its two buffers hold
  the write level's window through the build, so nothing is in flight under it. A third 32 KB buffer does not fit beside
  the tiles and tables in sm_86's 99 KB. Only the write level needs its bytes after the vote, so SPLIT holds that level
  alone (64 digits, 32 KB) and streams the rest as 128-byte runs through a small ring, each stage free once voted.
- AN sm_86 MBARRIER WAIT IS A POLLED SHARED WORD: in SPLIT the eight warps waiting on the write level's eight builders
  spun 30.7M of 116.5M warp instructions (26%); polls with `__nanosleep(64)` between them return the slots (resident
  0.56 -> 0.54 at k 4, 0.90 -> 0.88 at k 16).
- FORMS THAT LOST: popcount counts a (digit, word) with independent per-entry ranks (73M warp instructions at k 4
  against the walk build's 28M: per-cell work is paid at every density); one barrier a quad of level-sides on two
  slots (resident 0.55 / 0.84 but streaming 1.50 / 1.45, converted from bytes a cycle by the same cells' bytes moved:
  the held window leaves one slot to stream through); the read level built on
  eight warps beside the others' votes (k 16 1.20 -> 1.31: its latency doubles); a lane's walk batched four entries with
  predicated effects (resident k 16 0.88 -> 0.99: more instructions and registers than the overlap returns).
- OPEN: k 16 is 1.20x with its compute at 0.88x the floor; the build's ~156 instructions a live entry (both walks) and
  its latency at eight to sixteen warps are what is left to cut. The vote-only 1.07x is this geometry's floor on sm_86:
  one level-side in four reads 64-byte runs.

Record: `measure/lab/results/facts_block/sm_86.json`.

<a id="parts"></a>
## Part harnesses

An experiment may include the kernels' own headers (`tools/lab.py` puts `csrc/rola/src` on the include path and builds
with the production flags), so a component is measured as the kernel runs it, alone and on real inputs, before its
cost is read in the composed ledger (KERNEL_STANDARDS section 19).

<a id="fold-part"></a>
### `fold_part`

The carry's fold (`csrc/rola/src/carry/fold.cuh`) over one head's real write lists (`fold_part.inputs.py`: nl64k-alt-k4,
32 windows), the pool filled once, one CTA an SM each a 32-digit box: the composed shape (a window's units behind a
barrier), the same units back to back (the step's own cost) and the MMAs alone on fixed rows -- each against the
structural floor.

Two layouts of the same lists follow (`ordered.bin`, `packed.bin`). A slot's residue mod 8 is its pool row's bank
group, and an `ldmatrix` phase reads 8 rows, so rows sharing a residue in a phase conflict. ORDERED: each unit's
entries reordered so a step's 8-entry halves hold distinct residues, and a unit's first step (at most 16 entries)
dealt across both halves with its pads on the residues each half leaves free. 8-TOKEN STEPS: the step on m16n8k8
MMAs (half an m16n8k16's cycles, `mma_k8`), heads packed live-first with pads on free residues, a second step only for
a list past 8 entries.

Measured (sm_86, nl64k-alt-k4, against the 16-token floor; ncu wavefronts per 16-token-equivalent MMA):

| form | composed | steady | wavefronts | of which bank conflicts |
|---|---|---|---|---|
| as built | 1.179x | 1.064x | 4.69 | 1.57 |
| ordered, dealt heads | 1.157x | 1.060x | 3.69 | 0.69 |
| 8-token steps | 1.012x | 0.844x | 3.48 | 1.30 |

Ordering within a half cannot help at alt-k4 (a list's ~8 entries a window fill one half); dealing them across the
step's halves does. 8-token steps do 0.70x the tensor work (1.40x the MMAs at half the cycles); their full phases
conflict more than dealt halves, so their shared-memory saving is small.

Record: `measure/lab/results/fold_part/sm_86.json`.

<a id="fold-mchan"></a>
### `fold_mchan`

The value carve's fold (M = channel: a warp 16 channels of a box's 256 leaves, 4-warp CTAs, two an SM) on the same
lists, lab-local code: a 16 x 16 and an 8 x 32 box, composed, steady, the MMAs alone, whole loops with a barrier a
window, with heads prefetched, with lane-ready step operands (a lane's two row slots and two c pairs, 16 bytes), and
free-running on landed and drained mbarriers. Per-burst timing reads below the floor at two CTAs an SM (each CTA's
burst can have the tensor pipe alone); only the whole-loop modes are honest.

Measured (sm_86): the 16 x 16 box 1.45x its floor free-running (1.69x with a barrier a window), 8 x 32 1.26x. Not the
operand chain: 10.7 shared-memory wavefronts an HMMA at 16 x 16 (ncu) against the 8 an SM issues an fp32-accumulate
HMMA in, 28% of them bank conflicts over gathered rows, every warp reloading the same slice; 8 x 32 7.45.

Record: `measure/lab/results/fold_mchan/sm_86.json`.

<a id="read-image"></a>
### `read_image`

The carry's readout with the state's operand from a bf16 image in shared memory (lab-local code: no production readout
yet), on nl64k-alt-k4's real read lists (`read_image.inputs.py`; a read list holds slots only) and a synthetic dense
cell. One CTA an SM a box of 32 lists x 32 read digits x 32 channels; a warp reads digits w, w + 8, w + 16, w + 24,
holding a digit's 16 state registers while its tokens stream; read rows gathered from a 256-slot pool; bf16 MMAs with
fp32 accumulate; each token's outputs times its coefficient to s16 pairs added by `red.shared` into its panel row
(20-word rows); the drain at the pass's end, the union's rows only. A warp's heads (its digits' pass starts and first
16 slots) and the union count arrive a window ahead. Forms: tokens on M (16-token steps, the state the B operand) or
on N (8-token tiles, the state the A operand); the state a digit at a time, a ring of two digits, or every digit's
first step in one straight-line block (as the fold's).

Measured (sm_86, against each form's tensor floor: 2074 cycles a CTA-window at alt-k4 on M, 1628 on N, 65.5K dense):

| cell, form | the MMAs alone | + panel adds | + the drain |
|---|---|---|---|
| alt-k4, M, first steps straight-line | 1.55x | 1.86x | 2.44x |
| alt-k4, M, state ring | 1.54x | 2.00x | 2.52x |
| alt-k4, N, state ring | 2.08x (1.64x of M's floor) | 2.76x | 3.40x |
| dense, M | | | 1.16x |
| dense, N | | | 1.41x |

Tokens on N lose everywhere: an 8-token tile carries a 16-token step's fixed work for half its MMAs. The ring gains
nothing (two warps a scheduler hide the image's loads). Shared-memory wavefronts an MMA at alt-k4 (ncu): 4.8 the MMAs
alone, 7.0 with the panel's adds, 8.7 with the drain (0.8 / 1.8 / 2.6 of them bank conflicts: a step's tokens
collide on residues); dense 2.6.

Then, tokens on M with first steps straight-line, on lists dealt round-robin by residue within the 256-slot pass
(`ordered.bin`), outputs converted on the FMA pipe (one FFMA a value adds 1.5 x 2^23 with the coefficient and unit
folded in, a PRMT a pair) in place of F2I, half steps (a digit's last 8 tokens or fewer: 8 read rows and their adds
only, a declared warp-uniform branch choosing the form) and no barrier at the pass's end:

| alt-k4, tokens on M | + panel adds | + the drain | instructions an MMA (with adds) |
|---|---|---|---|
| F2I | 1.85x | 2.44x | 21.7 (2.0 on the conversion unit) |
| ordered lists | 1.85x | | 21.7 |
| ordered, half steps | 1.99x | 2.50x | 21.9 |
| ordered, FMA conversion | 1.65x | 2.17x | 18.7 (none) |
| ordered, FMA conversion, no pass barrier | 1.48x | | 18.8 |
| dense, FMA conversion | | 1.12x | 12.1 |
| ordered, FMA conversion, a token's panel base once | 1.60x | 2.12x | 16.2 |
| the same, free-running | 1.44x | | |
| ordered, FMA conversion, facts-form entries (byte offsets, swizzle folded), heads path at compile time | 1.62x | 2.13x | 13.2 |
| the same, free-running | 1.45x | | |
| tokens on N (8-token tiles, 0.69x the MMAs), facts form, exact pairs | 2.11x (1.65x of M's floor) | | 18.8 |
| the same, free-running | 1.78x (1.40x of M's floor) | | |
| dense, a token's panel base once | | 1.12x | 9.6 |

Ordering moves conflicts 1.76 -> 1.65 an MMA only: a digit's ~8 tokens collide on residues with no ninth to deal.
Half steps save ~0.8 wavefronts an MMA but their branch costs sm_86 more. What remains at alt-k4 is the output path
(the box's 32 lists make each MMA 64 output values, each a conversion and half an add, half of them padding rows'
at alt-k4) and each window's chain with ~4 steps a warp (tensor pipe idle ~38% with the MMAs alone).

The facts-form variants state their instruction budget by opcode class (KERNEL_STANDARDS §29; `Bench::budget`, gated
by `tools/lab.py run` on ncu's executed counts): the step itself now matches its operation count (an entry load,
three shuffles, four address operations, two read-row loads, 8 MMAs, 16 FFMAs, 8 PRMTs, two panel bases, 8 adds),
and the gate holds the rest over budget -- integer/address 4.3 an MMA against 1.8, shared loads 1.45 against 1.15,
select/compare 0.85 against 0.4: each digit's 64-bit list row computed on the common path, each lane's state offsets
rematerialized a digit, the heads' 64-bit copies a window. On sm_86 the facts form's per-digit choice of heads or
global path (a branch in the straight-line block) costs more than its 3.3 fewer instructions an MMA save; the count
is A100's limit, the block structure sm_86's.

The per-card prediction (`Bench::mma`; tensor, shared-memory and issue cycles an MMA from the same counts) puts the
alt-k4 readout SHARED-MEMORY BOUND ON BOTH CARDS: 8.1 cycles an MMA (7.1 wavefronts, each shared atomic's second cycle)
against 8.0 of tensor on sm_86 and 2.0 on A100; tokens on N spend 7.15 a tokens-on-M MMA's worth of work, about 12%
less. A digit's state read from the image (16 wavefronts a digit-window) is the largest single item at alt-k4. An
earlier run of the facts-form variants was wrong: the pad tokens' dummy words were 32 a CTA while a step's adds reach
60 bytes past a lane's base, so the last lanes' adds wrote into warp 0's heads and moved its counts (the tokens-on-N
form, whose pads add nonzero, ran twice its tiles); the lab checks no outputs, so only the counts showed it.

Record: `measure/lab/results/read_image/sm_86.json`.

<a id="mma-k8"></a>
### `mma_k8`

The tensor pipe's cost of an m16n8k8 MMA against an m16n8k16, two warps a scheduler, 32 independent accumulators.
Measured (sm_86, cycles an MMA a warp): fp32 accumulate 64.0 (k16) and 31.2 (k8); fp16 accumulate 32.0 and 15.6 --
k8 costs exactly half. Record: `measure/lab/results/mma_k8/sm_86.json`.
