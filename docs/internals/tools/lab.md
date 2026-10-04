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
shows each kernel's registers and leading opcodes (or the whole disassembly).

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
* `lab::WarpSpan` — a warp's start and end `clock64`; `Bench::span(warps)` is the mean block's span (last end less
  first start), the throughput when warps finish apart.
* The operand pieces the kernels use: `ldsm_t` (`ldmatrix.x4.trans`), `hmul2`, `pack_half2`, `smem`; the mbarrier as
  `ops.cuh` spells it: `mbar_init`, `mbar_arrive`, `mbar_wait` (polled), `mbar_test` (once).
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
the fp16 arm every staged form spills (12-36 LDL/STL) and is not run. The step alone is near its floor; the barrier's
realignment is the fp16 arm's loss in the owner part. Record: `measure/lab/results/fold_step/sm_86.json`.

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
