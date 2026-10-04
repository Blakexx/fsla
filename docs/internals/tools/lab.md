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
   CALL nobody declared; checks the GPU is quiet; runs the repetitions; prints median, min and max and flags a result
   whose samples spread past 5%.
3. `--record` writes `measure/lab/results/<name>/<sm_XX>.json` — the device, driver, toolkit, commit, the results with
   their samples, and each kernel's registers and leading opcodes. A rule cites this file.
4. The finding becomes a law in `docs/KERNEL_STANDARDS.md` (§26), naming the experiment, the record and the
   architectures it holds on.

`python tools/lab.py list` names the experiments by their QUESTION; `python tools/lab.py sass <name> [REGEX] [--full]`
shows each kernel's registers and leading opcodes (or the whole disassembly).

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
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
inline (8.3 cycles a shuffle); under a branch on the warp id (13.6) or with the role setting the trip count (13.9) it is
a CALL into a convergence subroutine, ~1.65x. Record: `measure/lab/results/collective_branch/sm_86.json`.
