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
2. `bazel run //tools:lab -- run <name>` compiles it with the pinned toolkit for the running card, reads the SASS and
   REFUSES to measure unless every declared expectation holds, no kernel touches local memory and no kernel holds a
   convergence call nobody declared (the SASS gate's classification, `tools/sass_gate.py`: ptxas's loop-exit
   `CALL.REL.NOINC` and guarded fallbacks are not ones); checks the GPU is quiet; runs the repetitions; prints median, min and max and flags a result
   whose samples spread past 5%.
3. `--record` writes `measure/lab/results/<name>/<sm_XX>.json` — the device, driver, toolkit, commit, the results with
   their samples, and each kernel's registers and leading opcodes. A rule cites this file.
4. The finding becomes a law in `docs/KERNEL_STANDARDS.md` (§26), naming the experiment, the record and the
   architectures it holds on.

`bazel run //tools:lab -- list` names the experiments by their QUESTION;
`bazel run //tools:lab -- sass <name> [REGEX] [--full]` shows each kernel's registers and leading opcodes (or the whole
disassembly). `bazel run //tools:lab -- ncu <name> --metric M [--metric ...] [--record]` reads
hardware counters per kernel (the median over its launches) into `<sm_XX>.counters.json`, for a
question about traffic rather than time.
`bazel run //tools:lab -- ledger <name> [--kernel REGEX]` attributes a kernel's executed instructions to its phases --
the `//:` headings of the experiment's source -- per declared unit (KERNEL_STANDARDS §29.13; the method:
[budgets.md](budgets.md)). `run` also refuses a budgeted kernel whose loops hold convergence instructions (`BSSY`,
`BSYNC`, `BRA.DIV`) beyond what it declares as `LOOP_CONVERGENCE`. `ledger <name> --static --arch sm_90a` needs no device and no counters (without `--arch` it reads the
device's): it prints each
kernel's instructions AS COMPILED, by the role that holds them (the `NAME_role` call they were inlined from), by
innermost loop (every backward branch closes one) and by phase and kind (global loads, stores and adds, shared
loads and stores, MMAs); a loop's cost a unit is its count times its trips, which the source states. It is the
ledger of a card that grants no profiler counters.

<a id="hazards"></a>THE LEARNINGS' CHECKS ([`LEARNINGS.md`](../LEARNINGS.md)). An experiment outside `tools/lab.py`'s `_PRE_TIMELINE` (the
experiments written before these checks; the list only shrinks) is refused by `run` and by `pack` unless its header
comment states a `TIMELINE:` -- each phase's stream, which load is a stage ahead of which use, the live registers --
or its source has a warp collective after a conditional's `?` (`collective_selected`: only the lanes the condition
takes reach a collective every lane must), and EVERY kernel of it, budgeted or exempt, is held to five counts, each zero unless the experiment
declares it (`Bench::expect(kernel, NAME, "<=", n)`): `LOOP_CONVERGENCE`; `LOAD_IN_TRIP`, the global loads whose
value an instruction of the same innermost trip reads; `STORE_RELOAD`, the shared loads of an address the same trip
stored to; `STORE_UNFENCED`, in a kernel that holds a wgmma, the barrier arrivals that follow a shared store with no
`fence.proxy.async` between them; `STEPS_SERIALIZED`, the toolkit's notices that it serialized the kernel's
asynchronous MMAs for want of registers (ptxas C7512, kept beside the build's binary).

<a id="pack"></a>A CARD ELSEWHERE. A box that is not started from the devbox image ([`setup.md`](../../setup.md)) carries a driver and
no toolkit, so an experiment for it is built and checked here and run there: `bazel run //tools:lab -- pack <name> --arch sm_90a --into DIR` compiles it for that architecture,
holds its SASS to the same checks as `run`, and writes the binary with a pack file beside it (the kernels' registers
and opcodes, the toolkit, the commit, the binary's digest); on the card, `tools/lab.py carried <name> --from DIR --into
OUT --no-counters HOST` runs the pack's binary by its digest through the sequence `run` runs (`measure`: the inputs, a
quiet GPU, the repetitions under a kill timeout, `--timeout`, the checks) and writes the record with the pack's build
facts. `--no-counters HOST` is the caller's DECLARATION that the named host grants no profiler counters: the record
then carries the written exception in their place (`counters_exception`: the host, what stands in --
`docs/KERNEL_STANDARDS.md` §16 addendum 4). `run` takes the same flag; without it a profiler that fails fails the
run. An experiment with budgeted kernels, whose gate reads executed counts, is neither packed nor recorded under
the exception.

An experiment that reads data -- a cell's routing, say -- has a `<name>.inputs.py` beside it whose `write(directory)`
writes the files and returns their names. `run` and `ncu` call it first (in the scratch space), pass the binary
`--inputs <directory>`, and the record carries each file's digest under `inputs`.

<a id="counters"></a>

## The standard counters

`run` reads the standard counter set of [`pipes.py`](pipes.md) for every kernel of the experiment (its first launch),
prints per card each pipe's cycles a unit (its MMAs, else its declared `Bench::units`, else a launch an SM) with the
binding one named, and on this card each pipe's busy share, L2 sectors and the three largest stall reasons; the record
keeps the counters under `counters` (KERNEL_STANDARDS §29.14). The tensor pipe is priced from a declared `Bench::mma`,
else from the work the tensor-pipe counter counts and the one MMA kind the SASS issues.

## The kit (`measure/lab/lab.cuh`)

* `lab::WarpClock` — a warp's cycles over a region, lane 0's `clock64`, stored at `(block, warp)`.
* `lab::WarpSpan` — a warp's start and end `clock64`; `Bench::span(warps)` (it exits on a block one of whose warps recorded no clock: a timed launch starts from cleared clocks) is the mean block's span (last end less
  first start), the throughput when warps finish apart.
* `lab::touch` — a line asked into the L2 cache ahead of the load that uses it (`prefetch.global.L2`). MEASURED on
  sm_90a under CUDA 13.0 (`carry_window`'s ahead rows): a plain `ld.global` whose result nothing reads compiled to no
  instruction at all; a row that asks states the asks its binary must hold.
* The operand pieces the kernels use: `ldsm` and `ldsm_t` (`ldmatrix.x4`, `.trans`), `hmul2`, `pack_half2`, `smem`;
  the mbarrier as `ops.cuh` spells it: `mbar_init`, `mbar_arrive`, `mbar_wait` (polled), `mbar_wait_suspended` (sm_90: `try_wait`, the thread may be suspended), `mbar_test` (once).
* `lab::sink(out, v)` — a fold of everything a thread computed, stored unconditionally.
* The atoms as one PTX instruction each: `mma_f16_f32` (and `_fresh`), `mma_f16_f16`, `mma_bf16_f32`, `mma_u8u8`,
  `mma_u8s8`.
* <a id="wgmma"></a>Hopper's atoms, each one PTX instruction of sm_90a (an experiment calling one compiles for sm_90a
  only): `wg_desc` (wgmma's unswizzled K-major shared operand: lines in bricks of 8, 16 bytes a line, the K halves and
  the bricks at stated offsets), `wg_fence`, `wg_commit`, `wg_wait<N>`, `wg_hold` (a register an asynchronous MMA
  wrote, held behind its wait), `wg_bf16_n16_ss`, `wg_bf16_n64_ss`, `wg_u8_n64_ss` (the integer step, u8 by u8 into s32), `wg_bf16_n64_rs` (A from registers), `wg_bf16_n72_rs` (the same 72 columns wide; `wg_bf16_n72_rs_mn` with B M|N-major), `wg_bf16_n8_ss_mn` (8 columns, both operands M|N-major), `wg_bf16_n16_ss_mna` (A alone M|N-major; its `add` false makes a step the accumulator's first, with nothing
  cleared), `wg_bf16_n16_ss_mn`, `wg_bf16_n64_ss_mn`,
  `wg_bf16_n256_ss_mn` (both operands M|N-major: a line is one K's 8 rows or columns; `wg_desc` describes either
  major's tile); the two
  narrowings of an fp32 pair to bf16, `bf16x2_trunc` (one `PRMT`) and `bf16x2_round` (one conversion) -- the common
  atoms' `hi_bf16x2` and `pack_bf16x2`, spelled here until the common atoms admit a hopper build; and the bulk reduction
  out of shared memory, `bulk_add_f32` (`cp.reduce.async.bulk`, fp32 added into global) with the writers' proxy fence
  before an asynchronous reader of their stores (`bulk_fence`: a bulk copy out, another warp's wgmma), `bulk_commit`,
  `bulk_wait<N>`, beside `red_add_f32` (one value by its thread), `red_add_u32` and `red_add_u64` (an integer) and
  `red_add_f16x2` (a pair of f16, each half its own sum, from `pack_half2`); the plain bulk copy out, `bulk_store`; the tensor copies in,
  `tensor_load_2d` and its multicast `tensor_cast_2d` (one copy landed in every CTA of a cluster a mask names); the bulk copy
  in, `bulk_load`, under a barrier whose
  arrival states its bytes (`mbar_expect`); and a warpgroup's registers a thread,
  `regs_down<N>` and `regs_up<N>` (`setmaxnreg`). The lab's step checks read `HGMMA` and the integer `IGMMA` as they read `HMMA`
  and `IMMA`; a wgmma kernel states `Bench::exempt`, a budgeted one being refused until the hopper counters' model
  names its per-MMA divisor.
* <a id="cluster"></a>Hopper's clusters (a kernel of them carries `__cluster_dims__`): `cluster_rank`,
  `cluster_addr` (this CTA's shared address as a mate holds it), `cluster_store_f32` and `cluster_add_f32`
  (a value stored or added into a mate's shared memory by its thread), `cluster_bulk_copy` and `cluster_bulk_add_u64`
  (a span of this CTA's shared memory copied, or added as 64-bit integers, into a mate's, its bytes completing the
  mate's barrier), `cluster_add_u64` (the warps' integer add), `cluster_fence` (a thread's stores and adds into
  a mate ordered for the cluster), `cluster_arrive` (at a mate's barrier, releasing to the cluster) with
  `cluster_wait` (acquiring from it) and `cluster_fence_init`,
  and `cluster_sync` (the cluster's own barrier: before any CTA reaches into a mate, and before any exits).
* `lab::Bench` — the host runner: `expect(kernel, op, cmp, count)` declares SASS (printed under `--expect`, run by the
  driver before anything is timed), `variant(launch, measure)` warms up and repeats a launch, `record(name, unit,
  samples)` prints a result, `cycles(warps)` and `mean(...)` read the warps' clocks. `--blocks N` holds the grid an experiment launches over `Bench::sms` to N CTAs, so a sanitizer
  can run an experiment at one CTA (racecheck's host memory grows with the CTAs it tracks).
* `Bench::budget(kernel, class, max)` — KERNEL_STANDARDS §29's instruction budget, per MMA, gated on ncu's executed
  counts by `run`; `Bench::units(kernel, unit, n)` makes a kernel with no MMAs budgeted per unit, its first launch
  doing `n` of them.
* `lab::kernel("tax", {3, 17})` — a kernel's demangled name as an expectation's pattern.

## What goes wrong, and what guards it

Every item below cost a wrong number before the lab existed; each is now a guard or a habit.

* <a id="build"></a>**A launch sized at the host pass's card.** The host pass compiles at the pinned card, so a
  harness that sized its launch from a plan at the compiling architecture launched the A100 with sm_86's shared bytes
  (2026-11-07, `read_alone`, `read_dense`). An experiment compiles for one architecture (`tools/lab.py`'s `build`:
  the extension's own `-gencode` form for that one row, C++20 as the extension) and `-DROLA_LAB_CC` names it to both passes, so `lab::kLabBuild` -- that
  architecture's own build ([`common/hardware.md#builds`](../common/hardware.md#builds)) -- is the one plan a harness
  reads on the device and on the host.
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
  Habit: an experiment about a kernel's step holds the kernel's live accumulators.
* **Runtime choices inside a timed loop.** A `kind` argument switched at run time inside the loop measured branches, not
  MMAs. Habit: every variant is a template parameter.
* **The operand pattern.** An instruction's cost beside the tensor pipe grows with the registers it reads: packed f16
  math with a constant operand measured ~4x cheaper than with two registers. Habit: write the operand pattern the
  kernels use.
* **Phase lock.** Two warps a scheduler can fall into a slow relative phase on some launches (up to 22% for a pure HMMA
  stream). Guard: repetitions; the driver flags spread.
* **A busy GPU.** Another consumer time-slices every kernel. Guard: the driver refuses above 5% utilization before a
  run (`--force` to override).

## <a id="parts"></a>Part harnesses

An experiment may include the kernels' own headers (`tools/lab.py` puts `csrc/rola/src` on the include path and builds
with the production flags), so a component is measured as the kernel runs it, alone and on real inputs, before its
cost is read in the composed ledger (KERNEL_STANDARDS section 19).

## <a id="experiments"></a>The experiments

Each experiment's question, method, record and verdict is its own page under [`docs/lab/`](../../lab/README.md), one
a source in `measure/lab/experiments/`. The anchors below are where the sources' citations land:

- <a id="tensor-tax"></a>[`tensor_tax`](../../lab/tensor_tax.md)
- <a id="collective-branch"></a>[`collective_branch`](../../lab/collective_branch.md)
- <a id="smem-rates"></a>[`smem_rates`](../../lab/smem_rates.md)
- <a id="row-span"></a>[`row_span`](../../lab/row_span.md)
- <a id="vote-stash"></a>[`vote_stash`](../../lab/vote_stash.md)
- <a id="facts-block"></a>[`facts_block`](../../lab/facts_block.md)
- <a id="facts-two"></a>[`facts_two`](../../lab/facts_two.md)
- <a id="issue-latency"></a>[`issue_latency`](../../lab/issue_latency.md)
- <a id="drain"></a>[`drain`](../../lab/drain.md)
- <a id="wgmma-regs"></a>[`wgmma_regs`](../../lab/wgmma_regs.md)
- <a id="bulk-drain"></a>[`bulk_drain`](../../lab/bulk_drain.md)
- <a id="reg-trade"></a>[`reg_trade`](../../lab/reg_trade.md)
- <a id="reg-five"></a>[`reg_five`](../../lab/reg_five.md)
- <a id="row-ring"></a>[`row_ring`](../../lab/row_ring.md)
- <a id="wgmma-rate"></a>[`wgmma_rate`](../../lab/wgmma_rate.md)
- <a id="byte-expand"></a>[`byte_expand`](../../lab/byte_expand.md)
- <a id="cluster-sum"></a>[`cluster_sum`](../../lab/cluster_sum.md)
- <a id="state-move"></a>[`state_move`](../../lab/state_move.md)
- <a id="fold-stream"></a>[`fold_stream`](../../lab/fold_stream.md)
- <a id="drain-rate"></a>[`drain_rate`](../../lab/drain_rate.md)
- <a id="drain-role"></a>[`drain_role`](../../lab/drain_role.md)
- <a id="state-image"></a>[`state_image`](../../lab/state_image.md)
- <a id="panel-merge"></a>[`panel_merge`](../../lab/panel_merge.md)
- <a id="wgmma-width"></a>[`wgmma_width`](../../lab/wgmma_width.md)
- <a id="intra-chain"></a>[`intra_chain`](../../lab/intra_chain.md)
- <a id="intra-window"></a>[`intra_window`](../../lab/intra_window.md)
- <a id="call-pair"></a>[`call_pair`](../../lab/call_pair.md)
- <a id="facts-lists"></a>[`facts_lists`](../../lab/facts_lists.md)
- <a id="gather-pass"></a>[`gather_pass`](../../lab/gather_pass.md)
- <a id="cast-cluster"></a>[`cast_cluster`](../../lab/cast_cluster.md)
- <a id="value-gather"></a>[`value_gather`](../../lab/value_gather.md)
- <a id="readout-role"></a>[`readout_role`](../../lab/readout_role.md)
- <a id="value-cast"></a>[`value_cast`](../../lab/value_cast.md)
- <a id="pool-fill"></a>[`pool_fill`](../../lab/pool_fill.md)
- <a id="load-rate"></a>[`load_rate`](../../lab/load_rate.md)
- <a id="carry-window"></a>[`carry_window`](../../lab/carry_window.md)
- <a id="add-rate"></a>[`add_rate`](../../lab/add_rate.md)
- <a id="slot-sum"></a>[`slot_sum`](../../lab/slot_sum.md)
