# Measurement — the instruments, the discipline, and the record

Every performance number this repository publishes is produced by one harness, taken
with one stopwatch, and written to one file with its provenance. This document is
what those three sentences mean.

**THE SUCCESSOR HAS LANDED (P72).** The chunk arm's harness is `benchmarks/bench_chunk.py`
over `benchmarks/{unit,integration}/`, and the owed list below is DELIVERED — item by item,
with the file that discharges it, in [the owed list](#what-the-chunk-arm-harness-owed-and-where-it-landed).
The table and the sections that follow describe the RETIRED tiled apparatus and remain the
predecessor's record; where a row says RETIRED, its successor is named beside it.

**HALF OF THIS HARNESS IS RETIRED, AND THIS DOCUMENT IS BOTH ITS SUCCESSOR'S
SPECIFICATION AND ITS PREDECESSOR'S RECORD.** Every fixture, every methodology
object and the driver itself named the tiled consumer — `probe/registry.py`'s
entries each named `consumer_forward_kernel` and a `bench.cells` cell — so when
that kernel was deleted (P67 D2) the harness had no subject and could not even
import. It was deleted as ONE capability rather than left broken
([`internals/DELETIONS.md`](internals/DELETIONS.md); revival `git show
c7eaeb9:<path>`).

| Layer | Where | Answers | State |
|---|---|---|---|
| fixtures | `benchmarks/bench/cells.py` | WHAT is measured | **RETIRED** — every cell was a tiled arm |
| methodology | `benchmarks/bench/pairing.py` | HOW: interleaved paired arms, median of per-pair ratios | **RETIRED** — the engine is what the successor most owes |
| preconditions | `benchmarks/probe/discipline.py` | what the box must be doing first | **RETIRED**; `settle_clocks` survives, moved into `bench/regression.py` |
| statistics | `benchmarks/bench/stats.py`, `bench/regression.py` | whether a difference is real | **SURVIVES**, gated by `tests/unit/test_perf_ledger.py` |
| record | `perf/ledger.jsonl`, `benchmarks/bench/ledger.py` | where the number lives afterwards | **SURVIVES** |
| driver | `benchmarks/rola_probe.py`, `benchmarks/probe/` | the process that runs all of it | **RETIRED** |

## What the chunk-arm harness owed, and where it landed

Stated as a list so it cannot be half-remembered, and marked DELIVERED only where a file
in the tree discharges it.

| owed | state | where |
|---|---|---|
| the interleaved paired-ratio engine (`_paired_one_kernel` and its ~7x bias-reduction rule) | **DELIVERED** | `benchmarks/bench/pairing.py` — `interleaved_ab`, and `PairedResult` REFUSES a statistic over samples that were not alternated |
| the discipline gates (the lock, the dirty-tree refusal, the clock-settle refusal, the extension's identity, the warmup floor) | **DELIVERED** | `benchmarks/bench/discipline.py`, `bench/pairing.py`, `bench/chunk_baseline.py` |
| their planted-violation tests | **DELIVERED** | `tests/unit/test_chunk_bench_harness.py` — one planted violation per gate, with positive controls |
| per-cell latency baselines for the chunk arm | **DELIVERED** | `benchmarks/cells/chunk_latency_baseline.json`, recorded whole by `bench_chunk.py baseline --write`; the gate is `tests/unit/test_chunk_latency_regression.py`, which reads and never writes |
| the frozen cells the baselines are keyed to | **DELIVERED** | `benchmarks/bench/chunk_cells.py` + `benchmarks/cells/chunk_manifest.json` |
| its own LOCKING STATEMENT | **DELIVERED** | below, and enforced by `require_gpu_lock` |
| a sanitizer measurement for the chunk arm | **STILL OWED** | never existed; see [the sanitizer gap](#the-sanitizer-gap) |

**THE LOCKING STATEMENT.** The chunk-arm harness does **not** take `/tmp/rola_gpu.lock`
itself. It is invoked WRAPPED —

```bash
flock /tmp/rola_gpu.lock python benchmarks/unit/bench_chunk_forward.py --tier landing
```

— and `bench.discipline.require_gpu_lock` CHECKS that it was, by finding the ancestor
process holding the lock path open. A bare invocation is a refusal, not a warning. This
puts the successor on the side of the rule that has no exception left: the `rola-probe`
self-lock exception retired with `rola-probe`, and nothing in the tree needs it back.
`bench_chunk.py list` is the one subcommand that reaches no device and needs no lock.

**The sanitizer gap.** No sanitizer measurement has ever been taken of the chunk arm, and
this harness does not add one: `compute-sanitizer` runs a kernel under conditions no timing
instrument reproduces, so it is a THIRD instrument like `ncu` and needs its own refusal
rules and its own row shape before a number from it can enter the record. Stating it here
is the honest half of the delivery — an undelivered item named is not a delivered one.

The surviving pre-P72 benches (`bench_layer.py`, `bench_scaling.py`, `bench_paging.py`) run
today and use the same statistics, but they are curve and attribution studies rather than
gated cells.

Everything below describes the discipline as it was BUILT and measured on the tiled
apparatus. It is the specification the successor is held to; where the successor's
implementation differs it is noted at the row.

## The stopwatch

There is one, it is named, and its name travels with every number.

| instrument | what it measures |
|---|---|
| **`cuda_events`** (committed) | device-side elapsed time between two events straddling the launch |
| `perf_counter` | host wall time around a full device sync on both sides — the launch and sync cost included |
| `ncu` | a counter, or a duration taken under profiling conditions: replayed, caches flushed, base clocks |

**A comparison across two of them is refused, not warned about.** The gap is not
noise; it is the launch and sync cost one instrument includes. Measured on this box
(RTX 3080 Ti, sm_86, 30 interleaved reps per instrument):

| cell | `cuda_events` | `perf_counter` | ratio |
|---|---:|---:|---:|
| `topo-D2-w64x64-BC64` global | 0.0725 ms | 0.2942 ms | 4.06× |
| `atscale-L2048-BC16` global | 12.8251 ms | 13.4350 ms | 1.05× |

The host cost is roughly constant (0.22 ms here), so it is nearly the whole
measurement on a sub-millisecond cell and a rounding error at scale. The
`ncu` duration for the same small cell reads 0.0906 ms — 25 % above the event time —
which is why an ncu duration is a third instrument and not a cross-check on the
first two.

`bench.pairing.same_instrument` implements the refusal; `bench.regression.check_instrument`
applies it to the latency gate, and a baseline that does not say which stopwatch took
it is refused as unknown rather than assumed compatible.

## The discipline, which the harness acquires

Not a checklist. `probe.discipline.disciplined` did each of these with no flag
that skipped any of them, and the successor owes the same — enforced, not
documented.

| step | what happens |
|---|---|
| GPU lock | `/tmp/rola_gpu.lock`, taken exclusively; if held, the harness says it is waiting and waits. A harness that takes the lock ITSELF must be invoked BARE — `flock` locks are held per open file description, so wrapping it in the lock it is about to take blocks it forever with no output. That was `rola-probe`'s standing exception to the flock rule; with the harness retired the exception is retired too (`docs/testing.md`), **SUCCESSOR: the other side of it.** The chunk-arm harness is WRAPPED and `bench.discipline.require_gpu_lock` refuses a bare invocation, so nothing in the tree takes this lock itself and the rule has no exception. `ROLA_GPU_LOCK` still MOVES the lock for a caller composing with a genuinely DIFFERENT outer lock. |
| memory fraction | `set_per_process_memory_fraction(0.65)`, inside the lock and before the fixture is built — the value the two device-heavy test modules already pin after two whole-machine VRAM freezes |
| clock settling | `settle_clocks()` — moved to `bench/regression.py` when `bench/instruments.py` went, because `bench_scaling.py` still needs it; **`settled: False` aborts the run.** Clocks cannot be locked from inside WSL2 at all, so targeting the plateau is the only available protocol, not a convenience |
| tree identity | `git rev-parse` + `git status --porcelain` + `ratify.csrc_digest()`; a dirty tree does not block the run, it stamps `dirty: true`, and dirty rows are excluded from baselines |
| extension identity | `import rola`, then resolve its `__file__` against this checkout's root; **an extension resolved from OUTSIDE this checkout ABORTS the run.** `rola` installs editable into a shared venv, so a probe launched from a worktree can silently import the MAIN tree's compiled extension unless the worktree leads it on `sys.path` — the row would then carry this tree's sha and csrc digest against another tree's kernel, agreeing with itself the way a true negative also would. The resolved path is stamped into every row's `env` regardless |
| tier | `light` or `landing`, stamped into every row — a light-tier number can never be quoted as a landing one by a reader who did not see the command |
| ledger write | automatic; there is no path that produces a number without recording where it came from |

An identity that does not describe the measured binary is worse than no identity —
which is why the extension check runs before the clock settles and before any row
exists, alongside the tree's sha and csrc digest rather than after them.

## The record

`perf/ledger.jsonl`: append-only, one JSON object per line, one line per measurement
of one arm. `perf/ledger.jsonl merge=union` in `.gitattributes` is what makes that
survive two branches appending — every conflict is "both appended at EOF".

```jsonc
{
  "schema": 1, "ts": "...", "sha": "...", "dirty": false,
  "csrc_sha256": "...",              // the KERNEL's identity, not just the tree's
  "probe": "consumer_forward", "probe_version": 1,
  "arm": "topo-D2-w64x64-BC64|global|none",
  "fingerprint": "...",              // sha256(probe, probe_version, arm, gpu, sm, driver, torch, ptxas)
  "env": { ... },                    // provenance + the settle_clocks() dict + temperatures + the resolved extension path
  "source": "cuda_events", "metric": "duration", "unit": "ms",
  "values": [0.0731, 0.0728, ...],   // RAW per-rep samples, IN RUN ORDER
  "paired_with": "...|raw|none", "round": 2,
  "tier": "landing", "run_id": "...", "cmd": "rola-probe duration ..."
}
```

Three properties are load-bearing:

* **Raw samples, no derived scalars.** The run order carries the drift signal, and a
  stored median beside its own samples is a second copy that can disagree with them.
  Medians and IQRs are computed on read.
* **`paired_with` and `round`.** The alternation structure IS the measurement design,
  so keeping it makes the natural test a paired one — strictly more powerful than an
  unpaired test, precisely because drift is common-mode within a round.
* **The fingerprint carries the box.** GPU, arch, driver, torch and assembler are in
  it from the first row. A comparison across an arch or a toolchain is impossible
  rather than discouraged; that is unrecoverable if left until later.

### Legacy rows

The banked markdown-era numbers were imported once, by `tools/ledger_import.py`, at
`probe_version: 0`. That field is inside the fingerprint, so a legacy row can never
enter a series with a disciplined one. Their warm-up, clock state and lock discipline
are unknown and the docket numbers were taken with a stopwatch that was not in the
tree. They are kept because they are the record, not because they are comparable, and
their `ts`, `sha` and `csrc_sha256` are null rather than invented.

### The committed latency baseline

`benchmarks/cells/latency_baseline.json` was the per-commit gate's reference,
recorded by `benchmarks/bench_regression.py --write` and never by a test. **Its
rows are the TILED consumer's and no chunk number may be compared against them**;
`benchmarks/cells/README.md` says so beside the files, which are parked rather
than deleted because a measurement is a record even when its subject is gone. Under the committed
instrument its arms read 0.0625–0.2376 ms with block IQRs of 0.001–0.005 ms — a
spread 30–60× tighter than the host timer's, because the launch and sync cost that
dominated it is no longer in the measurement.

**A tighter band exposes anything that was hiding inside the old one.** It exposed
one immediately: the gate had been reading its number from inside a benchmark
runner's round loop, and that runner's schedule — not the kernel — set the value. The
gate now takes the arm's median once, through the recorder's own entry point, so the
two sides are the same quantity by construction. `benchmarks/test_latency_regression.py`
recorded the measurement that forced it, and retired with the harness — the lesson
(the gate and the recorder must take the same quantity through the same entry
point) is CARRIED: `tests/unit/test_chunk_latency_regression.py` and
`bench_chunk.py baseline --write` both go through `bench.pairing.measure_arm` on a
fixture from the same registry, and neither owns a round loop of its own.

### The chunk arm's committed baseline

`benchmarks/cells/chunk_latency_baseline.json`, one row per `subject|cell` over the
registry in `benchmarks/bench/chunk_cells.py`. Recorded WHOLE, in one quiet locked
session, on a CLEAN tree — `bench.chunk_baseline.write` refuses a dirty one, because a
number taken over uncommitted edits describes a tree nobody can check out. It carries the
GPU, the arch, the driver, torch, the assembler, the ratification digest, the recorded
sha and the extension's own device-side build stamp; it carries no hostname, username or
path. The gate side reads it and never writes it.

## The flagging rule: three gates, all must fire

1. **Effect size** — `bench.regression.threshold_ms`: `median + 3·IQR/1.349`, derived
   from the window's own measured spread. `ALARM_SIGMA = 3.0` is the only free
   choice and it is stated as a false-alarm rate (~0.1 % one-sided per cell per run),
   never as a percentage of anything.
2. **Significance** — `bench.stats.paired_verdict`: an exact Wilcoxon signed-rank
   test over the per-round differences, at `ALPHA = 0.01`. **The round floor is
   derived, not chosen**: the smallest two-sided exact p reachable with `n` non-zero
   differences is `2/2ⁿ`, so below `ceil(log2(2/ALPHA)) = 8` rounds no outcome can be
   significant and the test is undefined rather than underpowered. A landing-tier run
   collects that many; anything shorter reports `insufficient_data`.
3. **Debounce** — `bench.stats.classify_flags`: a regression needs a trailing run of
   at least two violations. One thermally unlucky run on a box with 90 s of logged
   power capping is not a kernel change. A run of three that has since healed is
   reported as `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first
says the design cannot decide, the second says it decided.

## Running it

The five subcommands the retired driver carried, kept as the successor's surface
brief:

```bash
rola-probe duration <probe> --ab <arm>:<arm> --tier landing
rola-probe counters <probe> --arm <arm> --section <section>
rola-probe sanitize <probe> --tool memcheck
rola-probe sass --json bodies.json [--diff-against before.json]
rola-probe report [--probe <probe>] [--format md]
```

**No `flock` on those lines, and adding one HANGED the command** — the harness took
the lock itself, and a second acquisition of the same path from the parent process
blocks its child forever. `sass` and `report` took no lock at all because neither
runs a kernel.

It was not a console script, deliberately: the installed wheel ships `rola/`, not
`benchmarks/`, so a `rola-probe` on PATH would be a command that exists and cannot
import the harness it drives. The same holds of anything that replaces it.

The two commands that DO survive today are the ones that never needed the harness:
`python tools/ratify.py` and `python tools/sass_bodies.py` (below).

### Counters

The metric list was a committed section file, `benchmarks/probe/sections/*.section`,
not a flag string — two throwaway scripts holding the same 24-metric comma-joined
string is how the two drifted apart. Two ncu behaviours are handled by the harness
rather than remembered:

* `--section-folder` **replaces** the default search path instead of adding to it, so
  the stock folder is always passed alongside ours;
* counter capture runs at `--launch-count 1` with the timing loop disabled, because a
  warm-up loop pollutes counters the way it does not pollute a median.

The **replay mode is recorded in every row**. `kernel` replay restores only the memory
the kernel wrote, which is silently wrong for a kernel that carries state across
launches — the paged and stateful continuation arms — where `application` replay is
mandatory.

### SASS

`rola-probe sass` was a front door to `tools/ratify.py` and `tools/sass_bodies.py`
and re-implemented neither the disassembly nor the compile flags; **both tools are
in the tree and are the front door now.** Both binaries in a
comparison no longer need to have been built at the same path (P19,
`docs/build.md`'s path-trap section) — they must agree on `ratify.SASS_HASH_VERSION`,
which `--compare` checks and refuses on mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `python tools/ratify.py --census` there, read the spill
column — and never a flag on the shipped tool. `--force-ctas` was that flag; it
was deleted (P65 S2) once nothing in `csrc/` read its macro, because a tool that
can force a bound is a tool that can ratify one.

## What is deliberately not here

* **Clock locking.** `nvidia-smi -lgc` has no effect from inside WSL2; the plateau
  protocol is the correct design on this box, not a workaround for it.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in
  a landing. Any query artifact is derived and ignored.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in
  the surveyed field; replacing it would reintroduce the invented number the
  derivation exists to avoid.
