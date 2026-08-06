# Measurement — the instruments, the discipline, and the record

Every performance number this repository publishes is produced by one harness, taken
with one stopwatch, and written to one file with its provenance. This document is
what those three sentences mean.

| Layer | Where | Answers |
|---|---|---|
| fixtures | `benchmarks/bench/cells.py` | WHAT is measured |
| methodology | `benchmarks/bench/pairing.py` | HOW: interleaved paired arms, median of per-pair ratios |
| preconditions | `benchmarks/bench/instruments.py`, `benchmarks/probe/discipline.py` | what the box must be doing first |
| statistics | `benchmarks/bench/stats.py`, `bench/regression.py` | whether a difference is real |
| record | `perf/ledger.jsonl`, `benchmarks/bench/ledger.py` | where the number lives afterwards |
| driver | `benchmarks/rola_probe.py`, `benchmarks/probe/` | the process that runs all of it |

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

Not a checklist. `probe.discipline.disciplined` does each of these, and there is no
flag that skips any of them.

| step | what happens |
|---|---|
| GPU lock | `/tmp/rola_gpu.lock`, taken exclusively; if held, the harness says it is waiting and waits. **Invoke `rola-probe` BARE — never under an outer `flock` on this path.** `flock` locks are held per open file description, so wrapping the harness in the lock it is about to take blocks it forever with no output. `ROLA_GPU_LOCK` exists to MOVE the harness's lock for a caller composing with a genuinely DIFFERENT outer lock; pointing it at the path the caller already holds is the deadlock it exists to avoid. |
| memory fraction | `set_per_process_memory_fraction(0.65)`, inside the lock and before the fixture is built — the value the two device-heavy test modules already pin after two whole-machine VRAM freezes |
| clock settling | `bench.instruments.settle_clocks()`; **`settled: False` aborts the run.** Clocks cannot be locked from inside WSL2 at all, so targeting the plateau is the only available protocol, not a convenience |
| tree identity | `git rev-parse` + `git status --porcelain` + `ratify.csrc_digest()`; a dirty tree does not block the run, it stamps `dirty: true`, and dirty rows are excluded from baselines |
| tier | `light` or `landing`, stamped into every row — a light-tier number can never be quoted as a landing one by a reader who did not see the command |
| ledger write | automatic; there is no path that produces a number without recording where it came from |

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
  "env": { ... },                    // provenance + the settle_clocks() dict + temperatures
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

`benchmarks/cells/latency_baseline.json` is the per-commit gate's reference, recorded
by `benchmarks/bench_regression.py --write` and never by a test. Under the committed
instrument its arms read 0.0625–0.2376 ms with block IQRs of 0.001–0.005 ms — a
spread 30–60× tighter than the host timer's, because the launch and sync cost that
dominated it is no longer in the measurement.

**A tighter band exposes anything that was hiding inside the old one.** It exposed
one immediately: the gate had been reading its number from inside a benchmark
runner's round loop, and that runner's schedule — not the kernel — set the value. The
gate now takes the arm's median once, through the recorder's own entry point, so the
two sides are the same quantity by construction. `benchmarks/test_latency_regression.py`
records the measurement that forced it.

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

```bash
python benchmarks/rola_probe.py duration consumer_forward --ab global:raw --tier landing
python benchmarks/rola_probe.py counters consumer_forward --arm global --section RolaMemOcc
python benchmarks/rola_probe.py sanitize consumer_forward --tool memcheck
python benchmarks/rola_probe.py sass --json bodies.json [--diff-against before.json]
python benchmarks/rola_probe.py report [--probe consumer_forward] [--format md]
```

**No `flock` on those lines, and adding one HANGS the command** — the harness takes
the lock itself, and a second acquisition of the same path from the parent process
blocks its child forever. Everything else that reaches a device is wrapped;
`rola-probe` is invoked bare (`docs/testing.md`, "Running it"). `sass` and `report`
take no lock at all because neither runs a kernel.

`rola-probe` is not a console script. The installed wheel ships `rola/`, not
`benchmarks/`, so a `rola-probe` on PATH would be a command that exists and cannot
import the harness it drives.

### Counters

The metric list is a committed section file, `benchmarks/probe/sections/*.section`,
not a flag string — two throwaway scripts holding the same 24-metric comma-joined
string is how the two drifted apart. Two ncu behaviours are handled by the harness
rather than remembered:

* `--section-folder` **replaces** the default search path instead of adding to it, so
  the stock folder is always passed alongside ours;
* counter capture runs at `--launch-count 1` with the timing loop disabled, because a
  warm-up loop pollutes counters the way it does not pollute a median.

The **replay mode is recorded in every row**. `kernel` replay restores only the memory
the kernel wrote, which is silently wrong for a kernel that carries state across
launches — the paged and `use_cache` continuation arms — where `application` replay is
mandatory.

### SASS

`rola-probe sass` is a front door to `tools/ratify.py` and `tools/sass_bodies.py`; it
re-implements neither the disassembly nor the compile flags. Both binaries in a
comparison no longer need to have been built at the same path (P19,
`docs/build.md`'s path-trap section) — they must agree on `ratify.SASS_HASH_VERSION`,
which `--compare` checks and refuses on mismatch.

## What is deliberately not here

* **Clock locking.** `nvidia-smi -lgc` has no effect from inside WSL2; the plateau
  protocol is the correct design on this box, not a workaround for it.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in
  a landing. Any query artifact is derived and ignored.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in
  the surveyed field; replacing it would reintroduce the invented number the
  derivation exists to avoid.
