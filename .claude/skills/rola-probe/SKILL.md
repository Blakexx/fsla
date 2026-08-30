---
name: rola-probe
description: Run tools/probe_cells.py, the canonical carry-family perf measurement harness — fixed cells, explicit arm resolution, interleaved A/B, ncu
---
Read `tools/probe_cells.py`'s module docstring in full before using it; it exists because two prior
ad hoc probes disagreed by a large margin for reasons this tool closes (run-to-run drift,
un-interleaved A/B, timing the python wrapper's own repacking). See FINDINGS `## PROBE-HARNESS`,
`## K36`, `## R-C` for the incidents that motivated each design choice below.

## The fixed cells (never invent a new one ad hoc)
Read `DEFAULT_CELLS` (and the off-by-default extras) at the top of `tools/probe_cells.py` for the
current cell list and which ones run by default. Don't hand-roll a draw — the cell specs are the
frozen RNG-stream mirror of `tests/oracle/fixtures.py`'s `Cell`/`alt_cell`, kept as a SEPARATE copy on
purpose so cross-tip comparisons don't drift with whichever fixtures version a worktree happens to
carry. When you need to know what a cell means (topology, k_tok, window, level_modes), read its
definition in the file — don't rely on a remembered description.

## Running it — NEVER wrap it in an external `flock`
```bash
python tools/probe_cells.py \
  --binary worktree:<path>,venv:<path>,label:<name> \
  --binary worktree:<path>,venv:<path>,label:<name> \
  --cells <comma-list> --reps <n> --warmup <n> --rounds <n> --out <path>
```
Invoke it BARE. `tools/probe_cells.py` takes the GPU lock ITSELF, via `tools/gpu_lock.py`'s
`gpu_lock()` in `run_worker`/`run_ncu_cell` — the ONE lock primitive every GPU entry point in this
tree now uses (K46: `tests/conftest.py`, `tools/sanitize_oracle.py`, `bench.discipline.disciplined`
all take it the same way; the two-family split this section used to describe — some tools
self-locking, pytest/bench requiring an external wrap — is retired). An outer
`flock` wrapping this command on the same lock path still blocks the tool's own internal
acquisition forever (a real incident, K38: 17 CPU-idle minutes on a live GPU before the coordinator
killed it) — `gpu_lock()`'s reentrancy only protects a self-locking tool calling ANOTHER
self-locking tool, never an external `flock` on the same path, which stays banned everywhere (a
lint rule in `tools/lint/lint_standards.py` finds it in a committed doc/script).

Add `--ncu` for one oneshot kernel-name + metrics capture per (binary, cell) — read the script's
`--ncu` handling for the current metric list rather than assuming it's fixed.

## NEVER default dispatch — the arm/kernel-symbol assert
A cell names a DECLARATION (topology, k, m, d_v, window, chunk, level_modes); `resolve_arm` scans
THIS binary's own `rola.ops.carry.arms()` census for an exact match and REFUSES (raises) if nothing
matches — it never falls through to whatever the kernel's own dispatch would pick. Read
`resolve_arm`'s carve-preference order in the source rather than assuming it. This matters because
older binaries can still carry a silent "unbuilt declaration served a default body" fallback that a
later tip removed; the harness's own refusal is what stands between a mismatched request and a
wrong-body measurement on those binaries. When adding `--ncu`, the CSV's `Kernel Name` column is
the symbol assertion — confirm it names the kernel you expect (carved vs. uncarved symbol), never
assume.

## Interleaved A/B, never separated runs
`interleaved_rounds` runs A, B, A, B, A, B — never "all of A then all of B". This is the K36 lesson:
separated runs on this host read a false regression from thermal/schedule drift alone (see
FINDINGS `## K36` for the magnitude, historical); only interleaved same-cell A/B isolates a real
change. If you write a NEW comparison script, copy this pattern — don't average two non-interleaved
runs and call it a measurement.

## Where baselines live
Recorded numbers live in FINDINGS tables (`## PROBE-HARNESS`, `## R-A`, `## R-B`, `## R-C`, `## K36`
stage-probe table, `## STAGE-LEDGER`), keyed by git SHA and cell — never re-derive a cited baseline
by re-running it casually; the tool prints `git_sha` and device stamp on every run specifically so a
new number can be told apart from an old one. A stage's §16 perf-probe gate cites the PREVIOUS
stage's recorded number, not a fresh re-measurement of it.

## The measurements record (K38) — every run is ALSO written, not just printed
As of K38, a probe run (unless `--no-record`) writes each binary's rows through
`benchmarks/bench/store.py` into `benchmarks/results/<hw-profile>/<yyyy-mm>/<run_id>.jsonl` — the
committed record — and `python tools/rola_db.py rebuild` indexes it into the gitignored
`.rola/measurements.sqlite`. Pass `--stage <name>` so a later `rola-db baseline --stage <name>` can
find these rows; `rola-db last`/`history`/`compare` are how an agent finds a prior number by SQL
instead of grepping FINDINGS. Read `docs/measurement.md`'s "The measurements record (JSONL) and its
derived index" section for the schema before writing a NEW tool that measures anything — the writer
epilogue is one shared call site on purpose.

## Instruction counts alone are never a verdict
`--ncu`'s IPC and stall-reason columns are not decoration. §16 addendum 2: an instruction count that
FALLS while wall time RISES is RED, not a win — see FINDINGS for a worked example (e.g. the
M1a→S1-parent regression) of instructions falling while stalls drove time up. Report time + IPC +
stalls together every time, never instructions in isolation. A regression past the ratio bar
KERNEL_STANDARDS §16 sets, on either a dense or a sparse probe cell vs. the previous stage, is RED —
report it, Blake rules, don't tune around it.

- NEVER wrap tools/probe_cells.py in an external `flock /tmp/rola_gpu.lock`: it takes the GPU lock ITSELF (check the source — grep rola_gpu.lock); an outer flock on the same file self-deadlocks silently (K41 and K38 both hit it). Same rule for any tool that documents taking the lock.

## Addendum (F1b, 2026-08-29)
- THE `sha` COLUMN IS THE WORKTREE HEAD, NOT THE BINARY: commit before the probe you intend to record, or
  the K38 rows name the wrong commit (an uncommitted tree stamps its base).
- DRAM NOISE FLOOR (G1b-pad, 2026-08-29): single-oneshot ncu DRAM bytes drift ~0.6% on this host (a
  no-state control launch shows it) — a DRAM delta under 1% is not a finding; alignment questions are
  L1-REQUEST questions (l1tex requests/wavefronts per instruction), which the fixed metric list does not
  ask — add them explicitly when the mechanism is an alignment.
- DRAM BYTES ARE BIMODAL ON THIS HOST (G1c-dram, 2026-08-29): dram__bytes swings ±14% launch-to-launch
  even with NO state I/O (the --state none control) — never quote one draw; report p50 AND p75 over ≥ 60
  profiled launches per point, interleaved, and compare distributions.
- `--state paged` WITHOUT `--continuation` RUNS NO ENTRY SWEEP (make_call passes a null s_in): every
  paged number measured that way is the exit sweep alone and cannot show a read-side effect. Use the
  continuation shape (entry plane IS exit plane) for any state-traffic claim; tools/experiments/
  dram_by_activity.py is the committed activity-controlled harness (read-only / written / none variants).
