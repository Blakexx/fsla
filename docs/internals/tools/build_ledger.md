# `tools/build_ledger.py` — every measurement a build is judged by, in one run

KERNEL_STANDARDS §22's suite as one command. It drives the tools the gate already has and reads
their output back, so no number is typed twice: `tools/sass_gate.py` (§20 signatures),
`tools/life_ranges.py` (peak live registers per region beside the allocation), the scoped fp64
oracle, `tools/phase_ledger.py` per cell (cycles a warp a window per phase, per-warp rows),
`ncu` SourceCounters and pipe counters with `tools/region_ledger.py` per component, tensor
utilization as HMMA count × the measured cycles an HMMA holds the pipe over scheduler cycles, and
`tools/probe_cells.py` against the baseline run with its own schedule (`box` dense, `sparse-g32`
sparse; this tree `first`).

The report lands in `benchmarks/results/ledgers/<utc>-<sha7>-<tag>.json` and `.md`, and the
markdown carries the per-phase diff against the newest earlier report for the same cells. A step
that fails is a row with its output, never a stop. `--skip` names steps to leave out (a CPU-only
dry run is `--skip oracle,phases,profile,ab`).

The tree grows: a number computed by hand during a debug session is added as a step the same day
(§22 addendum). Candidates on the card: per-phase HMMA activity (a sub-phase clock), the edge
barriers' wait split by site.
