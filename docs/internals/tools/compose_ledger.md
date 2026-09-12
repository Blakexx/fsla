# `tools/compose_ledger.py` — a phase's time attributed to its parts by composition

KERNEL_STANDARDS §22 (9). The carry kernel's MMA phases are built from parts that a build may stub
(`docs/internals/carry/carry_kernel.md#parts`). A LADDER over a phase builds the kernel with none of
that phase's parts real and every other part real, then adds the phase's parts back in the order
named, one rung a build; the last rung is the kernel itself.

    python tools/compose_ledger.py --cells flagship-dense,nl64k-alt-k4 \
        --ladder readout:stream,loads,drain --ladder fold:pool,ring,loads

Each rung is read in wall time by `tools/phase_ledger.py` (one warm-up launch, then `--launches`):
a part's cost is the step between the rung without it and the rung with it. No constant enters and
no stall sample is interpreted. A rung is VALID only when two checks pass. The profiler's HMMA
count for each cell must equal the kernel's, since a stub that changed the workload measures a
different kernel. The build's device instructions must hash differently from the kernel's, since a
build that did not take the mask measures nothing. The kernel is rebuilt with every part real at
the end, and its instructions must hash to the value taken before the first rung. The kernel is then
read again: its before-and-after difference is the run's drift bound, and a rung step smaller than
that bound is not a reading. The first run needed it: another process had the GPU at 36-38%, and the
sparse cell read 45% slow with rungs that contradicted each other. Nothing in a single run's numbers
flagged that except the kernel disagreeing with the day's readings.

Why it exists: on 2026-09-12 five kernel changes were built on inferences from warp-side
instruments (stall samples, the phase clock, a utilization computed from an HMMA count and a
32-cycle constant), and their misses could not separate a wrong fix from a wrong premise. An
MMA-only variant then measured the phases' floor directly (readout 38.2K, fold 35.7K cycles a warp
a window at flagship-dense against 52.4K and 47.0K). The composer makes that measurement a
mechanical, per-part ladder.

The report lands in `benchmarks/results/ledgers/<utc>-<sha7>-compose.{json,md}`. Builds run
serially (the host's memory watchdog kills parallel ones). Every build is an iteration build and
never ships.
