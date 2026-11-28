# `dependent_launch`

Mirrors `measure/lab/experiments/dependent_launch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's launch is followed by the op's, which reads what it wrote. What does programmatic dependent launch save a pair of dependent kernels -- the second launched with the attribute, its prologue (what it reads that the first does not write) run while the first drains, then `griddepcontrol.wait` -- against two plain launches?

THE FORM. The first kernel, a CTA an SM of 256 threads, writes its own 64 KB slice of an 8.4 MB buffer, `griddepcontrol.launch_dependents` at its top, or (a third form) after its writes. The second, the same grid: its prologue stages and folds its own 32 KB of a weights table, then waits, then reads the first's slice. 64 pairs back to back in one stream, timed by events. Every word of the second is checked: it read its own pair's writes.

PREDICTION, run 20261128T014912Z-160980-6ad3db89 (written 2026-11-27, before it ran; its basis: the pair's bytes (8.4 MB written, 4.2 MB of weights and 8.4 MB read) at ~3 TB/s, ~7 us, and two launch gaps; the dependent second saves its gap and its prologue (the weights, ~1.4 us)):

- two plain launches: a pair: 8.00-14.00 us a pair.
- saved a pair by the dependent launch: 1.00-3.50 us.

PREDICTION, run 20261128T050218Z-236870-abd0066c (written 2026-11-28, before it ran; its basis: the first run's dependent pair (the trigger at the first's top) was 0.94 us slower than plain: the dependent's CTAs launched at once and shared the SMs and HBM with the first's writes; with the trigger after its writes they launch as the first's CTAs finish, so the pair keeps the launch's latency hidden (~0.3-1 us) and the prologue's overlap of the first's tail, without the contention; the other two forms repeat within 5%):

- two plain launches: a pair: 7.90-8.70 us a pair.
- the second launched dependent (griddepcontrol): a pair: 8.80-9.70 us a pair.
- saved a pair by the trigger after the writes: 0.10-1.00 us.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00gnjr2w7t9bny2p4, driver 580.196.07; 2026-11-28, the experiment at
0aac2c88, run 20261128T050218Z-236870-abd0066c; 7 repetitions, the 3 checked variants exact; the median, the widest
spread of a row 1.2%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| two plain launches: a pair | 8.36 us a pair | 7.90-8.70 us a pair | held |
| the second launched dependent (griddepcontrol): a pair | 9.28 us a pair | 8.80-9.70 us a pair | held |
| saved a pair by the trigger after the writes | -0.92 us | 0.10-1.00 us | MISSED |

The record's rows:
- two plain launches: a pair: 8.36 microseconds a pair.
- the second launched dependent (griddepcontrol): a pair: 9.28 microseconds a pair.
- the second launched dependent, the first's trigger after its writes: a pair: 9.28 microseconds a pair.

Record: `data:records/fsla/measure/lab/dependent_launch.run/20261128T050218Z-236870-abd0066c/output.json`.
