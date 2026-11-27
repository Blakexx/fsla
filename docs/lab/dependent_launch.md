# `dependent_launch`

Mirrors `measure/lab/experiments/dependent_launch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer's launch is followed by the op's, which reads what it wrote. What does programmatic dependent launch save a pair of dependent kernels -- the second launched with the attribute, its prologue (what it reads that the first does not write) run while the first drains, then `griddepcontrol.wait` -- against two plain launches?

THE FORM. The first kernel, a CTA an SM of 256 threads, writes its own 64 KB slice of an 8.4 MB buffer, `griddepcontrol.launch_dependents` at its top. The second, the same grid: its prologue stages and folds its own 32 KB of a weights table, then waits, then reads the first's slice. 64 pairs back to back in one stream, timed by events. Every word of the second is checked: it read its own pair's writes.

PREDICTION (written 2026-11-27, before the first run). A plain pair 8-14 microseconds (the bytes at ~3 TB/s, ~7 us, and two launch gaps); the dependent second saves its gap and its prologue: 1.0-3.5 us a pair, 10-30%.

NOT YET MEASURED: it runs on the producer's held H100 session.
