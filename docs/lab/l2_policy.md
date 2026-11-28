# `l2_policy`

Mirrors `measure/lab/experiments/l2_policy.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer streams what it reads once beside what every CTA reuses (a write head's weights from L2). Does an L2 policy -- the stream's loads evict-first, the table's evict-last (`createpolicy`, `L2::cache_hint`) -- keep the table and save time, against plain loads?

THE FORM. A CTA an SM of 512 threads; a trip loads 8 units (8 KB a CTA each) of a 1 GiB stream, each CTA its own slice once, and 8 units of a shared table every CTA walks round from its own offset; every word folded into a chain. Tables of 16 MB and 40 MB (the L2 is 50 MB), with and without the hints. Time a trip; the table's hit rate by the lab's `ncu` (`lts__t_sector_op_read_hit_rate.pct`: half the sectors are the table's, the stream's read once). Every chain is checked.

PREDICTION, run 20261128T014645Z-403597-16c6d19e (written 2026-11-27, before it ran; its basis: a table line is re-read after ~2x the table's bytes of distinct insertions, and LRU keeps it iff that is under the 50 MB L2; the trip HBM-bound on its misses):

- 16 MB: the plain trip over the hinted: 0.90-1.10 times.
- 40 MB: the plain trip over the hinted: 1.30-2.00 times.

PREDICTION, run 20261128T050251Z-715035-6e09bfc4 (written 2026-11-28, before it ran; its basis: the first run's trips, a kernel now a table size (the profiler's hit rate then each table's): the 16 MB plain trip 1.31x the hinted and the 40 MB 1.10x, repeating within 5%):

- 16 MB: the plain trip over the hinted: 1.24-1.38 times.
- 40 MB: the plain trip over the hinted: 1.05-1.16 times.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00gnjr2w7t9bny2p4, driver 580.196.07; 2026-11-28, the experiment at
0aac2c88, run 20261128T050251Z-715035-6e09bfc4; 7 repetitions, the 4 checked variants exact; the median, the widest
spread of a row 1.7%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| 16 MB: the plain trip over the hinted | 1.27 times | 1.24-1.38 times | held |
| 40 MB: the plain trip over the hinted | 1.13 times | 1.05-1.16 times | held |

The record's rows:
- 16 MB table beside a 1 GiB stream, no hint: a trip: 6,528.78 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 16 MB table beside a 1 GiB stream, no hint: the launch: 435.42 microseconds.
- 16 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: a trip: 5,147.07 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 16 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: the launch: 343.46 microseconds.
- 40 MB table beside a 1 GiB stream, no hint: a trip: 8,092.23 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 40 MB table beside a 1 GiB stream, no hint: the launch: 538.85 microseconds.
- 40 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: a trip: 7,131.56 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 40 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: the launch: 473.44 microseconds.

Record: `data:records/fsla/measure/lab/l2_policy.run/20261128T050251Z-715035-6e09bfc4/output.json`.
