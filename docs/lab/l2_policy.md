# `l2_policy`

Mirrors `measure/lab/experiments/l2_policy.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer streams what it reads once beside what every CTA reuses (a write head's weights from L2). Does an L2 policy -- the stream's loads evict-first, the table's evict-last (`createpolicy`, `L2::cache_hint`) -- keep the table and save time, against plain loads?

THE FORM. A CTA an SM of 512 threads; a trip loads 8 units (8 KB a CTA each) of a 1 GiB stream, each CTA its own slice once, and 8 units of a shared table every CTA walks round from its own offset; every word folded into a chain. Tables of 16 MB and 40 MB (the L2 is 50 MB), with and without the hints. Time a trip; the table's hit rate by the lab's `ncu` (`lts__t_sector_op_read_hit_rate.pct`: half the sectors are the table's, the stream's read once). Every chain is checked.

PREDICTION, run 20261128T014645Z-403597-16c6d19e (written 2026-11-27, before it ran; its basis: a table line is re-read after ~2x the table's bytes of distinct insertions, and LRU keeps it iff that is under the 50 MB L2; the trip HBM-bound on its misses):

- 16 MB: the plain trip over the hinted: 0.90-1.10 times.
- 40 MB: the plain trip over the hinted: 1.30-2.00 times.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00def8gx4f4d34esq, driver 580.196.07; 2026-11-28, the experiment at
00123881, run 20261128T014645Z-403597-16c6d19e; 7 repetitions, the 4 checked variants exact; the median, the widest
spread of a row 1.4%; the clock as the card left it):

| the figure | measured | the prediction | held |
|---|---:|---|---|
| 16 MB: the plain trip over the hinted | 1.31 times | 0.90-1.10 times | MISSED |
| 40 MB: the plain trip over the hinted | 1.10 times | 1.30-2.00 times | MISSED |

The record's rows:
- 16 MB table beside a 1 GiB stream, no hint: a trip: 6,780.56 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 16 MB table beside a 1 GiB stream, no hint: the launch: 449.82 microseconds.
- 16 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: a trip: 5,185.12 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 16 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: the launch: 344.80 microseconds.
- 40 MB table beside a 1 GiB stream, no hint: a trip: 8,020.30 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 40 MB table beside a 1 GiB stream, no hint: the launch: 534.72 microseconds.
- 40 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: a trip: 7,278.05 cycles a trip a CTA (64 KB of the stream, 64 KB of the table).
- 40 MB table beside a 1 GiB stream, the stream evict-first and the table evict-last: the launch: 484.96 microseconds.

Record: `data:records/fsla/measure/lab/l2_policy.run/20261128T014645Z-403597-16c6d19e/output.json`.
