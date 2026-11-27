# `l2_policy`

Mirrors `measure/lab/experiments/l2_policy.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The producer streams what it reads once beside what every CTA reuses (a write head's weights from L2). Does an L2 policy -- the stream's loads evict-first, the table's evict-last (`createpolicy`, `L2::cache_hint`) -- keep the table and save time, against plain loads?

THE FORM. A CTA an SM of 512 threads; a trip loads 8 units (8 KB a CTA each) of a 1 GiB stream, each CTA its own slice once, and 8 units of a shared table every CTA walks round from its own offset; every word folded into a chain. Tables of 16 MB and 40 MB (the L2 is 50 MB), with and without the hints. Time a trip; the table's hit rate by the lab's `ncu` (`lts__t_sector_op_read_hit_rate.pct`: half the sectors are the table's, the stream's read once). Every chain is checked.

PREDICTION (written 2026-11-27, before the first run). A table line is re-read after ~2x the table's bytes of distinct insertions: LRU keeps it iff that is under 50 MB. 16 MB: the table's hit rate at least 80% with or without the hint, the trips equal within 10%. 40 MB: at most 40% without the hint, at least 60% with it, the trip 1.3x-2.0x faster.

NOT YET MEASURED: it runs on the producer's held H100 session.
