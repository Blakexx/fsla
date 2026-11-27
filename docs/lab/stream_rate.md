# `stream_rate`

Mirrors `measure/lab/experiments/stream_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the H100 stream, a CTA an SM, from HBM and from L2: the read rate, the write rate and a copy's? The producer's floors (its outputs to HBM, its weights from L2) are priced on these, and no lab row recorded either (`measure/rates.json` had no bandwidth row).

THE FORM. A CTA an SM of 512 threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in flight a thread, in two halves at their own registers, each word folded into a chain a lane. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was read in another table. The sources: HBM (each CTA its own slice of a 1 GiB table once; the copy 512 MB into 512 MB); L2 DISTINCT (each CTA its own 128 KB slice of a 16.5 MB table 64 times); L2 SHARED (every CTA the whole of one 4 MB table 4 times from its own offset). Every chain, written word and copied word is checked exactly on a short launch. Each row records GB/s the device (events) and bytes a cycle an SM (the CTA's span).

PREDICTION (written 2026-11-27, before the first run). HBM read 2,850-3,350 GB/s (0.85-1.00 of the documented 3.35 TB/s); HBM write 2,700-3,350; HBM copy (read and write bytes) 2,600-3,300. L2, a CTA's own slice: 20-40 bytes a cycle an SM (A100's documented 5,120 bytes a clock over 108 SMs is 47 an SM; a stream reaches 50-85% of such a peak). L2, one shared table: 15-40.

NOT YET MEASURED: it runs on the producer's held H100 session.
