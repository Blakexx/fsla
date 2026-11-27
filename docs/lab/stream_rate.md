# `stream_rate`

Mirrors `measure/lab/experiments/stream_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the H100 stream, a CTA an SM, from HBM and from L2: the read rate, the write rate and a copy's? The producer's floors (its outputs to HBM, its weights from L2) are priced on these, and no lab row recorded either (`measure/rates.json` had no bandwidth row).

THE FORM. A CTA an SM of 512 threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in flight a thread, in two halves at their own registers, each word folded into a chain a lane. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was read in another table. The sources: HBM (each CTA its own slice of a 1 GiB table once; the copy 512 MB into 512 MB); L2 DISTINCT (each CTA its own 128 KB slice of a 16.5 MB table 64 times); L2 SHARED (every CTA the whole of one 4 MB table 4 times from its own offset). Every chain, written word and copied word is checked exactly on a short launch. Each row records GB/s the device (events) and bytes a cycle an SM (the CTA's span).

PREDICTION (written 2026-11-27, before the first run). HBM read 2,850-3,350 GB/s (0.85-1.00 of the documented 3.35 TB/s); HBM write 2,700-3,350; HBM copy (read and write bytes) 2,600-3,300. L2, a CTA's own slice: 20-40 bytes a cycle an SM (A100's documented 5,120 bytes a clock over 108 SMs is 47 an SM; a stream reaches 50-85% of such a peak). L2, one shared table: 15-40.

MEASURED (H100 SXM, 132 SMs, Nebius instance computeinstance-e00tq93nhkp40jq82t; 2026-11-27, the experiment at
998eae3c; 7 repetitions, every chain, written word and copied word exact; the median, the spread under 2.2%; the
clock as the card left it):

| the row | GB/s the device | bytes a cycle an SM | the prediction | held |
|---|---|---|---|---|
| HBM read, a CTA its own slice of a 1 GiB table | 3,162.7 | 13.00 | 2,850-3,350 GB/s | yes |
| HBM write, a CTA its own slice of a 1 GiB table | 3,185.6 | 13.22 | 2,700-3,350 GB/s | yes |
| HBM copy, 512 MB into 512 MB (read and write bytes) | 2,862.7 | 12.67 | 2,600-3,300 GB/s | yes |
| L2 read, a CTA its own 128 KB slice 64 times | 15,608.0 | 65.28 | 20-40 B/clk/SM | MISSED |
| L2 read, every CTA one 4 MB table 4 times | 10,242.1 | 43.01 | 15-40 B/clk/SM | MISSED |

WHAT IT SHOWS.
- HBM STREAMS AT 0.94-0.95 OF THE DOCUMENTED 3.35 TB/s, read or write (3,163 and 3,186 GB/s, 13.0 and 13.2 bytes a
  cycle an SM); a copy's read and write together 2,863 GB/s, 0.85.
- EVERY SM READING ONE 4 MB TABLE FROM L2 TAKES 43.0 BYTES A CYCLE AN SM (10.2 TB/s the device), over the 40 predicted.
- THE OWN-SLICE ROW IS NOT AN L2 RATE. A plain `ld.global` is cached in the SM's L1 on sm_90, and a 128 KB slice fits
  in an SM whose kernel carves no shared memory (its L1 up to 256 KB), so its passes after the first hit L1: 65.3
  bytes a cycle an SM is L1's, mostly. The row's form, not the machine, is what it measured.
- THE WRONG PRICES, named: (1) the L2 ceiling was priced from A100's documented 5,120 bytes a clock over 108 SMs (47 an
  SM) at 50-85%; the H100 delivers 43 an SM to a shared table, 0.91 of that peak, so the band's top (40) was low; (2)
  the own-slice row assumed a 128 KB slice misses L1, which it does not. Before this row is a rate, its loads go
  `ld.global.cg` (L2 only) or its slice past L1's capacity, and the price is predicted again.

Record: the run's sample 20261127T232424Z-209524-19f37764 of `@fsla//measure/lab:stream_rate.run`, outside the pinned data
until the data repository takes it.
