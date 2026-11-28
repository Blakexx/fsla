# `stream_rate`

Mirrors `measure/lab/experiments/stream_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the H100 stream, a CTA an SM, from HBM and from L2: the read rate, the write rate and a copy's? The producer's floors (its outputs to HBM, its weights from L2) are priced on these, and no lab row recorded either (`measure/rates.json` had no bandwidth row).

THE FORM. A CTA an SM of 512 threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in flight a thread, in two halves at their own registers, each word folded into a chain a lane. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was read in another table. The sources: HBM (each CTA its own slice of a 1 GiB table once; the copy 512 MB into 512 MB); L2 DISTINCT (each CTA its own 128 KB slice of a 16.5 MB table 64 times); L2 SHARED (every CTA the whole of one 4 MB table 4 times from its own offset). The two L2 walks run again PAST L1 (`ld.global.cg`, cached in L2 only: the level the producer's staged copies, `cp.async.cg` and the tensor copy, read; a slice past L1's capacity would instead push the walk's table toward L2's own 50 MB and mix HBM's misses into the row). Every chain, written word and copied word is checked exactly on a short launch. Each row records GB/s the device (events) and bytes a cycle an SM (the CTA's span).

PREDICTION (written 2026-11-27, before the first run). HBM read 2,850-3,350 GB/s (0.85-1.00 of the documented 3.35 TB/s); HBM write 2,700-3,350; HBM copy (read and write bytes) 2,600-3,300. L2, a CTA's own slice: 20-40 bytes a cycle an SM (A100's documented 5,120 bytes a clock over 108 SMs is 47 an SM; a stream reaches 50-85% of such a peak). L2, one shared table: 15-40.

THE SECOND RUN'S PREDICTION (written 2026-11-27, before it ran). The first run's two L2 bands missed, and the prices
were wrong: the ceiling came from A100's documented L2 (47 bytes a clock an SM) at 50-85%, where the H100 gave every
SM reading one shared table 43.0; and a CTA's own 128 KB slice was taken to miss L1, which plain loads on sm_90 do not
(the slice fits an SM's L1). From the H100's 43.0: the two plain walks repeat within 5% (the own slice 62-69, the
shared table 41-45 bytes a cycle an SM); PAST L1 the shared table stays 40-46 (its 4 MB a pass already overran L1,
so `.cg` removes nothing it used), and a CTA's own slice reads 34-48 (no L1 now; its lines spread over both L2
partitions, the far one's at a longer path, so up to a fifth under the shared table, not over it). HBM's three walks
repeat within 3%.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00tq93nhkp40jq82t, driver 580.196.07; 2026-11-27, the experiment at
998eae3c; 7 repetitions, the 5 checked walks exact; the median, the widest spread of a row
3.8%; the clock as the card left it):

| the walk | GB/s the device | bytes a cycle an SM | the prediction | held |
|---|---:|---:|---|---|
| HBM read, a CTA its own slice of a 1 GiB table | 3,162.7 | 13.00 | 2,850-3,350 GB/s | yes |
| HBM write, a CTA its own slice of a 1 GiB table | 3,185.6 | 13.22 | 2,700-3,350 GB/s | yes |
| HBM copy, a CTA its own slice of 512 MB into 512 MB | 2,862.7 | 12.67 | 2,600-3,300 GB/s | yes |
| L2 read, a CTA its own 128 KB slice 64 times | 15,608.0 | 65.28 | 20-40 B/clk/SM | MISSED |
| L2 read, every CTA one 4 MB table 4 times | 10,242.1 | 43.01 | 15-40 B/clk/SM | MISSED |

WHAT IT SHOWS.
- HBM READ, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,162.7 GB/s the device, 13.00 bytes a cycle an SM, inside its band.
- HBM WRITE, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,185.6 GB/s the device, 13.22 bytes a cycle an SM, inside its band.
- HBM COPY, A CTA ITS OWN SLICE OF 512 MB INTO 512 MB: 2,862.7 GB/s the device, 12.67 bytes a cycle an SM, inside its band.
- L2 READ, A CTA ITS OWN 128 KB SLICE 64 TIMES: 15,608.0 GB/s the device, 65.28 bytes a cycle an SM, over its band; plain loads are cached in the SM's L1 on sm_90 and a 128 KB slice fits an SM whose kernel carves no shared memory, so its passes after the first hit L1: the figure is L1's, mostly, and no L2 rate.
- L2 READ, EVERY CTA ONE 4 MB TABLE 4 TIMES: 10,242.1 GB/s the device, 43.01 bytes a cycle an SM, over its band.

Record: `data:records/fsla/measure/lab/stream_rate.run/20261127T232424Z-209524-19f37764/output.json`.
