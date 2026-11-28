# `stream_rate`

Mirrors `measure/lab/experiments/stream_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the H100 stream, a CTA an SM, from HBM and from L2: the read rate, the write rate and a copy's? The producer's floors (its outputs to HBM, its weights from L2) are priced on these, and no lab row recorded either (`measure/rates.json` had no bandwidth row).

THE FORM. A CTA an SM of 512 threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in flight a thread, in two halves at their own registers, each word folded into a chain a lane. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was read in another table. The sources: HBM (each CTA its own slice of a 1 GiB table once; the copy 512 MB into 512 MB); L2 DISTINCT (each CTA its own 128 KB slice of a 16.5 MB table 64 times); L2 SHARED (every CTA the whole of one 4 MB table 4 times from its own offset). The two L2 walks run again PAST L1 (`ld.global.cg`, cached in L2 only: the level the producer's staged copies, `cp.async.cg` and the tensor copy, read; a slice past L1's capacity would instead push the walk's table toward L2's own 50 MB and mix HBM's misses into the row). Every chain, written word and copied word is checked exactly on a short launch. Each row records GB/s the device (events) and bytes a cycle an SM (the CTA's span).

PREDICTION, run 20261127T232424Z-209524-19f37764 (written 2026-11-27, before it ran; its basis: HBM against the documented 3.35 TB/s; L2 from A100's documented L2 (5,120 bytes a clock over 108 SMs, 47 an SM) at 50-85% of it):

- HBM read, a CTA its own slice of a 1 GiB table: 2,850-3,350 GB/s the device.
- HBM write, a CTA its own slice of a 1 GiB table: 2,700-3,350 GB/s the device.
- HBM copy, a CTA its own slice of 512 MB into 512 MB: 2,600-3,300 GB/s the device.
- L2 read, a CTA its own 128 KB slice 64 times: 20-40 bytes a cycle an SM.
- L2 read, every CTA one 4 MB table 4 times: 15-40 bytes a cycle an SM.

PREDICTION, the run to come (written 2026-11-27, before it ran; its basis: the first run's two L2 bands missed on wrong prices: the ceiling came from A100's documented L2, where the H100 gave every SM reading one shared table 43.0 bytes a cycle; and a CTA's own 128 KB slice was taken to miss L1, which plain loads on sm_90 do not (it fits an SM's L1). From the H100's 43.0: the plain walks repeat within 5%; past L1 the shared table stays near 43 (its 4 MB a pass overran L1 already, so `.cg` removes nothing it used), and a CTA's own slice reads up to a fifth under it, never over (no L1 now, its lines over both L2 partitions); HBM's walks repeat within 3%):

- HBM read, a CTA its own slice of a 1 GiB table: 3,068-3,258 GB/s the device.
- HBM write, a CTA its own slice of a 1 GiB table: 3,090-3,281 GB/s the device.
- HBM copy, a CTA its own slice of 512 MB into 512 MB: 2,777-2,949 GB/s the device.
- L2 read, a CTA its own 128 KB slice 64 times: 62-69 bytes a cycle an SM.
- L2 read, every CTA one 4 MB table 4 times: 41-45 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times: 34-48 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times: 40-46 bytes a cycle an SM.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00tq93nhkp40jq82t, driver 580.196.07; 2026-11-27, the experiment at
998eae3c, run 20261127T232424Z-209524-19f37764; 7 repetitions, the 5 checked walks exact; the median, the widest spread
of a row 3.8%; the clock as the card left it):

| the walk | GB/s the device | bytes a cycle an SM | the prediction | held |
|---|---:|---:|---|---|
| HBM read, a CTA its own slice of a 1 GiB table | 3,162.7 | 13.00 | 2,850-3,350 GB/s the device | held |
| HBM write, a CTA its own slice of a 1 GiB table | 3,185.6 | 13.22 | 2,700-3,350 GB/s the device | held |
| HBM copy, a CTA its own slice of 512 MB into 512 MB | 2,862.7 | 12.67 | 2,600-3,300 GB/s the device | held |
| L2 read, a CTA its own 128 KB slice 64 times | 15,608.0 | 65.28 | 20-40 bytes a cycle an SM | MISSED |
| L2 read, every CTA one 4 MB table 4 times | 10,242.1 | 43.01 | 15-40 bytes a cycle an SM | MISSED |

WHAT IT SHOWS.
- HBM READ, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,162.7 GB/s the device, 13.00 bytes a cycle an SM, inside its band.
- HBM WRITE, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,185.6 GB/s the device, 13.22 bytes a cycle an SM, inside its band.
- HBM COPY, A CTA ITS OWN SLICE OF 512 MB INTO 512 MB: 2,862.7 GB/s the device, 12.67 bytes a cycle an SM, inside its band.
- L2 READ, A CTA ITS OWN 128 KB SLICE 64 TIMES: 15,608.0 GB/s the device, 65.28 bytes a cycle an SM, over its band.
- L2 READ, EVERY CTA ONE 4 MB TABLE 4 TIMES: 10,242.1 GB/s the device, 43.01 bytes a cycle an SM, over its band.

Record: `data:records/fsla/measure/lab/stream_rate.run/20261127T232424Z-209524-19f37764/output.json`.
