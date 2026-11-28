# `stream_rate`

Mirrors `measure/lab/experiments/stream_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the H100 stream, a CTA an SM, from HBM and from L2: the read rate, the write rate and a copy's? The producer's floors (its outputs to HBM, its weights from L2) are priced on these, and no lab row recorded either (`measure/rates.json` had no bandwidth row).

THE FORM. A CTA an SM of 512 threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in flight a thread, in two halves at their own registers, each word folded into a chain a lane. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was read in another table. The sources: HBM (each CTA its own slice of a 1 GiB table once; the copy 512 MB into 512 MB); L2 DISTINCT (each CTA its own 128 KB slice of a 16.5 MB table 64 times); L2 SHARED (every CTA the whole of one 4 MB table 4 times from its own offset). The two L2 walks run again PAST L1 (`ld.global.cg`, cached in L2 only: the level the producer's staged copies, `cp.async.cg` and the tensor copy, read; a slice past L1's capacity would instead push the walk's table toward L2's own 50 MB and mix HBM's misses into the row). Three more walks past L1 separate why a CTA's own slice reads slower than a shared table: the own slice on half the SMs, its 16 units interleaved over the table, and every CTA one shared 128 KB slice. Every chain, written word and copied word is checked exactly on a short launch. Each row records GB/s the device (events) and bytes a cycle an SM (the CTA's span).

PREDICTION, run 20261127T232424Z-209524-19f37764 (written 2026-11-27, before it ran; its basis: HBM against the documented 3.35 TB/s; L2 from A100's documented L2 (5,120 bytes a clock over 108 SMs, 47 an SM) at 50-85% of it):

- HBM read, a CTA its own slice of a 1 GiB table: 2,850-3,350 GB/s the device.
- HBM write, a CTA its own slice of a 1 GiB table: 2,700-3,350 GB/s the device.
- HBM copy, a CTA its own slice of 512 MB into 512 MB: 2,600-3,300 GB/s the device.
- L2 read, a CTA its own 128 KB slice 64 times: 20.0-40.0 bytes a cycle an SM.
- L2 read, every CTA one 4 MB table 4 times: 15.0-40.0 bytes a cycle an SM.

PREDICTION, run 20261128T013030Z-737802-5ab0223a (written 2026-11-27, before it ran; its basis: the first run's two L2 bands missed on wrong prices: the ceiling came from A100's documented L2, where the H100 gave every SM reading one shared table 43.0 bytes a cycle; and a CTA's own 128 KB slice was taken to miss L1, which plain loads on sm_90 do not (it fits an SM's L1). From the H100's 43.0: the plain walks repeat within 5%; past L1 the shared table stays near 43 (its 4 MB a pass overran L1 already, so `.cg` removes nothing it used), and a CTA's own slice reads up to a fifth under it, never over (no L1 now, its lines over both L2 partitions); HBM's walks repeat within 3%):

- HBM read, a CTA its own slice of a 1 GiB table: 3,068-3,258 GB/s the device.
- HBM write, a CTA its own slice of a 1 GiB table: 3,090-3,281 GB/s the device.
- HBM copy, a CTA its own slice of 512 MB into 512 MB: 2,777-2,949 GB/s the device.
- L2 read, a CTA its own 128 KB slice 64 times: 62.0-69.0 bytes a cycle an SM.
- L2 read, every CTA one 4 MB table 4 times: 41.0-45.0 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times: 34.0-48.0 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times: 40.0-46.0 bytes a cycle an SM.

PREDICTION, run 20261128T014429Z-424977-ce56362c (written 2026-11-28, before it ran; its basis: the second run's own slice past L1 missed under its band (30.95 bytes a cycle an SM, 0.73 of the shared table's 42.38): its price took the far L2 partition to cost at most a fifth. Three walks separate the explanations. IF a CTA's distinct lines are served from the one partition that holds them (half the SMs' far) while a table every SM reads is served near from both, as predicted here: the own slice on half the SMs reads as on all (a path an SM, no aggregate cap), interleaving its units over the table moves nothing (placement by address is already spread), and every CTA reading one shared 128 KB slice reads as fast as the shared 4 MB table (sharing is what lifts it). An aggregate cap on distinct lines would lift the half-the-SMs walk instead; a placement effect, the interleaved one; and the shared small slice under 34 would say sharing is not the cause. Every walk of the second run repeats within 5% (HBM's within 3%)):

- HBM read, a CTA its own slice of a 1 GiB table: 3,077-3,267 GB/s the device.
- HBM write, a CTA its own slice of a 1 GiB table: 3,094-3,286 GB/s the device.
- HBM copy, a CTA its own slice of 512 MB into 512 MB: 2,770-2,942 GB/s the device.
- L2 read, a CTA its own 128 KB slice 64 times: 62.0-68.5 bytes a cycle an SM.
- L2 read, every CTA one 4 MB table 4 times: 41.4-45.7 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times: 29.4-32.5 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times: 40.3-44.5 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times, half the SMs: 28.0-34.0 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), a CTA its own 128 KB interleaved over the table 64 times: 28.0-34.0 bytes a cycle an SM.
- L2 read past L1 (ld.global.cg), every CTA one 128 KB slice 64 times: 39.0-46.0 bytes a cycle an SM.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM e00def8gx4f4d34esq, driver 580.196.07; 2026-11-28, the experiment at
00123881, run 20261128T014429Z-424977-ce56362c; 7 repetitions, the 10 checked walks exact; the median, the widest spread
of a row 4.5%; the clock as the card left it):

| the walk | GB/s the device | bytes a cycle an SM | the prediction | held |
|---|---:|---:|---|---|
| HBM read, a CTA its own slice of a 1 GiB table | 3,166.9 | 12.95 | 3,077-3,267 GB/s the device | held |
| HBM write, a CTA its own slice of a 1 GiB table | 3,170.2 | 13.21 | 3,094-3,286 GB/s the device | held |
| HBM copy, a CTA its own slice of 512 MB into 512 MB | 2,868.8 | 12.75 | 2,770-2,942 GB/s the device | held |
| L2 read, a CTA its own 128 KB slice 64 times | 15,440.9 | 63.39 | 62.0-68.5 bytes a cycle an SM | held |
| L2 read, every CTA one 4 MB table 4 times | 10,252.7 | 42.90 | 41.4-45.7 bytes a cycle an SM | held |
| L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times | 7,318.7 | 30.81 | 29.4-32.5 bytes a cycle an SM | held |
| L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times | 10,113.4 | 42.58 | 40.3-44.5 bytes a cycle an SM | held |
| L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times, half the SMs | 5,286.1 | 51.87 | 28.0-34.0 bytes a cycle an SM | MISSED |
| L2 read past L1 (ld.global.cg), a CTA its own 128 KB interleaved over the table 64 times | 7,323.4 | 30.68 | 28.0-34.0 bytes a cycle an SM | held |
| L2 read past L1 (ld.global.cg), every CTA one 128 KB slice 64 times | 13,052.8 | 55.20 | 39.0-46.0 bytes a cycle an SM | MISSED |

WHAT IT SHOWS.
- HBM READ, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,166.9 GB/s the device, 12.95 bytes a cycle an SM, inside its band.
- HBM WRITE, A CTA ITS OWN SLICE OF A 1 GIB TABLE: 3,170.2 GB/s the device, 13.21 bytes a cycle an SM, inside its band.
- HBM COPY, A CTA ITS OWN SLICE OF 512 MB INTO 512 MB: 2,868.8 GB/s the device, 12.75 bytes a cycle an SM, inside its band.
- L2 READ, A CTA ITS OWN 128 KB SLICE 64 TIMES: 15,440.9 GB/s the device, 63.39 bytes a cycle an SM, inside its band.
- L2 READ, EVERY CTA ONE 4 MB TABLE 4 TIMES: 10,252.7 GB/s the device, 42.90 bytes a cycle an SM, inside its band.
- L2 READ PAST L1 (LD.GLOBAL.CG), A CTA ITS OWN 128 KB SLICE 64 TIMES: 7,318.7 GB/s the device, 30.81 bytes a cycle an SM, inside its band.
- L2 READ PAST L1 (LD.GLOBAL.CG), EVERY CTA ONE 4 MB TABLE 4 TIMES: 10,113.4 GB/s the device, 42.58 bytes a cycle an SM, inside its band.
- L2 READ PAST L1 (LD.GLOBAL.CG), A CTA ITS OWN 128 KB SLICE 64 TIMES, HALF THE SMS: 5,286.1 GB/s the device, 51.87 bytes a cycle an SM, over its band.
- L2 READ PAST L1 (LD.GLOBAL.CG), A CTA ITS OWN 128 KB INTERLEAVED OVER THE TABLE 64 TIMES: 7,323.4 GB/s the device, 30.68 bytes a cycle an SM, inside its band.
- L2 READ PAST L1 (LD.GLOBAL.CG), EVERY CTA ONE 128 KB SLICE 64 TIMES: 13,052.8 GB/s the device, 55.20 bytes a cycle an SM, over its band.

Record: `data:records/fsla/measure/lab/stream_rate.run/20261128T014429Z-424977-ce56362c/output.json`.
