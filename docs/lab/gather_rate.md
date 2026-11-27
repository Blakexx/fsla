# `gather_rate`

Mirrors `measure/lab/experiments/gather_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The gather fetches a token's rows scattered from a table, stores them into shared memory in the order they will be
read, and sends the image out to global memory by one bulk copy. What does that access pattern cost a row -- a
128-byte row -- when the table is one the card's cache holds (8 MB: [`load_rate`](load_rate.md),
[`state_move`](state_move.md)) and when it is not (2 GB), and when the image goes to a ring the cache holds or to
fresh memory?

THE FORM. A CTA an SM of 16 warps. A round is 64 DEPTH rows of 128 bytes (512 at DEPTH 8, 256 at DEPTH 4): a lane
loads 16 bytes, an instruction four rows, DEPTH instructions a warp in flight, each loaded a round before its use, except that in the LOAD form at DEPTH 8 the toolkit copies the fourth register of one instruction at the round's end, so that instruction's four rows are read in the round that issued them.
- LOAD: every word folded into a chain a lane; nothing stored.
- STORE: a round's rows stored into one of three images in shared memory, a warp's lanes on 32 banks.
- OUT: the image then stored to the CTA's span of global memory by one bulk copy, a round's copy waited two rounds on,
  so one barrier a round orders the images' reuse. The span is a RING of 256 KB a CTA (4 rounds at 512 rows, 8 at 256)
  the cache holds, or FRESH MEMORY, a span as long as the launch.
Both table sizes, 8 MB and 2 GB, take each form. Every form is checked exactly: the chain, the last round's image, or
every image the copies wrote, against the table's words.
The row's cost is the span of a CTA's 16 warps over the measured rounds' rows, in cycles a 128-byte row a CTA.

WHAT IT WILL DECIDE: the gather's floor a token is these rows' cycles times the token's lines, where the bundle fits
the cache and where it does not; which of the load, the store into shared memory and the copy out is the largest term
of the gather, to be split next; and whether a ring the cache holds is cheaper than fresh memory.

MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM rgfp2qac, driver 580.196.07; 2026-11-20, the experiment at
7d0d6925; 7 repetitions; the 10 checked rows pass, every word exact; the median, the widest
spread of a row 3.5%). Cycles a 128-byte row a CTA:

| form | 8 MB table | 2 GB table |
|---|---:|---:|
| LOAD: loaded and folded | 2.58 | 9.71 |
| STORE: and stored into shared memory | 3.71 | 9.80 |
| OUT, a ring: and copied out to a ring of 256 KB a CTA | 10.05 | 19.10 |
| OUT, fresh memory: and copied out to fresh memory | 12.30 | 21.06 |
| OUT, a ring, 256 rows in flight | 10.91 | 20.86 |

WHAT IT SHOWS.
- THE 8 MB TABLE, THE LOAD: 2.58 cycles a 128-byte row a CTA.
- THE 8 MB TABLE, THE STORE INTO SHARED MEMORY ADDED: 3.71 cycles a 128-byte row a CTA, 1.13 cycles above the 2.58 of LOAD.
- THE 8 MB TABLE, THE COPY OUT TO A RING ADDED: 10.05 cycles a 128-byte row a CTA, 6.34 cycles above the 3.71 of STORE.
- THE 8 MB TABLE, FRESH MEMORY AGAINST THE RING: 12.30 cycles a 128-byte row a CTA, 2.25 cycles above the 10.05 of OUT, a ring.
- THE 8 MB TABLE, 256 ROWS IN FLIGHT AGAINST 512 (THE RING): 10.91 cycles a 128-byte row a CTA, 0.87 cycles above the 10.05 of OUT, a ring.
- THE 2 GB TABLE, THE LOAD: 9.71 cycles a 128-byte row a CTA.
- THE 2 GB TABLE, THE STORE INTO SHARED MEMORY ADDED: 9.80 cycles a 128-byte row a CTA, the same as the 9.71 of LOAD.
- THE 2 GB TABLE, THE COPY OUT TO A RING ADDED: 19.10 cycles a 128-byte row a CTA, 9.30 cycles above the 9.80 of STORE.
- THE 2 GB TABLE, FRESH MEMORY AGAINST THE RING: 21.06 cycles a 128-byte row a CTA, 1.96 cycles above the 19.10 of OUT, a ring.
- THE 2 GB TABLE, 256 ROWS IN FLIGHT AGAINST 512 (THE RING): 20.86 cycles a 128-byte row a CTA, 1.76 cycles above the 19.10 of OUT, a ring.

Record: `data:lab/gather_rate/sm_90a.json`.
