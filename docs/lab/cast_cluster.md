# `cast_cluster`

Mirrors `measure/lab/experiments/cast_cluster.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`value_cast`](value_cast.md) found a multicast copy 3.6 times a CTA's own -- in clusters of 8, with a cluster-wide
barrier every 8 KB, on tiles the cache held. A fill by contiguous copies of data several CTAs share is the shape a multicast serves: the carry lab's rows
that land a column's expanded lists ([`carry_window`](carry_window.md)), and the intra's reader tiles, which share
their writer tiles ([`intra_window`](intra_window.md)). Does it pay in a SMALL cluster, with a
slot-sized tile and a hand-off that stops no thread but the issuer's, and on tiles the cache has not seen?

THE FORM. Clusters of 2, 4 and 8 CTAs, 4 warps each; a chunk is a tile of 20 KB, two or four in flight: a copy's delay from its issue to its landing is hidden by the tiles in flight, a limit on bytes is not. `each`: every CTA
issues its own tensor copy of the cluster's tile. `cast`: a CTA's readers done, one of its threads states the
bytes at its own barrier and arrives at a barrier of rank 0's, a count of the cluster's CTAs; rank 0's issuing
thread waits on it and issues one multicast copy to all. No cluster barrier; no thread of a mate waits on another
CTA. Each is timed on a few tiles the cache holds and on a fresh tile a chunk a cluster, where a cluster's CTAs
each asking for the same tile ask global memory for it. The last tiles are checked exactly in every CTA.

MEASURED (H100 80GB HBM3, 132 SMs, vast.ai instance 60598408, driver 595.91.07; 2026-11-14, the experiment at
46954a3b; 7 repetitions; every landed byte of the last tiles exact; the median, cycles a kilobyte a CTA):

| a cluster's CTAs | tiles in flight | own copy, cached | multicast, cached | own copy, fresh | multicast, fresh |
|---|---|---|---|---|---|
| 2 | 2 | 16.9 | 67.8 | 47.6 | 86.5 |
| 2 | 4 | 13.7 | 67.9 | 34.5 | 86.7 |
| 4 | 2 | 19.3 | 67.6 | 36.8 | 82.8 |
| 4 | 4 | 15.7 | 67.8 | 23.8 | 82.9 |
| 8 | 2 | 17.5 | 66.7 | 33.4 | 80.5 |
| 8 | 4 | 14.7 | 66.9 | 19.6 | 80.8 |

WHAT IT SHOWS.
- A MULTICAST COPY LOSES IN EVERY FORM: 68 cycles a kilobyte a CTA on tiles the cache holds and 80 to 87 on fresh
  ones, the same at 2, 4 and 8 CTAs a cluster and at 2 and 4 tiles in flight, where a CTA's own copy is 14 to 19
  and 20 to 48. More tiles in flight do not move it, so it is not a delay a pipeline hides: a multicast copy
  delivers at its own rate, a quarter of an own copy's. Its hand-off here stops no thread but the issuer's.
- THE CACHE ALREADY SHARES A FRESH TILE: a CTA's own copy of a tile its cluster-mates also ask for is 34.5, 23.8
  and 19.6 cycles a kilobyte at 2, 4 and 8 CTAs asking, 4 in flight -- the more that ask, the less each pays. What a
  multicast is for, one read of global memory for many CTAs, the cache does for own copies.
- OWN COPIES WANT MORE IN FLIGHT: 16.9 to 13.7 cached and 33.4 to 19.6 fresh at 8 CTAs, from 2 tiles in flight to
  4. The 33 cycles a kilobyte the carry's fill and [`value_cast`](value_cast.md) read as a multiprocessor's rate
  is two tiles' depth, not a ceiling.

Record: `data:records/fsla/measure/lab/cast_cluster.run/20261114T051506Z-ebcfe583/output.json`.
