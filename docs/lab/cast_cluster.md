# `cast_cluster`

Mirrors `measure/lab/experiments/cast_cluster.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`value_cast`](value_cast.md) found a multicast copy 3.6 times a CTA's own -- in clusters of 8, with a cluster-wide
barrier every 8 KB, on tiles the cache held. The carry's fill is now contiguous copies of data that the boxes of a
sparse digit's column share ([`carry_window`](carry_window.md)), and the intra's reader tiles share their writer
tiles ([`intra_window`](intra_window.md)): the shape a multicast serves. Does it pay in a SMALL cluster, with a
slot-sized tile and a hand-off that stops no thread but the issuer's, and on tiles the cache has not seen?

THE FORM. Clusters of 2, 4 and 8 CTAs, 4 warps each; a chunk is a tile of 20 KB, two or four in flight: a copy's delay from its issue to its landing is hidden by the tiles in flight, a limit on bytes is not. `each`: every CTA
issues its own tensor copy of the cluster's tile. `cast`: a CTA's readers done, one of its threads states the
bytes at its own barrier and arrives at a barrier of rank 0's, a count of the cluster's CTAs; rank 0's issuing
thread waits on it and issues one multicast copy to all. No cluster barrier; no thread of a mate waits on another
CTA. Each is timed on a few tiles the cache holds and on a fresh tile a chunk a cluster, where a cluster's CTAs
each asking for the same tile ask global memory for it. The last tiles are checked exactly in every CTA.

MEASURED: not yet run.
