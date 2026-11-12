# `value_cast`

Mirrors `measure/lab/experiments/value_cast.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A cluster's one use left to the carry is sharing what its CTAs all load
([`cluster_sum`](cluster_sum.md), [`state_move`](state_move.md)). What does a kilobyte of a tile cost a CTA when ONE
multicast tensor copy lands it in all eight CTAs of a cluster, beside every CTA issuing its own copy of it?

THE FORM. Clusters of 8 CTAs, 4 warps each. A chunk is a tile of 64 tokens x 128 bytes (8 KB) of a 8 MB table every
cluster reads; two tiles are in flight, a slot each by the chunk's parity.
- `each`: every CTA issues its own tensor copy of the chunk's tile.
- `cast`: the CTA of rank 0 issues one multicast copy to all eight; every CTA states the bytes at its own barrier
  and waits on it, with no word between the CTAs.
The tensor map is the driver's (`cuTensorMapEncodeTiled`, reached in the running driver's library). The last tiles
are checked exactly in every CTA.

UNMEASURED: no card has run it.
