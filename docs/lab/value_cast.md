# `value_cast`

Mirrors `measure/lab/experiments/value_cast.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A cluster shares no sum's cost ([`cluster_sum`](cluster_sum.md)) and no state's ([`state_move`](state_move.md)).
Does it share a load's? What does a kilobyte of a tile cost a CTA when ONE multicast tensor copy lands it in all
eight CTAs of a cluster, beside every CTA issuing its own copy of it?

THE FORM. Clusters of 8 CTAs, 4 warps each. A chunk is a tile of 64 tokens x 128 bytes (8 KB) of a 8 MB table every
cluster reads; two tiles are in flight, a slot each by the chunk's parity. A chunk: every thread waits on the tile
two chunks back and reads a word of it; one thread states the chunk's bytes at its CTA's barrier; then the slot's
readers' edge, and the copy into the slot.
- `each`: every CTA issues its own tensor copy of the chunk's tile, past its own CTA's barrier.
- `cast`: the CTA of rank 0 issues one multicast copy to all eight, past the CLUSTER's barrier: a copy into a
  mate's slot is ordered after that mate's readers and its stated bytes by nothing less.
The last tiles are checked exactly in every CTA.

MEASURED: not yet run in this form. (An earlier form issued a copy into a slot with no edge from the slot's
readers; its numbers are withdrawn.)
