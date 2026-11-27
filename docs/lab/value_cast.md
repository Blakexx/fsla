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

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60316069, driver 615.71.09; 2026-11-12, the experiment at
864c6e3f; 7 repetitions, every landed tile exact in every CTA; SM-cycles a kilobyte a CTA, the median, the
spread under 0.7%):

| the form | cycles a kilobyte a CTA |
|---|---|
| every CTA its own copy | 38.3 |
| one multicast copy a cluster | 137.4 |

WHAT IT SHOWS. A MULTICAST COPY COSTS A RECEIVING CTA 3.6 TIMES ITS OWN COPY: 137.4 cycles a kilobyte against
38.3. The copy itself lands in every CTA it names, as a CTA's own does; what the multicast adds is its issue edge,
the cluster's barrier a chunk, where a CTA's own copy waits on its own. A cluster shares no load's cost, as it
shares no sum's ([`cluster_sum`](cluster_sum.md)) and no state's ([`state_move`](state_move.md)). A larger chunk
would spread the cluster's barrier over more bytes and bring the multicast towards the own copy, never under it by
this form; chunks past 8 KB are UNMEASURED.

Record: `data:records/fsla/measure/lab/value_cast.run/20261112T060215Z-7bb6c049/output.json`.
