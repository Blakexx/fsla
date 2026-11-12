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

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60311802, driver 595.91.07; 2026-11-12, the experiment at
b05e4bcc; 7 repetitions, every landed tile exact in every CTA; SM-cycles a kilobyte a CTA, the median, the
spread under 0.1%):

| the form | cycles a kilobyte a CTA |
|---|---|
| every CTA its own copy | 82.5 |
| one multicast copy a cluster | 83.3 |

WHAT IT SHOWS. A MULTICAST COPY SAVES A RECEIVING CTA NOTHING: 83.3 cycles a kilobyte against 82.5 for its own
copy. What a copy costs a CTA is its landing, and a multicast copy lands in every CTA it names. A cluster shares no
load's cost, as it shares no sum's ([`cluster_sum`](cluster_sum.md)) and no state's ([`state_move`](state_move.md)).
Both forms here, an 8 KB tensor copy with every thread waiting on its barrier, read well over
[`state_move`](state_move.md)'s plain bulk copy in (33 a kilobyte at 64 KB); what of that is the tensor copy and
what the waits is not separated.

Record: `measure/lab/results/value_cast/sm_90a.json`.
