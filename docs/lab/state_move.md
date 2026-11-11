# `state_move`

Mirrors `measure/lab/experiments/state_move.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

[`cluster_sum`](cluster_sum.md) showed that a token's partial costs as much into a cluster-mate's shared memory as
straight to global memory, so a carry whose CTAs exchange something must exchange less of it. A CTA's STATE is
that: it changes once a window, not once a token. What does a kilobyte of state cost a CTA to move -- into its
cluster-mates' shared memory by bulk copies, or out to global memory and back in by one bulk copy each way? And
what does a token's 128-byte value cost, loaded by itself, the rate
[`carry_floor`](../internals/tools/carry_floor.md) only estimates?

THE FORM. Clusters of 8 CTAs, 4 warps each, as many clusters as the card's SMs hold. A round moves 8 blocks a CTA,
of 1 KB or of 8 KB each (8 KB or 64 KB a round).
- `stage`: the round's words stored into the CTA's own shared memory -- the base every moving form carries.
- `mates`: the stage, then a bulk copy of one block to each CTA of the cluster (itself among them), whose bytes
  complete the receiver's barrier; the receiver reads its image, fences at cluster scope and releases its mates into
  the next round.
- `out`: the stage, then one bulk copy of it to the CTA's own span of global memory, waited.
- `out and in`: that, then one bulk copy of the span back into the image.
- `in`: one bulk copy from global memory into the image, with no stage.
- `rows`: 64 loads of 128 bytes a round, from scattered rows of an 8 MB table, issued by one thread or by 64.
The last round's image is checked exactly, word for word.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a bulk copy into a mate is one `UBLKCP.S.S`, out to
global memory one `UBLKCP.G.S`, in from it one `UBLKCP.S.G`.

UNMEASURED: no card has run it.
