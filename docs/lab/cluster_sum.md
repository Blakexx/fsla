# `cluster_sum`

Mirrors `measure/lab/experiments/cluster_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A token's output is a sum of partials that different CTAs hold, and each partial added to global memory by itself is
the carry's largest cost in [`carry_floor`](../internals/tools/carry_floor.md). What does a token cost a CTA when a
cluster sums its partials in one CTA's shared memory first, by each of three forms, beside every CTA draining its
own? And can one collector take a cluster's partials, or must the collecting be spread?

THE FORM. Clusters of 8 CTAs, 4 warps each, as many clusters as the card's SMs hold. A round is a partial of 64
tokens × 64 channels a CTA, from registers.
- `direct`: every CTA adds its partial to its cluster's sum in global memory.
- `slots`: every CTA's warps store its partial into its own slot of the collector's shared memory; the collector
  adds a token's 8 slots in registers and adds the sum to global memory. The fp32 sum itself.
- `atomic`: every CTA's warps add fp32 into the collector's one panel; the collector reads it, clears it, drains it.
- `bulk slots`: the slot form with the partial staged in the CTA's own shared memory and sent by one bulk copy a
  collector, whose bytes complete the collector's barrier.
- `bulk integers`: the partial staged as 64-bit integers at a fixed scale and sent by one bulk reduction into the
  collector's one panel, converted back by the collector.
Each cluster form with ONE collector (rank 0 takes every token) and with EIGHT (a token's collector is its index
modulo 8). Barriers a round order sources and collectors, and the cluster's own barrier its start and its end. Every
sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a warp's store into a cluster-mate's shared memory is a
generic store (`ST.E`). A warp's float add there is `ATOM.E.ADD.F32` on one path with compare-and-swap loops on two
others, chosen by the address at run time. The bulk copy is one `UBLKCP.S.S`, the bulk integer reduction one
`UBLKRED.S.S.ADD.U64`.

THE FIRST RUN (H100 SXM, vast.ai instance 60237584, 2026-11-11, the experiment at 44bf7b0f; the driver REFUSED TO
RECORD it, so these are the session's printed rows and no record holds them): every CTA draining its own 23.5 cycles a
token a CTA; warp-stored slots 88.9 with one collector and 30.4 with eight; the warps' atomic fp32 adds 398 and 395;
and that run's fourth form, a warp's 64-bit integer add into the panel (`red.shared::cluster.add.u64`), 215 and 76.7
WITH WRONG SUMS -- 688 of 65,536 with eight collectors, 84 with one on a second launch. So: a warp's store into a
mate's shared memory costs about what its add to global memory does, and one collector is three times worse than
eight; the warps' float add into a mate's shared memory is seventeen times slower than a store; and the warps'
integer add there is not atomic on this card. That last form is removed from the experiment for the two bulk forms.

UNMEASURED: the experiment as it is now -- the bulk forms, and the other rows again in one record.
