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
- `slots`: every CTA stores its partial into its own slot of the collector's shared memory; the collector adds a
  token's 8 slots in registers and adds the sum to global memory. The fp32 sum itself.
- `atomic`: every CTA's warps add fp32 into the collector's one panel; the collector reads it, clears it, drains it.
- `s64`: the same with 64-bit integers at a fixed scale, converted back by the collector.
Each of the last three with ONE collector (rank 0 takes every token) and with EIGHT (a token's collector is its index
modulo 8). A barrier a round in each direction orders sources and collectors, and the cluster's own barrier its start
and its end. Every sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a store into a cluster-mate's shared memory is a generic
store (`ST.E`). A float add there is `ATOM.E.ADD.F32` on one path with compare-and-swap loops on two others, chosen
by the address at run time; the 64-bit integer add is `ATOM.E.ADD.64` with a compare-and-swap loop beside it, after
an `F2I.S64` a value.

UNMEASURED: no hopper card has run it (`bazel run //tools:lab -- pack cluster_sum --arch sm_90a`, then `carried` on
the card, under its kill timeout).
