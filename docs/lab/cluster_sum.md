# `cluster_sum`

Mirrors `measure/lab/experiments/cluster_sum.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

A token's output is a sum of partials that different CTAs hold, and each partial added to global memory by itself is
the carry's largest cost in [`carry_floor`](../internals/tools/carry_floor.md). What does a token cost a CTA when a
cluster sums its partials in one CTA's shared memory first, by each of five forms, beside every CTA draining its
own -- and how much of that is the round's fixed cost? And can one collector take a cluster's partials, or must the
collecting be spread?

THE FORM. Clusters of 8 CTAs, 4 warps each, as many clusters as the card's SMs hold. A round is a partial of 16 or of
64 tokens × 64 channels a CTA, from registers: a form's cost a token at two round sizes separates what it pays a
round from what it pays a token.
- `direct`: every CTA adds its partial to its cluster's sum in global memory.
- `slots`: every CTA's warps store its partial into its own slot of the collector's shared memory; the collector
  adds a token's 8 slots in registers and adds the sum to global memory. The fp32 sum itself.
- `atomic`: every CTA's warps add fp32 into the collector's one panel; the collector reads it, clears it, drains it.
- `bulk slots`: the slot form with the partial staged in the CTA's own shared memory and sent by one bulk copy a
  collector, whose bytes complete the collector's barrier.
- `bulk integers`: the partial staged as 64-bit integers at a fixed scale and sent by one bulk reduction into the
  collector's one panel, converted back by the collector.
- `warp integers`: the warps' own 64-bit integer adds into that panel.
- `hand-over`: the barriers of a round of the warps' forms (an arrival a CTA at the collector, an arrival a collector
  at every CTA) with no partial sent or summed. A bulk form's round completes its collector's barrier by bytes, with
  a fence for the bulk engine and a re-arm, and no row isolates that.
Each cluster form with ONE collector (rank 0 takes every token) and with EIGHT (a token's collector is its index
modulo 8). Barriers a round order sources and collectors -- every thread that stored or added into a mate fences at
cluster scope, an arrival releases to the cluster, a wait acquires from it -- and the cluster's own barrier its
start and its end. A mate's addresses are mapped once, before the clock. Every row's timed round includes its
collectors' drain to global memory. Every sum is checked exactly.

WHAT THE SASS SAYS (sm_90a, the pinned toolkit; no device): a warp's store into a cluster-mate's shared memory is a
generic store (`ST.E`). A warp's float add there is `ATOM.E.ADD.F32` on one path with compare-and-swap loops on two
others, chosen by the address at run time; its 64-bit integer add is `ATOM.E.ADD.64`. The bulk copy is one
`UBLKCP.S.S`, the bulk integer reduction one `UBLKRED.S.S.ADD.U64`.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60238483, driver 580.173.02; 2026-11-11, the experiment at b870d1e2 --
rounds of 64 tokens only, a mate's address mapped at each use, an arrival's release the one cluster-scope ordering;
7 repetitions, every sum exact; cycles a token a CTA, the median, the spread under 0.5%):

| the form | one collector | eight collectors |
|---|---|---|
| every CTA drains its own | 26.0 | |
| fp32 slots stored by the warps | 152.1 | 127.6 |
| fp32 atomic adds into one panel | 442 | 471 |
| fp32 slots by a bulk copy | 151.7 | 123.6 |
| 64-bit integers by a bulk reduction | 252.2 | 135.0 |
| 64-bit integer adds by the warps | 275.4 | 165.1 |
| the round's barriers alone, no partial | 102.0 | 106.9 |

WHAT IT SHOWS. NO CLUSTER FORM IS CHEAPER THAN EVERY CTA DRAINING ITS OWN at this round. The barriers of one round
cost a CTA about 6,800 cycles -- 107 a token at 64 tokens -- which is four times what the partial costs straight to
global memory. Less that row, a warp-stored slot costs 21 cycles a token a CTA with eight collectors against the
direct 26, its collectors' drain included; one collector is 50. Less it too, the warps' fp32 add into a mate is seventeen
times its store. The warps' 64-bit integer add sums exactly here, under barriers that release to and acquire from
the cluster.

Two earlier runs of this experiment, at barriers without the cluster's scope, read 31 (warp slots, eight collectors),
40 (bulk slots) and 78 (bulk integers) with every sum exact, and wrong sums from the warps' integer add (688 of
65,536); the first of them has no record, and what made those sums wrong was not isolated.

NOT YET READ: the same forms at 16 tokens a round (the round's cost against the token's), with the addresses mapped
once and every writer fencing -- the experiment as it stands now.

Record: `measure/lab/results/cluster_sum/sm_90a.json` (the experiment at b870d1e2).
