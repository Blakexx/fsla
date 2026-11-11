# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum slots | s64] [--round 64] [--dv 64] [--tokens 65536] [--json floor.json]

WHY IT EXISTS. The hopper carry's design is a choice of box and cluster, and each choice moves several costs at
once: a box read is a partial to sum, a box written is a set of operands to load, and a cluster shares one or the
other. The model makes that trade one calculation. Its inputs are the declaration's structure alone (the levels'
widths and which side is sparse on which), so a result holds for any draw of that structure, as an expectation.

WHAT IT LEAVES OUT: the hand-over between producer and consumer warps, the den, the divergent carve's register swap,
paging, and every other kernel. Its rates are the H100's; no blackwell card is priced (none has been measured), and
a form that names an sm_90 instruction states no sm_100 form here -- owed with that family's first lab.

THE CLUSTER'S SUM takes one of two forms, each summing a token's partials inside a cluster before one fp32 drain to
global memory. `slots`, the default, is the fp32 sum the format states
([precision.md](../../architecture/precision.md)):
each box stages its partial and sends it by a bulk copy into its own slot of a collector's shared memory, and the
collector adds a token's slots in registers. `s64` sends 64-bit integers by a bulk reduction, at a scale agreed a
window -- outside the fp32 sum the format states, counted for comparison ([`arith_model`](arith_model.md#sums)
measures its accuracy). Both are MEASURED by [`cluster_sum`](../../lab/cluster_sum.md) at two round sizes, and the
model reads that record's line: a cost a round a CTA whatever the round holds (`--round` partials a CTA hand over
between barriers; beyond the lab's 64 the line is UNMEASURED), and a cost a partial. The lab's partial includes its
collectors' drain, an eighth of one, which the model's drains count again: a clustered row is over by that much. The
warps' own fp32 add into a mate's shared memory, read at eighteen times a store, is no form here.

MODELLED (2026-11-11; the H100's records for the pipe, the shared access rate, the drain and the cluster's sum; a load
request, the issue rate, the clock and the memory bandwidth ESTIMATED; `--read-sparse 0 1 --write-sparse 1 0 --k 4
--box 16 16` unless the row says, the 16^4 row `--read-sparse 0 0 1 1 --write-sparse 1 1 0 0 --k 2 --box 4 4 4 4`;
SM-cycles a token, low to high, and the 64K call):

| the shape | boxes read | the sum (partials, rounds, drains) | fold values | total | the call |
|---|---|---|---|---|---|
| 256 x 256, no cluster | 58.2 | 0, 0, 1,176 to 2,726 | 320 to 640 | 1,590 to 3,460 | 440 to 950 us |
| ... `--cluster 16 1`, rounds of 64 | 58.2 | 916, 5,710, 74 to 170 | 88 to 176 | 6,880 to 7,060 | 1.9 ms |
| ... `--cluster 16 1 --round 512` | 58.2 | 916, 714, 74 to 170 | 88 to 176 | 1,880 to 2,070 | 520 to 570 us |
| 64 x 64 (4,096 leaves), no cluster | 10.9 | 0, 0, 221 to 512 | 80 to 160 | 320 to 690 | 90 to 190 us |
| 16^4 (two sparse levels a side, 2 live digits, boxes 4^4), no cluster | 49.0 | 0, 0, 989 to 2,293 | 1,280 to 2,560 | 2,360 to 4,940 | 650 to 1,360 us |
| 256 x 256 dense on both sides, no cluster | 256 | 0, 0, 5,169 to 11,981 | 20,480 to 40,960 | 31,500 to 58,800 | 8.7 to 16 ms |

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and the 256 x 256 alt row's cost a token does not move.

WHAT IT SHOWS. The tensor work is 64 of the sparse rows' cycles; the rest is bytes, counted a (token, box): the
output's partials and the fold's values. A CLUSTER DOES NOT PAY: a partial costs as much into a mate's shared memory
as to global memory, and the round's barriers come on top -- at the lab's round the clustered row is four times the
unclustered one, and at a round of 512 partials (unmeasured, and more shared memory than the slots have) it is no
better than the unclustered row's low end. So the output's sum is priced by the count of partials a token, 58 at
these boxes, and at 65,536 tokens the unclustered row stands around FlashAttention's floor, not under it. What
lowers it is fewer partials a token -- a CTA that sums a token's boxes in its own registers before it drains -- and
that is no term of this model yet. Dense on both sides at as many leaves as tokens is over FlashAttention's floor by
the tensor work alone (4,096 against 2,049).
