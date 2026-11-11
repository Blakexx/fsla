# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum none | slots | s64] [--round 64] [--dv 64] [--tokens 65536] [--json floor.json]

WHY IT EXISTS. The hopper carry's design is a choice of box and cluster, and each choice moves several costs at
once: a box read is a partial to sum, a box written is a set of operands to load, and a cluster shares one or the
other. The model makes that trade one calculation. Its inputs are the declaration's structure alone (the levels'
widths and which side is sparse on which), so a result holds for any draw of that structure, as an expectation.

WHAT IT LEAVES OUT: the hand-over between producer and consumer warps, the den, the divergent carve's register swap,
paging, and every other kernel. Its rates are the H100's; no blackwell card is priced (none has been measured), and
a form that names an sm_90 instruction states no sm_100 form here -- owed with that family's first lab.

THE FOLD'S VALUES. A token's value reaches each box it writes once, at the cheaper of its share of the window's
whole value tile loaded packed and a 128-byte load of its own ([`state_move`](../../lab/state_move.md), read on
working sets the cache holds); then it is gathered into list order in shared memory, ESTIMATED at the shared access
rate. A list's leaves in a box run whole wgmma steps, the fold's in eights and the readout's in sixteens.

THE CLUSTER. By default (`--sum none`) a cluster shares the fold's value loads and nothing else: every box drains
its own partial; that sharing is the low end of a row's loads and is ESTIMATED (no lab has landed a load across a
cluster). THE CLUSTER'S SUM, asked for, takes one of two forms, each summing a token's partials inside a
cluster before one fp32 drain to global memory. `slots` is the fp32 sum the format states
([precision.md](../../architecture/precision.md)):
each box stages its partial and sends it by a bulk copy into its own slot of a collector's shared memory, and the
collector adds a token's slots in registers. `s64` sends 64-bit integers by a bulk reduction, at a scale agreed a
window -- outside the fp32 sum the format states, counted for comparison ([`arith_model`](arith_model.md#sums)
measures its accuracy). Both are MEASURED by [`cluster_sum`](../../lab/cluster_sum.md) at two round sizes, and the
model reads that record's line: a cost a round a CTA whatever the round holds (`--round` partials a CTA hand over
between barriers; beyond the lab's 64 the line is UNMEASURED), and a cost a partial. The lab's partial includes its
collectors' drain, an eighth of one, which the model's drains count again: a row with a cluster's sum is over by
that much. The warps' own fp32 add into a mate's shared memory, read at eighteen times a store, is no form here.

MODELLED (2026-11-11; the H100's records for the pipe, the shared access rate, the drain, the cluster's sum and the
value load; the list gather, the clock and the memory bandwidth ESTIMATED; 4 live digits a sparse level unless the
row says, 65,536 tokens, no cluster unless the row says; SM-cycles a token, low to high, and the call):

| the shape | boxes read / written | tensor (readout, fold) | the output's sum | fold values (loaded, gathered) | total | the call |
|---|---|---|---|---|---|---|
| 256 x 256, read sparse on level 2, write on level 1; boxes 16 x 16 | 58.2 / 58.2 | 32, 32 | 1,176 to 2,726 | 1,051, 110 | 2,428 to 3,978 | 0.7 to 1.1 ms |
| ... boxes 64 x 4 | 15.6 / 175.0 | 32, 64 | 316 to 731 | 1,051, 439 | 1,929 to 2,345 | 532 to 647 us |
| ... boxes 256 x 1 (the whole read-dense level) | 4.0 / 256.0 | 32, 256 | 80 to 186 | 1,051, 1,756 | 3,203 to 3,309 | 883 to 913 us |
| ... boxes 256 x 1, clusters of 8 sharing the value tile | 4.0 / 256.0 | 32, 256 | 80 to 186 | 131 to 1,051, 1,756 | 2,284 to 3,309 | 630 to 913 us |
| ... boxes 16 x 16, `--cluster 16 1 --sum slots` | 58.2 / 58.2 | 32, 32 | 7,565 to 7,662 | 66 to 1,051, 110 | 7,832 to 8,913 | 2.2 to 2.5 ms |
| 256 x 256, BOTH sides sparse on level 2; boxes 256 x 1 | 4.0 / 4.0 | 32, 32 | 80 to 186 | 130, 7 | 309 to 415 | 85 to 114 us |
| 256 x 256, read sparse on level 2, write DENSE; boxes 256 x 1 | 4.0 / 256.0 | 32, 2,049 | 80 to 186 | 1,051, 439 | 4,543 to 4,648 | 1.3 ms |
| 64 x 64 (4,096 leaves), sparse on opposite levels; boxes 16 x 16 | 10.9 / 10.9 | 8, 8 | 221 to 512 | 66, 27 | 337 to 628 | 93 to 173 us |
| ... boxes 64 x 4 | 3.6 / 16.0 | 8, 16 | 73 to 170 | 66, 110 | 280 to 377 | 77 to 104 us |
| 16^4, two sparse levels a side (2 live digits), boxes 4^4 | 49.0 / 49.0 | 128, 64 | 989 to 2,293 | 1,051, 439 | 2,699 to 4,003 | 0.7 to 1.1 ms |
| ... boxes 16 x 16 x 1 x 1 (the read-dense levels whole) | 3.8 / 256.0 | 32, 256 | 76 to 176 | 1,051, 1,756 | 3,198 to 3,298 | 882 to 910 us |
| 512 x 512 (262,144 leaves), sparse on opposite levels; boxes 256 x 1 | 8.0 / 960.0 | 64, 512 | 161 to 373 | 4,203, 3,513 | 8,508 to 8,720 | 2.3 to 2.4 ms |
| 256 x 256 dense on both sides; boxes 16 x 16 | 256.0 / 256.0 | 2,049, 2,049 | 5,169 to 11,981 | 1,051, 7,026 | 19,099 to 25,911 | 5.3 to 7.1 ms |

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and a 256 x 256 row's counts a token do not move (its rates were read
on working sets the cache holds; a million tokens' values are not such a set, and that is UNMEASURED).

WHAT IT SHOWS. The tensor work is the least of it; the rest is bytes.
- WHERE THE TWO SIDES ARE SPARSE ON DIFFERENT LEVELS, NO BOX SHAPE BRINGS A TOKEN CLEARLY UNDER FLASHATTENTION'S
  FLOOR at as many leaves as tokens. A box square on both sides pays the output's sum (58 partials a token). A box
  the whole of the read side's dense level pays 4 partials, and then every token writes every box: each loads the
  window's whole value tile, gathers it into list order, and runs the fold's one leaf a list in a step of eight.
  The shapes between trade one for the other and stay near 2,000.
- WHERE BOTH SIDES ARE SPARSE ON THE SAME LEVEL the read-aligned box is the write-aligned one too, and every term is
  small: 310 to 415.
- A DENSE WRITE SIDE is the fold's whole dense product, over the floor by itself.
- A CLUSTER'S SUM does not pay (`cluster_sum`); a cluster sharing the value tile is the one use left for it, and its
  low end is ESTIMATED.
- The model's unit is the box. A CTA's boxes that share one value tile and one gather halve those two terms; that
  pairing is not counted. Dense on both sides at as many leaves as tokens is over the floor by the tensor work
  alone.