# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum slots | atomic | s64] [--dv 64] [--tokens 65536] [--json floor.json]

WHY IT EXISTS. The hopper carry's design is a choice of box and cluster, and each choice moves several costs at
once: a box read is a partial to sum, a box written is a set of operands to load, and a cluster shares one or the
other. The model makes that trade one calculation. Its inputs are the declaration's structure alone (the levels'
widths and which side is sparse on which), so a result holds for any draw of that structure, as an expectation.

WHAT IT LEAVES OUT: the hand-over between producer and consumer warps, the den, the divergent carve's register swap,
paging, and every other kernel. Its rates are the H100's; no blackwell card is priced (none has been measured), and
a form that names an sm_90 instruction states no sm_100 form here -- owed with that family's first lab.

THE CLUSTER'S SUM takes one of three forms, each summing a token's partials inside a cluster before one fp32 drain to
global memory. `slots`, the default, is the fp32 sum the format states ([precision.md](../../architecture/precision.md)):
each box stores its partial into its own slot of the collector's shared memory and the collector adds a token's slots
in registers. `atomic` has each box's warps add fp32 straight into the collector's panel. `s64` adds 64-bit integers
at a scale agreed a window -- outside the fp32 sum the format states, counted for comparison
([`arith_model`](arith_model.md#sums) measures its accuracy).

READ FROM THE COMPILER (ptxas of toolkit 13.0, sm_90a; compiled once, no device, no lab): a warp's
`red.shared::cluster.add.f32` into another CTA's shared memory compiles to the generic atomic add
(`ATOM.E.ADD.F32`), predicated, with compare-and-swap loops as its fallback paths; into the CTA's own shared memory
(`red.shared::cta`) it is a compare-and-swap loop alone; the integer add is one `ATOM.E.ADD.S32`. The asynchronous
reductions into a cluster's shared memory the toolkit's wrappers list are integer. [`cluster_sum`](../../lab/cluster_sum.md) has since MEASURED these on the H100, and it REFUTES this model's cluster
terms: a partial into a mate's shared memory costs about what it costs straight to global memory (21 against 26
cycles a token a CTA), the float add seventeen times a store, and a round's barriers 6,800 cycles a CTA. The
three forms' terms here still differ only in bytes and instructions counted at one shared rate; every clustered row
below is therefore TOO LOW in its sum, and stands until the model reads that record.

MODELLED (2026-11-11; the H100's records for the pipe, the shared access rate and the drain; a load request, the
issue rate, the clock and the memory bandwidth ESTIMATED; alt = read side sparse on one level, write side on the
other, 4 live digits; boxes 16 x 16; `--sum slots`; SM-cycles a token, low to high, and the 64K call):

| the shape | boxes read | clusters read | the sum (partials, drains) | fold values | total | the call |
|---|---|---|---|---|---|---|
| 256 x 256 alt, no cluster | 58.2 | 58.2 | 0, 1,176 to 2,726 | 320 to 640 | 1,590 to 3,460 | 440 to 950 us |
| ... a cluster the 16 boxes one read range's tokens read | 58.2 | 3.64 | 200, 74 to 170 | 88 to 176 | 450 to 670 | 125 to 185 us |
| ... a cluster 8 x 2 | 58.2 | 6.62 | 200, 134 to 310 | 82 to 165 | 510 to 800 | 140 to 220 us |
| ... a cluster 4 x 4 | 58.2 | 10.9 | 200, 221 to 512 | 60 to 120 | 570 to 950 | 160 to 260 us |
| 64 x 64 alt (4,096 leaves), clusters of 4 | 10.9 | 2.73 | 38, 55 to 128 | 29 to 59 | 145 to 250 | 40 to 70 us |
| 16^4 alt (two sparse levels a side), the same clusters | 58.2 | 3.64 | 200, 74 to 170 | 352 to 703 | 990 to 1,470 | 270 to 405 us |
| 256 x 256 dense on both sides, clusters 4 x 4 | 256 | 16 | 878, 323 to 749 | 1,280 to 2,560 | 8,300 to 10,200 | 2.3 to 2.8 ms |

The second row under the other forms: `atomic` 350 to 570 (its partials 100), `s64` 450 to 730.

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and the 256 x 256 alt row's cost a token does not move.

WHAT IT SHOWS. The tensor work is 64 of the alt rows' cycles; the rest is bytes, counted a (token, box): the partials
into the cluster, the drains, the values. On these rows a cluster is worth a factor of three to five -- a factor
`cluster_sum` does not bear out -- and its shape trades the
readout's drains against the fold's value loads -- which side to favour rests on the ESTIMATED load request, the
least known rate here. A deeper routing at the same box reads the same boxes with four times the leaves, and its
fold sits in four times the lists a box. Dense on both sides at as many leaves as tokens is over FlashAttention's
floor by the tensor work alone (4,096 against 2,049).
