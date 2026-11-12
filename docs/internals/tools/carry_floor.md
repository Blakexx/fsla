# `tools/carry_floor.py` — the hopper carry's floor, counted

No hopper carry exists. This model prices one before it does: for a declared routing shape, a box and a cluster it
counts what a token costs an SM, term by term, so a design is compared by one calculation and not by arithmetic redone
each time the question is asked. Its rates are of two kinds and every line of its report says which: MEASURED, read at
run time from an H100 lab's record under `measure/lab/results/` (the tool refuses a record or a row that is gone), and
ESTIMATED, a constant of the file with its reason. A total built on an estimate is an estimate.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \
        --box 16 16 --cluster 16 1 [--sum none | slots | s64] [--round 64] [--dv 64] [--tokens 65536] [--json floor.json]

WHY IT EXISTS. The hopper carry's design is a choice of box and cluster, and each choice moves several costs at
once: a box read is a partial to sum and a box written is a set of operands to load. The model makes that trade
one calculation. Its inputs are the declaration's structure alone (the levels'
widths and which side is sparse on which), so a result holds for any draw of that structure, as an expectation.

WHAT IT LEAVES OUT: the hand-over between producer and consumer warps, the den, the divergent carve's register swap,
paging, and every other kernel. Its rates are the H100's; no blackwell card is priced (none has been measured), and
a form that names an sm_90 instruction states no sm_100 form here -- owed with that family's first lab.

THE FOLD'S OPERANDS are the cheaper of two forms, and each row says which. LISTED: a (token, list, box) entry's
value and coefficients are fetched from scattered rows into the list's tiles by producer warps, MEASURED whole, the
fold's steps beside them ([`pool_fill`](../../lab/pool_fill.md): 5.4 to 6.9 cycles an entry by 8 to 2 producers, at
16 leaves a list; other widths are priced the same, UNMEASURED). TILED: every box takes the window's whole value
tile packed ([`state_move`](../../lab/state_move.md); every token's bytes), expands its coefficients into shared
memory and gathers the values into list order, that copy ESTIMATED at the shared access rate. Both are read on
working sets the cache holds.
THE FOLD'S TENSOR TIME is its steps': a token in a list of a box is a sixteenth of a step as wide as the
list's leaves in the box, a step's own time MEASURED at 8, 16, 32 and 64 columns and whole 64-column steps past
that ([`wgmma_width`](../../lab/wgmma_width.md)); the readout's leaves run whole steps of sixteen.

THE CLUSTER. A cluster shares no load -- a multicast copy costs each CTA what its own does
([`value_cast`](../../lab/value_cast.md)) -- so a cluster with no sum (`--sum none`, the default) changes nothing
and is refused. THE CLUSTER'S SUM takes
one of two forms, each summing a token's partials inside a
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

MODELLED (2026-11-12; the H100's records for the pipe, the shared access rate, the drain, the cluster's sum, the
fold's step time and its operands; the tiled form's gather, the clock and the memory bandwidth ESTIMATED; 4 live
digits a sparse level unless the row says, 65,536 tokens, no cluster unless the row says; SM-cycles a token, low to
high, and the call):

| the shape | boxes read / written | tensor (readout, fold) | the output's sum | the fold's operands | total | the call |
|---|---|---|---|---|---|---|
| 256 x 256, read sparse on level 2, write on level 1; boxes 16 x 16 | 58.2 / 58.2 | 32, 80 | 1,133 to 1,469 | 345 to 438 (listed) | 1,604 to 2,033 | 442 to 561 us |
| ... boxes 64 x 4 | 15.6 / 175.0 | 32, 288 | 304 to 394 | 1,379 to 1,753 (listed) | 2,017 to 2,480 | 556 to 684 us |
| ... boxes 256 x 1 (the whole read-dense level) | 4.0 / 256.0 | 32, 1,152 | 77 to 100 | 2,821 (tiled) | 4,096 to 4,119 | 1.1 ms |
| ... boxes 16 x 16, `--cluster 16 1 --sum slots` | 58.2 / 58.2 | 32, 80 | 7,562 to 7,583 | 345 to 438 (listed) | 8,033 to 8,147 | 2.2 ms |
| 256 x 256, BOTH sides sparse on level 2; boxes 256 x 1 | 4.0 / 4.0 | 32, 32 | 77 to 100 | 22 to 27 (listed) | 177 to 205 | 49 to 57 us |
| 256 x 256, read sparse on level 2, write DENSE; boxes 256 x 1 | 4.0 / 256.0 | 32, 2,048 | 77 to 100 | 1,379 to 1,753 (listed) | 3,550 to 3,947 | 1.0 to 1.1 ms |
| 64 x 64 (4,096 leaves), sparse on opposite levels; boxes 16 x 16 | 10.9 / 10.9 | 8, 20 | 213 to 276 | 86 to 110 (listed) | 330 to 417 | 91 to 115 us |
| 16^4, two sparse levels a side (2 live digits), boxes 4^4 | 49.0 / 49.0 | 128, 288 | 954 to 1,236 | 1,379 to 1,753 (listed) | 2,763 to 3,418 | 762 to 943 us |
| 512 x 512 (262,144 leaves), sparse on opposite levels; boxes 16 x 16 | 122.1 / 122.1 | 64, 160 | 2,376 to 3,080 | 690 to 876 (listed) | 3,318 to 4,208 | 0.9 to 1.2 ms |
| 256 x 256 dense on both sides; boxes 16 x 16 | 256.0 / 256.0 | 2,049, 5,120 | 4,982 to 6,457 | 8,955 (tiled) | 21,983 to 23,459 | 6.1 to 6.5 ms |

FlashAttention's tensor floor at the same 65,536 tokens and width is 2,049 SM-cycles a token, 565 us the call; at
1,048,576 tokens it is 16 times that a token, and a 256 x 256 row's counts a token do not move (its rates were read
on working sets the cache holds; a million tokens' values are not such a set, and that is UNMEASURED).

WHAT IT SHOWS. A token's cost is bytes moved and steps run, a (token, box) each; the products themselves are little.
- WHERE THE TWO SIDES ARE SPARSE ON DIFFERENT LEVELS, at as many leaves as tokens, THIS MODEL'S FORM DOES NOT BEAT
  FLASHATTENTION'S FLOOR: a box square on both sides is 1,604 to 2,033 against 2,049, which is parity, and parity
  is a loss. Seven tenths of it is the output's sum -- 58 fp32 partials a token, each drained by its own box --
  and a fifth the fold's operands, 64 entries a token. The form this model prices is not the design's last word
  on the sum: one f16 line a token for two boxes ([`drain_role`](../../lab/drain_role.md),
  [`readout_role`](../../lab/readout_role.md)) is not a term here yet. A box the whole of the read side's dense
  level pays 4 partials, and then every token writes every box: 1,024 (token, list, box) steps' sixteenths a
  token, each of an 8-column step ([`wgmma_width`](../../lab/wgmma_width.md)), beside its operands.
- WHERE BOTH SIDES ARE SPARSE ON THE SAME LEVEL the read-aligned box is the write-aligned one too, and every term is
  small: 177 to 205.
- A DENSE WRITE SIDE is the fold's whole dense product: its tensor time alone is about the floor.
- A CLUSTER BUYS NOTHING: its sum does not pay (`cluster_sum`) and it shares no load (`value_cast`).
- The model's unit is the box; a CTA's two boxes sharing a drain line are not counted. Dense on both sides at as many leaves as tokens is over the floor by the tensor work
  alone.