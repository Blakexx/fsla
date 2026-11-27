# `tools/arith_model.py` — the bf16 family's arithmetic, modelled

The op's accuracy class is a format decision ([precision.md](../../architecture/precision.md#quality) records it);
this model prices the family before any kernel of it exists: rows u8 at every level, values and MMA operands bf16,
every sum fp32. Its truth is the fp64 oracle's -- summed a window at a time
so a 64K cell fits, and checked against `fsla.ops.naive.naive_fsla` over the first two windows on every run -- and it
rounds only where such a kernel would (the tool's docstring lists the four places), each alone and all together.

    bazel run //tools:arith_model -- carry-alt-k4-256x256-T1024 carry-dense-256x256-T65536 [--window 1024] [--json model.json]

MEASURED (2026-11-09, one head, the median / the 99th percentile a token):

| cell (window) | all | fold alone | state as operand alone | read coefficient alone | intra alone |
|---|---|---|---|---|---|
| carry-alt-k4-256x256-T1024 (512) | 0.134% / 0.202% | 0.015% / 0.066% | 0.015% / 0.066% | 0.015% / 0.062% | 0.114% / 0.202% |
| carry-dense-256x256-T1024 (512) | 0.132% / 0.191% | 0.001% / 0.003% | 0.000% / 0.002% | 0.001% / 0.004% | 0.132% / 0.191% |
| carry-alt-k4-256x256-T65536 (1,024) | 0.115% / 0.174% | 0.067% / 0.098% | 0.066% / 0.097% | 0.057% / 0.081% | 0.021% / 0.162% |
| carry-dense-256x256-T65536 (1,024) | 0.021% / 0.167% | 0.003% / 0.003% | 0.001% / 0.015% | 0.002% / 0.003% | 0.020% / 0.167% |

WHAT IT SHOWS. A rounding of a (token, leaf) coefficient or of a state value averages over the leaves a token reads,
so the fold, the state and the read coefficient cost little where a token reads many leaves (dense) and more where
it reads few (alt-k4). The intra rounds once a token pair, and the cells' values are random-signed, so a token's
output is a small remainder of its terms (about 7% of their size at carry-dense-256x256-T1024) while the roundings do not
cancel: the intra carries the error on a two-window cell and the worst tokens everywhere (a window's early tokens,
whose output is a few intra terms). With no rounding on the model is the truth exactly.

## <a id="sums"></a>The cross-box sum

WHY IT IS MODELLED. The readout's output is a sum over boxes that different CTAs hold, and
[precision.md](../../architecture/precision.md) states that sum's width as the plan's, fp32 or f16. On sm_90 the ASYNCHRONOUS reductions into another
CTA's shared memory are integer (the toolkit's PTX wrappers list them; its float bulk add goes to global memory
only), so an integer partial is the one a cluster can sum with no warp's work. The fp32 way to sum in a cluster, a slot a box
added by one CTA, is [`carry_floor`](carry_floor.md)'s, counted beside this one; the warps' own atomic float add
there is measured by [`cluster_sum`](../../lab/cluster_sum.md) at eighteen times a store and is no form of either. The model
asks what an integer sum costs in accuracy, under each way of choosing its scale, before any kernel takes one: each
box's fp32 partial summed in fp32, as 32- or 64-bit integers at a scale from a bound, or normalized by its cluster's
den (the tool's docstring states each form). A box is `--box FIRST SECOND` digits a level; a cluster is the boxes
sharing one range of level `--along`. Four stresses move what a scale must hold: `--scale-spread` (write scales rising
over the call), `--value-spread` (value scales rising), `--leaf-spread` (write scales apart across one cluster's
leaves) and `--spike` (the call's first 16 tokens far over the rest).

    bazel run //tools:arith_model -- carry-alt-k4-256x256-T65536 --along 1 [--scale-spread 40 | --value-spread 16 | --leaf-spread 24]

MEASURED (2026-11-11, one head, boxes 16 x 16, clusters along the read-sparse level; the sum's own error, the worst
token; the fp32 sum reads 4e-05% to 7e-05% in every column):

| the integer sum | carry-alt-k4-256x256-T65536 | ... write scales x 2^40 | ... value scales x 2^16 | carry-tied-k4-256x256-T1024, leaves 2^24 apart |
|---|---|---|---|---|
| 32 bits, one scale a call | 99% | 161% | 110% | |
| 32 bits, a window's coarse bound (largest token x tokens so far) | 0.28% | 1.3% | 0.85% | |
| 32 bits, the cluster's state a window | 0.0001% | 0.0001% | 0.0001% | 2.0% |
| 32 bits, the cluster's planned bound a window | 0.007% | 0.002% | 0.003% | 4.2% |
| 32 bits, normalized in the cluster | 0.0002% | 0.00006% | 2.3% | 0.00001% |
| 64 bits, one scale a call | 0.000008% | 107% | 0.0002% | |
| 64 bits, a window's coarse bound | 0.000008% | 0.000008% | 0.000008% | 0.00001% |
| 64 bits, the cluster's state a window | 0.00002% | 0.00002% | 0.00002% | 0.00001% |
| 64 bits, normalized in the cluster | 0.00002% | 0.00002% | 0.00003% | 0.00001% |

THE SPIKE (the first 16 tokens at 2^B over the rest; the 64-bit forms, the worst token):

| B, cell | a window's coarse bound | the cluster's state a window | normalized in the cluster |
|---|---|---|---|
| 40, carry-tied-k4-256x256-T1024 | 0.21% | 0.0002% | 0.000007% |
| 48, carry-tied-k4-256x256-T1024 | 27% | 0.045% | 0.000007% |
| 40, carry-alt-k4-256x256-T4096 | 0.000008% | | 0.00002% |

The 64-bit normalized sum under value scales rising 2^32, 2^40 and 2^48 (carry-alt-k4-256x256-T4096): 0.00002%, 0.0002%,
0.03%.

WHAT IT SHOWS. On carry-alt-k4-256x256-T65536 as drawn the 64-bit sums add less than the fp32 sum's own error; the 32-bit sums at
the cluster's state scale and normalized add about two and three times it, and at the cluster's planned bound about
a hundred times it (0.007%) -- all inside the family's 0.2%; a 32-bit call-wide or coarse scale is outside it. Every 32-bit form has a stress that takes it past 0.2%. Among the 64-bit forms the coarse
bound gives out at a spike near 2^40 where later tokens' reads miss the spike's leaves (none in the alternating cell,
whose every reader meets every writer); the cluster's state scale holds to a spike or a leaf spread near 2^48; the
normalized sum holds everywhere but a value-scale spread near 2^48. A token of carry-alt-k4-256x256-T65536 reads 3.66 of its
clusters.

<a id="f16"></a>WHY f16 IS MODELLED: [`drain_role`](../../lab/drain_role.md) prices an f16 line at half an fp32 partial's cost, and
[precision.md](../../architecture/precision.md) lets the plan state an f16 sum across CTAs on this measurement
of what it gives up. The model's CLUSTER -- the boxes sharing one range of a level -- is a fold group when
`--along` is the write side's sparse level.

THE SUM AS f16 (`sum f16 <bound>`; 2026-11-11, carry-alt-k4-256x256-T65536, boxes 16 x 16, clusters of 16 boxes along
level 0, one box a term; a token's relative error, median / 99th percentile / largest). The family's every other
rounding together is 0.115% / 0.174% / 0.267% on this cell, and the fp32 sum alone 0.000015%.

| the sum | no stress | scales spread 2^24 | values spread 2^24 | leaves spread 2^24 | a spike of 2^24 |
|---|---|---|---|---|---|
| f16 in a cluster at its `planned` bound, the clusters in fp32 | 0.023 / 0.034 / 0.047% | 0.019 / 0.029 / 0.038% | 0.020 / 0.031 / 0.045% | 0.023 / 0.033 / 0.048% | 0.025 / 0.038 / 0.054% |
| the same at the cluster's `state` bound | the same | the same | the same | the same | the same |
| f16, every box into one sum, the `coarse` bound | 0.118 / 0.263 / 0.478% | 0.095 / 0.212 / 0.388% | 0.110 / 0.259 / 0.470% | 0.118 / 0.264 / 0.479% | 0.121 / 0.269 / 0.531% |

WHAT IT SHOWS. A cluster's few partials added in f16 and the clusters merged in fp32 cost a fifth of the family's
own error, under every stress, and a bound that only counts what could have been written into the cluster
(`planned`: the write mass so far, no state read) does exactly as well as the state's own magnitude -- f16's
exponent takes the slack. EVERY box of a token into ONE f16 sum costs as much as the rest of the family together
(58 rounded adds a token here), and with the call-wide bound more (0.174 / 0.539 / 1.25%, no stress). A sum a
group needs a panel a group and a merge that costs what the fp32 line does ([`panel_merge`](../../lab/panel_merge.md)),
so the format's f16 form is one line a CTA, its boxes added in fp32 first (below). Not modelled: the den's column kept apart from the num's, and cells other than this one.

ONE f16 SUM FOR A TOKEN, ITS CTAS' PAIRS ADDED IN fp32 FIRST (`sum f16 coarse, pairs`; carry-alt-k4-256x256-T65536, boxes 16 x 16,
`--along 1`, where neighbours in the box order are the two boxes a CTA holds along the read side's dense level): 29
rounded adds a token in place of 58 -- 0.083% a token at the median, 0.204% at the 99th percentile, 0.40% at most,
against 0.115 / 0.314 / 0.74% with every box its own add. With `--along 0` the neighbours share no token and the
pairing changes nothing (0.112%).

