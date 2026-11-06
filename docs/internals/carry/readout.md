# `carry/readout.cuh` — the value-carve readout

Mirrors `csrc/rola/src/carry/readout.cuh`. The readout of a LISTED read side, a VALUE CARVE: every warp reads every
digit of the box and every leaf of it, for its own channels and its own tokens -- where the fold is a state carve (its
warps own lists, `fold.md`). The fold hands its state over through the image (`handoff.md#image`); the readout's
output goes to the panel and from there to `num` and `den` (`panel.md`). A side with no list level (the dense
readout: its nest of accumulator groups, operand scaling) is the next component, and `Plan` refuses an arm whose
`read_factors` is not 1.

THE OUTPUT: each token's partials over the CTA's box, `P(t, c) = sum_{d, r} x(t, d) a(t, r) S(d, r, c) / kappa` --
`S` the state before the token's window (digit `d`, list `r`), `x` the read coefficient of the digit as stored (a
sparse level's byte / 256), `a` the read row of the list level as stored (f16), `kappa` the state's byte scale (the
fold's operand an f16 dense level, its coefficient on the byte scale: 1/255, `rola.ops.prefill.carry_factor`). The
read side's per-token scales and `kappa` are common to `num` and `den` and cancel in the epilogue. Gated per slot
against the fp64 reference of this arithmetic (`tests/oracle/readout_reference.py`'s `readout`,
`tests/oracle/test_carry_readout.py`): both orders of the alternating box and both same-split boxes, both accumulate
forms, both reduce dtypes, each image mode.

<a id="warps"></a>
## The warps

At the 32-channel box the CTA's 8 warps are 2 TOKEN GROUPS x 4 CHANNEL GROUPS of 8 channels (`kTokenGroups`,
`kChannelGroups`; the planner's `rola.engine.carve.token_groups` and `channel_groups`): warp `w` is channel group `w
mod 4` of token group `w / 4`. A channel group is one MMA's N; a token group a chunk's slots on one side of
`group_split` -- a multiple of 64, the granule the facts pass's prefix rows count a list's tokens at, so a list's run
splits where its prefix rows hold the count below the split (`Run::n0`, `parts.md#entries`). A chunk of at most 64 slots
is token group 0's alone. The groups' slots are disjoint, so each group has its own panel rows and den words and no
two warps add into one word: the panel needs no atomics (sm_80/86's shared atomic add is integer only). All 8 warps
read the same digit's slices at the same time, so the image is read by the CTA as one stream, a digit at a time.

<a id="chunk"></a>
## A chunk

`read_chunk<P, F32, F16>` reads a chunk's slots over digits `lo .. hi - 1` (the whole box, or a half where the image is
stored in halves, `schedule.md#planner`), `F32` the panel's dtype (the call's reduce dtype), `F16` the MMAs' accumulate
form (fp16 where the card's fp32 accumulate runs at half rate,
[`carry_api.md#carry-readout-accumulate`](carry_api.md#carry-readout-accumulate)).

A DIGIT `d`: its run in the warp's token group (count and place by three shuffles from lane `d`, `chunk_runs`), its
slice -- the 32 lists x 8 channels of the warp's channel group, one `ldmatrix.x4.trans` (`slice_row`) -- the den
image's slice for the digit's block (`den_row`), the digit's exponent as `2^(e_d - kImageTop - kCoefTop)` (the image
holds the digit at `2^(kImageTop - e_d)` of the fold's state, itself `2^kCoefTop` the state: `handoff.md#maxima`) and its
den factor. A TILE is 16 of the run's entries, tokens on M: lane `l`'s entry `l & 15` (a lane past the run loads the
tile's first and adds nothing), its row from the pool by two `ldmatrix.x4` (`chunk_addr`: 16 tokens x 32 lists), two
m16n8k16 MMAs against the slice and two against the den slice (`tile_mma`). Then each of the lane's two tokens (rows
`g`, `g + 8`): its entry by two shuffles, its slot (`slot_of`: the entry's row offset, the union's, taken back by the
chunk's first slot modulo 2^16), its coefficient (the entry's second word, f16), its factor -- `c 2^(e_d - kImageTop -
kCoefTop) 2^(-s - 1)` on the f16 panel (`s` the token's scale byte, `token_scale`: its exponent field built directly,
so the step has no branch) or without the scale on fp32 -- and the panel's load-add-store at the slot's word
(`panel.md#words`), predicated past the run (`ops::store_shared_u32_if`, `store_shared_f32x2_if`). The den: the
column of the digit (`d & 1` of the lane's pair, a select), times the coefficient and the digit's den factor, added into
the slot's den word `cg` by the channel group the digit names (`d mod 4`), lane `tq == (d & 7) >> 1`.

THE DEN'S MMAs RUN ON EVERY WARP: a warp-uniform `if` around them is if-converted to predicated HMMAs, which the ptxas
gate refuses (`../../KERNEL_STANDARDS.md#s20`), so every channel group issues them and the owner's predicated stores
keep them -- the den is a quarter of the channels' tensor work a warp, the other three groups' a waste the den's own
budget prices (lab [`read_value`](../../lab/read_value.md): the den's tile +25%).

THE f16 PANEL AT THE TOKEN'S SCALE BYTE: the scale pass's byte bounds the token's `num` under `2^(s + 15)` with 2%
headroom ([`scale.md`](../scale/scale.md)), so every piece and every partial sum of the token's panel word lies within f16's range at
`2^(-s - 1)`, the top bit kept for the cross-CTA sum; a piece under the frame's 2^-24 rounds into the subnormals, which
the reference's bound carries (a term a piece and an add). The fp32 panel is absolute. Each piece is rounded once into
the panel's dtype and each add once more.

MEASURED (lab [`read_value`](../../lab/read_value.md), sm_86, this form without the per-token scale): shared memory
binds it -- 10.1 shared-pipe cycles an MMA at 16 entries a digit a window (the rows each channel group gathers, its
slice, the entries and the panel's load and store), ~1.6x that floor alone, its stalls the tile chain's fixed latencies.
THE SASS GATE reads its step loop's pipe cover at 22.4% (sm_86): a named debt held at that floor
(`tools/budgets/step_floors.json`, `tools/sass_gate.py`'s `_STEP_COVER_DEBT`), whose replacements are the tiles
pipelined a tile ahead, composition with the fold (each scheduler's two warps in different phases) and sm_90's
`wgmma`. Its part-harness budget comes with the pipelined tiles.

ACROSS CARDS: sm_80 and sm_86 run this form (`mma.sync` m16n8k16, `ldmatrix`); sm_86's half-rate fp32 accumulate takes
the fp16 form. sm_90 and sm_100 take the same slices and entries; their asynchronous MMAs (`wgmma`, `tcgen05`) consume
the slice from shared memory without the `ldmatrix`, the panel's adds unchanged.
