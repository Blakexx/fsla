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

<a id="factors"></a>
## The entries' factors

`read_factors`, once a chunk: after the chunk's copies land (a warp's own wait) and before the rendezvous that opens
its readout, each warp rewrites the entries of its own digits `warp + 8 j` -- the runs its fill copied
(`parts.md#entries`), so no other warp's copies are read and no barrier is added -- in place: its row offset and its
coefficient (f16, the high half, the den's) in the first word, its NUM FACTOR in the second, fp32: `c 2^(e_d -
kImageTop - kCoefTop)` (the image holds digit `d` at `2^(kImageTop - e_d)` of the fold's state, itself `2^kCoefTop` the
state: `handoff.md#maxima`), on the f16 panel times the token's scale `2^(-s - 1)` (`token_scale`, its exponent
field built directly). The same products in the same order as at a tile, once a chunk where they were once a channel
group: the four channel groups of a token group read every entry, and the token's scale byte, the exponent build and
the coefficient's conversion were a quarter of a tile's instructions.

<a id="chunk"></a>
## A chunk

`read_chunk<P, F32, F16>` reads a chunk's slots over digits `lo .. hi - 1` (the whole box, or a half where the image is
stored in halves, `schedule.md#planner`; the kernel's one call site, its passes a loop), `F32` the panel's dtype (the
call's reduce dtype), `F16` the MMAs' accumulate form (fp16 where the card's fp32 accumulate runs at half rate,
[`carry_api.md#carry-readout-accumulate`](carry_api.md#carry-readout-accumulate)). It runs two PASSES (`read_pass`):
the num's over every digit with entries in the warp's token group, then the den's over the digits its channel group
names (`d mod 4`) -- the den's MMAs only where they are kept, a quarter of the num's (a warp-uniform `if` around a
tile's den MMAs is if-converted to predicated HMMAs, which the ptxas gate refuses, `../../KERNEL_STANDARDS.md#s20`;
two loops are not).

A PASS walks its tiles in digit order: a TILE is 16 of a digit's entries in the token group, tokens on M, its digit
the next whose bit the pass's mask holds (a ballot of the lanes' counts; the next found by `lowest_bit` on every tile,
so no branch forms around it). `tile_in` loads a tile's operands -- its count and place by two shuffles from lane
`d`, lane `l`'s entry `l & 15` (a lane past the run loads the tile's first and adds nothing), its rows from the pool by
two `ldmatrix.x4` (`chunk_addr`: 16 tokens x 32 lists), the digit's slice (the 32 lists x 8 channels of the warp's
channel group, one `ldmatrix.x4.trans`, `slice_row`; the den's pass the den image's block, `den_row`, and the digit's
den factor) -- a TILE AHEAD: each trip of the pass is two tiles, each's loads issued before the other's MMAs and adds,
so no operand moves between registers and the trip's one exit is at its end (no branch among its MMAs; the second
tile's adds predicated off where it is past the pass's last). `tile_out`: two m16n8k16 MMAs (`tile_mma`), then each of
the lane's two tokens (rows `g`, `g + 8`) -- its entry by two shuffles, its slot (`slot_of`: the entry's row offset,
the union's, taken back by the chunk's first slot modulo 2^16) -- both tokens' loads, then both adds, then both
stores, so the two read-modify-writes overlap: the num's pair at its factor into the panel's word (`panel.md#words`),
the den's column of the digit (`d & 1` of the lane's pair, a select) at its coefficient times the digit's den factor
into the slot's den word `cg`, by the lanes holding the column (`tq == (d & 7) >> 1`). Every load whose store is
predicated off is predicated too (`ops::load_shared_u32_if`), so a dead lane reads no word a live lane writes; a warp
sync a tile orders one tile's adds before the next's, whose tokens may be the same in other lanes.

THE f16 PANEL AT THE TOKEN'S SCALE BYTE: the scale pass's byte bounds the token's `num` under `2^(s + 15)` with 2%
headroom ([`scale.md`](../scale/scale.md)), so every piece and every partial sum of the token's panel word lies within f16's range at
`2^(-s - 1)`, the top bit kept for the cross-CTA sum; a piece under the frame's 2^-24 rounds into the subnormals, which
the reference's bound carries (a term a piece and an add). The fp32 panel is absolute. Each piece is rounded once into
the panel's dtype and each add once more.

MEASURED (the part harness [`read_alone`](../../lab/read_alone.md), sm_86, cycles a warp-window at nl64k-alt-k4's
draw, the f16 panel at fp16 accumulate): 23.9K as first built (the den on every warp, each token's factor a channel
group, one tile at a time); the den on its own channel group 21.4K; the factors once a chunk and the tiles a tile ahead
14.5K; the two tokens' adds overlapped 13.5K -- against its floors there (tensor 4.0K, shared 8.1K at the first form's
traffic). In the composed kernel (`tools/phase_ledger.py`, the readout's one call site) the readout phase 13.1K of a
31.6K window, from 24.9K of 41.8K. THE SASS GATE holds its step loops at their floor (`tools/budgets/step_floors.json`,
`tools/sass_gate.py`'s `_STEP_COVER_DEBT`); what replaces the debt is composition with the fold (each scheduler's two
warps in different phases) and sm_90's `wgmma` (a warpgroup's operand read once).

ACROSS CARDS: sm_80 and sm_86 run this form (`mma.sync` m16n8k16, `ldmatrix`); sm_86's half-rate fp32 accumulate takes
the fp16 form. sm_90 and sm_100 take the same slices and entries; their asynchronous MMAs (`wgmma`, `tcgen05`) consume
the slice from shared memory without the `ldmatrix`, the panel's adds unchanged.
