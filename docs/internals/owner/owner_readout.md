# `csrc/rola/src/owner/owner_readout.cuh` — the readout, as the skeleton prices it

Mirror doc for the owner carry's readout in the skeleton: a warp's read digits over a pass of the read pool, into the
shared panel. The digits' state (the transfer's output) is placeholder registers and the panel is s32; the counts,
loads and adds are real. Vocabulary: [`owner.md`](owner.md).

<a id="operands"></a>

## The operands

An n-tile is 8 tokens. B is their read-dense rows by plain `ldmatrix` — a lane's 4 bytes are 4 consecutive leaves of one
token, the IMMA's column layout off a token-major row, no transpose — K the leaves, one or two k32 steps. A is the
digit's state, channels on M (two m-tiles) by leaves on K, two state digits, resident: 64 registers at both carves. A
digit's products for a tile are 2 m-tiles x the k-steps x 2 digits of IMMAs, and its den share is two DP4A a k-step
over the same B bytes against the leaves' masses, summed over a token's four lanes.

<a id="forms"></a>

## The forms

Per digit: each digit's entries in the pass, 8 a tile; a tile's products weighted by each token's factor (its
read-sparse byte for the digit) and added into the panel, one shared add an element. Merged (and a dense side): the
warp union's entries, the read rows loaded once a tile, every digit's weighted products summed in registers and the
tile added once — the ruled form's one panel add a token.

<a id="measured"></a>

## Measured

Per head (4 heads, CUDA events), the lists and read fills included: `nl64k-alt-k4` 1.23-1.36 ms at `(16, 64)` and
1.60-1.66 at `(32, 32)`; `nl64k-dense` 7.8 and 8.6 (the int8 tensor floor ~3.7). Before the merged form summed its
digits, dense ran 8.8 and 13.0: an add a digit made the panel's shared adds, not the IMMAs, the readout's cost, twice as
heavy a IMMA at `(32, 32)`.
