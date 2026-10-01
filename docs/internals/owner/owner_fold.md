# `csrc/rola/src/owner/owner_fold.cuh` — the fold, as the skeleton prices it

Mirror doc for the owner carry's fold in the skeleton: a warp's write lists over a pass of the write pool, into the
resident s32 state. The operands' formation (the write-sparse factor and the scales) is not built — the fold's
arithmetic line owns it — so the counts, the gathers and the loads are real and the products are not the model's.
Vocabulary: [`owner.md`](owner.md); the pool rows: [`owner_fill.md`](owner_fill.md).

<a id="operands"></a>

## The operands

A k-step is 32 tokens. Each lane addresses one token's pool row (an entry of the warp's list buffer, or the zero row
past the run's end), and `ldmatrix.x4.trans` gathers the 32 rows: the value digits as B (channels on N, four n-tiles
of the channel group, a byte permute a register for the even and odd channels) and the dense slice as A (leaves on M,
one m-tile at `(16, 64)`, two at `(32, 32)`). The IMMAs are m-tiles x four n-tiles x two value digits (`v_hat` signed,
`v_lo` unsigned) into the list's accumulators. The state is 128 registers a lane at both carves (eight lists of one
m-tile, or four of two).

<a id="forms"></a>

## The forms

Per list: each list's entries in the pass, k-step by k-step, the value operand loaded for every list's step. Merged
(and a dense side): the warp union's entries, the value operand loaded once a step and each list's A and IMMAs over the
whole union. A list's (or the union's) cursor carries across the window's passes; a pass's end is found by ballots over
the sorted entries.

<a id="measured"></a>

## Measured

Per head at `nl64k-alt-k4` (4 heads, CUDA events), over the fill part it contains: `(16, 64)` +1.10 ms per list, +0.85
merged; `(32, 32)` +0.38 per list, +0.41 merged — the same 512 IMMAs a CTA-window, but a `(16, 64)` k-step carries
half the IMMAs for the same per-step loads and scans, over three passes against two. At `nl64k-dense` both carves add
~4.1 ms, ~1.1x the int8 tensor floor (~3.7 ms a head). The cost model's choice ran slower than either forced form: its
weights are placeholders until calibrated against these.
