# The pre-pass: the int8 carry's skeleton at two columns a leaf

The pre-pass computes, per token and before the carry, the two numbers the carry's packed
panel and its epilogue need: `den_t`, the inter denominator against the window-start mass, and
`scale_t`, the per-token bound the packed s16 panel and the packed output are expressed in
(the queue card, the S8 budget line, ruled 2026-09-27). It is the same kernel shape the int8
carry takes, with the state two s32 columns a leaf (mass, bound) instead of 32 channels, so it
is built first: every component of the frame is measured here with a trivial epilogue before
the 32-channel state, the hi/lo transfer and the panel go in.

The declared arithmetic is `tests/oracle/int_reference.prepass_reference`; the op gate is
`tests/oracle/test_prepass_kernel.py`.

<a id="params"></a>
## The parameter block

`PrepassParams` (`csrc/rola/src/prepass/params.cuh`): the tall owner set's addressing block
(`CarryGeomRT`), the two sparse masks (bit `l` set when level `l` is sparse on the read or the
write side -- the levels whose digit rows the words are staged for), the bundle's u8 rows on
both sides as `[BH, L, sum_l B_l]` planes with their per-level maxima `[BH, L, D]`, the write
gain and the value scale `[BH, L]`, the two outputs `[BH, L]` REDUCED into, and the class-1
liveness words `[BH, 2, sum_l B_l, ceil(L / 32)]`. The phases are the skeleton's components,
one ledger row each: frame, words, lists, fill, fold, publish, rescale, readout, drain.

<a id="plan"></a>
## The tall plan

`TallPlan<D>` (`plan.cuh`): eight warps, 1024 leaves a CTA and 128 a warp
(`geom.md#tall-owner-set`), two columns, the 512-token window in sixteen rounds and sixteen
32-token tiles, the K block of 32 leaves and four blocks a read warp. The pool takes a chunk
of 64 tokens in two barriered slots; a row is the write-sparse level's entries over the
region's digits (128 bytes), the write-dense entries (16), and the two row scales, `g_t` and
`s_t` (16). Each read warp has a two-slot ring of a tile's read rows (the dense sub-block's
32 entries, the sparse entries, two scales). The transfer is every leaf's two columns as s16
against the leaf's max plus that max as a float; the panel a 128-token range's two fp32 sums,
two parities.

<a id="smem"></a>
## The shared ledger

Words (two parities) at 0; the pool; the rings; the transfer; the panel; the barriers (pool
full and empty a slot, a ring barrier a warp-slot, the words landed, the publish and the done a
parity); the phase ledger; the list counts. The plan's `static_assert` holds the total under
the architecture's per-CTA maximum, about 78 KB at depth two.

<a id="frame"></a>
## The frame

One warp stages the next window's liveness words while the current one runs (`stage_words`:
cp.async of the staged rows' sixteen words, landed at the words barrier). Every warp counts its
lists from the words (`list_counts`: its sixteen fold lists, one a write-sparse digit, and the
union of its four read digits, by popcount). The chunks are filled a slot ahead and folded:
`pool_fill` stages a chunk's rows by cp.async and arrives at the slot's full barrier when
landed; `fold` walks the warp's lists over the slot and arrives at its empty barrier. The fold
warps `publish` their rows into the transfer and every thread arrives at the publish barrier;
after it the read warps `rescale` their blocks (the block scale and live min from the 32 leaf
maxima, one a lane). The ranges are read out tile by tile into the panel (`readout_tile`: the
tile's rows staged into the ring, the partials, `red.shared.add.f32` into the panel) and
`drain`ed by `RED.F32` into the global sums, the panel zeroed for its next range.

THE COMPONENTS ARE STUBS WITH THEIR EDGES until each lands against its budget line: the fill
stages the window's tokens in order rather than the union-live compaction, the fold touches the
pool once a step and issues no MMA, the publish writes zeros, the readout adds zeros. What the
skeleton measures is therefore the frame: the words, the list counts, the fill traffic, the
publish and rescale traffic, the panel atomics, the drain and the barriers -- the
fixed-a-window row of the ledger, the one term no probe had measured.

<a id="call"></a>
## The call

`derive_prepass_call` (`prepass_host.cuh`) refuses shapes and dtypes at the boundary and derives
the tall owner set through `derive_carry_geom` with `bc` the tall leaf count, both stream counts
eight, run placement and no warp-level page floor, the mode word from the sparsity declaration.

<a id="api"></a>
## The API

`prepass_forward(rows_read, rows_write, scale_read, scale_write, gain, vscale, den, scale,
widths, level_modes, liveness)`, `prepass_ledger_bind(ledger?)`, `prepass_plan(depth)`
(`prepass_api.cuh`, `prepass.cu`). The Python surface is `rola/ops/prepass.py`: it packs the
bundle's rows and scales, folds the liveness words from the u8 rows (a digit is live iff its byte
is nonzero, the format's support rule) and reduces into zeroed planes. Depth two is built on this
line.

<a id="budgets"></a>
## Budgets

The S8 budget line on the queue card: tensor a quarter of the carry's a window, issue the carry's
list and A-row cost, target at most 0.35x the carry's MMA-phase cycles a window, 8 B a token
written. The measured frame row is recorded on the card when the skeleton runs.
