# `csrc/rola/src/intra/intra.cuh` / `intra.cu` — the intra's constants, operand block and translation unit

Mirror doc for the header every intra source includes and for the one translation unit that
builds the arms. The kernel is [`intra_kernel.md`](intra_kernel.md); the host entries this unit
defines, with their refusals, are [`intra_api.md`](intra_api.md).

<a id="constants"></a>

## The constants

The shape of the walk, all fixed: `kWindow` 512 tokens — the carry's window, because the two
kernels run one grid — cut into `kTile` = 64-token tiles, `kWindowTiles` = 8; a block is
`kRowTiles` = 2 tiles, `kBlockRows` = 128 readers, and a window `kBlocks` = 4 of them; a warp
owns `kWarpRows` = 16 readers, so a CTA is `kWarps` = 8 warps. The values are `kValueWidth` = 64
channels and the depth at most `kMaxLevels` = 4. A copy moves `kChunk` = 16 bytes (an
`ldmatrix` row); a writer stage is `kStageDigits` = 128 digits of one level; a slab is
`kSlabDigits` = 32 digits, one IMMA k-step and the gate's grain. `kStageBuffers` = 3 stage
buffers, a stage copied two ahead of its use ([`intra_kernel.md`](intra_kernel.md#pipeline)). `IntraPhase` is the phase ledger's column
order.

<a id="intraparams"></a>

## `IntraParams` — the operand block, by value

Per level a `RowView` of each side's rows (base, and the batch, token and head strides in
bytes) and a `ColumnView` of the write scale; the gain and the value scale; the value planes;
the tile unions; the output planes; the ledger; `H`, `L`, the window count, the weight digit
count and each side's dense-level bitmask. No width is in it: the arm is the width. It is a
kernel parameter of under a kilobyte, so a launch allocates nothing to learn its shape.

<a id="the-unit"></a>

## The translation unit

`ROLA_INTRA_ARMS_X` is the built set, `(2, 256)`, one macro the dispatch and `intra_arms`
both expand, so neither can name an arm the other does not. The host entry turns each tensor
into a view after checking it (`row_view`, `column_view`: dtype, shape, 16-byte starts and
strides), fills the block and launches `intra_kernel<D, B>` on a `(windows × 4, B·H)` grid of
256 threads. An arm's footprint is past the 48 KB a kernel gets unasked, so its first launch
checks it against the device's opt-in and sets the attribute once.

The build stamp's kernel lives in a NAMED namespace, `stampdet`: nvcc hashes the build path
into an anonymous namespace's kernel names, and a manifest ratified in one worktree would
refuse the identical kernel built in another.
