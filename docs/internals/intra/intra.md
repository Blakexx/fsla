# `csrc/rola/src/intra/intra.cuh` / `intra.cu` — the intra's constants, operand block and translation unit

Mirror doc for the header every intra source includes and for the one translation unit that
derives a call's geometry from its widths and builds the two arms, one a depth. The kernel is [`intra_kernel.md`](intra_kernel.md); the host entries this unit
defines, with their refusals, are [`intra_api.md`](intra_api.md).

<a id="constants"></a>

## The constants

The shape of the walk, all fixed: `kWindow` 512 tokens — the carry's window, because the two
kernels run one grid — cut into `kTile` = 64-token tiles, `kWindowTiles` = 8; a block is
`kRowTiles` = 2 tiles, `kBlockRows` = 128 readers, and a window `kBlocks` = 4 of them; a warp
owns `kWarpRows` = 16 readers, so a CTA is `kWarps` = 8 warps. The values are `kValueWidth` = 64
channels. The rows are u8, which serve depth at most `kMaxLevels` = 2, each level at most
`kMaxLevelWidth` = 256 digits. A copy moves `kChunk` = 16 bytes (an `ldmatrix` row, and the
granule a level's width is a whole number of); a stage is at most `kStageDigits` = 128 digits of
one level, so a level has one or two (`kLevelStages`) and a call at most `kMaxStages` = 4; a slab
is `kSlabDigits` = 32 digits, one IMMA k-step and the gate's grain. `kStageBuffers` = 3 stage
buffers, a stage copied one ahead of its use, which leaves two stages of slack between the warps
([`intra_kernel.md`](intra_kernel.md#pipeline)). `kMinPitchStages` = 3: a reader row is at least
three 128-byte stage slots, so a warp's 16 rows hold its deposit's staging tile (16 rows of 68
floats). `kColumns` = 2 + `kMaxLevels` writer columns. `IntraPhase` is the phase ledger's
column order.

<a id="pieces"></a>

## The pieces

`kPieceTiles` = 4 caps a CTA's sweep: block `b`'s tiles `0 .. 2b + 1` cut into pieces of at most
four, so the diagonal pair lands in the block's last piece. `Pieces` builds the window's table
at compile time, longest first and the heavier block first among equals — `(3, 0..3)`,
`(3, 4..7)`, `(2, 0..3)`, `(1, 0..3)`, `(2, 4..5)`, `(0, 0..1)` — and the kernel reads its entry from
constant memory by `blockIdx.x` modulo six. Why four is [`intra_kernel.md`](intra_kernel.md#the-grid).

<a id="stages"></a>

## The stages: a call's geometry from its widths

Each level is read at its rows' own width `w`, a whole number of 16-digit granules from 16 to
256, padded only to the 32-digit slab in shared memory: a level has `ceil(w / 32)` slabs, numbered
across the levels in order (the gate word's bits, `slab0` a level's first), and its slabs fall
into stages of four, the last stage of a level holding the rest. A `Stage` is a level's stage:
its `index` -- its slot in the reader row, its reader part and its place in the copy order --
its `slab0` and its slab `mask`. A 16-digit tail (a width of 16 modulo 32) is a slab's second
chunk, which the copies zero-fill rather than read; a stage's empty slots past its slabs are
filled the same way and never read. `geometry()` derives all of it and the footprint,
`sizeof(Smem)` plus 128 reader rows of `pitch` = `max(stages, kMinPitchStages)` slots of 128
bytes; the same function answers `intra_footprint`. The footprint grows with the stages, largest
at (256, 256).

The facts pass's union table is laid out by its own widths, the topology's padded ones, which
are never narrower than the rows: a level's union groups start at the layout's `group_base`
(`group0`), and its slabs read the first `ceil(w / 32)` of them.

<a id="intraparams"></a>

## `IntraParams` — the operand block, a grid constant

Per level a `RowView` of each side's rows (base, and the batch, token and head strides in
bytes), a `ColumnView` of the write scale, its width, first slab, first union group, stage
count and slab mask, and its stages; the gain and the value scale; the value planes; the tile
unions; the output planes; the ledger; the stage and slab counts, the union table's group
count and the reader row's pitch; `H`, `L`, the window count, the weight digit count and each
side's dense-level bitmask. It is a `__grid_constant__` kernel parameter of under a kilobyte, so
a launch allocates nothing to learn its shape and a level's fields are read where they lie.

<a id="the-unit"></a>

## The translation unit

The arms are the depth, `intra_kernel<1>` and `intra_kernel<2>`; the widths are the call's.
The host entry derives the geometry from the rows' widths, turns each tensor into a view after
checking it (`row_view`, `column_view`: dtype, shape, 16-byte starts and strides), fills the
block and launches the depth's arm on a `(windows × 6, B·H)` grid of 256 threads with the
call's footprint. A footprint past the 48 KB a kernel gets unasked is checked against the
device's opt-in and the arm's attribute raised to it, once a size: only a call wider than any
before it pays the attribute call.

The build stamp's kernel lives in a NAMED namespace, `stampdet`: nvcc hashes the build path
into an anonymous namespace's kernel names, and a manifest ratified in one worktree would
refuse the identical kernel built in another.
