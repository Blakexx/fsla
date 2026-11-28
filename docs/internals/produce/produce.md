# The producer's fused forward: its skeleton

`csrc/fsla/src/produce/produce.cuh` is the producer's ONE LAUNCH from the hidden states to the op's operands
(CONSTITUTION R-1). It is built as KERNEL_STANDARDS [§19](../../KERNEL_STANDARDS.md#s19) asks: the skeleton first --
the grid, the roles, the ring, the job table, every output and the density's ticket -- each component a stub at its
place, then one component at a time against its line of [`tools/budgets/produce.json`](#budget). The skeleton's stubs
write every output's zeros; its gate is the lab's [`produce_skeleton`](../../lab/produce_skeleton.md): every element of
every output written, for every routing of the test matrix, under the card's kill timeout. Nothing in the library
launches it yet: the CUDA call raises `not_built("produce")` until the components land.

<a id="plan"></a>
## The plan

`Plan` is the launch's one `__grid_constant__` parameter: the shape (`B`, `T`, `H_w`, `G`, `D`, hidden, `d_v`), each
level's width, which level-sides are MASKED (their activation can make exact zeros: their masks are written and their
density counted, api.md §2.6), the job table, and every output's pointer. The outputs are api.md's layouts today:

| output | layout |
|---|---|
| rows, a level-side | u8 `[B, T, H, b_l]`, `H` the side's heads (`H_w`, or `H_w G` with read head `w G + g`) |
| scales, a level-side | fp32 `[B, T, H]` |
| the lo plane, a level-side (a gradient on) | i8 `[B, T, H, b_l]`, beside the rows (the owner's ruling of 2026-11-27) |
| masks, a masked level-side | i32 `[B, H, ceil(T / 32), b_l]`, token-packed; the read side's one mask over every read head |
| values | bf16 `[B, T, H_w, d_v]`, raw, no scale (the owner's ruling of 2026-11-28) |
| density | u8 `[2, D]`, row 0 the read side, row 1 the write side; 255 for a side with no mask |

The lo plane is a compile-time variant (`produce_cta<LO>`), on exactly when a gradient is needed. The rows are
token-major as api.md states them; the tile-major order ARCHITECTURE.md §1 names for hopper's readers is the format
component's to write, with the reader that asks for it.

<a id="jobs"></a>
## The job table

A job is one ROW SPAN of one write head's columns: a write level (`kWriteJob`, its `tied` bits naming the read heads
whose planes take copies of its rows), an untied read entry (`kReadJob`, its read head), or the values
(`kValueJob`). One table serves every write head; its order is the write levels, the read entries by head then level,
the values. A plan holds at most `kMaxLevels` levels and `kMaxGroup` read heads a write head; the table's length is at most a write job a level, an untied read job a (read head, level) and the values'.

<a id="cta"></a>
## The grid and a CTA

Grid `B ceil(T / 128) x H_w`: a CTA owns one sequence's token tile of 128 and one write head, through every job of the
write head, so no CTA waits on another. Tiles are a sequence's own, not the flattened `B T`'s: a warpgroup's 64
tokens then start a mask word, so each word has one writer (a flattened tile would split a word across CTAs where
`T` is not a multiple of 32). Its cost is a partial tile a sequence, not one a call.

<a id="roles"></a>
## The roles

Two MATH warpgroups (threads 0-255), 64 tokens of the tile each, then one LOADER warp (`kThreads = 288`), a CTA an SM.

<a id="ring"></a>
## The ring

Three stages of a tile's `x` `[128 x 64]` and a job's weights `[256 x 64]` in bf16 (48 KB each, 144 KB), then each
stage's FULL barrier (one arrival: the loader's; the projection's form states its bytes) and EMPTY barrier (eight: a
math warp's lane 0 each). <a id="math"></a>A math warpgroup waits FULL, releases EMPTY, a stage a K-step, through a
job's K-steps, then runs the job's epilogue; the loader waits EMPTY from its fourth stage on. The skeleton stages no
bytes; the projection component fills the stages by the tensor copy (`stage_box_2d`,
[`tma_box`](../../lab/tma_box.md), [`projection_tile`](../../lab/projection_tile.md)).

<a id="epilogue"></a>
## The epilogue

A job's epilogue on a warpgroup's share: a write job its write side's level and each tied read head's copy, a read
job its read head's level, the value job the values. A share past its sequence's end (a tile's second half) writes nothing, its mask words included: a warpgroup's 64 tokens start a word, so no word has two writers.
<a id="format"></a>`store_level` writes one head's rows, scales, lo plane and mask words of a level-side (the solve's,
format's and mask's stubs: zeros). <a id="stores"></a>`zero_runs` is the stubs' one store: runs of bytes a stride
apart, 16 bytes a store where a chunk is whole and aligned, a byte a store otherwise.

<a id="density"></a>
## The density's ticket

A CTA's live digits a masked level-side (counted in shared memory past the ring's barriers) are added to the counts (`u64 [2][kMaxLevels]`); then a ticket (an atomic
counter): the last CTA to finish turns the counts into bytes, `(2 * 255 * live + total) / (2 * total)` (the library's
rounding, api.md §2.6), and zeroes the counts and the ticket for the next launch, so no host read and no clear between
launches; the op allocates them zeroed once.

<a id="budget"></a>
## The budget

[`tools/budgets/produce.json`](../../../tools/budgets/produce.json) states, before the code, each component's warp
instructions a CTA, its registers and each phase's cycles at the design's cell, with the arithmetic of each line
(`basis`). The skeleton is write-bound: a CTA's outputs, 165,888 bytes at the cell, take 12,800 cycles at an SM's
share of HBM's write rate against ~18K warp instructions of issue.
