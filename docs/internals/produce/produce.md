# The producer's fused forward: its skeleton

`csrc/fsla/src/produce/produce.cuh` is the producer's one launch from the hidden states to the op's operands (the owner's
ruling of 2026-11-27: one fused forward for inference and training). It is built as KERNEL_STANDARDS
[§19](../../KERNEL_STANDARDS.md#s19) asks: the skeleton first -- its grid, roles, ring, job table, every output and its
tickets -- with each component a stub at its place, then one component at a time against its line of
[`tools/budgets/produce.json`](#budget). The stubs write every output's zeros, so the skeleton fixes the final output
forms and phase structure before any real component lands. Its gate is the lab's
[`produce_skeleton`](../../lab/produce_skeleton.md): every element of every output written for every routing of the test
matrix, under the card's kill timeout, and the empty pipeline read by its timeline. Nothing in the library launches it
yet: the CUDA call raises `not_built("produce")` until the components land.

<a id="plan"></a>
## The plan and the outputs

`Plan` is the launch's one `__grid_constant__` parameter: the shape, each level's width, which level rows are MASKED,
the job table, and every output. A level has G + 1 ROWS, read head `g` at `g` and the write side at `G` (the owner's
order of 2026-11-27). A level-side's rows, scales and lo plane are one tensor over its heads (read head `g` of write head `w` is
query head `w G + g`); its masks are one a row (the owner's 2026-11-28 rulings):

| output | layout | why |
|---|---|---|
| rows, a level-side | u8 `[B, H, ceil(T / 64), ceil(b_l / 16), 64, 16]` (`H_w` on the write side): TILE-MAJOR, a (sequence, head)'s 64-token tiles, each its 16-digit blocks, a block a token's 16 bytes a line, pads zero ([api.md](../../api.md#tile-major)) | the public form is the layout its readers load cheapest (the owner's 2026-11-28 ruling on a level's public form); a warpgroup's 64 tokens are one tile, one contiguous run |
| scales, a level-side | fp32 `[B, T, H]` | a row's fp32 max |
| the lo plane, a level-side | u8, the rows' layout, written only with a gradient on (`produce_cta<..., LO>`) | the backward's residual: `lo = round(255 (r + 0.5))`, `p = (hi + lo / 255 - 0.5) max / 255` (the owner's 2026-11-28 ruling on the lo byte) |
| masks, a masked (row, level) | i32 `[B, H_w, ceil(T / 32), b_l]`, token-packed | one per (head, level), the write side included (the owner's 2026-11-28 ruling on masks); a tied read head reads the write side's |
| values | bf16 `[B, T, H_w, d_v]`, token-major, raw | no scale a token (the 2026-11-28 values ruling); its readers take a token's values as one run |
| growth | fp32 `[B, H_w, d_v]`: a channel's sum over the call's tokens of `|v| x` the token's largest write coefficient | the f16 output line's bound (the 2026-11-28 bound ruling) |
| density | u8 `[G + 1, D]`, 255 where a row's level has no mask | the prefill's own input (api.md §2.6) |

A read row is a simplex, so its consumer applies its scale at expansion; the producer writes it as every other.

<a id="jobs"></a>
## The job table

A job is one ROW SPAN of one write head's columns: a write level (its `tied` bits naming the read heads whose planes take
copies of its rows, the owner's ruling of 2026-11-27), an untied read entry (its read head), or the values. One table serves every write head, so a
write head's work is one loop over it; a job's width is its level's, read from the plan, never a second copy.

<a id="cta"></a>
## The grid and a CTA

A 1-D grid, a CTA a sequence's 128-token tile and a write head, tiles fastest so a write head's weights are served from L2
to its resident tiles. Tiles are a sequence's own, not the flattened `B T`'s: a warpgroup's 64 tokens then start a mask
word, so every word has one writer and no CTA waits on another (R22). A share past its sequence's end writes nothing.

<a id="roles"></a>
## The roles

Two MATH warpgroups, 64 tokens each, then one LOADER warp: the warpgroups issue the projection's steps on one tensor pipe
and run their epilogues on their own registers, the loader keeps the ring full. The loader is one warp, not a warpgroup,
so no register moves with its role today (`setmaxnreg` works on whole warpgroups): the register trade is the solve
component's decision, priced by its registers line.

<a id="ring"></a>
## The ring

Stages of a tile's `x` `[128 x 64]` and a job's weights `[256 x 64]` in bf16, each with a FULL barrier (the loader's
arrival; the projection's form states its bytes) and an EMPTY barrier (a math warp's arrival each). The stages are the
`Ring` tile of the CTA's one `SmemLayout`: it takes what the card leaves, so the depth follows the card's budget (four
stages at one CTA an SM on the H100, two at two) and degrades smoothly where shared memory is short (R12, R22). The
skeleton stages no bytes. <a id="math"></a>A math warpgroup passes each K-step's two edges, then runs the job's epilogue.

<a id="epilogue"></a>
## The epilogue

A write job writes its write row and copies the rows (never the mask) to each tied read head; a read job writes its read
head's row; the value job writes the values and adds the growth's part. <a id="format"></a>`store_level` is the format's
and mask's place, <a id="values"></a>`store_values` the values', and <a id="stores"></a>The stores are the STORE COMPONENT. Its shape
answers what the skeleton's runs measured: a stub that divided a chunk's index by its run's chunks spent 29 of its ~46
instructions a 16-byte store on the division, and once that was gone the CTA was still latency-bound -- about two warps
a scheduler, each store's address reused by the next trip of a rolled loop -- with DRAM at half its peak. So a thread
keeps several independent 16-byte stores in flight, each with its own address, and walks a run with no division; a
contiguous run (a row tile, a lo tile, a share's mask words) is the warpgroup's in 16-byte chunks, and a share's token
runs (scales, values) are each walked by consecutive threads so a store instruction covers whole runs, not one token a
lane. The token runs' stores are 4-byte words at the least: a plan's `d_v` is even (the host refuses an odd one), and a run
wider than the warpgroup's threads is walked in a lane loop.

<a id="density"></a>
## The tickets

Two joins with no host read and no clear between launches. A (sequence, write head)'s last CTA turns the growth's
running sums into the output and zeroes them; the launch's last CTA turns the live counts into the density bytes,
`(2 * 255 * live + total) / (2 * total)` (api.md §2.6), and zeroes them. The op allocates the sums, counts and tickets
zeroed once.

<a id="trace"></a>
## The tracer

`produce_cta` takes a tracer: the library's `NoTrace` is nothing; the lab's stamps each phase (`Site`) and each ring
barrier's wait and arrival through warpline, so the composed kernel is read by its timeline (KERNEL_STANDARDS
[#stepped](../../KERNEL_STANDARDS.md#stepped), [`lab.md#trace`](../tools/lab.md#trace)).

<a id="launch"></a>
## The launch

The library holds the CTA, `produce_cta<Bt, CTAS, LO, Tr>`, as the carry holds its own: no `__global__` is the library's until the op
launches one, and the lab's `produce_skeleton` defines the launch (`CTAS` CTAs an SM, reserving its layout's ring). The kernel's
`tools/coverage.json` entry comes with the library's launch; until then the lab kernel carries the floor's work.

<a id="floor"></a>
## The floor

[`measure/floors/produce.work.json`](../../../measure/floors/produce.work.json) declares the skeleton's work at the cell
`layer-alt-entmax1.5-256x256-T65536-h4`, its outputs' 323.0 MB written to HBM, and
[`h100.rates.json`](../../../measure/floors/h100.rates.json) the H100's stream rates from the lab's `stream_rate`:
warpline's floor is 101.9 us, `hbm.write` binding. Its `kernel_floor` waits on the kernel's stamped run manifest
(the lab kernel's run).

<a id="budget"></a>
## The budget

[`tools/budgets/produce.json`](../../../tools/budgets/produce.json) states each component's warp instructions a CTA at the
cell -- its trips times the SASS price of its trip, read from the built kernel -- its registers, and each phase's cycles.

## The blackwell form

NOT STATED: the roles, the ring and the tickets carry over; the steps become `tcgen05` into tensor memory. Designed when
blackwell is built (ARCHITECTURE.md §9).
