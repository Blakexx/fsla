# `csrc/rola/src/produce/produce.cuh` + `produce.cu` — the producer, one launch

RULED 2026-09-29: the producer is a declared configuration compiled into ONE kernel launch -- from a stretch of
tokens to the bundle the ops take: every routing level's projection, solve and unit-max rows, the write gain, and the
value digits. The layer declares its routing (`rola.Routing`), owns the weights, and compiles a producer plan once
(`rola/ops/produce.py`); `job_table` turns the plan into the rows this kernel reads, so every width, kind and offset
is a runtime field and the family is ONE instantiation.

The components land one at a time into the skeleton, each gated (KERNEL_STANDARDS §19); a component not yet landed
has its real signature, tenant and edges and does the cheapest thing that keeps the composition's shape (zero digits
and scales, unit gains). LANDED: the projection. `tests/integration/test_produce_kernel.py` holds the skeleton's
claim (every element of every output is written, whatever the routing, and the arrival counters are back at zero
after the call) and each landed component's gate.

## <a id="jobs"></a>The job table

One int32 row per job (`JobField`): the kind, the head (the first of a value job's heads, and their count), the
solve (`SolveKind`), the level's width, the `route_W` columns of its one or two logit rows, the packed-plane columns
its rows land at on each side (`-1` for a side it does not write), the level, and whether each side's fold applies.
A ROUTE job is one head's one solve: a union level's two logit rows (read and write), a tied level's one row filling
both planes, or one side of an independent level. A VALUE job is the heads whose channels fit one job (`256 / d_v`
of them, at most `kMaxColumns` channels). A row is at most `kMaxWidth` leaves: the widest a warp solves.

THE GRID is token tiles of `kTokenTile` tokens by jobs, the tiles fastest: the blocks resident together are one job's
tiles, so at prefill its weights are read from DRAM once and served to the rest from L2.

## <a id="accumulate"></a>The projection

The job's columns for the tile's tokens, fp32 (+ the routing's bias), into the block's logit rows in dynamic shared
memory -- `kTokenTile` rows of `kLogitStride` words, four past `kMaxColumns` so an accumulator store's 32 lanes land
in 32 banks. For a route job the columns are the solve's logit rows (both of a union's, gathered from their two
`route_W` spans); for a value job its heads' channels.

THE PRODUCT IS TRANSPOSED onto the tensor cores: the job's COLUMNS are the MMA's M (a 16-column tile) and the tile's
TOKENS its N (four groups of eight), `m16n8k16` bf16 with fp32 accumulate. Both weights are read along the hidden
axis, which the kernel takes contiguous (`route_W` through `rola.ops.produce.hidden_contiguous`; a linear's weight
already is): a column's hidden run and a token's row of `x` are each loaded straight into registers in 16-byte chunks,
four lanes a 64-byte run, and the chunks ARE the fragments. The MMA pairs k elements in a register in a fixed slot
order; the chunk holds eight consecutive k, so the kernel permutes the MMA's k order to the chunk's -- slot pairs
`(2q, 2q+1)` and `(2q+8, 2q+9)` take the chunk's elements 0-1 and 2-3 in the first k-step, 4-5 and 6-7 in the second
-- the same permutation on both operands, which leaves the sum unchanged. A k-block is 32 hidden elements (four
lanes' chunks, two k-steps); two k-blocks are in flight a warp. A chunk past the hidden size (which the host requires
to be a multiple of 8) loads nothing and contributes zeros, as does a column past the job's or a token past the call's.

THE K-LOOP IS A BURST (`common/burst.cuh`, KERNEL_STANDARDS §23): a k-block's chunks are one operand set, the next
set's loads issue before the current set's MMAs, and nothing data-dependent sits inside. What varies is decided once
a block, outside it, by `csrc/rola/src/common/structure_switch.cuh`'s `uniform_switch`: the warps' share (below) and the tile's LIVE TOKEN GROUPS (1, 2 or 4; three
round up to four, which only a call's last tile can be) -- nine bodies. So tokens not present cost no MMA: at one
decoded token the projection is the weight stream plus one group's MMAs; at prefill every group is live and each
weight chunk feeds 32 dot products.

<a id="tiling"></a>THE WARPS' SHARE (`tiling`) is a function of the job's column count alone: `per_warp` column tiles
a warp (1, 2 or 4), `groups` warps across the columns, and, where the columns alone would leave warps idle (at most four groups), `slices` warps
across the hidden axis. A sliced job's warps write their partials past the job's columns in each token row
(`kPartialBase`), and one pass sums them in slice order -- the same logits on every run.

The bias is read once per column a lane owns, by a predicated load: the element stores are predicated, never a branch
around a load (measured at 2050 of 5300 cycles of one k-block's projection before).

MEASURED (sm_86 RTX 3080 Ti, 816 GB/s copy; widths (256, 256) union, hidden 256, d_v 64, other components stubs, a
CUDA graph of launches): one SM streams at most ~32 B/clk (a stand-alone probe: 8 rows x 64 B a warp load or 512
contiguous bytes, the latter ~15 % faster), and 36 SMs at most ~700 GB/s (each SM's outstanding misses cap its share),
so at one token the projection is the weight stream at the design's CTA count: H 16 16.0 us (8.9 MB), H 32 25.9 us;
H 4 is one SM's stream of its 256 KB job, 9.9 us. At T = 4096 (H 16) 964 us, 38 TFLOP/s: the 32-token tile's MMAs
against a 68 TFLOP/s peak, the stub epilogues included.

## <a id="route-row"></a>A token's row

One warp a token row: the solve on the row, then the bundle's form of it -- the row against its max as u8 digits
(depth at most two) or f16, the fp32 max as the row's scale, and the row's mass for the finisher. SKELETON: zero
digits, scale and mass.

## <a id="finish"></a>The head's finish

The route jobs of one (token tile, head) each write their masses, fence, and add one to the pair's arrival counter;
the last to arrive forms the head's `read_mass` (the product of the read masses whose fold applies) and `g_write`
(the gain logit's softplus times the write masses), and resets the counter, so no launch zeroes it between calls.
SKELETON: unit read mass and gain.

## <a id="value-rows"></a>The value digits

Per token and head: the channel max as the scale, and each channel's digits against it -- s8 alone, or the s16's s8
high byte and u8 low byte. SKELETON: zeros.
