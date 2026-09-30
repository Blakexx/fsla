# `csrc/rola/src/produce/produce.cuh` + `produce.cu` — the producer, one launch

RULED 2026-09-29: the producer is a declared configuration compiled into ONE kernel launch -- from a stretch of
tokens to the bundle the ops take: every routing level's projection, solve and unit-max rows, the write gain, and the
value digits. The layer declares its routing (`rola.Routing`), owns the weights, and compiles a producer plan once
(`rola/ops/produce.py`); `job_table` turns the plan into the rows this kernel reads, so every width, kind and offset
is a runtime field and the family is ONE instantiation.

`rola.ops.produce.produce` runs this kernel on CUDA when no gradient flows (`run_kernel` allocates the outputs;
`_kernel_bundle` views its packed planes as the bundle's per-level rows and scales); the torch composition behind the
same call is the fp64 and host reference, and until the native backward the path a gradient takes.

The components land one at a time into the skeleton, each gated (KERNEL_STANDARDS §19); a component not yet landed
has its real signature, tenant and edges and does the cheapest thing that keeps the composition's shape (zero digits
and scales, unit gains). LANDED: every component -- the projection, the solve, the bundle, the values. `tests/integration/test_produce_kernel.py` holds the skeleton's
claim (every element of every output is written, whatever the routing, and the arrival counters are back at zero
after the call) and each landed component's gate.

## <a id="jobs"></a>The job table

One int32 row per job (`JobField`), its ADDRESSING: the head (the first of a value job's heads, and their count), the
level's width, the `route_W` columns of its one or two logit rows, the packed-plane columns its rows land at on each
side (`-1` for a side it does not write), the level, and whether each side's fold applies. Its CONTROL -- the kind, the
solve (`SolveKind`), the solve row's elements a lane, the sides it writes and whose fold applies -- is one SHAPE WORD
(int16) a job in the launch's parameters
(`ProduceParams::shapes`, `rola.ops.produce._shape`), read by block index: a value ptxas knows is warp-uniform, so a
branch on it compiles its collectives inline alone, where a branch on a loaded job field gave each a guarded fallback
subroutine as well (692 of them: cold code, KERNEL_STANDARDS §20). At most `kMaxJobs` jobs a launch.
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

One warp a token row: the solve on the row, then the bundle's form of it. Every warp makes the same passes over the
tile's rows (a count of its live tokens, uniform), a row past them solving and reducing zeros with its stores off, so
no collective sits under a branch ptxas cannot prove warp-uniform.

<a id="bundle"></a>THE BUNDLE'S FORM (`write_side`), for each side the job writes (a shape bit each): the row against
its max as u8 digits (depth at most two, round to nearest even of `unit x 255`) or f16 units, the max clamped at
`1e-30` as `quantize_unit_max_rows` clamps it; the max as the side's scale; and the level's mass for the finisher
where the fold applies (1 elsewhere, so the finisher multiplies every level without the plan's flags). A side reduces
only what the solve did not give: a one-activation row's max is its top element's own value (`top`: `(-tau/2)^2`,
`-tau` or `1 / sum`, the same expressions the elements are), so only a union's sides reduce their max, and only a
folded side its mass -- the fold bits ride the job's shape word, so the choice is warp-uniform. The WRITE fold
is applied -- its scale is the max over the mass, the stored level on the simplex -- and the READ fold is carried:
its scale stays the max and its mass rides `read_mass`, as the composition's `_fold` does. Gate: every row's digits
(within one), f16 units (within one unit in the last place at 1), scales (within twice the solve's bound), the read
mass and the write gain (within 2^-12 relative, measured 4.8e-5 and 2.7e-6) against the fp64 composition on the same
bf16 operands, with and without a gain column or a routing bias.

EVERY OUTPUT IS STORED HEAD-MAJOR, `[B, H, T, ...]` (`slot_of`: the flat token split into its batch and step), and the
bundle's `[B, T, H, ...]` tensors are views of it. At one token the two orders are the same bytes; over a prefill a
head's token rows are contiguous, and every prefill reader walks one head's tokens -- the facts pass measured 212
us a head at `(256, 256)`, `L = 64K`, `H = 16` on token-major rows (an 8 KB token stride, DRAM at 360 GB/s) against
113 us head-major (676 GB/s), the rate of a single head (`docs/internals/facts/facts.md#the-grid`).

<a id="solve"></a>THE SOLVE (`solve_row`) holds the row striped, element `lane + 32 i` in a lane's `i`th register,
`IPT` = the width's power-of-two class over 32 (one of four bodies, chosen once a job by its shape word). A union row
is the midpoint of its read and write logits (entmax on it) and their overflow-free difference (the split); a tied or
independent row is its one logit row. Softmax, entmax 1.5 and 2 and both unions are the closed registry's kerneled
activations, with the reference `rola/routing/entmax/production.py`'s `_torch_union_split` and `_torch_activation`.
The union's split reuses the entmax family's `union_write` (`docs/internals/entmax/entmax.md#union-write`) with its
fast policy, and the read side is the value times the sigmoid, both on the special-function unit: the fp32 levels
feed u8 or f16 digits, a thousand times coarser than those units' error. Gate: every route job's sides against the
fp64 reference solve on the kernel's own logits, per slot within `(width + 16) 2^-23` (measured at most 0.37 of it,
the 1.5-union's write side).

<a id="threshold"></a>THE THRESHOLD IS NEWTON FROM THE LEFT (`newton_threshold`), not a sort. With the row shifted by
its max, `f(tau) = sum_i g(z_i - tau) - 1` (`g(d) = max(d, 0)^2 / 4` at alpha 1.5, `max(d, 0)` at 2) is convex and
decreasing, and `f(-2) >= 0` (`f(-1)` at alpha 2): Newton from there never passes the root, and each step needs only
the count, sum and sum of squares of the support `{z > lo}`. The closed form on that support -- the formula the sorted
solve evaluates at its chosen rank -- IS the root once no member lies below it, which a vote decides (a vote is
warp-uniform to ptxas, so the loop's exit outlines nothing). At alpha 2 the steps are Michelot's iteration. Measured
in fp32 over Gaussian, uniform, tied and Cauchy rows of widths 16 to 256 at scales 0.3 to 30: at most 8 steps, 1 to 5
typically, within 2.2e-6 of the fp64 sorted threshold (8e-8 at alpha 2); `kNewtonSteps` caps it at 16, and a capped
result is still the closed form on a support whose extra members lie within the last step of the root. WHY NOT THE
SORT: at one decoded token a CTA has one live row, so its warp's whole solve is the kernel's latency, and the sort's
network is ~36 dependent shuffle stages of long straight-line code executed once from a cold instruction cache (~30
cycles an instruction there, ncu's no-instruction stall the top one); the sorted threshold measured 13.8K cycles a
256-wide row, twelve-round multisection 10K, Newton 2.7K. The sort's bitwise tie-order independence (its sums run in
sorted order) does not carry over -- these sum in lane order, deterministic run to run; the gate is per slot against
fp64.

<a id="phase-ledger"></a>THE PHASE LEDGER (`rola.ops.produce.ledger_bind`, `[jobs, 8 warps, 9 phases]` int64 cycles,
booked over the tiles): job, project, barrier, row load, threshold, levels, bundle, finish, values -- a warp's cycles
between the kernel's laps (`common/phase_clock.cuh`). MEASURED (the same card, one launch, widths (256, 256), a route
job's warp 0 a tile): at one token H 16, project 27.7K (the weight stream under 36 CTAs), row load 3.6K, threshold
2.7K, levels 5.9K, bundle 2.4K (the skeleton's stores), finish 1.3K; at T = 4096, a tile of 32 rows, project 24.2K,
row load 3.9K, threshold 4.3K, levels 10.4K, bundle 5.3K. The row load and the levels are straight-line code (cold
at one token), and at prefill the solve's special-function and latency chains sit beside the projection without
overlapping it (one CTA an SM): the next levers, priced when the bundle's form lands.

## <a id="finish"></a>The head's finish

The route jobs of one (token tile, head) each write their masses, fence, and add one to the pair's arrival counter;
the last to arrive -- a vote on the count ends the others, so the collectives past it are warp-uniform -- reads the
masses coherently (`__ldcg`) and forms the head's `read_mass`, the product of its levels' read masses, and `g_write`,
the write gain's softplus (`softplus` as torch's, linear past 20) times the write masses; then it resets the counter,
so no launch zeroes it between calls. The gain's logit (`gain_logit`) is the routing's trailing gain column against
the token's row, a warp's lanes over the hidden axis in 16-byte chunks, plus the column's routing bias and
`gain_bias`; a routing without a gain column has a gain of 1.

## <a id="value-rows"></a>The value digits

A warp a (token, head), its lanes over the channels: the channel max |v| (clamped at `1e-30`) as the scale, and each
channel's digits against it at the digit count's full scale (127, or 32767 split into the s16's s8 high byte and u8
low byte), round to nearest even -- `rola.routing.quant.quantize_values`, on the fp32 projection rather than the bf16
the composition's linear rounds it to. Gate: every digit within one plus the projection's fp32 error at full scale,
every scale within that error, against the fp64 composition (measured: at most one digit, on 0.1-0.2 % of the
channels; scales within 2.4e-7 relative).
