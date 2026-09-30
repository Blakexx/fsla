# `csrc/rola/src/produce/produce.cuh` + `produce.cu` — the producer, one launch

RULED 2026-09-29: the producer is a declared configuration compiled into ONE kernel launch -- from a stretch of
tokens to the bundle the ops take: every routing level's projection, solve and unit-max rows, the write gain, and the
value digits. The layer declares its routing (`rola.Routing`), owns the weights, and compiles a producer plan once
(`rola/ops/produce.py`); `job_table` turns the plan into the rows this kernel reads, so every width, kind and offset
is a runtime field and the family is ONE instantiation.

This file is the SKELETON: each component below has its real signature, its shared-memory tenant and its edges, and
does the cheapest thing that keeps the composition's shape (zero digits and scales, unit gains). The components land
one at a time into it, each gated against the reference producer (KERNEL_STANDARDS §19).
`tests/integration/test_produce_kernel.py` holds the skeleton's claim: every element of every output is written,
whatever the routing, and the arrival counters are back at zero after the call.

## <a id="jobs"></a>The job table

One int32 row per job (`JobField`): the kind, the head (the first of a value job's heads, and their count), the
solve (`SolveKind`), the level's width, the `route_W` columns of its one or two logit rows, the packed-plane columns
its rows land at on each side (`-1` for a side it does not write), the level, and whether each side's fold applies.
A ROUTE job is one head's one solve: a union level's two logit rows (read and write), a tied level's one row filling
both planes, or one side of an independent level. A VALUE job is the heads whose channels fit one job (`256 / d_v`
of them, at most `kMaxColumns` channels). The grid is jobs by token tiles of `kTokenTile` tokens.

## <a id="accumulate"></a>The projection

The job's columns for the tile's tokens, fp32, into the block's `logits` tile: for a route job the solve's logit
rows, for a value job its heads' channels. SKELETON: zeros.

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
