# `csrc/rola/src/intra/intra_kernel.cuh` — the intra kernel

Mirror doc for the kernel that computes every pair of tokens inside one window, over the
bundle as the producer wrote it: the Gram of each level's u8 digits by int8 tensor ops, the
value product in u8 weights against the s8 values, the rows reduced into the output planes the
carry reduces into beside it. The carry's readout sees the state as of a window's start, so
the within-window pairs are this kernel's and only this kernel's. Its constants and operand
block are [`intra.md`](intra.md), its host seam [`intra_api.md`](intra_api.md); the declared arithmetic it is graded against is
`intra_bundle_reference` in `tests/oracle/int_reference.py`, itself graded against the fp64
intra of the bundle's dequantized levels; `tests/integration/test_intra_kernel.py` holds the
kernel to it per slot.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B` = a level's
width in digits (the built arm: `D = 2`, `B = 256`); `L` = tokens; `BH` = batch × heads;
**window** = the carry's 512 tokens; **tile** = 64 consecutive tokens, eight to a window;
**block** = two tiles, the 128 READERS a CTA owns; **stage** = 128 digits of one level for one
writer tile; **slab** = 32 digits of one level, one IMMA k-step; **IMMA** = one
`mma.sync.m16n8k32` on 8-bit operands, 4096 multiply-adds.

## What it computes

For a window, a reader `t` and a writer `t' <= t` in it:

    G_l[t, t']  = Σ_d r_l[t, d] · w_l[t', d]                 level l's digits, exact s32
    A[t, t']    = f(t') · Π_l G_l[t, t'],    f(t') = g(t') · Π_l sw_l(t') / 255^(2D)
    den[t]     += Σ_t' A[t, t']
    num[t]     += Σ_t' A[t, t'] · s(t') · q(t') / LV       q the value digits, LV their levels

`r_l`, `w_l` are the rows' u8 digits (a row's unit value times 255), `sw_l` the write levels'
scales, `g` the write gain, `s` the value scale. The READER's factor `Π_l sr_l(t)` is not
applied: it is per token and cancels in `num / den`, the output contract the carry keeps too,
and the readout divides by `den + READOUT_EPS / r(t)` so the defined readout is kept exactly
(`rola.ops.prefill.readout`). `num` takes one declared rounding, the value product's weights
([below](#value-product)); `den` takes none but fp32's.

<a id="arms"></a>

## The arm

`Arm<D, B>` derives everything from the two template axes: a reader row is `D · B` bytes,
a level `B / 128` stages, a tile `D · B / 128` stages (four at the built arm), `D · B / 32`
slabs (sixteen), and `kUnit = 255^(-2D)`, the integer Gram's scale. The built set is one
macro in `intra.cu` (`ROLA_INTRA_ARMS_X`, `(2, 256)`), so the dispatch and the arm list read
the same rows. The rows at depth above two are f16 (`produce/produce.md#bundle`); their arms —
the Gram as f16 HMMA on the narrow levels — are not built, and the seam refuses them by name.

<a id="the-grid"></a>

## The grid: a CTA a block, the heaviest first

A CTA is one (batch-head, window, block): `blockIdx.y` the batch-head, `blockIdx.x` the window
times four plus the block, heaviest first. Block `b`'s readers pair with writer tiles `0 ..
2b + 1`, so the window's four blocks carry 2, 4, 6 and 8 tiles; launching the eight-tile block
first lets the short ones fill in behind it. Eight warps a CTA, each owning 16 readers: warps
0–3 the block's first tile, 4–7 its second. A warp's rows are its own for the whole sweep, so
its output is final in registers at the end and nothing is exchanged between warps. At ~100 KB
of shared memory and 210 registers a thread, one CTA runs an SM.

<a id="smem"></a>

## Shared memory

| tenant | bytes | holds |
|---|---|---|
| `reader` | `128 · D · B` = 64 KB | the block's reader rows, loaded once, every level |
| `stage[3]` | 3 × 8 KB | three writer stages in flight |
| `v_hat`, `v_lo` | 2 × 4 KB | the writer tile's value digits |
| `column` | `(2 + D) · 64 · 4` | the writer tile's gain, value scale and write scales |
| `den` | 512 | a warp's 16 row sums, added across tiles |
| `gate` | 64 | the slab gate's table |
| barriers, clock | ~0.5 KB | full/empty a stage buffer, one a reader part, full/empty the values |

The reader region is dead after the sweep and holds the deposit's staging tile.

<a id="swizzle"></a>

## The swizzle

A row is 16-byte chunks; chunk `c` of row `r` sits at `c ^ (r & 7)` (`swizzle8`), so the eight
rows an `ldmatrix` reads at one chunk lie in eight bank groups, and a copy's consecutive
threads along one token's run still write whole chunks. The value rows are 64 bytes — four
chunks — and pair rows the same way (`swizzle4`: `c ^ ((r >> 1) & 3)`), because `ldmatrix.trans`
reads eight consecutive tokens at one chunk.

<a id="copies"></a>

## The copies

Every copy is `cp.async` of one 16-byte chunk, consecutive threads along a token's contiguous
run, arriving on its buffer's mbarrier with `cp.async.mbarrier.arrive.noinc` (each of the
256 threads is one counted arrival). A thread computes a copy's address from the CTA's
(batch, head, window) and its own place at the copy (`Copier`), a few multiply-adds, rather
than holding addresses in registers across the loop. The rows are read where the bundle holds
them, by their strides, on 16-byte boundaries (the seam refuses others); the producer's rows
are head-major, so a CTA's tokens are one contiguous run a row.

<a id="stages"></a>
<a id="pipeline"></a>

## The pipeline

The reader rows land once, in four parts of one stage's digits each, each on its own
mbarrier: part `S` leaves beside tile 0's stage `S`, so the first Gram waits for one part and
one stage, not for 64 KB. The writer side streams: a tile's four stages through three
buffers, each stage copied two stages ahead of its use (`kStageBuffers - 1`) and pulled into
L2 two stages beyond that (`prefetch_l2`). A buffer's `full` barrier counts the 256 copies'
arrivals; its `empty` barrier counts the eight warps' releases, and the copy that refills a
buffer waits on it. The tile's value digits and writer columns follow the same protocol on
one buffer (`values_full`, `values_empty`), issued at a tile's first stage and read at its
last. A warp waits only for the data it reads and releases each buffer right after its last
read of it; no CTA barrier sits inside the sweep.

<a id="slab-gate"></a>

## The slab gate

A tile pair whose product is exactly zero is not computed, and a slab whose digits meet in
no token pair is not multiplied. The certificate is the facts pass's TILE UNIONS
([`facts/liveness_contract.md`](../facts/liveness_contract.md#tile-unions)): per side,
32-digit group and 64-token tile, the digits live anywhere in the tile. Warp 0 reads, a lane a
slab, the reader side's two unions for the block and the writer side's eight for the window
(40 bytes a lane) and builds the table `gate[tile][row tile]`: bit `s` set when slab `s`'s
reader and writer unions meet. A level with no bit set makes the pair's product zero, so the
whole word clears, and a pair above the diagonal is zero too. A side's dense level (the
topology's declaration, the statics the facts pass took) answers all-ones and is not read.

The table drives everything else, uniformly across the CTA:

* a stage is copied only when either row tile reads one of its slabs, and the tile's values
  only when either row tile's pair is live; an uncopied stage's arrival is immediate;
* a warp waits on a stage only when it reads one of its slabs, and computes the tile only when
  its pair is live; a stage with every slab live runs as one straight-line block, a stage with
  some as a branch a k-step, and a level's first step gated off zeroes the accumulator in its
  place;
* tile 0's copies leave before the table exists, so they are copied whatever it says and
  every warp awaits them: a copy that leaves is always awaited by a warp before its buffer is
  released, so none is in flight when a buffer refills.

The gate cannot move a number: a skipped slab's products are all zero, and a skipped pair adds
exact zeros to `num` and `den`. `test_the_slab_gate_is_an_exact_certificate` holds the gated
run to the ungated run's bits on a clustered draw where the gate skips pairs and slabs.

<a id="gram"></a>

## The Gram

A stage is four k-steps (`gram_step`); a k-step is the warp's 16 reader rows against the
stage's 64 writer tokens over one slab: one `ldmatrix.x4` of the reader rows, four of the
stage, eight IMMAs into the level's `int32[8][4]` accumulator. Both operands are token-major
rows read by plain `ldmatrix` — no transpose anywhere. A level's first k-step starts its
accumulator from zero (`imma_u8_k32_fresh`), so the accumulator is dead between levels. At a
level's last stage the s32 sums convert to fp32 — exactly, since a level's sum is at most
`255² · 256 < 2^24` — and multiply into the running level product (`fold_level`).

<a id="epilogue"></a>

## The epilogue

At the tile's last stage the level product becomes `A`: each lane reads its two writer columns'
factors (`f = g · Π sw_l · kUnit` and the value scale `s`) from shared memory and the quad's
columns reach the lanes that hold them by shuffles; on the diagonal tile a column above its
row is zero. The row sums go to `den` (two quad shuffles, one shared add a row) before
anything is quantized, and `A · s` — the value product's `P` — replaces the accumulator.

<a id="value-product"></a>

## The value product

`P` against each row's maximum over the tile (the quad's two shuffles), as the declared digits:
`a = min(rint(P · LW / m), LW)`, `LW` = 65535 at two weight digits (the default) or 255 at
one. The two digit bytes are packed by `prmt` from the accumulator's own registers into the
IMMA A fragment — the accumulator's layout IS the A fragment's under a fixed permutation of
k, lane `(g, tq)` holding tokens `{2tq, 2tq + 1, 8 + 2tq, 9 + 2tq}` of each n-tile — and the
values are loaded to match it: `ldmatrix.trans` of the token-major value rows and one `prmt` a
register (`0x6420` the even bytes, `0x7531` the odd). Per 16-channel group, the digit products
accumulate in s32 by their weight — hi × `v_hat`, (lo × `v_hat`, hi × `v_lo`), lo × `v_lo`, two
weight digits by two value digits — and one scaled fp32 add a group (`unit = m / (LW · LV)`)
lands them in the output accumulator. The four channel groups are a real branch apart
(`rola::uniform_switch`): unrolled together, the compiler hoisted every group's value loads
over the others and spilled.

A weight moves by at most half its unit, so a slot's error is bounded by `m / (2 LW) · Σ |q| /
LV` a tile, the reference's `bound`; the kernel's weights are within one unit of the exact
ones, and the gate allows three bounds plus fp32 accumulation against the slot's envelope.

<a id="deposit"></a>

## The deposit

After the sweep a CTA barrier frees the reader region, and each warp stages its 16 rows × 64
channels there (a 68-float pitch, so the accumulator's pair stores are bank free), then
reduces each row into `num` as two 128-byte `red.global.add.f32` runs, lanes on consecutive
channels, and its 16 row sums into `den`. A RED, not a store: the carry reduces into the same
planes on the same stream, and the caller owns zeroing them.

<a id="phase-ledger"></a>

## The phase ledger

`rola_intra_ledger_bind` binds a `[8][7]` int64 buffer (warp × `reader`, `gate`, `wait`,
`gram`, `epilogue`, `value`, `deposit`) every CTA's warps add their cycles into
(`rola.ops.intra.ledger_bind`); a bound ledger moves no output. A phase is booked at the lap
that ends it, so a producer's wait on an `empty` barrier lands in whichever phase laps next.

## The budget line and where the time goes

The line (ruled 2026-09-30, on the int8 rebuild's plan card), per head at `(256, 256)`, `d_v` 64, `L = 64K`:
the Gram 2.36M IMMAs (~68 µs at sm_86's 16 cycles an IMMA a scheduler), the value product half
that (~34 µs), the bytes ~75 MB (~94 µs); floor ~115 µs, budget 1.3× (~150 µs). The line left
out the output's read-modify-write (`num` and `den` reduced, ~34 MB a head), which puts the
byte floor at ~139 µs and the budget at ~180 µs.

Measured (sm_86, CUDA events at boost clocks; `(256, 256)`):

| cell | measured | against |
|---|---|---|
| `L = 64K`, 16 batch-heads, a head (every digit live) | 276 µs | ~180 µs budget (1.5×) |
| `nl64k-dense` (one sequence) | 286 µs | |
| `nl64k-alt-k4` | 294 µs | |
| `flagship-dense` (`L = 1K`) | 65 µs | |
| `flagship-alt-k4` | 64 µs | |
| `flagship-cohort-k16` | 49 µs | ~5 µs |

The ledger at `L = 64K`, 16 batch-heads: data waits 27%, Gram 27.5%, value product 20%,
deposit 11%, epilogue 7%, reader 4%, gate 3.5% of the warps' cycles. The Gram runs at ~1.2×
its tensor floor and the value product at ~1.8× its own; one CTA an SM leaves nothing to fill
the waits.

At `L = 1K` the grid is eight CTAs, and the time is one CTA's: the heaviest block issues ~11.5K
IMMAs on one SM, a ~24 µs tensor floor, and runs 64 µs. On `flagship-cohort-k16` every tile
pair is live and the gate keeps 2.5 of 16 slabs a pair, so the Gram's IMMAs fall to a sixth
and the time only to 49 µs: the line's ~5 µs assumed the gate would skip most of a cohort draw's
work, and the value product, the epilogue and the chain of stages stay. Reaching it takes the
heaviest block's sweep split across SMs — the output's REDs allow a reader row to take its
pairs from several CTAs — at a price in reader bytes the large-`L` regime pays.

The gate's own cost, from the ledger with the unions read and with every level declared dense
(nothing read): 1.1% for the prologue's barrier, 2.5% for the unions' round trip, which tile
0's stages could hide.

## Other architectures

On sm_80 the form is the same, and the IMMA issues every 8 cycles, so the byte floor binds
harder. On sm_90 the writer stages move by TMA boxes and the Gram and value product are
`wgmma` from shared memory (u8 × s8 at k 32), the reader side resident as now; the
warp-uniform gate carries over unchanged. sm_100's `tcgen05` takes the same shared-memory
operands with the accumulator in tensor memory.

## Where the mechanism came from

The bf16 intra it replaces (deleted 2026-09-30, [`DELETIONS.md`](../DELETIONS.md)) walked
the same 128-reader blocks and lower-triangular tiles, gated slabs on the producer's support
words, and deposited through a staging tile by REDs; those stay. What is new is the input —
the bundle's u8 rows, half the frame's bf16 bytes a digit — which lets the whole reader block
sit in shared memory once instead of being re-staged for every column tile (the frame read
3.6× its ideal bytes), the int8 Gram exact in s32, the mbarrier pipeline, the value product in
u8 weights with the accumulator reused as its operand, and the gate's certificate from the
facts pass's tile unions rather than a word per token.
