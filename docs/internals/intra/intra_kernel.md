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

Every copy is `cp.async` of one 16-byte chunk, arriving on its buffer's mbarrier with
`cp.async.mbarrier.arrive.noinc` when it lands. ONE WARP issues a whole stage (16 copies a
lane) or a tile's values (8 value-digit copies a lane a plane, 8 writer-column entries), so a
buffer's `full` barrier counts that warp's 32 lanes. A lane's place in a copy never changes — a
stage's rows `lane / 8 + 4 i` at chunk `lane % 8`, a value plane's rows `lane / 4 + 8 i` at chunk
`lane % 4` — so `Copier` computes each 64-bit base once per CTA, and a copy adds its tile's rows
with one unsigned 32 × 32 → 64 multiply-add (`rows_of`; the seam bounds a token stride to 31
bits, and the two value planes share one layout). A stage row `j + 4 i`'s swizzled chunk is
`(lane % 8) ^ j ^ 4 (i & 1)`, one address bit flipped between even and odd `i`. Recomputing the
(batch, token, head) address at every copy, the first form, cost 512 instructions a warp tile
pair against the 80 its budget allows (the [region ledger](#region-ledger)). The rows are read
where the bundle holds them, by their strides, on 16-byte boundaries; the producer's rows are
head-major, so a CTA's tokens are one contiguous run a row.

<a id="stages"></a>
<a id="pipeline"></a>

## The pipeline

The reader rows land once, in four parts of one stage's digits each, each on its own
mbarrier: part `S` leaves beside tile 0's stage `S`, so the first Gram waits for one part and
one stage, not for 64 KB. The writer side streams a tile's four stages through three buffers,
each stage copied ONE stage ahead of its use (`kCopyAhead`), which leaves two stages of slack
between the warps: a buffer is refilled once all eight warps have released its use three
stages back (`empty`, eight arrivals).

**The first warp to reach a refill issues it.** Each warp, at a stage's start, claims the
next stage's copy with an `atomicMax` of its absolute index on the buffer's claim word; the
warp that raises it waits alone for the buffer's release and issues the copy. The tile's
values move the other way: **the last warp to release a tile's values copies the next
tile's** (an arrival counter a tile parity), so nobody waits on the value buffer at all, and
the values have the next tile's four stages to land before they are read.

**Every warp waits for every stage and every tile's values, read or not.** An mbarrier wait
tests one phase bit, so it is sound only for a waiter within a phase of the barrier. With the
copies claimed, a warp that neither reads nor claims waits on nothing; one that skipped its
waits (the block's first row tile through its last tile, above its diagonal) ran ahead, claimed
a refill, and read a release two phases old as done, overwriting a stage another warp was
still reading — intermittently wrong rows, found by repeated launches
(`test_the_intra_is_bit_identical_across_launches`, which fails on that form). Waiting on every
phase keeps every warp within a phase of every barrier; a stage no warp reads was not copied,
and its wait is immediate.

Why this shape, measured (ncu, base clock; `L = 64K` at four heads, `flagship-dense` at `L =
1K`):

| form | 64K, 4 heads | flagship-dense |
|---|---|---|
| every thread copies its share of a stage two ahead; the tile's values tested, then awaited | 1.37 ms | 84 µs |
| the same, copied one ahead (two stages of slack) | 1.24 ms | 75 µs |
| one warp copies: the first to a stage's refill, the last to release a tile's values | 1.16 ms | 72 µs |

The shares form bound every warp's data to the slowest warp's copy share, and every thread
waited on the buffer's release before copying, which locked the warps together a stage at a
time. The waits were not memory latency: 64 heads that alias one head's data (head stride 0,
every CTA reading the same L2-resident 4.5 MB) ran exactly as fast as 64 distinct heads.

An L2 prefetch of each stage two stages beyond its copy was measured and removed: it lost in
every regime (the 4-head cell 1.42 against 1.37 ms, `flagship-dense` 87.5 against 83.8 µs), and
with the value product and epilogue skipped the Gram's pipeline was no faster at six or ten
stages ahead than at two.

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
* a warp computes the tile only when its pair is live, and multiplies only the slabs it reads
  (every warp still waits for every stage, [above](#pipeline)); a stage with every slab live
  runs as one straight-line block, a stage with some as a branch a k-step, and a level's first
  step gated off zeroes the accumulator in its place;
* tile 0's copies leave before the table exists, so they are copied whatever it says.

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
`a = rint(P · LW / m)`, `LW` = 65535 at two weight digits (the default) or 255 at one — no
clamp is needed, since `P ≤ m` and `m · (LW / m)` rounds within an ulp of `LW`. The two digit bytes are packed by `prmt` from the accumulator's own registers into the
IMMA A fragment — the accumulator's layout IS the A fragment's under a fixed permutation of
k, lane `(g, tq)` holding tokens `{2tq, 2tq + 1, 8 + 2tq, 9 + 2tq}` of each n-tile — and the
values are loaded to match it: `ldmatrix.trans` of the token-major value rows and one `prmt` a
register (`0x6420` the even bytes, `0x7531` the odd). Per 16-channel group, the digit products
accumulate in s32 by their weight — hi × `v_hat`, (lo × `v_hat`, hi × `v_lo`), lo × `v_lo`, two
weight digits by two value digits. Below hi × `v_hat`'s weight the sum is exact in s32 (at most
1.6e9 over a tile), so it is formed by one integer multiply-add and converted once; at one
digit either way the whole sum is. One scaled fp32 add a group (`unit = m / (LW · LV)`) lands
it in the output accumulator. The four channel groups are a real branch apart
(`rola::uniform_switch`): unrolled together, the compiler hoisted every group's value loads
over the others and spilled.

A weight moves by at most half its unit, so a slot's error is bounded by `m / (2 LW) · Σ |q| /
LV` a tile, the reference's `bound`; the kernel's weights are within one unit of the exact
ones, and the gate allows three bounds plus fp32 accumulation against the slot's envelope.

<a id="deposit"></a>

## The deposit

After its last tile each warp stages its 16 rows × 64 channels in its own 16 reader rows —
which no other warp reads, so no CTA barrier is needed and a warp of the block's first row
tile deposits while the second row tile's warps finish their last tile — at a 68-float pitch
(the accumulator's pair stores are bank free), then reduces each row into `num` as two 128-byte
`red.global.add.f32` runs, lanes on consecutive channels, and its 16 row sums into `den`. A RED, not a store: the carry reduces into the same
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

Measured (sm_86, `(256, 256)`): the 16-head cell by CUDA events at boost clocks; the rest by ncu
at its locked base clock, since one sequence's events swing ±10% between runs of one binary.

| cell | at the kernel's landing | now | against |
|---|---|---|---|
| `L = 64K`, 16 batch-heads, a head (events) | 276 µs | 211 µs | ~180 µs budget (1.17×) |
| `L = 64K`, 4 batch-heads (ncu, base clock) | 1.48 ms | 1.16 ms | |
| `nl64k-dense` (ncu) | | 320 µs | |
| `nl64k-alt-k4` (ncu) | | 332 µs | |
| `flagship-dense`, `L = 1K` (ncu) | | 72 µs | |
| `flagship-alt-k4` (ncu) | | 73 µs | |
| `flagship-cohort-k16` (ncu) | | 61 µs | ~5 µs at boost |

<a id="region-ledger"></a>

**The region ledger** (`tools/region_ledger.py` against `tools/budgets/intra.json`: ncu's
per-instruction counts and stall samples attributed to the component each instruction inlines
from, per warp tile pair) is the instrument for the instruction budget. At the kernel's landing
it read 2,450 instructions a unit against a design count of ~1,200: the copies 512 (budget 80),
the value product 724 (520), the stage loop's own control 418, the Gram 315 (288), the
epilogue 219 (180). After the copy rewrite: 2,109 — the copies 292, the value product 629, the control 439, the
Gram 313, the epilogue 219, the barrier spin 85. With one warp copying ([the pipeline](#pipeline)): 2,165 — the
copies 193, the value product 631, the control 458, the Gram 303, the epilogue 218, and the
spin 226 (every warp now waits on every stage).

**Where the time goes.** Neither DRAM (34–49% of peak) nor L2 (44%) nor the tensor pipe
(35% at landing) is saturated; the warps wait on each other and on latency, ~6.7 cycles an
issued instruction with two warps a scheduler. Two experiments at landing bounded it: with the
value product skipped the kernel ran ~200 µs a head (events, 16 heads), and with the epilogue
skipped too ~170 µs — the Gram's pipeline alone, 2.7× its ~62 µs tensor floor. That was not
memory: the aliased-heads run above changed nothing. It was the coupling of the copy protocol
([the pipeline](#pipeline)), whose two fixes took the 4-head cell from 1.37 to 1.16 ms. What
remains is the lockstep of the warps' phases: their epilogues and value products coincide and
leave the tensor pipe idle together, and the value product costs about twice the time of its
IMMAs. The phase ledger now: Gram 28%, value product 26%, waits 25%, epilogue 9%, gate 6%,
deposit 3%, reader 3% of the warps' cycles.

At `L = 1K` the grid is eight CTAs, and the time is one CTA's: the heaviest block issues ~11.5K
IMMAs on one SM, a ~24 µs tensor floor at boost. On `flagship-cohort-k16` every tile pair is live
and the gate keeps 2.5 of 16 slabs a pair, so the Gram's IMMAs fall to a sixth and the time only
by a fifth: the line's ~5 µs assumed the gate would skip most of a cohort draw's work, and the
value product, the epilogue and the chain of stages stay. Reaching it takes the heaviest
block's sweep split across SMs — the output's REDs allow a reader row to take its pairs from
several CTAs — at a price in reader bytes the large-`L` regime pays.

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
