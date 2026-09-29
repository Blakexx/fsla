# `prepass/prepass_kernel.cuh` — the pre-pass kernel: the skeleton and its frame

Mirrors `csrc/rola/src/prepass/prepass_kernel.cuh`. The family overview is `prepass.md`.

THE FRAME IS REAL AND THE COMPONENTS ARE STUBS WITH THEIR EDGES (KERNEL_STANDARDS section 19).
Each component below lands against its own budget line on the queue card; until it does, its stub
touches its operands and produces zero, so what the skeleton measures is the frame.

<a id="smem"></a>
## The shared addresses

`Smem<TP>` names every region of the plan's ledger (`plan.md#ledger`) as a shared address:
the words a parity, the pool a slot, a warp's ring slot, the transfer's values and maxima, a warp's
panel row, and the barriers.

<a id="rows"></a>
## The region's staged rows

`StagedRows` lists the sparse levels' digit rows on each side, read side first, and names each
side's sparse level and its first staged row as scalars. Every index into it is compile-time
(`static_for` over the sides and the level bound), so the table is registers: the first build's
runtime-indexed table cost 360 bytes of local memory, the ptxas signature section 20 forbids.

<a id="words"></a>
## `words`

One warp stages the next window's words while the current one runs: the staged rows' sixteen
words each by cp.async, lane = word, arriving at the words barrier when landed.

<a id="lists"></a>
## `lists`

One lane a fold list: lane `k` sums the popcounts of its digit's sixteen words. The first form
ran sixteen serial shuffle reductions and cost 6.3K cycles a window; one lane a list costs 1.7K.
The read side's union over the warp's four sparse digits is ORed a round, lane = round, and
reduced once: the token-major loop's tile count.

<a id="fill"></a>
## `fill`

TWO HALVES, TWO CHUNKS OF LOADS IN FLIGHT. `fill_issue` loads chunk `c + 2`'s rows into registers
before chunk `c` is folded (chunk `c + 1`'s were loaded a chunk earlier and are landed after the
fold, so a load has two folds to arrive under): a lane one 16-byte piece of one token's write-sparse row (lane = (token `4 warp + (lane &
3)`, piece `lane >> 2`): a quad is four consecutive tokens at one piece, 64 contiguous bytes of a
line, the copy ranking's fast pattern) and, on warps 4-7, up to two 16-byte pieces of write-dense
rows (a span of 8 by 8-byte halves on warp 4). `fill_land` runs after the fold, when the loads have
landed: each word is a token's four digits, and `quad_transpose` (two shuffle rounds, two byte
selects; every lane takes part, a lane past the span stores nothing) leaves lane `i` with digit
`4 j + i`'s four tokens, stored as one word of that digit's row;
warp 0 forms the chunk's factors `P_g(t)`, `P_gs(t)` from the token's gain, value scale and write
maxima it loaded with the rows (`#zeta`), a lane a token, into the factor plane. Nothing in the fill
is an asynchronous copy any more; the slot's full barrier counts one release arrival a thread. THE
STUB'S REMAINDER: the compaction to the union-live tokens is the component's own step.

<a id="fold"></a>
## `fold`

The declared arithmetic is the reference's (`tests/oracle/int_reference.prepass_reference`): per window,
column `m` in {g, g s_t} and write-dense digit `d`, `V_m(t, d) = (P_m(t) a_wd[t, d] + 128) >> 8` with
`P_m(t) = rint(2^23 col_m m_ws m_wd / (255 zeta_m))`, `acc_m(s, d) = sum_t a_ws[t, s] V_m(t, d)` by u8
IMMA, the state gaining `acc_m zeta_m / (2^15 255)` at the window's end. The per-token factor sits on
the DENSE side: the sparse digits enter the MMA as the bundle's raw bytes.

<a id="tiles"></a>
A warp's 128 leaves are sixteen TILES of one list (a write-sparse digit) and eight write-dense digits:
`WarpRows` gives lane `j` its tile's list row, the list's region-relative digit and the tile's first
dense digit (the dump's map). THE FOLD'S SHAPE `FoldShape<Q>`
at a write-dense span of `8 Q`: `16 / Q` lists a warp ON THE SIXTEEN ROWS OF ONE m-TILE, A their raw
write-sparse words (no formation), `Q` n-tiles of eight write-dense digits, each formed into FOUR B
columns `[g_hi, g_lo, gs_hi, gs_lo]`; an accumulator an (n-tile, column). The shape is a TEMPLATE
PARAMETER OF THE KERNEL beside the depth, chosen on the host from the declared span (`prepass.cu`,
`R9`): one kernel holding every shape sat at 254 registers (the largest shape's accumulators live on
every path) and issued at a third of its rate.

<a id="zeta"></a>
`window_zeta` scans the window's tokens once a warp (a lane a token, sixteen rounds, every load issued
before any reduce) for the two window maxima and returns `kz_m = 2^23 / (255 zeta_m)`. The factors `P_m(t)` are formed by the fill (`#fill`), so the pool needs no per-token planes; the
lane's list digits (`list_digits`) are computed once a launch, and a chunk's list word is loaded a
chunk ahead so the union reduce never waits on its load.

`fold_chunk` runs the chunk's two k16 steps, a step skipped when the union of the warp's list words
has no live token in its sixteen. A lane's four tokens are CONSECUTIVE (`16 st + 4 kg + i`): ONE
WORD of every digit row in the digit-major pool. A step loads the two factor columns (two 16-byte
loads), the two list words that are its A registers as they are (rows `lane >> 2` and `+ 8`; a list
past the shape's count reads the zero row), and for each n-tile the dense digit's word, unpacked once
(four `prmt`) and formed into `P a + 128` for each column (four IMAD), the hi column packing byte 2 of
the products and the lo column byte 1 (three `prmt` each); then an IMMA a column -- four IMMAs, five
shared loads and ~35 instructions a step at Q = 1. A token not in a list has a zero byte there, so
nothing is gathered and nothing is padded: the fold's cost is the chunk's tokens, at every sparsity.

WHY THIS SHAPE: the first form ran a list's own k16 steps over gathered live tokens (the carry's form,
where 32 channels on M pay for the gathers); with the pre-pass's four digit rows a list the gathers
were twelve shared loads and a dependent chain an IMMA, two warps a scheduler could not hide it (58K
then 65K cycles a warp a window at alt-k4), and the chunked pool padded each list's step with its
one live token in sixteen. Lists on M over the token-major pool turned them into 40 byte loads a
step, still shared-pipe bound (30K); the digit-major pool made them ten word loads but left the
formation on the sparse side, 32 elements a lane a step (~120 instructions, 14K cycles: two warps a
scheduler make a warp's instruction count its time); the factor on the dense side forms 4 elements
a lane into four columns instead (~35 instructions a step).

`window_close` joins each column's lo accumulator to its hi in the same lane, times the column's
unit, into the fp32 state in shared memory (leaf = list x span + n-tile x 8 + n), and clears the
accumulators. `dump_state` writes a warp's state to the call's plane by canonical leaf
(level 0 slowest) when one is bound: the fold's gate reads it against the reference's mass and bound.

<a id="transfer"></a>
## `publish` and `rescale`

A fold warp publishes its leaves' columns as s16 against each leaf's max and the max as a float
into the transfer; after the publish barrier a read warp loads its four blocks, takes each
block's scale as the max of its 32 leaf maxima (one a lane, five shuffles) and keeps the hi bytes
in registers. THE STUBS write and reduce zeros.

<a id="readout"></a>
## `readout`

The ring runs a tile ahead: `readout_issue` stages tile `t + 1` (the dense sector of a token by a
quad of 8-byte copies, the region's whole read-sparse row by two 8-byte copies and the scales a lane
a token) before `readout_tile` waits tile `t`. THE STUB stores zeros into the warp's own panel row. With no work to hide behind, the wait
is one global-to-shared round trip a tile, about 1.3K cycles, sixteen tiles a window: the first
frame measurement's largest row, and the reason the readout component's ring must run more than
one tile ahead of an 8-IMMA tile.

<a id="drain"></a>
## `drain`

The range's eight warp rows summed a token, one column a thread, to the global sums by RED.F32;
the rows are overwritten by the next range.

<a id="frame"></a>
## The frame

Per window: the window's zeta, wait the words, count the read lists, stage the next window's words,
then the chunks a slot ahead with two chunks of row loads in flight (each owner walking the chunks
from its own offset), the window close, publish, the publish barrier, rescale, and the ranges read
out a tile ahead into a double-buffered panel and drained. Every CTA-wide barrier counts warps.
Measured with the fold landed and the other components stubbed, cycles a warp a window on
flagship-alt-k4 (two windows, 64 CTAs): frame (the pool-full waits) 13.0K, words 2.1K, lists (with
zeta) 3.2K, fill 10.4K, fold 7.6K, publish 0.9K, rescale 0.1K, readout 19.4K, drain 3.6K, 60.3K in
all; flagship-cohort-k16 56.7K with fold 4.4K (most of its chunks skipped whole). The fold alone, in
a standalone probe over a prefilled pool, runs 334 cycles a chunk a warp with eight warps (167 a
scheduler against the eight IMMAs' 128 of tensor pipe) and 249 with four. THE FRAME'S REMAINING COST
IS THE LOCK-STEP: every warp both lands a chunk and folds one between two CTA-wide barriers, so a
chunk costs the slowest warp's land plus the slowest warp's fold plus two barrier latencies (~2K
cycles a chunk against ~500 of work); removing the fill's global loads altogether did not move it,
nor did staggering the owners' chunk order. The form that removes it is warp specialization
(producer warps land, consumer warps fold; KERNEL_STANDARDS section 5's form (b) on sm_80), the
readout component's ring the other exposed row.
