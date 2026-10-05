# `csrc/rola/src/facts/facts.cuh` / `facts.cu` — the facts pass

Mirror doc for the one pass that reads a call's bundle before the carry and the intra
do: both sides' class-1 liveness words, their tile unions and the fp16 fold's per-(window,
list) exponents, in one launch. What the words ARE — their layout, the folds over them and why each is the
grain it is — is [`liveness_contract.md`](liveness_contract.md), which this kernel does
not restate and must not contradict; its host seam is [`facts_api.md`](facts_api.md).
`rola.engine.facts.liveness.liveness_words` is the pure-torch MODEL the words are graded
against bit for bit at every sparse level, `list_exponents` in `tests/oracle/fold_reference.py` is the exponents'
definition, and `tests/integration/test_facts_pass.py` holds the kernel to both.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B_l` =
level `l`'s PADDED width; `b_l` = its LOGICAL width, the rows' own; `L` = tokens in the
call; `BH` = batch × heads; `rows = Σ_l B_l`; **word** = 32 consecutive tokens of one
row; **window** = the carry's 512 tokens, 16 words; **unit** = one (side, level,
32-digit group) of a window.

## What it reads, and what it does not

The bundle as the producer wrote it (`docs/api.md` §2.1): each level's rows `[B, T, H, b_l]`, u8 digits at every
depth, read where they lie — a view of the producer's packed plane or a tensor of its own, by its strides, under the
ROW ALIGNMENT LAW ([`liveness_contract.md`](liveness_contract.md#alignment)): every level's rows start on 32 bytes
and step batch, token and head by multiples of 32. The pass also reads the write levels' scales, the write gain and
the value scale for the list exponents — a few words a token.

A DENSE level (the bundle's `sparse` flag unset, set at production for an activation with no exact zeros;
`rola.ops.facts.call_statics`) is neither read nor written: its words are the width mask over the live tokens, which
every reader takes from the side's statics ([`liveness_contract.md`](liveness_contract.md#statics)), so its rows and
groups of `words` and `unions` are left as allocated. The one exception is the LIST level's write rows, read for their
maxima whatever the side declares. Every sparse level's rows are read once, whole, whatever their support: the read
is density-blind, so the pass costs the same on a sparse and a dense draw of a sparse level.

<a id="the-grid"></a>

## The grid: two blocks an SM, taking items in order

The work is items, per (batch, window, head) — the window is the carry's, `kWindowWords` = 16 words, and the host
entry refuses any other, because it is the lists' grain. A window's units (each a (side, level, 32-digit group)) are
the list level's write units first, then the other sparse levels', dealt into two kinds of item:

* a LIST item, which first stages the window's columns ([the lists](#lists)): on a voted list level two units, four
  warps each (four words a warp) — a list unit's word costs ~3× a vote unit's, so its item is half as wide; on a dense
  list level, whose units vote and store nothing, every warp one unit over its whole window (`list_item_units`, set
  by the host from the declared structure), so the columns are staged once a window and each warp's ring streams 16
  words between the item's two barriers;
* a VOTE item: four sparse units, two warps each (eight words a warp).

`kFactsBlocksPerSm` = 2 blocks of eight warps an SM take the items IN ORDER from the call's counter (`counters[0]`, an
atomic a block; the last block out resets both counters, so they are zero between calls). Thread 0 decomposes an
item once — its place in its window, the window, the batch-head, its batch and head — into shared memory, and no warp
divides. Taken in order, heads innermost, the blocks at work together read one window of every head: the rows
`[B, T, H, b_l]` hold a token's heads side by side (a DRAM page's bytes together in time), and the columns `[B, T, H]`
a token's heads in one 32-byte sector, which a head-major order fetched from DRAM once a head (84 MB read against 67
MB of rows at `nl64k-dense`, four heads; heads innermost 75 MB, and `nl64k-alt-k4` 61 → 51 µs a head). Taken
dynamically, a block's load is the items it actually draws rather than a static deal of a few heavy-tailed ones.
MEASURED at `nl64k-alt-k4` (one batch-head): a static stride of 160 blocks over 8 items a window dealt a quarter of
the blocks all the list items (118 µs); coprime striding 88; the ordered counter, with list items half as wide, 63.

Why two blocks of eight warps, measured on the previous form at `(256, 256)` on sm_86: more resident warps than 16
an SM made the rows' stream slower, not faster — the DRAM serves fewer bytes a page visit when more streams
interleave (4 blocks an SM 164 µs a head, 3 147, 2 113); rows read head-major (the producer writes `[B, H, T, …]`) ran
113 against 212 token-major at 16 heads.

<a id="load"></a>

## The load: a ring of words a warp

A warp reads its part of a unit through a RING of `kStages` = 4 word slots in dynamic shared memory: a word is 32
tokens × the unit's 32-byte run, 1 KB. Each lane issues two 16-byte `cp.async` quanta a word (`LaneCopies`: lanes
`2i` and `2i + 1` one token's halves; a pointer set once a part and stepped a word at a time), zero-filled past `L` and
past the rows' width (`ops::stage_tile_n`: a partial quantum's first bytes, nothing past them read), so the ring holds
three words in flight ahead of the one being read with no registers held. Before the ring, a lane issued 32 byte loads a
word, each with a 64-bit address and a predicate: ~5.5 instructions a byte, and one word's loads at a time.

A token's 32-byte row lands at row `4 (t % 8) + t / 8` of its slot (`ring_row`): the four token blocks a lane group
reads together sit in four consecutive rows, so a warp's 4-byte reads meet 32 banks. A DENSE list level's rows are f16
(the dense write levels are; [`liveness_contract.md`](liveness_contract.md#floor)), a unit's 32 digits 64 bytes a token:
its ring is `kDenseStages` = 2 word slots of 32 tokens × 64 bytes in the same bytes, a lane copying two quanta a
token half, the same rows by `ring_row` so a lane group's 8-byte reads stay on distinct banks; a dense level is only
ever read for its lists, so the vote keeps the u8 ring. A whole row moves, so each copy
keeps its global run's 32-byte phase — the condition for a 16-byte lane pair to be one L2 sector request
(docs/internals/common/ops.md#stage-tile-n). The alignment law is what makes the quanta legal: a 16-byte copy needs a
16-byte source, and a level slot on 32 bytes keeps a token's run in one sector.

The word loop's trip count is a constant (a word past the call's end is zero-filled and stores nothing), so its
collectives sit at the top level of a constant loop (KERNEL_STANDARDS §20).

<a id="vote"></a>

## The vote: a lane a (token block, digit quad)

Lane `(tk, dq)` = `(lane / 8, lane % 8)` reads token `8 tk + i`'s digits `4 dq .. 4 dq + 3` as one word, `i` 0..7. The
same reads feed both the vote and the lists (`word_lanes`). THE VOTE: `((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x` sets bit 7 of
each byte iff the byte is nonzero (three operations for four digits); shifted right by `7 - i` it lands at bit `i` of
its byte, so after the 8 tokens byte `k` of the lane's register holds digit `4 dq + k`'s 8 token bits. A 4 × 4 byte
transpose across the four token blocks — two exchanges (`shfl_xor` 16 then 8), a byte permute each — leaves lane
`(tk, dq)` digit `4 dq + tk`'s 32 token bits: the class-1 word, masked by the digit's width-mask bit. ~60 instructions a
word against ~260 for byte loads and a test a byte, and against ~100 for a token-lane form (two 16-byte reads, a
multiply-gathered nonzero mask, a five-round bit transpose).

Each 64-token tile's union is one `__reduce_or_sync` of the lanes' `bit d` where their two words are nonzero. A dense
list unit runs the same loop over its whole window with the vote, its tile rows and the unions compiled out.

<a id="transpose"></a>

### The store: a unit's tile, a row's window one run

The layout wants a row's words contiguous (`[BH][side][row][word]`). The unit's warps write their words into one shared
tile (32 rows × 17 words; the odd stride makes the row writes conflict-free), and after the barrier each warp stores its
share of the rows two an instruction: lanes 0–15 one row's 16 words and lanes 16–31 the next's, each a contiguous 64-byte
run — whole sectors, 64-byte aligned when `L` is a whole number of windows, which the prefill requires.

<a id="lists"></a>

## The list exponents: the fp16 fold's scales

    m(t)          = g_write(t) × Π_l scale_write_l(t)      (the gain first, then the levels in order)
    K[w, r, 0]    = ceil(log2(max over the window's tokens of b(t, r) × m(t)))
    K[w, r, 1]    = ceil(log2(max over the window's tokens of b(t, r) × m(t) × s(t)))     (s = the value scale)

per window `w` and digit `r` of the LIST LEVEL — the carry's level R, whose write digits are its write lists
(`rola.ops.facts.list_level`; none past depth two) — `b` the token's write coefficient on the byte scale: a sparse
level's u8 digit, a dense level's f16 times 255 (exact; its floor stored,
[`liveness_contract.md`](liveness_contract.md#floor)), read four halves a lane by one 8-byte load
(`dense_list_lanes`), floored at -126 (the smallest normal's exponent: a list no
token writes). Stored int16 `[BH, windows, B_list, 2]`, a lane's pair one word. The fp16 fold puts each list's largest
coefficient at a fixed power of two by them, so one scale a list keeps every coefficient in fp16's range (ruled
2026-10-03); it replaced zeta, the int fold's per-window maximum, which nothing read once that fold was gone.

A LIST item first issues its threads' column loads (`load_columns`: two tokens a thread, every load before any
use), then starts its warps' rings (`start_part`), then stages the columns in shared memory (`stage_columns`: `m` in
level order, a pad slot after every 8 tokens so the four token blocks a lane group reads together meet distinct banks,
and past them the window's largest of each), so the two loads overlap behind one barrier with the columns at the head
of the item's requests — issued behind the ring's 24 KB they waited on it (`nl64k-dense`, four heads: 126 → 118 µs). In the word loop, lane `(tk, dq)` converts each of its
four bytes to fp32 through the mantissa (a byte permute with an immediate selector, `0x4b0000bb`, minus 2^23) and maxes
`b × m` and `b × m × s` into its four digits' pairs — six operations an element, ~210 a word. At the part's end the four
token blocks' maxima join across lanes, the unit's parts join in shared memory after the barrier, and its first part's
warp stores a lane's `(K_m, K_v)` pair, the unit's 32 one 128-byte run. A dense level's floor is in its stored rows,
so the maxima take it as read.

<a id="factor-products"></a>

## The factor products: the fold's `(F, m)`

    m(t) = ((g_write(t) × scale_write_0(t)) × scale_write_1(t))      F(t) = m(t) × s(t)

multiplied in fp32 in that order: the products `stage_columns` already forms for the lists. A window's first list
item (`k == 0`) stores each token's `(F, m)` as one float2 to the `[BH, L, 2]` plane `fm`, so the fold reads a token's
two products in one 8-byte load where it read four factor planes and multiplied (ruled 2026-10-04: precompute what the
fold's restage would multiply). A past-the-end token stores nothing; a call without a list level leaves the plane
unwritten. At 8 bytes a token-head it adds no `O(L·N)` term.

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[8][3]` int64 buffer (warp × `columns`, `vote`, `store`)
that every block's warps add their cycles into; `rola.ops.facts.ledger_bind` is the
Python side. A bound ledger moves no output (`tests/integration/test_facts_pass.py`). The `columns` phase books from
an item's start — the wait for the item's barrier included — to its word loop, so it carries a block's waiting on its
slowest warp as well as the list items' staging (`nl64k-alt-k4`, one batch-head: columns 49%, vote 44%, store 7%).

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView` (base, the batch, token and head strides in digits, and the rows' width), per write
level and for the gain and value scale a `ColumnView`, the outputs, the item counters, the depth, the padded widths and
row bases, per side the dense bitmask and the digit mask's words, the unit table (`unit_code(side, level, group)`, at
most 64, the list level's units first) and the item shape (`list_units`, `vote_units`, `list_item_units`,
`list_items`, `vote_items`, `window_items`). It is a kernel
parameter of under a kilobyte, so the pass takes no allocation to learn its shape. No
logical width is in it: `b_l` reaches the mask at the seam and bounds nothing else — the
host entry refuses a mask that names a digit past a level's rows, so the kernel never
reads beyond them.

## The budget line and the floor

The int8 rebuild's facts line (ruled 2026-09-30): per head at `(256, 256)`, `L = 64K`, the rows 2 × 512 B a
token (67 MB, ~84 µs at 800 GB/s), the words 8 MB (~10 µs), the old zeta ~1 µs: a DRAM floor
of ~95 µs, budget 1.3× (~125 µs), every cell priced as if both sides were read whole.
Measured (ncu, kernel time): at 16 batch-heads all-sparse 98 µs a head (DRAM at 88% of
peak), alt 59, all-dense 19 — a dense level unread; the carry cells (one sequence):
`nl64k-alt-k4` 70 µs, `nl64k-dense` 23 µs, `flagship-cohort-k16` (`L = 1K`) 12.8 µs,
against the line's "~1.5 µs, launch-bound" — which was not a floor: this kernel reading
nothing runs 5.3 µs, and the rest is the warps' load chains
([the grid](#the-grid)). Reading half of each token row (alt) costs a page visit all
the same, so alt runs at 59 µs where its bytes alone would take 54.

Writing the [tile unions](liveness_contract.md#tile-unions) (2026-09-30) added 131 KB of
stores a head at `L = 64K` and a ballot a tile to the vote: all-sparse 99 µs a head, alt 62,
all-dense 23 (from 98, 59, 19), `L = 1K` all-sparse 13.4 µs (from 12.9). The dense case
pays most because its vote loads nothing and the per-tile ballot is then most of its
loop; a dense level's union is its width mask on every live tile, one ballot a unit
rather than a tile.

THE LIST EXPONENTS (2026-10-03) MISSED THEIR LINE. The line, stated before the code: `nl64k-alt-k4` <= 78 µs (its
list rows already read, ~6 added instructions a (digit, token) on 8 of its 16 reading units, the zeta item gone),
`nl64k-dense` <= 40 µs (the list level's write rows a new 16.8 MB read: a DRAM floor ~31 µs), `flagship-cohort-k16`
<= 14 µs. Measured (ncu kernel time, one batch-head, from 71 / 25 / 12.8 with zeta): 88 / 58 / 19.6 µs, 18.5M
instructions at alt with the scheduler issuing on 58% of its cycles — the pass is issue-bound as much as it waits on
its rows: a vote (digit, token) costs ~8 instructions (a byte load's 64-bit address and predicate, its test), a list
one ~10 more (its conversion, two products, two maxima, the column load). Variants measured worse: two units a list
item at four warps each (alt 98, cohort 14.6), 32-bit load offsets in either form (alt 107-115, 20-25M instructions).
THE LEVER is the load structure, not the arithmetic: the rows staged by `cp.async` 16-byte copies (two a lane a word in
place of 32 byte loads, several words in flight without registers) and read from shared memory — the sm_90 TMA form
below, on sm_80/86. Taken the same day: the rebuilt pass below.

THE REBUILT PASS (2026-10-03, Blake: "a first class optimized kernel"). The line, stated before the code: alt <= 65
µs (DRAM floor ~52: 33.5 MB rows + 8.4 MB words), dense <= 41 µs (floor ~31), cohort-k16 <= 10 µs (an empty launch is
5.3). Measured (ncu kernel time; one batch-head, then four, a head's share):

| cell | zeta form | list exponents, byte loads | rebuilt | at 4 heads (DRAM % of peak) |
|---|---|---|---|---|
| `nl64k-alt-k4` | 71 µs | 88 µs | 62.5 µs, 10.6M instr | 63 µs (83%) |
| `nl64k-dense` | 25 µs | 58 µs | 45.6 µs, 8.1M instr | 40 µs (83%) |
| `flagship-cohort-k16` | 12.8 µs | 19.6 µs | 11.6 µs | — |

At four heads both cells are DRAM-bound at 83% of peak: the pass is first class where prefill runs. One batch-head is
latency-bound: dense's list items (4 a window) each wait one DRAM latency on their columns before their words, and too
few items are in flight to hide it (11% over the line); cohort-k16 is two windows, its critical path one list item after
the launch (16% over). The columns could reach the list warps through the ring (4-byte copies a word) in place of a
staging barrier, which a 3-stage ring with two blocks an SM measured no better.

DENSE LEVELS UNWRITTEN (2026-10-03, Blake: "We don't need to write the dense levels' words"). A dense level's words
are its width mask over the live tokens; every fold takes them from the statics, so the pass writes none: no mask
items, and a dense list level's units only their exponents. The line, stated before the code: `nl64k-dense` loses its
8.4 MB of writes a head, leaving its 16.8 MB of list rows (floor ~21 µs a head), line <= 27 at four heads;
`nl64k-alt-k4` loses half (floor ~47), line <= 61; cohort-k16 no regression. Measured (ncu kernel time, a head's
share at four heads / one batch-head), each step on the last:

| step | `nl64k-dense` | `nl64k-alt-k4` | `flagship-cohort-k16` |
|---|---|---|---|
| before (the rebuilt pass above) | 40 / 45.6 µs | 63 / 62.5 µs | 11.6 µs |
| dense words unwritten | 34.7 / 38.3 (DRAM 71%) | 61.0 / 59.2 | 11.5 |
| items heads innermost ([the grid](#the-grid)) | 31.5 | 50.9 (DRAM 86%) | — |
| column loads ahead of the ring ([the lists](#lists)) | 29.4 | 51.0 | — |
| a dense list level's item a warp a unit | 27.4–28.4 / 31.4–32.2 (DRAM 72–74%) | 49.6–50.3 / 57.2 | 10.6–11.3 |

Dense meets its line within the ±2% run-to-run spread; alt and cohort are under theirs. Measured and not taken:
four units a list item on every list level (dense 33 at four heads, but alt 87 and cohort 14.2: the voted list items
lose their balance); the next item's counter taken during the current one (no change outside the spread).

The sm_80 form is this one, re-measured for its residency point (HBM's page behaviour
is not GDDR6X's). On sm_90 the ring's slots fill by TMA: a word's `[32 tokens × 32 digits]` box
in one instruction in place of 64 `cp.async` quanta; the rest is unchanged.

## Where the mechanism came from

The liveness pass it replaces (`csrc/rola/src/facts/liveness.{cu,cuh}`, deleted
2026-09-30 with its Python op) read the bf16 planes the prefill packed from
the levels, a lane a token, a warp ballot a digit; its layout, its dense-level mask and
its past-the-end dead vote are kept unchanged. What is new is the input — the bundle's
own u8 rows, half the bytes of the bf16 planes, with no packing pass before it —
the tile that keeps the stores whole, and the fold's scales (zeta then; the list exponents now). The rebuilt load
(2026-10-03) replaced the digit-lane byte loads the old input's lack of an alignment law asked for: the alignment law,
the ring, the shared (token block, digit quad) lanes.
