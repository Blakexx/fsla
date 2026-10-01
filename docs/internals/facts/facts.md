# `csrc/rola/src/facts/facts.cuh` / `facts.cu` — the facts pass

Mirror doc for the one pass that reads a call's bundle before the carry and the intra
do: both sides' class-1 liveness words, their tile unions and the fold's per-window
scales, zeta, in one launch. What the words ARE — their layout, the folds over them and why each is the
grain it is — is [`liveness_contract.md`](liveness_contract.md), which this kernel does
not restate and must not contradict; its host seam is [`facts_api.md`](facts_api.md).
`rola.engine.facts.liveness.liveness_words` is the pure-torch MODEL the words are graded
against bit for bit, `fold_zeta` in `tests/oracle/int_reference.py` is zeta's
definition, and `tests/integration/test_facts_pass.py` holds the kernel to both.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B_l` =
level `l`'s PADDED width; `b_l` = its LOGICAL width, the rows' own; `L` = tokens in the
call; `BH` = batch × heads; `rows = Σ_l B_l`; **word** = 32 consecutive tokens of one
row; **window** = the carry's 512 tokens, 16 words; **unit** = one (side, level,
32-digit group) of a window.

## What it reads, and what it does not

The bundle as the producer wrote it (`docs/api.md` §2.1): each level's rows
`[B, T, H, b_l]`, u8 digits at depth ≤ 2 or f16 units deeper, read where they lie — a
view of the producer's packed plane or a tensor of its own, by its strides. There is no
alignment law: a level may start at any byte and be any width. The pass also reads the
write levels' scales, the write gain and the value scale for zeta — a few words a token.

A DENSE level (the bundle's `sparse` flag unset, set at production for an activation with
no exact zeros; `rola.ops.facts.call_statics`) is not read: its words are the width mask over the live
tokens ([`liveness_contract.md`](liveness_contract.md#statics)). Every other level's rows
are read once, whole, whatever their support: the read is density-blind, so the pass
costs the same on a sparse and a dense draw of a sparse level.

<a id="the-grid"></a>

## The grid: two blocks an SM, walking items

The work is items: per (batch-head, window) its units four at a time, then its zeta.
The window is the carry's — `kWindowWords` is 16 words, and the host entry refuses any
other — because it is zeta's grain. `kFactsBlocksPerSm` = 2 blocks of eight warps an SM
walk the items (the grid is `min(items, 2 × SMs)`), the windows of one batch-head
consecutive, so the blocks at work together read one head's neighbouring windows. An
item's eight warps take its four units two ways each: warp `w` votes unit `w mod 4`'s
half `w / 4` of the window's words into that unit's shared tile, the block meets at a
barrier, and each warp stores its half of its unit's rows; a barrier closes the item.
A warp's half of a window is four 64-token tiles, so it writes those tiles' unions itself
([the vote](#vote)).

Why this shape, measured at `(256, 256)` on sm_86 (ncu kernel time; `L = 64K` over 16
batch-heads, a head's share, and `L = 1K` over one):

* **Residency is a design point, not "as many as fit".** More resident warps than 16 an
  SM made the rows' stream slower, not faster — the DRAM serves fewer bytes a page visit
  when more streams interleave: with a block a window as the item, 4 blocks an SM ran 164
  µs a head, 3 ran 147, 2 ran 113.
* **Rows are read head-major.** The producer writes the bundle `[B, H, T, …]`
  (`docs/internals/produce/produce.md#bundle`); on token-major rows at 16 heads (an 8 KB
  token stride) the same pass ran 212 µs a head against 113.
* **The warps of a block read the same tokens.** A block's warps each on a different
  column of the same token rows keeps a DRAM page's bytes together in time; a block's
  warps on different tokens of one column (one unit an item, its window split eight ways)
  ran 129 µs a head, and independent warp-sized items (drifting apart) 172.
* **A unit is split among warps, because a warp's loads are a chain.** A lane holds one
  word's 32 loads in flight (byte loads into 32 registers; two words in flight spill at
  the 128-register bound), so a warp that votes a whole window's 16 words walks 16 round
  trips — 21.8 µs at `L = 1K`, where a call has two windows. The sweep over units an
  item (`kItemUnits`, the warps splitting each unit's window `8 / kItemUnits` ways):

  | units an item | 64K all-sparse | 64K alt | 64K all-dense | 1K all-sparse |
  |---|---|---|---|---|
  | 8 | 110 µs | 105 µs | 17.0 µs | 21.8 µs |
  | **4** | **98** | **59** | **19.4** | **12.9** |
  | 2 | 104 | 64 | 26.1 | 8.6 |
  | 1 | 121 | 76 | 36.8 | 6.4 |

  Four is the best at scale and gives up the small call: a warp's eight words are eight
  round trips at `L = 1K`. The lever that would remove the chain rather than trade it is
  wider loads — four digits a lane — which needs every level's rows to start and stride
  on four digits (the padded widths would give it); it is recorded on the card, not taken.

<a id="unit"></a>

## A unit: a warp's lanes over 32 digits, its tokens walked

A unit's lane `i` owns digit `32g + i` of level `l` on one side — the warp's lanes are
32 consecutive digits of one row, and the unit walks the window's tokens. That
orientation is the one that makes every load a single contiguous run across the warp:
for token `t` the lanes read `rows_l[b, t, h, 32g .. 32g + 31]`, one 32-byte (u8) or
64-byte (f16) run whatever the level's offset or width. The alternative — a lane per
token, the warp walking digits — reads 32 rows 1 byte each per instruction and leans on
L1 to recover the rest of each sector; the old pass measured that leaning fail at 2.5–
3.8× the unique bytes.

<a id="vote"></a>

### The vote

`vote_part` walks a warp's half of the window a word at a time: a word's 32 loads
(`load_word`, unrolled so all are in flight), then 32 nonzero tests (`digit_live`: a u8
digit, or an f16 unit's magnitude, so `-0` is dead) OR-ed into the lane's word at the
token's bit, into the lane's row of the unit's tile. A token past `L` is not loaded and
votes dead. A lane whose digit the mask does not name (a pad digit, or one past `B_l`)
loads nothing and votes zero; a dense level's words are the mask over the live tokens.
At each tile's second word — or the call's last word, closing a one-word tile — the warp
ballots its lanes' OR of the tile's words, and lane 0 stores the word: the tile union
([`liveness_contract.md`](liveness_contract.md#tile-unions)), four a warp's part, one
ballot and one store each.

<a id="transpose"></a>

### The store: a unit's tile, a row's window one run

The layout wants a row's words contiguous (`[BH][side][row][word]`), and a lane's votes
are its row's. So the unit's two warps vote into one shared tile (32 rows × 17 words;
the odd stride makes the lanes' row writes conflict-free), and after the barrier each
warp stores its 16 rows two an instruction: lanes 0–15 one row's 16 words and lanes
16–31 the next's, each a contiguous 64-byte run — whole sectors, 64-byte aligned when
`L` is a whole number of windows, which the prefill requires. Every word is written
once.

<a id="zeta"></a>

## Zeta: the fold's window scales

    m(t)       = g_write(t) × Π_l scale_write_l(t)      (the gain first, then the levels in order)
    zeta[0]    = max over the window's tokens of m(t)
    zeta[1]    = max over the window's tokens of m(t) × s(t)      (s = the value scale)

stored `[BH, windows, 2]` fp32 and floored at the smallest normal fp32 — a zeta is a
divisor (the carry's int8 fold: `P_m(t) = rint(2^23 col_m(t) / (255 zeta_m))`, the window's largest
column at full scale), and a window of dead tokens keeps the floor. `m` is the product
`int_reference.fold_zeta` calls `g m_ws m_wd`: every write level's row maximum, sparse
or dense. A zeta item's 256 threads take two tokens each, every column load issued before the
first product, reduce by warp shuffles and a shared row of eight, and two threads store. The product ORDER is part of the contract
with the fold that divides by it: a consumer that recomputes `m(t)` in the same order
gets its own maximum token at exactly `zeta`.

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[8][3]` int64 buffer (warp × `vote`, `store`, `zeta`)
that every block's warps add their cycles into; `rola.ops.facts.ledger_bind` is the
Python side. A bound ledger moves no output (`tests/integration/test_facts_pass.py`). At
`L = 64K`, 16 batch-heads, all sparse: vote 92%, store 2.5%, zeta 6% of the warps'
cycles — the pass waits on its rows.

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView` (base, and the batch, token and head strides in digits),
per write level and for the gain and value scale a `ColumnView`, the outputs, the
depth, the padded widths and row bases, per side the dense bitmask and the digit mask's
words, and the unit table (`unit_code(side, level, group)`, at most 64). It is a kernel
parameter of under a kilobyte, so the pass takes no allocation to learn its shape. No
logical width is in it: `b_l` reaches the mask at the seam and bounds nothing else — the
host entry refuses a mask that names a digit past a level's rows, so the kernel never
reads beyond them.

## The budget line and the floor

The int8 rebuild's facts line (ruled 2026-09-30): per head at `(256, 256)`, `L = 64K`, the rows 2 × 512 B a
token (67 MB, ~84 µs at 800 GB/s), the words 8 MB (~10 µs), zeta ~1 µs: a DRAM floor
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

The sm_80 form is this one, re-measured for its residency point (HBM's page behaviour
is not GDDR6X's). On sm_90 the rows move by TMA: a unit's `[tokens × 32 digits]` box
lands in shared memory in one instruction, the vote reads it from there, and the
register-bound load chain that shapes the item split here goes away.

## Where the mechanism came from

The liveness pass it replaces (`csrc/rola/src/facts/liveness.{cu,cuh}`, deleted
2026-09-30 with its Python op) read the bf16 planes the prefill packed from
the levels, a lane a token, a warp ballot a digit; its layout, its dense-level mask and
its past-the-end dead vote are kept unchanged. What is new is the input — the bundle's
own u8/f16 rows, half the bytes of the bf16 planes, with no packing pass before it —
the digit-lane orientation that input's lack of an alignment law asks for, the tile
that keeps the stores whole, and zeta.
