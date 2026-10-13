# `csrc/rola/src/facts/facts.cuh` / `facts.cu` — the facts pass

Mirror doc for the one pass that reads a call's bundle before the carry and the intra do: both sides' class-1
liveness words and their tile unions, the window's columns (the factor products and the gains), the write list
level's exponents, and each side's lists in the carry's final form, in one launch. What the words ARE — their layout,
the folds over them and why each is the grain it is — is [`liveness_contract.md`](liveness_contract.md), which this
kernel does not restate and must not contradict; its host seam is [`facts_api.md`](facts_api.md).
`rola.engine.facts.liveness.liveness_words` is the words' pure-torch MODEL, `rola.engine.facts.lists.final_lists` the
lists', `list_exponents` in `tests/oracle/fold_reference.py` the exponents' definition, and
`tests/integration/test_facts_pass.py` holds the kernel to all three.

Vocabulary, defined here and used unqualified below: `D` = routing depth; `B_l` = level `l`'s PADDED width; `b_l` =
its LOGICAL width, the rows' own; `L` = tokens in the call; `BH` = batch × heads; `rows = Σ_l B_l`; **word** = 32
consecutive tokens of one row; **window** = the carry's 512 tokens, 16 words; **group** = 32 consecutive digits of a
level; **unit** = one (side, level, group) of a window, the work of one CTA.

## What it reads, and what it does not

The bundle as the producer wrote it (`docs/api.md` §2.1): each level's rows `[B, T, H, b_l]`, u8 digits, read where
they lie — a view of the producer's packed plane or a tensor of its own, by its strides, under the ROW ALIGNMENT LAW
([`liveness_contract.md`](liveness_contract.md#alignment)): every level's rows start on 32 bytes and step batch, token
and head by multiples of 32. It also reads the write levels' scales, the write gain and the value scale for the
columns — a few words a token.

A DENSE level (the bundle's `sparse` flag unset; `rola.ops.facts.call_statics`) is neither read nor written: its words
are the width mask over the live tokens, which every reader takes from the side's statics
([`liveness_contract.md`](liveness_contract.md#statics)), so its rows and groups of `words` and `unions` are left as
allocated. Every sparse level's rows are read once, whole, whatever their support: the read is density-blind.

<a id="the-grid"></a>

## The grid: a CTA a unit

`kFactsWarps` = 8 warps a CTA, one CTA a UNIT, four CTAs an SM on sm_86 (its 23 KB of shared memory a CTA; its
registers a thread held to four CTAs by `kFactsCtasMin`, 46 used, which allow five) and five on sm_80 (registers): a
unit is a plan-fixed load
and its work, independent of every other unit, so the hardware's block scheduler balances the units' unequal costs and
several CTAs' loads and work overlap on an SM (R22's independent units). The grid is `(per_window, windows, BH)`: a
window's units by x, innermost, so the CTAs at work together read one head's window of every level -- the bundle is
stored head-major (`[B, H, T, ...]`), a contiguous block of each side's plane -- and no CTA divides to find its unit
(the host refuses more than 65,535 windows or batch-heads).

<a id="the-plan"></a>

## The plan: a window's units

`plan_units` (host) lists a window's units once a call, the same for every window (`FactsParams::unit`, coded by
`unit_code`: kind, side, level, group): for each side and level in order, a `kUnitList` unit a group of the side's LIST
LEVEL (its one sparse level, `rola.ops.facts.list_level`) and a `kUnitVote` unit a group of every other sparse level;
and when the write side lists nothing (no sparse level, or two), one `kUnitColumns` unit for the window's columns. A
side with two or more sparse levels gets its words and no lists: they would be COMPOSITE, keyed by a digit of each
level, which this pass does not build.

<a id="shared-memory"></a>

## The CTA's shared memory

`FactsSmem` by offset, dynamic, 22.6 KB:

| tenant | bytes | live |
|---|---|---|
| the window: 512 tokens × the group's 32 bytes | 16,384 | the load to the lists' last coefficient |
| the bit tile (`tile_at`) | 2,048 | the vote to the end |
| the columns `(m, F)` (float2) | 4,096 | a write unit's load to the end |
| the union's words | 64 | the vote to the ranks |
| each pass's bound (its first token) | 16 | the ranks to the end |
| the warps' maxima | 64 | the load |

<a id="load"></a>

## The load

A unit's window is 512 tokens × 32 bytes, the group's digits of each token's row: two `cp.async` 16-byte quanta a
token, a lane pair a token (one L2 sector request: [`ops.md`](../common/ops.md#stage-tile-n)), token `t` at region
`t / 32` (1 KB each) row `ring_row(t % 32)` = `4 (t % 8) + t / 8` of 32 bytes, so the vote's lanes meet 32 banks.
Bytes past the rows' width or a token past `L` are zero-filled (`ops::stage_tile_n`), never read. A write list unit
loads its window's columns beside it into registers (two tokens a thread), so their latency hides under the window's.

<a id="vote"></a>

## The vote: a lane a (token block, digit quad)

Lane `(tk, dq)` = `(lane / 8, lane % 8)` reads token `8 tk + i`'s digits `4 dq .. 4 dq + 3` as one word, `i` 0..7.
`((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x` sets bit 7 of each byte iff the byte is nonzero (`nonzero_bytes`); shifted right
by `7 - i` it lands at bit `i`, so byte `k` of the lane's register holds digit `4 dq + k`'s 8 token bits, and a 4 × 4
byte transpose across the token blocks (two `shfl_xor` exchanges) leaves lane `(tk, dq)` digit `4 dq + tk`'s 32 token
bits, masked by the digit's width-mask bit (`digit_in`). Warp `w` votes the window's words `2w` and `2w + 1`, and ORs
each over the warp (one warp reduction) into the union's word.

<a id="tiles"></a>

## The tiles and the bits out

A unit's words go into a shared TILE (`tile_at`: digit `d` word `j` at `16 d + (j ^ (d / 2 % 16))`, so a warp's
column writes, a digit a lane, and its row reads, a row a half-warp, each meet 32 banks); after the barrier each
digit's window of words goes out as one 64-byte run (a half-warp a digit) and each 64-token tile's union, a digit a
lane, as one ballot (warp `w` tile `w`).

<a id="columns"></a>
<a id="factor-products"></a>

## The window's columns

    m(t) = g_write(t) × Π_l scale_write_l(t)      (the gain first, then the levels in order)      F(t) = m(t) × s(t)

`s` the value scale, a token past `L` zero. Every write list unit loads them (the coefficients and the list maxima
read them); the unit of group 0 of the write list level — or the window's `kUnitColumns` unit when the write side lists
nothing — also stores each token's `(F, m)` to the `[BH, L, 2]` plane `fm` (the factor products: the fold reads a
token's two in one 8-byte load; at 8 bytes a token-head it adds no `O(L·N)` term) and the gains.

<a id="gains"></a>

## The gains

Each token's `(m, F)` as an f16 pair at its window's powers of two (`gains16`) and the exponents (`gains_exp`): the
window's largest `m` and `F` (each warp's largest posted in shared memory before the vote's barrier, joined after it)
give `gains_shift(top)` = `-4 - floor(log2 top)` (the largest in [2^-4, 2^-3)), clamped to [-126, 127], 0 for a zero
largest.

<a id="lists"></a>

## The list exponents: the fp16 fold's scales

    K[w, r, 0]    = ceil(log2(max over the window's tokens of b(t, r) × m(t)))
    K[w, r, 1]    = ceil(log2(max over the window's tokens of b(t, r) × m(t) × s(t)))

per window `w` and digit `r` of the WRITE LIST LEVEL, `b` the token's u8 digit, floored at -126 (the smallest normal's
exponent: a list no token writes). Stored int16 `[BH, windows, B_list, 2]`. The fp16 fold puts each list's largest
coefficient at a fixed power of two by them (ruled 2026-10-03). They come out of the write side's maxima walk (each
lane's largest over its entries, joined over the digit's eight lanes; [the lists out](#lists-out)).

<a id="ranks"></a>

## The ranks: one slot map a (window, side, group)

Every slot-addressed consumer -- the readout's pool and panel rows, the fold's pool rows, the den pass's staged rows --
reads one slot map a union (ruled 2026-10-12). A union token's SLOT is its RANK, its place among the union's tokens in
token order (ruled 2026-10-13). After the vote's barrier every warp scans the 16 union words' counts (a 16-lane shuffle
scan), and its two words' tokens go out to `union_tok` at their slots -- a word's tokens at consecutive slots, so the
stores coalesce -- every slot past the union `kGap` (`carry/carry.cuh`: token 0 with bit 15 set, so a row fetch masked
to the window stays inside it). The union's EXTENT (`union_count`, int32) is its size in whole steps of `kStep` = 8:
the carry stages the extent's slots and zero-fills a gap's row, so every pad's slot (below) is a staged row. Each
pass's first token is its BOUND, posted in shared memory (`kWindowTokens` for a pass the union does not reach): pass
`q` holds the tokens from bound `q` to bound `q + 1`, which the lists count without walking.

The slots are not colored. A coloring that put each step's 8 tokens on distinct residues mod 8 (the bank groups of
every slot-indexed row) made the carry's gathers conflict-free, but it padded each list's passes to its most-used
color and was most of this pass's time; on sm_86 the composed carry ran as fast on rank slots as on the best coloring,
its time following the lists' places rather than its shared-memory conflicts (2026-10-13). An architecture whose carry
is bound by its gathers' wavefronts would take the coloring back as one more slot-choosing step, measured in the
composed carry first.

<a id="lists-out"></a>

## The lists out

Per list (a digit of the list level), its tokens in PASSES of the carry's pass for the side — the readout's
`pass_slots` for the read side, the fold's `unit_slots` for the write side, the arm's form on the device's card
(`rola.ops.carry.form`), a power of two from 128 to 512 (the host refuses another: a list's passes and its end must fit
its `kListAt` = 8 starts): pass `q` its tokens of slot `q P .. q P + P - 1` in token order, its last step filled with
PADS, so each pass starts on a step. A list's entries are at most the union's extent, at most 512. `at[q]` is pass
`q`'s start, and from the last pass on the list's end.

An entry is 8 bytes, the carry's `uint2`:
- READ: the slot's pool row offset in its pass with the row's swizzle folded in (`local × row + 16 × chunk0`, `local`
  the slot less its pass's first, `chunk0` the carry's `pool_chunk` of chunk 0: an 8-chunk row XORs by the residue, a
  row of an odd number of 64-byte quarters by the residue's high bits), 0; the coefficient `f16(byte / 256)`, the slot's
  panel row offset past the dummy rows (`kPadBytes + panel_row(local)`). A pad at place `c` of its step: the pass's row
  `c` and dummy panel row `c`, coefficient 0.
- WRITE: the slot, 0; `c_v = f16(fp32(b F) 2^(4 - E_v))`, `c_m = f16(fp32(b m) 2^(4 - E_m))`, `E = max(K, -123)` the
  list's exponent. A pad at place `c` of its step: the pass's slot `c`, both coefficients 0.

A warp's HEADS a (window, group): its 4 lists `w + 8 j`'s first 16 entries, copied as they are written.

THE LANES, every warp its 4 lists `w + 8 j` (its own heads), 8 lanes a list, lane `s` the list's words `2s` and
`2s + 1`. Each lane counts its tokens before each pass's bound (a popcount of each word under a mask), so its tokens in
each pass; a segmented scan of those counts (four u16 fields in two registers) gives each pass's tokens before the
lane's and, at the list's last lane, each pass's tokens, so each pass's start in whole steps; lane `s` writes `at[s]`.
On the write side a walk over the lane's entries takes the list maxima, and their exponents. A walk then places the
lane's `i`-th entry at its pass's start plus the pass's tokens before the lane's, less the lane's tokens in earlier
passes, plus `i` -- its slot `pre + popc(union word & below)` from the lane's two union words and their ranks before,
its pass the slot's high bits -- and writes it; lane `q` writes pass `q`'s pads.

<a id="access-table"></a>

## The access table (KERNEL_STANDARDS §30)

| buffer | writes | reads | wavefronts |
|---|---|---|---|
| window | `cp.async` 16-byte quanta, a lane pair a token | the vote: lane `(tk, dq)` row `4 i + tk` word `dq`; the walks: a byte at (token, digit) | the vote 1.0 a load (32 banks by `ring_row`); a walk's byte loads land on random banks, ~1.5; the copies' cost is their global pattern (`ldgsts-copy-ranking`) |
| tile | the vote: a digit a lane, one word | the bits out: a half-warp a digit row; the lists: a lane a list's two words | 1.0 except the lists' loads: their 4 lists share a bank half, 4.0 against 2.0 ideal, two loads a lane a unit |
| union words | the vote: lane 0, a word | the ranks: lane `i` word `i % 16` | 1.0 (a broadcast pair) |
| bounds | the ranks: the lane holding a pass's first slot | the lists: every lane, three words | broadcast, 1.0 |
| columns | a lane a token | the walks: a float2 at a token | random, ~1.5 |

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[8][4]` int64 buffer (warp × load, vote, ranks, lists) that every CTA's warps add
their cycles into; `rola.ops.facts.ledger_bind` is the Python side. An unbound ledger issues no lap; a bound ledger
moves no output (`tests/integration/test_facts_pass.py`).

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView`, per write level and for the gain and value scale a `ColumnView`, the outputs (a
`ListsOut` a side), the depth, the padded widths, row and group bases, each side's list level and pass (log2), the read
pool row's bytes, the window's unit codes, per side the digit mask's words. Under a kilobyte and a half. No logical
width is in it: `b_l` reaches the mask at the seam and bounds nothing else — the host entry refuses a mask that names a
digit past a level's rows.

<a id="budget"></a>

## The budget line and the floor

THE LINE (ruled 2026-10-12, restated 2026-10-13): the whole pass at <= 1.15x its required bytes' DRAM floor on the
lab's `facts_block` cell (rows `[4 level-sides, 65,536, 4, 256]` u8, 268 MB, k live digits a token-head; the write side
lists level 0 and the read side level 1, the other level of each voted). Its required bytes -- the rows, the columns,
the words and tile unions, every list's entries (its tokens and its pads), heads, starts, union and extent, the
exponents, the factor products and gains -- were 449 µs at k 4 and 578 µs at k 16 at the lab's flat read (854 GB/s)
with the colored lists' places, so the line is 516 and 665 µs. Rank slots pad less: the floor is now 445 and 520 µs.

MEASURED (`measure/harness/bench_facts_block.py`, RTX 3080 Ti at its boost clock, the sm_86 even arm's form, medians):

| | k 4 | k 16 |
|---|---|---|
| the pass | 482 µs (1.08x the floor) | 599 µs (1.15x) |
| the line | 516 µs | 665 µs |
| the colored form it replaced | 707 µs | ~1,270 µs |
| phase share, cycles over every warp: load, vote, ranks, lists | 0.46, 0.24, 0.07, 0.23 | 0.35, 0.23, 0.07, 0.35 |

At k 16 the pass moves 273 MB from DRAM and writes 175 MB (ncu, one launch), its required bytes: no list sector is
written twice. Four CTAs an SM keep 62% of the warp slots active.

Executed warp-instructions a window-head (its 32 units: 8 write list, 8 read list, 16 vote), by component
(`tools/region_ledger.py`, one launch):

| component | k 4 | k 16 |
|---|---|---|
| `lists_out` | 103,900 | 170,600 |
| `vote` | 35,400 | 35,400 |
| the kernel's own lines (columns' maxima, barriers) | 32,200 | 32,200 |
| `bits_out` | 22,800 | 22,800 |
| `issue_window` | 21,500 | 21,500 |
| `ranks` | 17,900 | 17,900 |
| `load_column` | 11,800 | 11,800 |

Everything but the lists is fixed a unit. The lists' walks run each warp's longest lane: 37.6 trips a unit at k 4 and
82.4 at k 16 on this cell (a write unit walks twice, its maxima then its entries), so a list unit costs ~3,000 fixed,
and a trip ~186 summed over a write unit's two walks and a read unit's one.

Per architecture: sm_80 runs the same form at five CTAs an SM (registers); on sm_90 the window fills by TMA (a
`[512 tokens × 32 bytes]` box of the head's plane in one instruction) under an mbarrier, the rest unchanged. Scaling:
O(L × rows) bytes read, O(L k) entries, O(L B / 32) unions and heads; nothing O(L N).
