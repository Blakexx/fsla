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

## The grid: one block an SM, two teams of eight warps

`kFactsWarps` = 16 warps, one block an SM (`kFactsBlocksPerSm` = 1), in two TEAMS of `kTeamWarps` = 8 that share
nothing but the SM and run each its own pipeline; the teams sync on their own named barriers (`bar.sync 1 + team`).
The work is UNITS, one per (span of `kSpanDigits` = 128 digits, batch, window, head), the window the carry's
`kWindowWords` = 16 words (the host entry refuses any other: it is the lists' grain). A unit's place is the span
outermost, then batch, window and head innermost, so the blocks at work together read one window of every head: the
rows `[B, T, H, b_l]` hold a token's heads, and a head's spans, side by side (a DRAM page's bytes together in time),
and the columns `[B, T, H]` a token's heads in one 32-byte sector, which a head-major order fetched once a head (84 MB
read against 67 MB of rows at `nl64k-dense`, four heads; heads innermost 75 MB). The unit index runs span innermost,
then head, window, batch (`nl64k-alt-k4`: 363 against 350 µs with the span outermost). Block `b` takes units `b`, `b +
grid`, ... : a static deal, the grid `min(units, SMs)`.

A unit is the span's ITEMS ([the plan](#the-plan)). Team 0 takes the write list level: a sparse one as `kHalfDigits`
= 64-digit HALVES held whole in the held window ([the load](#load)), a dense one as a ring item whose maxima it
keeps; team 1 takes every other level's span as ring items. A team's warp `w` owns the window's tokens `64 w ..` in
every item. The teams are split by data, so a team's build is covered only by the other team's stream; that is the
form's limit, measured at `facts-alt-k4` (the overlap gate): 352 µs against the previous form's 300.

Why one SM-wide pipeline of this size, measured on the earlier form at `(256, 256)` on sm_86: more resident warps than
16 an SM made the rows' stream slower, not faster (the DRAM serves fewer bytes a page visit when more streams
interleave); rows read head-major ran 113 against 212 token-major at 16 heads.

<a id="the-plan"></a>

## The plan: a span's items in order

`plan_units` (host) builds, once a call, each span's item list, every unit of the span the same: a code per item
(`plan_code`: kind, side, level, first group, groups, team). Kinds: `kItemHalf` a write list half (team 0),
`kItemDenseMax` a dense write list level's span for its maxima (team 0; since 2026-10-10 a write side without one sparse level hands the kernel its first dense level as the COLUMNS LEVEL, so every call gets its columns and gains, and the op discards that level's exponents), `kItemReadList` the read list level's span
(team 1, with lists), `kItemVote` every other sparse level's span (team 1). Team 0's halves are interleaved with team
1's items (the first half, half the ring items, the second half, the rest) so the two streams' builds fall apart. The
plan rides in `FactsParams`, `kMaxPlan` codes a span, `kMaxSpans` spans.

<a id="shared-memory"></a>

## The block's shared memory

`FactsSmem` by offset, dynamic: the held window (`kHalfBytes` = 32 KB, 16 regions of 2 KB; a dense list level streams
through it as team 0's rings instead and keeps its maxima past them), team 0's columns (`kWindowTokens` float2 = 4 KB),
its two half tiles (`tile_at` layout, 64 digits × 16 words each), its half union words and its maxima; team 1's rings
(8 warps × 2 stages × 1 KB), its two span tiles (128 digits × 16 words) and their union words; the builders' tables
(16 bytes a (digit, pass), 32 keys a warp) and marks. The static shared memory beside it holds the streams' barriers
and the phase clock; the sum fits sm_86's 99 KB (a static assertion), and the host entry checks the compiled
kernel's own static bytes against the device's opt-in limit.

<a id="load"></a>

## The load: a held window and per-warp rings

Every copy is `cp.async` 16-byte quanta (`ops::stage_tile_n`: a partial quantum's first bytes, nothing past them
read), zero-filled past `L` and past the rows' width; the alignment law is what makes the quanta legal, and a whole
row moves so each copy keeps its global run's 32-byte phase (a 16-byte lane pair one L2 sector request,
docs/internals/common/ops.md#stage-tile-n).

A RING ITEM (a span of a level): each warp streams its tokens through its own ring of `kRingDepth` = 2 stages of
`kStageBytes` = 1 KB, a stage 8 tokens' 128-byte runs of the span (a dense list level's f16 rows: 4 tokens of 256
bytes); `Stream` is a warp's place in its team's items, its lane's source and run prepared once an item, a stage's
step a token block. A stage is refilled as soon as it is voted, signalled by the warp's own `mbar_arrive_when_landed`
barrier object, waited on with `mbar_wait_sleep` (the poll sleeps: on sm_86 a tight poll takes the issue slots of the
warps doing the work).

A WRITE HALF is held whole: 64 digits × the window's 512 tokens in the held window, a warp's 32-token blocks at
`kHalfRegion` = 2 KB each, a token's two 32-digit groups' 16-byte quanta in the 1 KB rows of their groups at row
`ring_row(t)` = `4 (t % 8) + t / 8` (the four token blocks a lane group reads together sit in four consecutive rows,
so a warp's 4-byte reads meet 32 banks). The build needs the bytes for the coefficients, so the window is refilled only
once the half's build ends; there is no room on 99 KB for a second one.

<a id="vote"></a>

## The vote: a lane a (token block, digit quad)

Lane `(tk, dq)` = `(lane / 8, lane % 8)` reads token `8 tk + i`'s digits `4 dq .. 4 dq + 3` as one word, `i` 0..7.
THE VOTE: `((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x` sets bit 7 of each byte iff the byte is nonzero (three operations for
four digits, `nonzero_bytes`); shifted right by `7 - i` it lands at bit `i` of its byte, so after the 8 tokens byte
`k` of the lane's register holds digit `4 dq + k`'s 8 token bits. A ring stage is 8 tokens (`ring_flags`), a warp's
four stages per 32-token block giving four such registers that two byte permutes fold into the digit's 32 token bits;
a half's vote reads the held window instead and transposes across the four token blocks by two `shfl_xor`
exchanges (`half_vote`). Either way a digit's word is masked by the digit's width-mask bit (`digit_in`).

<a id="tiles"></a>

## The tiles and the bits out

An item's words are written into a shared TILE (`tile_at`: digit `d` word `j` at `16 d + (j ^ (d / 2 % 16))`, so a
warp's column writes, a digit a lane, and its row reads, a row a half-warp, each meet 32 banks), and after the team
barrier `bits_out` stores each digit's window of words as one 64-byte run (two digits a store, a half-warp each) and
each (group, 64-token tile)'s union, a digit a lane, as one ballot. A half's tile alternates between two so the
next half's vote does not wait on the previous store. Whole sectors, 64-byte aligned when `L` is a whole number of
windows, which the prefill requires.

<a id="columns"></a>
<a id="factor-products"></a>

## The window's columns

    m(t) = g_write(t) × Π_l scale_write_l(t)      (the gain first, then the levels in order)      F(t) = m(t) × s(t)

Team 0 loads a unit's 512 tokens' `(m, F)` pairs into shared memory (`load_column`; `s` the value scale, a token past
`L` zero), a warp its own 64 tokens (so a dense list level's stage loop reads only what its warp wrote), before its first item: the build reads them for the coefficients, the gains' pass for the window's largest.
The load is exposed at a unit's start (~20% of team 0's time at `nl64k-alt-k4`). The span-0 unit also stores each
token's `(F, m)` as one float2 to the `[BH, L, 2]` plane `fm` (the factor products), so the fold reads a token's two
products in one 8-byte load where it read four planes (ruled 2026-10-04: precompute what the fold's restage would
multiply); at 8 bytes a token-head it adds no `O(L·N)` term.

<a id="gains"></a>

## The gains

The unit of span 0 stores each token's `(m, F)` as an f16 pair at its window's powers of two (`gains16`) and the
exponents (`gains_exp`): the window's largest `m` and `F` (`store_gains`, every team-0 warp over the 512 columns, the
warps' stores split by 64-token slice) give `gains_shift(top)` = `-4 - floor(log2 top)` (the largest in [2^-4, 2^-3)),
clamped to [-126, 127], 0 for a zero largest. The first item's team barrier is the only synchronisation it needs.

<a id="lists"></a>

## The list exponents: the fp16 fold's scales

    K[w, r, 0]    = ceil(log2(max over the window's tokens of b(t, r) × m(t)))
    K[w, r, 1]    = ceil(log2(max over the window's tokens of b(t, r) × m(t) × s(t)))

per window `w` and digit `r` of the WRITE LIST LEVEL — the write side's one sparse level, the carve's `fold.lists`
(`rola.ops.facts.list_level`; none for a side with no sparse level or with two, whose lists are composite) — `b` the
token's write coefficient on the byte scale: a sparse
level's u8 digit, a dense level's f16 times 255 (exact; its floor stored,
[`liveness_contract.md`](liveness_contract.md#floor)), floored at -126 (the smallest normal's exponent: a list no token
writes). Stored int16 `[BH, windows, B_list, 2]`. The fp16 fold puts each list's largest coefficient at a fixed power
of two by them, so one scale a list keeps every coefficient in fp16's range (ruled 2026-10-03). A sparse list level's
maxima come out of the build's first walk (one `atomicMax` a run of a digit into the team's maxima); a dense one's
from its ring stages (`dense_stage`: lane `l` digits `4 l ..` as f16 at the byte scale, the stage's largest `b m`, `b
F`), joined over the team's warps by `atomicMax` and read after the team barrier.

<a id="lists-out"></a>

## The lists out

A side's LIST LEVEL is its one sparse level (`rola.ops.facts.list_level`, the carve's `lists` for the side). A side
with no sparse level has no lists (every token of the window is a slot); a side with two or more gets its words and no
lists, `None` in `Facts`, and `side_lists` refuses them: they would be COMPOSITE, keyed by a digit of each level, which
this pass does not build. A list level gets, per 512-token
window and 32-digit group, a UNION (the group's live tokens ranked, `union_tok`, `union_count`) and per digit a LIST:
the union's slots of its tokens in slot order, in PASSES of `kPassSlots` = 128 slots, each pass round-robin by
slot & 7 (`round_robin`: the entry's place from the pass's residue counts), compact, at most `kListCap` = 512
entries, the pass starts and the end `kListAt` = 8 words a list (`at`). A read entry is its slot (`slot`); a write entry
one 8-byte record (slot, 0, c_v, c_m), the pair at 2^(`kCoefTop` - E), E the list's exponent floored at `kUnitMin`.

`build` runs one 32-digit group's lists on four warps (`kParts`), `part` a quarter of its digits: lane `(dl, q)` digit
`dl`'s words `4 q ..`, the warp's CELLS (`16 dl + j` word `j` of digit `dl`). The union's tokens go out first; the
warp's entries are then dealt evenly over the lanes (lane `L` walks entries `L total / 32` on, found from the lanes'
counts). A first walk totals each digit's pass by residue into the warp's TABLE (and on the write side maxes each
digit's `b m`, `b F` over its entries); a segmented scan of the lanes' last runs gives each lane the counts before its
first; the table's starts, the exponents and scales come out per (digit, pass); the second walk ranks each entry among
its residue, places it and writes it — on the write side with its coefficient pair from its byte and the token's
columns. A build is ~1,000-1,400 instructions a warp for ~64 entries at `facts-alt-k4`, ~7 cycles an instruction
(dependency waits, not shared-memory queueing).

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[16][4]` int64 buffer (warp × columns, vote, lists, store; warps 0-7 team 0,
8-15 team 1) that every block's warps add their cycles into; `rola.ops.facts.ledger_bind` is the Python side. An unbound
ledger issues no lap at all (the clock is gated); a bound ledger moves no output (`tests/integration/test_facts_pass.py`). Instrumenting a build with `clock64` and global
atomics inflates its cycles by 25-35%, so the ledger books only phase boundaries.

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView` (base, the batch, token and head strides in bytes, the rows' width), per write level and
for the gain and value scale a `ColumnView`, the outputs (`ListsOut` per side), the depth, the padded widths and row
bases, per side the dense bitmask and the digit mask's words, the unit count and the plan. It is a kernel parameter
of under a kilobyte, so the pass takes no allocation to learn its shape. No logical width is in it: `b_l` reaches the
mask at the seam and bounds nothing else — the host entry refuses a mask that names a digit past a level's rows, so
the kernel never reads beyond them.

## The budget line and the floor

The line, per head at `(256, 256)`, `L = 64K`: the rows' bytes read once (67 MB both sides read whole, ~84 µs at
800 GB/s) and the words written (8 MB, ~10 µs), a DRAM floor of ~95 µs at 1.3x; a dense level unread and unwritten
takes its bytes off the floor. Measured (ncu kernel time, a head's share at four heads / one batch-head, sm_86):

| cell | the line | measured |
|---|---|---|
| `nl64k-dense` | <= 27 µs at four heads (16.8 MB of list rows, floor ~21) | 27.4–28.4 / 31.4–32.2 µs (DRAM 72–74%) |
| `nl64k-alt-k4` | <= 61 µs (floor ~47) | 49.6–50.3 / 57.2 µs |
| `flagship-cohort-k16` (`L = 1K`) | no regression past ~10 µs (an empty launch is 5.3) | 10.6–11.3 µs |

At four heads the pass is DRAM-bound where prefill runs. One batch-head is latency-bound: dense's list items (four a
window) each wait one DRAM latency on their columns, and too few are in flight to hide it; cohort-k16 is two windows,
its critical path one list item after the launch. What the load is: the rows staged by `cp.async` 16-byte copies into
per-warp rings and read from shared memory, the lanes shared over (token block, digit quad), the items heads
innermost, the columns loaded ahead of the ring, a dense list level's item a warp a unit ([the grid](#the-grid),
[the load](#the-load-a-held-window-and-per-warp-rings)). Byte loads a lane cost ~8 instructions a (digit, token) and
left the pass issue-bound (18.5M instructions at alt); 32-bit load offsets and two units a list item at four warps
measured worse; four units a list item on every list level helped dense and cost alt and cohort.

The sm_80 form is this one, re-measured for its residency point (HBM's page behaviour is not GDDR6X's). On sm_90 the
ring's slots fill by TMA: a word's `[32 tokens × 32 digits]` box in one instruction in place of 64 `cp.async` quanta;
the rest is unchanged.
