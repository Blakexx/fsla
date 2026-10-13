# `csrc/rola/src/facts/facts.cuh` / `facts.cu` — the facts pass

Mirror doc for the one pass that reads a call's bundle before the carry and the intra do: both sides' class-1
liveness words and their tile unions, the window's columns (the factor products and the gains), the write list
level's exponents, and each side's lists in the carry's final form, in one launch. What the words ARE — their layout,
the folds over them and why each is the grain it is — is [`liveness_contract.md`](liveness_contract.md), which this
kernel does not restate and must not contradict; its host seam is [`facts_api.md`](facts_api.md).
`rola.engine.facts.liveness.liveness_words` is the words' pure-torch MODEL, `rola.engine.facts.lists.final_lists` the
lists', `rola.engine.facts.coloring` the coloring's, `list_exponents` in `tests/oracle/fold_reference.py` the
exponents' definition, and `tests/integration/test_facts_pass.py` holds the kernel to all four.

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

`kFactsWarps` = 8 warps a CTA, one CTA a UNIT, three CTAs an SM on sm_86 and sm_80 (its registers a thread held to
three CTAs by `kFactsCtasMin`; sm_86's 31 KB of shared memory a CTA allows three as well): a unit is a plan-fixed load
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

`FactsSmem` by offset, dynamic, 31.3 KB:

| tenant | bytes | live |
|---|---|---|
| the window: 512 tokens × the group's 32 bytes | 16,384 | the load to the lists' last coefficient |
| the bit tile (`tile_at`) | 2,048 | the vote to the end |
| each token's color, then its slot (u16) | 1,024 | the claims to the end |
| the union by slot (u16) | 1,024 | the slots to the end |
| the columns `(m, F)` (float2) | 4,096 | a write unit's load to the end |
| SCRATCH: each token's digits, the claim queue (a ring of 1,024), each digit's color counts and capacity, its singles before each word and its singles' color ends; then, over the dead digits, the placement's tables | 6,272 | the coloring, then the lists |
| the union's words, each word's color counts, the union's color counts, the claim rounds' appends, the warps' maxima | 464 | through the unit |

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
bits, masked by the digit's width-mask bit (`digit_in`). Warp `w` votes the window's words `2w` and `2w + 1`.

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
coefficient at a fixed power of two by them (ruled 2026-10-03). They come out of the lists' first walk (each lane's
largest over its entries, joined over the digit's eight lanes).

<a id="coloring"></a>

## The coloring: one slot map a (window, side, group)

Every slot-addressed consumer -- the readout's pool and panel rows, the fold's pool rows, the den pass's staged rows --
reads one colored slot map a union (ruled 2026-10-12). A slot's residue mod 8 is the bank group of every row indexed
by it, and the carry steps a list 8 tokens at a time, so each union token gets a COLOR and its SLOT is
`8 × rank + color`, `rank` its place among its color's tokens in token order: the RESIDUE LAYOUT (ruled 2026-10-13),
where the slot is the color and no consumer is told one. The bounds: each digit at most `ceil(n / 8)` tokens a color
(`n` its tokens; its CAPACITY, soft: a token past it costs its list a step, never a conflict) and the union at most
`kUnionCap` = 64 a color (hard: a union's slots end by 512).

THE INPUTS, every warp its two words: each token's digits from a 32 × 32 bit transpose of its word, three ballots for
the union's words (tokens in one digit or more, two or more, three or more), each digit's capacity; then the QUEUE of
the multi-digit tokens (three-or-more first, each in token order, a token's place its kind's tokens in the earlier
words plus its rank in its word) and, on warps 1-4, each digit's singles before each word (the deal's index base).

THE CLAIMS, 32 a batch (a lane each): a token sums its digits' color counts and takes the least-used color allowed by
every digit's capacity (full where a count reached it) and the union's cap (ties to the union's least-used, then
`(c - t) & 7`), claiming it on the union and every digit by shared atomics; a claim that finds a bound over is
withdrawn and its token requeued. Up to `kClaimRounds` = 4 rounds run every warp a batch while more than a batch is
queued, losers appended past the round's queue -- a ring of 1,024 holding at most a window's 512 tokens, so a round's
appends never reach its reads; then warp 0 takes batches to the queue's end, a batch that keeps nothing followed by its
first token alone, which always keeps (the rounds may keep nothing; warp 0's batches cannot, so the claims end). A
token no color allows is BLOCKED and takes the union's least-used color under the cap.

THE SINGLES (in one digit), warp 0, lane `d` digit `d`: the union fills to a level, the least whose quotas
(`max(0, level - n_c)`, `n_c` a color's tokens so far) hold every single, pass by pass -- each digit wants its singles
at its room's colors (under its capacity) that have quota left, in the rotation `(d + i) & 7`; each color grants in
digit order while its quota lasts (a prefix of the wants over the lanes, two colors a 32-bit word); a pass that grants
nothing raises the level -- and past the union's cap the rest go by quota alone, in digit order over the colors' room.
Each digit's singles take their colors in color order, its tokens in token order: the digit's color ends are posted and
each single's color is dealt in the slots phase from its index among its digit's singles.

The coloring's model (`coloring.color`) colors the multi-digit tokens serially in token order and the singles by the
same passes; the kernel's claims race, so its coloring is graded by `coloring.check` and the residue layout, never by
equality. Measured on the kernel at 4-of-256 rows: no bound broken, ~12 gap slots a union, 0.02 tokens a union over a
capacity; at 16-of-256 ~20 gaps and ~4 over -- the price of the rounds' concurrent claims, which see each other's
counts late (warp 0's batches alone left 0.3-0.8 over at twice the claims' time; the model 15 gaps and none). A gap's
row is zero-filled by the carry, so a gap costs it no read; a token over its capacity costs its list a step in every
CTA reading the union.

THE SLOTS, every warp its two words: each token's color (a multi-digit token's claimed, a single's dealt), three ballots
giving each color's tokens in the word; the words' color counts meet in shared memory, and each token's slot is
`8 × (its color's tokens in the earlier words + its rank in the word) + color`. The union by slot (`union_tok`) holds
each slot's token, a GAP's `kGap` (`carry/carry.cuh`: token 0 with bit 15 set, so a row fetch masked to the window stays
inside it), and the EXTENT `8 × max_c n_c` its slots (`union_count`, int32).

<a id="lists-out"></a>

## The lists out

Per list (a digit of the list level), its tokens in PASSES of the carry's pass for the side — the readout's
`pass_slots` for the read side, the fold's `unit_slots` for the write side, the arm's form on the device's card
(`rola.ops.carry.form`), a power of two from 128 to 512 (the host refuses another: a list's passes and its end must fit
its `kListAt` = 8 starts) — each pass in ROUND order: step `i` takes each color's `i`-th token, colors in order, a color
with no `i`-th token a PAD, so 8 consecutive entries never repeat a residue. A pass of a list whose most-used color has
`S` tokens there holds `8 S` entries; a list's entries are at most 512 (`S` is at most a pass's slots / 8). `at[q]` is
pass `q`'s start, and from the last pass on the list's end.

An entry is 8 bytes, the carry's `uint2`:
- READ: the slot's pool row offset in its pass with the row's swizzle folded in (`local × row + 16 × chunk0`, `local`
  the slot less its pass's first, `chunk0` the carry's `chunk_at` of chunk 0: an 8-chunk row XORs by the residue, a row
  of an odd number of 64-byte quarters by the residue's high bits), 0; the coefficient `f16(byte / 256)`, the slot's
  panel row offset past the dummy rows (`kPadBytes + panel_row(local)`). A pad: its color's row (`local` = the color)
  and dummy panel row, coefficient 0.
- WRITE: the slot, 0; `c_v = f16(fp32(b F) 2^(4 - E_v))`, `c_m = f16(fp32(b m) 2^(4 - E_m))`, `E = max(K, -123)` the
  list's exponent. A pad: the pass's first slot of its color, both coefficients 0.

A warp's HEADS a (window, group): its 4 lists `w + 8 j`'s first 16 entries, copied as they are written.

THE WALKS, every warp its 4 lists `w + 8 j` (its own heads), 8 lanes a list, lane `s` the list's words `2s` and
`2s + 1`. The first walk counts each entry into its (list, pass)'s color count (a shared byte) and the lane's own
color counts, and on the write side takes the list maxima. A segmented scan of the lanes' color counts gives each lane
the list's tokens of each color before its words; lane `s` = pass `q` turns its pass's counts into the pass's steps,
start and the colors' counts in the earlier passes, and writes the starts (`at`). The second walk places each entry at
`start_q + 8 × (its rank among the list's tokens of its color, less those in earlier passes) + color` and writes it;
lane `q` writes its pass's pads.

<a id="access-table"></a>

## The access table (KERNEL_STANDARDS §30)

| buffer | writes | reads | wavefronts |
|---|---|---|---|
| window | `cp.async` 16-byte quanta, a lane pair a token | the vote: lane `(tk, dq)` row `4 i + tk` word `dq`; the walks: a byte at (token, digit) | the vote 1.0 a load (32 banks by `ring_row`); a walk's byte loads land on random banks, ~1.5; the copies' cost is their global pattern (`ldgsts-copy-ranking`) |
| tile | the vote: a digit a lane, one word | the bits out: a half-warp a digit row; the transposes and tile unions: a digit a lane; the walks and singles: a lane a list's words | 1.0 except the walks' loads: their 4 lists share a bank half, 4.0 against 2.0 ideal, two loads a lane a unit |
| color / slot | the claims: a u16 at a token; the slots: a lane a token | the walks: a u16 at a token | the slots 1.0; the scattered ones random, ~1.5 |
| union | the slots: a u16 at a slot | the copy out: 4 bytes a lane | out 1.0; a word's slots spread over residues, ~2.0 |
| columns | a lane a token | the walks: a float2 at a token | random, ~1.5 |
| scratch (coloring) | the queue: a u16 a token; the claims: atomics at a (digit, color) pair; the color ends: 16 bytes a digit | a digit's counts as one 16-byte load; the deal: a digit's index base and ends | per claim, random |
| scratch (tables) | an atomic a walk-1 entry at its (list, pass, color half) | a lane a pass; a walk-2 entry's 8-byte load | a list's 8 lanes share its few words: same-address atomics serialize, ~2 an entry at sparse |

<a id="phase-ledger"></a>

## The phase ledger

`rola_facts_ledger_bind` binds a `[8][7]` int64 buffer (warp × load, vote, inputs, claims, singles, slots, lists) that
every CTA's warps add their cycles into; `rola.ops.facts.ledger_bind` is the Python side. Warps 1-7 book their wait for
warp 0's claims and singles into those phases. An unbound ledger issues no lap; a bound ledger moves no output
(`tests/integration/test_facts_pass.py`).

<a id="factsparams"></a>

## `FactsParams` — the operand block, by value

Per side and level a `RowView`, per write level and for the gain and value scale a `ColumnView`, the outputs (a
`ListsOut` a side), the depth, the padded widths, row and group bases, each side's list level and pass (log2), the read
pool row's bytes, the window's unit codes, per side the digit mask's words. Under a kilobyte and a half. No logical
width is in it: `b_l` reaches the mask at the seam and bounds nothing else — the host entry refuses a mask that names a
digit past a level's rows.

<a id="budget"></a>

## The budget line and the floor

THE LINE (ruled 2026-10-12): the whole pass, coloring included, at <= 1.15x its required bytes' DRAM floor on the lab's
`facts_block` cell (rows `[4 level-sides, 65,536, 4, 256]` u8, 268 MB, k live digits a token-head; the write side lists
level 0 and the read side level 1, the other level of each voted), stated then as <= 389 µs at k 4 and <= 444 µs at
k 16 on the RTX 3080 Ti from the lab's floor (338 and 386 µs). That floor left out the words (33.5 MB) and priced the
lab's smaller lists; this pass's required bytes -- the rows, the columns, the words and tile unions, every list's
entries (its tokens and its pads), heads, starts, union and extent, the exponents, the factor products and gains --
are 449 µs at k 4 and 578 µs at k 16 at the lab's flat read (854 GB/s), so 1.15x is 516 and 665 µs.

MEASURED (`measure/harness/bench_facts_block.py`, sm_86, the sm_86 even arm's form, best of runs; unlocked clocks vary
runs by up to 15%): 707 µs at k 4 (1.58x the floor) and ~1,270 µs at k 16 (2.2x). The line is missed. The phase ledger,
cycles a unit on warp 0 (a vote unit load and vote only):

| phase | k 4 | k 16 |
|---|---|---|
| load | 3,300 | 3,250 |
| vote and bits out | 2,280 | 2,320 |
| coloring inputs and queue | 4,100 | 4,380 |
| claims | 7,700 | 29,200 |
| singles | 6,640 | 9,930 |
| slots | 3,330 | 3,300 |
| lists | 6,160 | 12,250 |

The pass is latency-bound: a list unit's coloring is a chain of dependent shared-memory steps (a claim batch ~2-3K
cycles), and three CTAs an SM leave three such chains an SM where the coloring's lab priced twelve
(`docs/lab/color_union.md`: its cost a box-window at 12 warps an SM). The serial singles of the first form (one single a
digit a round) cost 18K cycles at k 4 and the claims 13K before the claims ran on every warp.

Per architecture: sm_80 runs the same form at three CTAs an SM (registers); on sm_90 the window fills by TMA (a
`[512 tokens × 32 bytes]` box of the head's plane in one instruction) under an mbarrier, the rest unchanged. Scaling:
O(L × rows) bytes read, O(L k) entries, O(L B / 32) unions and heads; nothing O(L N).
