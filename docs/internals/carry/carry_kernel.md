# `carry/carry_kernel.cuh` — the carry body

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`. The plan it runs on is
[`box.md`](box.md); the bytes are [`smem_ledger.md`](smem_ledger.md); the launch is
[`carry.md`](carry.md). The design's rulings are on card P81; this page is what the code
does, and why each piece is shaped the way it is.

<a id="frame"></a>
## The frame, in one picture

One CTA owns one BOX of `BC = 256` leaves for the whole call, eight warps, one CTA per SM.
The state for those leaves is fp32 in registers: a warp OWNS two of the box's sixteen
sixteen-leaf boxes (`w` and `w + 8`) and every channel of them. The call walks the sequence
in WINDOWS of 512 tokens; per window, in order:

1. the SNAPSHOT: the state's high halves written to shared memory once, the readout's operand;
2. the READOUT over the read side's sorted tiles, each warp its own tiles;
3. the HEAD of the next window: the read side's sort, the write side's words;
4. the FOLD over the write side's union pool, each warp its own boxes.

The readout reads the snapshot and its own staged tiles; the fold reads the pool and writes
registers. Nothing in them touches until the window end: snapshot before either, fold
complete before the next snapshot, one CTA barrier at each. The head's sort has three sync
points of its own. Those five barriers are the window's.

<a id="carve"></a>
## The carve, and why it is two different carves

The coefficient of token `t` on leaf `(b, pos)` is a Kronecker product,
`inner[t][pos] · outer[t][b] · gain[t]`, and it is never materialized.

- **The fold** is `S[leaf][ch] += Σ_t C[t][leaf] · V[t][ch]`: the contraction is tokens,
  the outputs are leaves and channels. Partitioning leaves across warps (the STATE CARVE)
  leaves every warp's result complete in its own registers, and lets one modulated A serve
  all eight n-tiles of V: the per-box work (two shuffles for the box's scalar pairs, four
  packed multiplies on A) is paid once per eight HMMAs.
- **The readout** is `y[tok][ch] = Σ_leaf C[tok][leaf] · S[leaf][ch]`: the contraction is
  leaves. Partitioning leaves would make every warp's y a partial needing a cross-warp sum;
  partitioning TOKENS (the POSITION CARVE) keeps each row complete in one warp. The price is
  that the readout needs every box's state, not just its own: the snapshot.

The rule underneath: partition an output axis and results are complete; partition the
contraction axis and they have to be summed. The value carve (channels, an output of both)
needs neither the snapshot nor a sum, but pays its per-box work once per HMMA; that is the
`kNT` knob's other end and the record is in FINDINGS §46.

<a id="layout"></a>
## A side's layout

`SideLayout` (`carry/layout.cuh`) classifies each level of a side's carve order by where its
digit falls against the sixteen-leaf position bits: INNER levels lie within them (their
runs make the box's tile, box independent), OUTER levels above (a scalar a box), a level
across bit four STRADDLES. It is derived on the host (`make_side_layout`) and handed to the
kernel in the parameter block, so every field is a constant-bank operand and none a register
— the earlier in-kernel struct cost 32 registers a side. Every array is indexed with a
compile-time level: a runtime index would put it in local memory.

The shipped arm's layouts are PLAIN: one inner level of sixteen digits, one outer level of
a digit a box. A composed or straddling arm is refused at compile time in the fill, the fold
and the readout (`static_assert`), where it would compose its runs per token.

<a id="factor-rows"></a>
## The factor rows

`abase[l]` is the box's first amplitude column at level `l`, which is also the liveness
words' digit row base. `run_bases` selects the inner and outer levels' byte bases from it
statically; `stage_box_words` stages both sides' words a window ahead by `cp.async`, ROUND-MAJOR
(`words(side, rho)`: a round's `kBoxRows` digit words contiguous), so the head's any-digit OR and
its sixteen box words are eight 16-byte loads a side rather than thirty-two scalar ones -- the
row-major form was MIO-bound (FINDINGS §46).

<a id="order"></a>
<a id="read-order"></a>
## The head

Read side: a counting sort of the window's tokens by FIRST LIVE BOX, the dead (and those
past the window) last; the ORDER (`order`, rank → token, u16), the TILE MASKS (`tilemask`, a
tile's live boxes) and the live count. A warp takes two (round) groups, lane = token: the
round's box words (below), the token's mask bit by bit, its bucket, the bucket counts by
`__match_any_sync`; warp 0 scans the count table in place into bases (a lane a bucket); the
scatter; the tile masks by reduction over the order. The `p.order` dial chooses the identity
order instead (one bucket for the live): the sort's A/B.

Write side: per round the box words, each warp's WORD (`warpwords`: tokens live in a box it
owns), the UNION word (`unionwords`: live in any box) and, by warp 1, the union's prefix by
round (`prefix`) and its total. No write-side sort, no lists: the fold compacts as it walks.

<a id="box-words"></a>
### The box words

`box_words` forms a round's sixteen box words from the staged liveness words: an AND over
the outer levels' digit words at the box's digits, over the inner levels' any-digit words,
and over the straddling level's class. The plain layout is two runtime row bases and static
offsets; a composed or straddling layout walks its level lists in uniform loops. It never
branches on a lane value, so the collectives that follow stay provably converged
(KERNEL_STANDARDS §20).

<a id="state"></a>
## The state

`State::c[kDealt][kNT][4]`: the owned boxes' state as m16n8k16 C-layout tiles (lane (r, q):
leaves `r`, `r + 8` of the box at channels `2q`, `2q + 1` of the n-tile). `State::m[kDealt][4]`:
each box's masses as an MMA accumulator whose column 0 is the mass (lane (r, 0): leaves `r`
in `[0]`, `r + 8` in `[2]`); the fold accumulates it with a ones column, so no reduction
ever forms it.

<a id="state-io"></a>
### The sweeps

The state crosses the pages as whole 128-byte rows, staged in shared memory: the high
halves in the snapshot, the low halves across the pool region (free at both edges of a call;
`kRegionBytes >= kSnapshotBytes`), the masses in the mass row. `state_copy<Store>` is the
CTA's coalesced copy between the pages and the stages, one 16-byte chunk an item dealt by
thread so eight lanes write one row, the masses a leaf a thread; `state_gather` joins a
warp's boxes off the stages into its accumulators, and `snapshot_publish<true>` is its
mirror at the exit (the window's publish is `<false>`, hi only). A page this call only reads
gives hi alone; a page without a slot, or inactive, is zero; the exit stores only the pages
this call writes. Once a call each, a rendezvous on each side of the stage.

The first form scattered 2-byte halves straight from the accumulator fragments: 128 stores a
lane, each address register's rewrite waiting on the previous store's scoreboard. At
alt-k4 that exit was 30% of the kernel's stall samples (12% dense), invisible to the phase
ledger until its launch carried a state plane.

<a id="snapshot"></a>
## The snapshot

`snapshot_publish`: each warp writes its boxes' high halves into the snapshot as
`[leaf in read order][channel]` pairs (`chan_row_off`: the 16-byte chunk XORed with the row,
so the readout's `ldmatrix` over eight rows is bank free), and their masses into the mass
row in read order. 32 KB once a window, skipped when the read side has no live token.

<a id="pool"></a>
## The pool

The write side's union-live tokens, by rank, in chunks of `kPoolTok` (80 at DV = 64) across
`kPoolSlots` (3) slots, each slot a token's V row, inner run, outer run and GAIN PAIR (the
aligned word of the gain row holding it; the walker takes the half its token's parity
names), and a ZERO ROW (row `kPoolTok`) the tail of a list pads from. Slots are under
`mbarrier` pairs: `full` completes at `kThreads` landed-copy arrivals
(`cp.async.mbarrier.arrive.noinc`), `empty` at `kWarps` arrivals, a warp's after its LAST
GATHER from the slot, not after its MMAs — a row is free once loaded into registers.
`PoolCursor` carries a use count a slot for the parities, and a window's chunks take slots
from 0 in turn so chunk 0 lands in slot 0, the slot free under the readout
(`kPrefetchFirst`).

`pool_fill`: the chunk's rounds by two ballots over the prefix, dealt to warps (`r0 + w`,
`r0 + w + warps`; a warp without a round skips it behind `ops::once`); a round is two passes
of sixteen tokens, lanes `j` and `j + 16` a token copying alternate 16-byte halves so each
instruction's pair covers whole sectors; a lane's token is live by the union word's bit,
its rank by popcount, in the chunk by range; an idle lane's zero-size copies go to the zero
row (a zero-size `cp.async` ZERO-FILLS its destination — the first form overwrote the next
warp's rows). One copy group; the arrive when it lands.

<a id="fold"></a>
## The fold

`fold`, per chunk: the chunk's rounds by the same two ballots; per round each lane's token
is KEPT when the warp's word has it and its rank is in the chunk; kept lanes queue their
pool row with the token's gain (`walk`, a 64-entry ring a warp) at the slot their prefix
popcount names; every sixteen queued, `fold_fragment`. The chunk two ahead is filled before
the walk (chunk 1 before the first chunk); the tail pads from the zero row; the slot is
released after the last gather.

`fold_fragment`: the sixteen entries shuffled to the `ldmatrix` lane maps (`run_lane_row`,
`v_lane_row`); A = the inner runs transposed (`ldmatrix.trans`: lane (r, q) holds positions
`r`, `r + 8` at tokens `2q..`, `2q + 8..`), the outer runs transposed the same way (lane
(r, q) holds boxes `r`, `r + 8`: pairs `[0]` (box r, low tokens), `[1]` (box r + 8, low),
`[2]` (box r, high), `[3]` (box r + 8, high)), V as B (four `ldmatrix.trans` for eight
n-tiles); the outer pairs scaled by the tokens' gains (two shuffles of the entries); then a
box: its pairs by shuffle from the lanes holding boxes `b & 7`, A scaled by them (four
packed multiplies), `kNT` HMMAs into the box's state and one against a ones column into
its mass. About 102 instructions and 18 HMMAs a fragment at DV = 64.

<a id="readout"></a>
## The readout

`readout`: tile `k` to warp `k % warps`; `readout_issue` stages the tile's inner and outer
runs into the warp's private ring a tile ahead (`rring`; lanes `l`, `l + 16` a row; a rank
past the live count zero-filled), one copy group a tile, `cp.async.wait_group 1` and a warp
edge before the take. No CTA barrier in the phase.

`readout_tile`: A = the inner tile by `ldmatrix` once; per live box (the tile mask's set
bits): the rows' outer factors off the outer tile, A scaled per row (four packed
multiplies), B the box off the snapshot (`ldmatrix.trans`, two n-tiles a load), `kNT`
HMMAs into y and one against the box's mass column (`massrow`, lanes `r == 0`) into den.
Then the rows out: four rows a pass through the drain stage (`drain`), row-major, so each
reduction instruction covers one 128-byte line (reductions cost by the lines they touch;
fragment layout costs ~3x — FINDINGS §37); den by reduction from the lanes holding it; dead
ranks leave nothing. `num` and `den` are summed across owner CTAs, so every row leaves by
reduction in every form.

<a id="edges"></a>
## Edges

`Edge::kCounts` and `Edge::kOrder` are the head's named barriers; `ops::rendezvous` the
window's two; `warp_edge` (barrier `kWarpEdge + warp`) orders a warp's private rows across
its lanes and publishes its landed copies. The pool's slots are `mbarrier` pairs, never a
CTA barrier.

<a id="phase-ledger"></a>
## The phase ledger

`PhaseClock` accumulates a warp's cycles per phase (`CarryPhase`) in its shared row by
lane 0 (`%clock64` laps), added into `CarryParams::ledger` at the kernel's end when bound
(`carry_ledger_bind`). A debug instrument; unbound in production.

<a id="budgets"></a>
## Budgets

The budget of each component is the design's own operation count, written in
`benchmarks/bench/carry_model.py` and generated into `tools/budgets/carry.json` per cell
(instructions per CTA-window; the formulas name every operation they count). The region
ledger (`tools/region_ledger.py`) attributes a launch's executed instructions and stalls to
the component each SASS instruction inlines from, with barrier spins split out, and is red
over budget. Flagship dense: head 4.0K, fill 6.0K, fold 21.5K (4,608 HMMAs), readout 17.6K
(4,608 HMMAs); alt-k4: 15.3K in all, the head the largest block.

<a id="part-harness"></a>
## The part harness

`benchmarks/unit/bench_carry_parts.py` builds `benchmarks/unit/carry_parts/carry_parts.cu`
(one driver kernel a component, each the kernel's own prologue plus that component, on the
call the kernel would run) and checks each against a pure reference: the head's order,
masks, words and prefix; the fill's slots byte for byte, zero row included; the fold's state
and masses against the Kronecker fold with the kernel's bf16 arithmetic emulated. `--ncu`
adds executed instructions per CTA-window beside the budget. KERNEL_STANDARDS §19.
