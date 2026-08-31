# The carry box's shared-memory ledger

Mirrors the `constexpr` offsets in `csrc/rola/src/carry/box.cuh` and the `Smem` accessors
in `carry/carry_kernel.cuh`. Numbers below are the flagship arm, `D = 2`, `DV = 64`,
`warps_per_cta = 8`, hence `BC = 256` leaves, `LdX = 72`, `LdO = 68`, `LdV = 64`.

The ledger's purpose is that shared memory is allocated BY NAME and BY MOMENT. A stage
that needs bytes states which tenant it is and when it is alive; a stage that quietly grew
an allocation would move the residency the whole design is built on.

## What is allocated, today

| offset | tenant | bytes | owner |
|---|---|---|---|
| 0 | the transit region, `BC * LdX * 2` | 36,864 | the exchange between the two sides' carves |
| 36,864 | the union fold's value ring, `kSegmentSlots * kSeg * LdV * 2` | 2,048 | the union fold's staging |
| 38,912 | the union fold's column blocks, `kColSlots * kSpanTotal * kTokStride * 2` | 3,072 | the union fold's transpose |
| 41,984 | the union fold's segment gains, `kFactorSlots * kSeg * 2` | 64 | the deep index staging |
| 42,048 | the union fold's raw factor rows, `kFactorSlots * kSeg * kRawStride * 2` | 2,304 | the deep index staging |
| 44,352 | THE READ BLOCK, `max(union, sparse)` — the union readout's two shared raw slots (2,304) or the sparse readout's `warps_per_cta` private rings (9,472) | 9,472 | whichever form the readout runs |
| 53,824 | the box's page slots, `BC/16 * 4` | 64 | the prologue |
| 53,888 | the fill's ONE claim cursor | 4 | both union bodies' fills |
| 53,892 | the stream counters, `2 * warps_per_cta * 4` | 64 | the sparse bodies' `numProduced`/`numConsumed` |
| | **total** (16-byte aligned) | **53,968** | |

The frame's three edges and the streams' three take no bytes at all: they are NAMED BARRIERS
(`carry_kernel.md#edges`), which is what an edge with equal participation costs.

At sm_86's 101,376-byte per-CTA maximum this leaves the parked plane and the fill rule the
rest of the budget, and the residency is one CTA per SM by the launch bound rather than by
the allocation.

<a id="region"></a>
## The region, and the aliasing structure per moment

The transit region is the largest tenant and the only one whose bytes are re-let:

| moment | who lives in the region's bytes |
|---|---|
| the entry sweep | as many state page images as fit: `floor(36,864 / (16 * LdO * 4))` = 8 images of 4,352 bytes = 34,816 bytes, in 2 rounds over the box's 16 pages |
| the out leg, PHASE 1, the return leg | the transit region itself — `BC` rows of `LdX` bf16 channels, the accumulator's halves in the owner-local canonical order — and, from the pull to the store-back, the UNION readout's fan-in panel in the first 34,816 of those bytes |
| PHASE 2, union fold | nothing: the union fold's own tenants have their own bytes |
| PHASE 2, SPARSE fold | the eight private stream rings, `warps_per_cta * (kVTileBytes + kRawSlotBytes + kGainSlotBytes)` = 25,856 bytes |
| the exit sweep | the state page images again |

<a id="streams"></a>
### The sparse form's stream rings

A stream is a leaf range of `G` leaves inside ONE warp's sub-box, so its ring is that warp's
alone and no other warp reads or writes it. `warps_per_cta` streams are LIVE at a time — one
per warp — and the catalog (`leaves_per_warp / G` per warp) is walked in turns, which is what
keeps the resident cost `warps_per_cta * (one segment)` instead of `catalog * (one segment)`.
The ledger is sized at the FINEST grain the nesting law admits (`G = 16`, the page rectangle);
a coarser runtime `G` uses the same bytes with fewer turns.

| ring | per live stream, at the flagship | why |
|---|---|---|
| the fold's value tile | `kSeg * DV * 2` = 2,048 | the A operand, drained by one `ldmatrix.trans` burst |
| the fold's raw factor rows | `kSeg * kRawStride * 2` = 1,152 | the coefficient, expanded in place — the sparse body has NO column block |
| the fold's segment gains | `kSeg * 2` = 32 | folded into level 0's factor at the expansion |
| the readout's raw factor rows | `kSeg * kRawStride * 2` = 1,152 | the B operand, read as it lies |
| the readout's token ids | `kSeg * 2` = 32 | the window-relative token each compacted row carries, which the direct deposit is addressed by |

THE FOLD'S RINGS ALIAS THE REGION and the readout's do not. The fold's are the bigger pair
and PHASE 2 is the moment the region is otherwise idle; the WAR against the rejoin, which
reads the region, is closed by ONE crossing of `StreamEdge::kEmpty` at the sparse fold's
opening — the sparse body's own edge, at one crossing per window against the union fold's
two per segment. The readout's rings cannot alias the region, because the UNION readout's
panel may be living there in the same phase under a mixed schedule; they take the READ BLOCK,
which the union readout's own two shared slots would otherwise hold, and the two forms never
coexist. `kSparseFoldBytes <= kReadBlockOffset` is asserted in `box.cuh`, so a value width
whose rings outgrow the region (DV = 128) overflows into the union fold's feed, which is dead
whenever the fold is sparse.

The page slots, the claim cursor and the stream counters are PERSISTENT: they are alive from the prologue to the
exit and are never re-let. So is the union bodies' raw factor ring — a tenant that must survive a
phase change is not a region tenant, and both cover slots fill their ring while the region
still holds the transit.

Parking the accumulator's low plane in shared memory holds the region through PHASE 1, which
is why the output blocks are a priced row on this table rather than a surprise later.

## Reserved by name, for the stages that fill them

These are named here so a later stage lands as an entry on this table rather than as an
allocation nobody sized. None is allocated today.

| tenant | size, at the flagship arm | who fills it |
|---|---|---|
| the lists and ballot words | `W/32` words per side per level run, ≈ 2 KB | the liveness walk |
| the parked low plane | `BC * (DV + 1) * 2` = 33,280 bytes | the register-relief stage |

<a id="panel"></a>
## PHASE 1's fan-in panel

One SLICE PER WARP of `[kPanelTokens][kLdO]` fp32, aliased into the region's first bytes.
The CTA's warps meet here before any of them reaches `num`/`den`, so the planes see one
contribution per owner box per token rather than one per warp (`carry_kernel.md#fan-in`).
Each warp writes its own slice WHOLE — an accumulator element belongs to exactly one lane of
one warp — so the slices are disjoint, nothing contends, and there is no atom in the path.

`kPanelTokens` is DERIVED: the widest power-of-two token count up to `kSeg` whose
`warps_per_cta` slices fit inside the region. The region's size is `BC * kLdX * 2` and `BC`
falls as the value width rises, while a slice's width rises with it — so the grain is the
batch at the shipped value widths and halves at the widest declared one. A `static_assert`
refuses an arm where even one n-tile of slices would not fit.

The slice's stride is the page image's, and the reason is the same congruence: for a fixed
channel m-tile a warp's 32 lanes address four TOKEN rows (differing by two, hence by
`2 * kLdO ≡ 8` words) times eight consecutive channels, so the four rows land on four bank
quadruples and the eight channels fill each — 32 banks, once. The fan-in's own reads are
consecutive channels across consecutive threads, which is conflict free at any stride.

<a id="read-ring"></a>
## The read side's raw ring

`kReadSlots = 2` slots of the SAME `[kSeg][kRawStride]` bf16 shape the write side stages,
because the runs a box reaches are the BOX's and only the amplitude plane differs. It carries
no values (the readout's A operand is the state) and no gains (the gain is a write-side
quantity), and it needs no column block: the readout's B fragment pairs two adjacent LEAVES
at one token, which the packed plane already stores adjacent, so the expansion reads these
rows as they lie (`carry_kernel.md#read-expansion`). Two slots is what makes ONE edge a batch
enough — the slot being staged and the slot being expanded alternate.

<a id="factor-rows"></a>
## The deep index staging: what it holds, and why a ring rather than a window

The row this table reserved for the deep index staging is FILLED, at `kFactorSlots =
kSegmentSlots + 1` segment slots — the factor ring runs one segment ahead of the value ring,
so it carries one slot more — rather than at the window's depth the reservation estimated.
Two reasons, and neither is the byte count alone. A window's worth at the flagship arm is
`512 * 66` = 34 KB, which is above the reservation and enough to displace the readout's own
tenants and the parked plane. And DEPTH BUYS ONLY LATENCY COVER, never traffic: a token's
factor row is read by exactly one segment, so staging it earlier moves no bytes. What the
cover slots need from this tenant is that it is PERSISTENT — that production reads only
routing data, never the current phase's accumulator and never the region — and a ring holds
that exactly as a window would.

Per slot it holds a segment's `kSeg` factor rows AS THEY LIE in the packed plane
(`[kSeg][kRawStride]` bf16, each token's `D` contiguous runs of the box's own digits) and
its `kSeg` gains. The rows are staged by `cp.async`; the gains are not (`carry_kernel.md#column-block`).

`kRawStride` is `kSpanTotal` rounded up to `4 (mod 32)` halfwords, and that congruence is
the transpose's access pattern rather than a round number: the transpose reads word
`t * (kRawStride / 2) + p` for token `t` and column pair `p`, so `kRawStride / 2 = 2 (mod 8)`
spreads a warp's eight token pairs over eight bank quadruples and its four column pairs
inside the quadruple — 32 lanes, 32 banks, once each. The same stride keeps every row eight-byte
aligned, which is the widest quantum a level run can be staged in.

<a id="column-block"></a>
## The column block

`[kSpanTotal][kTokStride]` bf16, TWO blocks — the expansion runs one segment ahead of the
deposits, so the block it reads and the block the transpose is filling are never the same —
aliased into the region beside the value ring.
It stores each amplitude column's `kSeg` tokens ADJACENT, which is the order the MMA's B
fragment pairs them in.

`kTokStride` is `kSeg` rounded up to `8 (mod 16)` halfwords: the expansion reads word
`c * (kTokStride / 2) + t / 2`, so `kTokStride / 2` being four times an odd number puts eight
consecutive columns on eight distinct bank quadruples, and the four token pairs of a warp
fill each quadruple. At `kSeg = 16` that is 24 halfwords, a 50% pad, and it is what makes the
expansion's every load conflict free.

<a id="v-tile"></a>
## The V tile's layout and its swizzle

A segment's value rows are stored TOKEN MAJOR: `[kSeg][LdV]` bf16, row = token, column =
value channel. There is ONE tile per stream per slot, not one per warp: every warp's A
operand is the same `V^T` and only its coefficient differs, so the segment carries the
values once and the box fetches each value row once instead of `kWarps` times. The fold's A operand is CHANNEL major, so the transposing tile load reads
this layout directly and there is no transpose pass — the same arrangement flash attention
uses to feed a value tile as an operand.

The 16-byte group index is XORed with the token row (`v_group_offset`), which is a bijection
of the tile and leaves every group of eight lanes addressing eight distinct bank quadruples.
Unswizzled, a 128-byte row stride would put all sixteen rows in the same banks and the tile
load would serialize eight ways. The swizzle is GF(2)-linear, so it distributes over the
row/group split and every displacement stays an immediate.

The tile is written by the predicated staging quantum: a token past the window's end
zero-fills rather than reading out of range, so a partial window carries zeros and never a
neighbour's bits.

<a id="page-image"></a>
## The state page image, and the pad its readers earn

`[16][LdO]` fp32 with the mass in column `DV`. `LdO` is `DV + 1` rounded UP to the next
value congruent to 4 modulo 32, and that congruence is derived from the access pattern
rather than picked:

The image's reader is the sweep that lifts a warp's accumulator fragments out of it. For a
fixed channel m-tile, a warp's 32 lanes address four PAGE ROWS (the four values of
`lane & 3`, whose slots differ by two) times eight CHANNELS (`lane >> 2`), and a lane's
word index is `row * LdO + channel`. The bank is that index modulo 32, so `LdO mod 32` is
exactly the stride the four rows are spread by:

| `LdO mod 32` | rows land on | banks reached by the 32 lanes |
|---|---|---|
| 1 (`DV + 1` at `DV = 64`) | `0, 2, 4, 6` | 14 of 32 — a two-to-three-way conflict |
| 4 | `0, 8, 16, 24` | all 32, exactly once — conflict free |

Measured: the pad takes the flagship cell's shared bank conflicts from 524,288 to 4,096 and
the small cells' from 2,048 to 16. The global side is untouched by it — `state_page::load`
and `::store` address the PAGE, not the image — and their own image access is a pair of
adjacent columns per lane, which is conflict free at any `LdO`.

The residue that remains belongs to the OTHER member of the generated sub-box set, whose
four slots share one page row rather than four: no single leading dimension can serve both,
and the pattern's own answer is a member-dependent swizzle, exactly as the transit region
has. That is a measured follow-up, not a gap in this one.
