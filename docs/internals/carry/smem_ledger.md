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
| 0 | the transit region, `BC * LdX * 2` -- and, sharing its bytes, THE POOL: `(kPoolPos + 1) * (kVRowBytes + kAmpRowBytes)` rows plus `kSideStreams * kListBytes` of lists and the fill scratches -- and, in PHASE 1's sparse form, THE READ POOL in the same bytes: `(kPoolPos + 1)` amplitude rows, its lists and scratches, and the slot -> token table; the tenant is the larger of the two, `kRegionTenantBytes` | 54,288 | the exchange between the two sides' carves; PHASE 2's rows |
| after the region | THE READ BLOCK: the union readout's two shared raw slots | 2,304 | the union readout |
| 53,824 | the box's page slots, `BC/16 * 4` | 64 | the prologue |
| 53,888 | the fill's ONE claim cursor | 4 | both union bodies' fills |
| after the claim | THE LIVE WORDS, per side: each stream's word per round, `kSideStreams * kRounds * 4`, and its exclusive prefix, `kSideStreams * kRounds * 2` (`kLiveBytes` each) | 2 x 768 | cover slot A writes the read side's, cover slot B the write side's; each fill walks its own once |
| | **total** (16-byte aligned, the flagship arm on sm_86) | **67,616** | |

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
| PHASE 2 | the pool: `kPoolPos + 1` value rows, `kPoolPos + 1` amplitude rows and the eight lists -- filled only after the rejoin has read the region, dead again at the window line |
| the exit sweep | the state page images again |

<a id="streams"></a>
### The streams: lists and live words, no rings

A stream is a leaf range of `G` leaves inside ONE warp's sub-box. It owns no ring: its
operands are the side's pool rows, shared by every stream that names a position, and what
is the stream's own is its LIST of pool slots (`kListEntries` halfwords, `kListBytes`), minted
by the fill, and its LIVE WORDS (one 32-bit word per round and a 16-bit exclusive prefix),
written at the cover slot in front of the phase. The ledger is sized at the FINEST grain the
nesting law admits (`G = 16`, the page rectangle: `kSideStreams` streams per side); a coarser
runtime `G` uses fewer of the same lists.

The page slots, the claim cursor and the live words are PERSISTENT: they are alive from the
prologue to the exit and are never re-let. So is the union bodies' raw factor ring — a tenant that must survive a
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

The panel's REASON is measured, not assumed: the form without it — every warp reducing into
the planes directly — is 1.39x slower at the flagship and 1.10-1.14x at the small cells, because
it multiplies the global reductions by `warps_per_cta` while saving only 6% of the section's
instructions (`carry_kernel.md#fan-in`). These 34,816 bytes buy an 8.73x cut in L2 atomic traffic.

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

<a id="pool"></a>
## The pool

PHASE 2's rows, keyed by position, in the region's bytes (`kPoolVOffset`, `kPoolAmpOffset`,
`kListOffset`); PHASE 1's sparse form lays its READ POOL in the same bytes (`kReadAmpOffset`,
`kReadListOffset`, `kReadScratchOffset`, `kPoolTokenOffset`, `kStageOffset`): amplitude rows
only, at the same capacity, the slot -> window-token table the deposit addresses by, and
`warps_per_cta` STAGING TILES of `(kSeg + 1) * kStageStride` fp32 (`kStageBytes`, 4,624 at
the flagship: sixteen token rows of `DV + 4` floats, the pad keeping a fragment's 64-bit pair
stores two-way at most and a token's row contiguous, plus a row of the tile's token ids)
through which a tile's partial is transposed for its coalesced reductions.
The region tenant is the largest of the fold pool, the read pool and the region itself. Capacity `kPoolPos` is `pool_positions` of the budget the other tenants leave:
the largest power of two whose rows, plus the dead row and the lists, fit -- 256 at the
flagship arm on sm_86, where the pool outgrows the transit region and the region tenant is
sized to it (`kRegionTenantBytes`). Per position: a value row, `kVRowBytes = DV * 2`, and an
amplitude row, `kAmpRowBytes = kAmpStride * 2`. The window is folded in batches of the pool,
so the capacity is the fold's reap floor (`carry_kernel.md#pool-tile`) and the lever for it is
the budget: fewer channels per CTA, or the arch's larger file.

<a id="factor-rows"></a>
### The amplitude rows

`[kPoolPos + 1][kAmpStride]` bf16: each position's `D` contiguous runs of the box's own
digits, as they lie in the packed plane, at `span_base` per level; the outermost run carries
the gain (`carry_kernel.md#pool`). `kAmpStride` is `amp_stride(kSpanTotal)`: the span
rounded up to a whole ODD number of sixteen-byte quanta, so every row is an `ldmatrix`
address and a `cp.async` target at the widest quantum, and eight consecutive rows land on
eight distinct bank quadruples -- 80 bytes at the flagship's 64-byte span.

<a id="v-tile"></a>
### The value rows and their swizzle

`[kPoolPos + 1][DV]` bf16, row = position, column = value channel: TOKEN MAJOR, which the
fold's transposing fragment load reads as the channel-major A operand directly. Each value
row is in the pool ONCE, whichever warps' lists name it: the box fetches a value row once,
not `kWarps` times, and nothing is duplicated per stream. The 16-byte group index is XORed
with the row's residue (`v_group_offset`), a GF(2)-linear bijection of the row, so eight
consecutive rows' `ldmatrix` hits eight distinct bank quadruples; gathered rows at arbitrary
residues collide two-way at most, which is what the measured 3.8% conflict share is.

### The lists

`kSideStreams` lists per side of `kListEntries` halfword slots (`kListBytes` each), one per
stream of the side's finest grain, minted by the fill and consumed by the stream's pass; behind them
`warps_per_cta` fill scratches of one round's live lanes (`kFillScratchBytes`). Never a
stream's operands.


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
