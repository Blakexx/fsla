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
| 0 | the transit region, `BC * LdX * 2` -- and, sharing its bytes in PHASE 2 only, THE POOL: `(kPoolPos + 1) * (kVRowBytes + kAmpRowBytes)` rows plus `kSideStreams * kListBytes` of lists and the fill scratches | 36,864 (region) / 62,160 (pool) | the exchange between the two sides' carves; PHASE 2's rows |
| 36,864 | THE READ POOL, behind the region: `(kReadPos + 1)` amplitude rows, the fill scratches (a round's ranked lanes and their slots, `kFillScratchBytes` a warp), the slot -> token table, each slot's k-step mask (`kPoolGroupsOffset`, a word a slot) | about 22,700 | PHASE 1's fill |
| after the pool | THE READ SIDE'S LISTS, one a stream: the batch's slots live in it by rank, `kSideStreams * kReadPos * 2` (`kReadListOffset`) | 8,192 | the stream form's fill mints, its tile walk reads (`carry_kernel.md#stream-form`) |
| after the lists | THE STAGING TILES, one a warp: `(kSeg + 1) * kStageStride * 4` | 8 x 2,720 | PHASE 1's write-out |
| 80,224 | the box's page slots, `BC/16 * 4` | 64 | the prologue |
| after the page slots | THE LIVE WORDS, per side: each stream's word per round, `kSideStreams * kRounds * 4`, and its exclusive prefix, `kSideStreams * kRounds * 2` (`kLiveBytes` each) | 2 x 1,536 | cover slot A writes the read side's, cover slot B the write side's; each fill walks its own once |
| after the live words | THE K-STEP WORDS: the read side's word per k-step of the box in the read order, per round (`kKstepWordBytes`) | 1,024 | the read walk writes; the fill's masks read (`carry_kernel.md#stream-form`) |
| after the k-step words | THE BOX WORDS: one side's class-1 words for one window, every digit row of the box by round (`kBoxWordsBytes`) | 2,048 | staged a slot ahead by `cp.async`; the cover walks read (`carry_kernel.md#box-words`) |
| after the live words | THE READ MEMO: per k-step of the box, each lane's region row (`kKb * 32 * 4`), the mass lanes' four rows (`kKb * 64`) and the amplitude columns (`kKb * kMemoColStride * 2`) | 3,200 | the prologue derives it once; every readout stream reads its k-steps' entries |
| after the memo | THE GAIN TABLE, the window's write-side gains by window token, `kWindow * 2` | 1,024 | cover slot B copies it; the fold's fill reads it |
| after the gain table | THE UNIFORM TABLES, per side: the union's word and exclusive prefix per round, each stream's count before the batch and before its end (`kUniformWords`) | 2 x 256 | every warp writes them alike; every loop bound reads them (`carry_kernel.md#uniform`) |
| | **total** (16-byte aligned, the flagship arm on sm_86) | **100,864** | |

The tenant at offset 0 is the larger of the region and PHASE 2's pool; the read pool and
the staging tiles follow it because the region is READ through the whole of PHASE 1
(`carry_kernel.md#readout`) and nothing may alias it there. The frame's three edges and the
streams' take no bytes at all: they are NAMED BARRIERS (`carry_kernel.md#edges`), which is
what an edge with equal participation costs.

At sm_86's 101,376-byte per-CTA maximum the residency is one CTA per SM by the launch bound
rather than by the allocation.

<a id="region"></a>
## The region, and the aliasing structure per moment

The transit region is the largest tenant and the only one whose bytes are re-let:

| moment | who lives in the region's bytes |
|---|---|
| the entry sweep | as many state page images as fit: `floor(36,864 / (16 * LdO * 4))` = 8 images of 4,352 bytes = 34,816 bytes, in 2 rounds over the box's 16 pages |
| the out leg, PHASE 1, the return leg | the transit region itself — `BC` rows of `LdX` bf16 channels, the accumulator's top halves in the owner-local canonical order — read in place by the readout as its B operand and by the rejoin |
| PHASE 2 | the pool: `kPoolPos + 1` value rows, `kPoolPos + 1` amplitude rows and the eight lists -- filled only after the rejoin has read the region, dead again at the window line |
| the exit sweep | the state page images again |

<a id="streams"></a>
### The streams: lists and live words, no rings

A stream is a leaf range of `G` leaves inside ONE warp's sub-box. It owns no ring: its
operands are the side's pool rows, shared by every stream that names a position, and what
is the stream's own is its LIVE WORDS (one 32-bit word per round and a 16-bit exclusive
prefix), written at the cover slot in front of the phase, and -- on the write side -- its
LIST of pool slots (`kListEntries` halfwords, `kListBytes`), minted by the fill. The read
side mints no list: a warp's slice of positions is its tile, and the words answer how many
of a stream's positions fall in it. The ledger is sized at the FINEST grain the
nesting law admits (`G = 16`, the page rectangle: `kSideStreams` streams per side); a coarser
runtime `G` uses fewer of the same lists.

The page slots, the live words, the read memo, the gain table and the uniform tables are
PERSISTENT: they are alive from the prologue to the exit and are never re-let.

<a id="memo"></a>
The READ MEMO (`carry_kernel.md#memo`) is three tables by k-step of the box: `memo_rows(K)`,
one 32-bit shared address a lane; `memo_mass(K)`, four addresses for each of the four mass
lanes; `memo_cols(K)`, `2 + (D - 1)` halfword columns.

## Reserved by name, for the stages that fill them

These are named here so a later stage lands as an entry on this table rather than as an
allocation nobody sized. None is allocated today.

| tenant | size, at the flagship arm | who fills it |
|---|---|---|
| the lists and ballot words | `W/32` words per side per level run, ≈ 2 KB | the liveness walk |
| the parked low plane | `BC * (DV + 1) * 2` = 33,280 bytes | the register-relief stage |

<a id="panel"></a>
<a id="read-ring"></a>
<a id="pool"></a>
## The pool

PHASE 2's rows, keyed by position, in the region's bytes (`kPoolVOffset`, `kPoolAmpOffset`,
`kListOffset`, and one gain halfword a position at `kPoolGainOffset`). PHASE 1's READ POOL lies BEHIND the region (`kReadAmpOffset`,
`kReadScratchOffset`, `kPoolTokenOffset`, `kStageOffset`), because the region is read through
the phase: amplitude rows only, in batches of `kReadPos = kSlices * warps_per_cta * kSeg`
positions (two slices a warp, 256 at the flagship), the slot -> window-token table the
write-out addresses by, and `warps_per_cta` STAGING TILES of `(kSeg + 1) * kStageStride`
fp32 (`kStageBytes`, 2,720 at the flagship: sixteen position rows of `32 + 8` floats -- one
32-channel segment of one slice at a time, a mass column a slice, and the pad that keeps a
fragment's 64-bit pair stores bank free and a row contiguous -- plus a row holding the
slices' token ids by (slice, batch parity)) through which a slice's rows are transposed, a
segment at a time, for their one-line reductions.
The region tenant is the largest of the fold pool and the region with the read tenants behind it (`kRegionTenantBytes`). Capacity `kPoolPos` is `pool_positions` of the budget the other tenants leave:
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

`kSideStreams` lists on the write side of `kListEntries` halfword slots (`kListBytes` each),
one per stream of the side's finest grain, minted by the fill and consumed by the stream's
pass; behind each side's rows `warps_per_cta` fill scratches of one round's live lanes
(`kFillScratchBytes`). Never a stream's operands.


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
