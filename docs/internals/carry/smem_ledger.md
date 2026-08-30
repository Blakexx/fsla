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
| 36,864 | the frame's three edge barriers, `3 * 8` | 24 | the frame |
| 36,888 | the box's page slots, `BC/16 * 4` | 64 | the prologue |
| | **total** | **36,952** | |

At sm_86's 101,376-byte per-CTA maximum this leaves the fill rule most of the budget, and
the residency is one CTA per SM by the launch bound rather than by the allocation.

<a id="region"></a>
## The region, and the aliasing structure per moment

The transit region is the largest tenant and the only one whose bytes are re-let. It is
allocated once at its full size, and every phase-local tenant lives INSIDE it, because no
two of them are alive at the same moment:

| moment | who lives in the region's bytes |
|---|---|
| the entry sweep | as many state page images as fit: `floor(36,864 / (16 * LdO * 4))` = 8 images of 4,352 bytes = 34,816 bytes, in 2 rounds over the box's 16 pages |
| the out leg, the return leg | the transit region itself — `BC` rows of `LdX` bf16 channels, the accumulator's halves in the owner-local canonical order |
| PHASE 1 | the read stream's segment slots and the readout's output block |
| PHASE 2 | the V ring, one `kSeg x LdV` bf16 tile per warp per slot = 2,048 bytes, `8 * 1 * 2,048` = 16,384 bytes today; and the write stream's segment slots |
| the exit sweep | the state page images again |

The edge barriers and the page slots are PERSISTENT: they are alive from the prologue to
the exit and are never re-let.

Two consequences worth stating because they are easy to get wrong later. First, a tenant
that must survive a phase change is NOT a region tenant — the deep index staging that runs
a window ahead of the phase that consumes it is persistent by construction and takes its
own bytes. Second, parking the accumulator's low plane in shared memory holds the region
through PHASE 1, which displaces PHASE 1's own tenants into bytes of their own; that is
what the parked plane costs, and it is a priced tenant on this table rather than a surprise.

## Reserved by name, for the stages that fill them

These are named here so a later stage lands as an entry on this table rather than as an
allocation nobody sized. None is allocated today.

| tenant | size, at the flagship arm | who fills it |
|---|---|---|
| read stream segments | one 16-token slot per accumulator tile; the register fragments are the second buffer where the MMA sources from registers | the fold-and-stream stage |
| the readout's output block | `C * (DV + 1) * 4` ≈ 8,320 bytes, or none where the reduction is native | the fan-in stage |
| write stream segments | one 16-token slot per concurrently active accumulator box, each `LdX * 2` plus its coefficient row | the stream stage |
| the deep index staging | the window's union list and its un-expanded factor rows, a window ahead: ≈ 48 bytes per token, ≈ 24 KB at `W = 512` | the fold stage |
| the lists and ballot words | `W/32` words per side per level run, ≈ 2 KB | the liveness walk |
| the parked low plane | `BC * (DV + 1) * 2` = 33,280 bytes | the register-relief stage |

<a id="v-tile"></a>
## The V tile's layout and its swizzle

A segment's value rows are stored TOKEN MAJOR: `[kSeg][LdV]` bf16, row = token, column =
value channel. The fold's A operand is CHANNEL major, so the transposing tile load reads
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
