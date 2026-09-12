# The carry box's shared-memory ledger

Mirrors the `constexpr` offsets in `csrc/rola/src/carry/box.cuh` and the `Smem` accessors in
`carry/carry_kernel.cuh`. Numbers are the flagship arm, `D = 2`, `DV = 64`,
`warps_per_cta = 8`: `BC = 256` leaves, sixteen boxes, `kPoolTok = 80`.

Shared memory is allocated BY NAME and BY MOMENT: a tenant states which phase it is alive
in, and the two phases' private tenants alias one region because the phases are sequential.
A tenant written by `cp.async` states two edges: the edge its issue follows (the last read
of the bytes is before it) and the wait that lands it.

<a id="tenants"></a>
## The tenants, top to bottom

| tenant | bytes | moment |
|---|---|---|
| liveness words, both sides (`kWordsOffset`) | 4,096 | staged a window ahead; read by the head |
| read order, two parities (`kOrderOffset`) | 2,048 | written by the head, read by the readout |
| tile masks, two parities (`kTileMaskOffset`) | 128 | likewise |
| count table (`kRCountOffset`), masks by rank (`kMaskOffset`) | ~1,600 | the head's working set |
| warp words, union words, prefix, two parities (`kWarpWordsOffset`, `kUnionOffset`, `kPrefixOffset`) | ~1,300 | written by the head, read by the fill and the fold |
| live counts (`kLiveOffset`) | 16 | |
| snapshot (`kSnapshotOffset`) | 32,768 | written at the window start, read by the readout |
| mass table (`kMassTableOffset`), mass row (`kMassRowOffset`), row map (`kRowMapOffset`) | ~5,600 | |
| page slots, phase ledger, counter, pool barriers, walk rings (`kSlotOffset`..`kWalkOffset`) | ~2,600 | |
| THE REGION (`kRegionOffset`) | 47,664 | below |

Total `kSmemBytes` ≈ 95.8 KB of the 99 KB maximum.

<a id="pool"></a>
## The region: the pool, and the readout's blocks over it

The region is `kPoolSlots` (3) pool slots of `kPoolSlotBytes` (15,888) each: `kPoolRows`
(81) rows — `kPoolTok` tokens and the ZERO ROW — laid out as arrays: V rows
(`kPoolVOffset`, 128 B a row, the 16-byte chunk XORed with the row), inner run tiles
(`kPoolInnerOffset`, 32 B a row, the second chunk XORed with the row's bit two), outer run
tiles (`kPoolOuterOffset`), gain pairs (`kPoolGainOffset`, a word a row). The slot size is
the largest the remainder holds, never a literal (`kPoolTokFit`).

During the readout the fold is not running, so the readout's private blocks — a warp's
`kRRSlots` (2) ring slots of a tile's inner and outer runs (1 KB each) and its drain stage
(`kDrainRows` = 4 rows of y, fp32) — alias the pool, from `kReadOffset`. When they fit past
slot 0 (`kPrefetchFirst`), slot 0 stays untouched and the window's first pool chunk lands
under the readout; else they cover the region and the first chunk waits. The plan's
`static_assert` refuses a block set the region does not hold.

<a id="barriers"></a>
## The pool's barriers

`kPoolBarOffset`: a `full` and an `empty` `mbarrier` a slot. `full` completes at
`kThreads` landed-copy arrivals; `empty` at `kWarps` arrivals, each warp's after its last
gather from the slot. Parities come from `PoolCursor`'s use count a slot.
