# The carry box's shared-memory ledger

Mirrors the `constexpr` offsets in `csrc/rola/src/carry/box.cuh` and the `Smem` accessors in
`carry/carry_kernel.cuh`. Numbers are the flagship arm, `D = 2`, `DV = 64`,
`warps_per_cta = 8`: `BC = 256` leaves, sixteen boxes, `kPoolTok = 64`.

Shared memory is allocated BY NAME and BY MOMENT: a tenant states which phase it is alive
in. The fold and the readout run interleaved, so their tenants (the pool, the rings) do not
alias each other; the sweeps' stages, alive only at a call's edges, alias the whole region.
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
| page slots, phase ledger, counter, take words, pool and ring barriers, walk rings (`kSlotOffset`..`kWalkOffset`) | ~2,800 | |
| THE REGION (`kRegionOffset`) | 50,080 | below |

Total `kSmemBytes` = 100,512 B, 98.2 KB of the 99 KB maximum.

<a id="pool"></a>
## The region: the readout's blocks, then the pool

The region opens with the readout's private blocks (`kReadOffset`, `kReadBytes` = 24,576): a
warp's `kRRSlots` (2) ring slots of a tile's inner and outer runs (1 KB each) and its drain
stage (`kDrainRows` = 4 rows of y, fp32). Then the pool (`kPoolOffset`): `kPoolSlots` (2)
slots of `kPoolSlotBytes` (12,752) each: `kPoolRows` (65) rows — `kPoolTok` tokens and the
ZERO ROW — laid out as arrays: V rows
(`kPoolVOffset`, 128 B a row, the 16-byte chunk XORed with the row), inner run tiles
(`kPoolInnerOffset`, 32 B a row, the second chunk XORed with the row's bit two), outer run
tiles (`kPoolOuterOffset`), gain pairs (`kPoolGainOffset`, a word a row). The slot size is
the largest the remainder holds, never a literal (`kPoolTokFit`).

The two streams run at once, so the blocks and the pool do not alias. The sweeps' low-half
stage (32 KB, `carry_kernel.md#state-io`) covers the region at a call's edges, when neither
stream is alive (`kRegionBytes >= kSnapshotBytes`, asserted).

<a id="barriers"></a>
## The pool's barriers

`kPoolBarOffset`: a `full` and an `empty` `mbarrier` a slot. `full` completes at
`kThreads` landed-copy arrivals; `empty` at `kWarps` arrivals, each warp's after its last
gather from the slot. Parities come from `PoolCursor`'s use count a slot. `kRingBarOffset`:
a warp's ring slot's barrier, full at its 32 lanes' landed copies (`RingCursor`).
`kTakeOffset`: a warp's take word, the tile its lane 0 dealt itself off the counter.
