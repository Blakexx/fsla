# `prepass/prepass_kernel.cuh` — the pre-pass kernel: the skeleton and its frame

Mirrors `csrc/rola/src/prepass/prepass_kernel.cuh`. The family overview is `prepass.md`.

THE FRAME IS REAL AND THE COMPONENTS ARE STUBS WITH THEIR EDGES (KERNEL_STANDARDS section 19).
Each component below lands against its own budget line on the queue card; until it does, its stub
touches its operands and produces zero, so what the skeleton measures is the frame.

<a id="smem"></a>
## The shared addresses

`Smem<TP>` names every region of the plan's ledger (`plan.md#ledger`) as a shared address:
the words a parity, the pool a slot, a warp's ring slot, the transfer's values and maxima, a warp's
panel row, and the barriers.

<a id="rows"></a>
## The region's staged rows

`StagedRows` lists the sparse levels' digit rows on each side, read side first, and names each
side's sparse level and its first staged row as scalars. Every index into it is compile-time
(`static_for` over the sides and the level bound), so the table is registers: the first build's
runtime-indexed table cost 360 bytes of local memory, the ptxas signature section 20 forbids.

<a id="words"></a>
## `words`

One warp stages the next window's words while the current one runs: the staged rows' sixteen
words each by cp.async, lane = word, arriving at the words barrier when landed.

<a id="lists"></a>
## `lists`

One lane a fold list: lane `k` sums the popcounts of its digit's sixteen words. The first form
ran sixteen serial shuffle reductions and cost 6.3K cycles a window; one lane a list costs 1.7K.
The read side's union over the warp's four sparse digits is ORed a round, lane = round, and
reduced once: the token-major loop's tile count.

<a id="fill"></a>
## `fill`

A chunk's rows into a slot by cp.async, four threads a token; the slot is full at every thread's
landed arrival, empty at one arrival a warp after the fold (the barrier counts warps, so the
arrival is lane 0's). THE STUB stages the window's tokens in order; the compaction to the
union-live tokens is the component's own step.

<a id="fold"></a>
## `fold`

`State` is the warp's 128 leaves' two columns, the accumulator of the transposed fold's m-tile:
sixteen lists, one n-tile, four registers each, indexed at compile time. THE STUB walks each
list's k16 step count and touches the pool once a step; no MMA.

<a id="transfer"></a>
## `publish` and `rescale`

A fold warp publishes its leaves' columns as s16 against each leaf's max and the max as a float
into the transfer; after the publish barrier a read warp loads its four blocks, takes each
block's scale as the max of its 32 leaf maxima (one a lane, five shuffles) and keeps the hi bytes
in registers. THE STUBS write and reduce zeros.

<a id="readout"></a>
## `readout`

The ring runs a tile ahead: `readout_issue` stages tile `t + 1` before `readout_tile` waits tile
`t`. THE STUB stores zeros into the warp's own panel row. With no work to hide behind, the wait
is one global-to-shared round trip a tile, about 1.3K cycles, sixteen tiles a window: the first
frame measurement's largest row, and the reason the readout component's ring must run more than
one tile ahead of an 8-IMMA tile.

<a id="drain"></a>
## `drain`

The range's eight warp rows summed a token, one column a thread, to the global sums by RED.F32;
the rows are overwritten by the next range.

<a id="frame"></a>
## The frame

Per window: wait the words, count the lists, stage the next window's words, fill chunk 0, then a
slot ahead fill `c + 1` while `c` is folded, publish, wait the publish barrier, rescale, and read
out each range a tile ahead and drain it. The measured frame with every component stubbed, cycles
a warp a window on the flagship cells (two windows, 64 CTAs): frame 7.2K, words 1.4K, lists 1.7K,
fill 7.2K, fold 1.4K, publish 0.2K, rescale 0.2K, readout 21.8K, drain 2.1K — 43K in all. The
fill's and the readout's rows are latency exposed by construction (nothing to hide the round trip
behind) and fall as the MMA components land; the frame's barrier waits, the words, the lists and
the drain are the fixed cost the design carries.
