# `csrc/rola/src/pipeline.cuh` — the consumer's item-stream pipeline

The intra-CTA phase protocol: three counters, two release/acquire primitives and one
spin, plus the edge DAG and the deadlock argument that rest on them. A warp that
reaches a phase edge early executes another WORK ITEM instead of stalling on a
rendezvous — that is the whole point, and it is why these are counters and not
`__syncthreads()`.

The phases themselves are [`consumer_kernel.md`](consumer_kernel.md); the shared
memory the control block occupies is priced by [`smem_layout.md`](smem_layout.md).

## Reference conventions

The same in all four consumer-kernel documents — [`consumer_kernel.md`](consumer_kernel.md),
[`smem_layout.md`](smem_layout.md), [`mma_tile.md`](mma_tile.md),
[`pipeline.md`](pipeline.md):

- **`§N` is ALWAYS a SPEC reference.** `ROLA_V3_SPEC.md` unless another spec is named
  at the site; `GLA_MASS_DECAY_SPEC.md` §12 is the decay numerics contract. It never
  means a section of these documents.
- **A section of these documents is referenced by LINK**, never by its number, so a
  reference resolves without the reader knowing which of the four holds it.
- Section numbers run continuously across the four documents, so a number identifies
  one section — but it is the link, not the number, that is the reference.

## <a id="item-stream"></a>36. The control block and its two primitives

The tile's intra-CTA phase ordering is carried by COUNTERS rather than by `__syncthreads()`. The
point is not to spell a barrier differently: a warp that reaches a phase edge early goes and
executes another WORK ITEM instead of stalling on a rendezvous, which is what makes the fold a
fluid pool. A rendezvous there leaves 6 of 8 warps idle through the Gram (measured; provenance:
r1 finding A).

**A WORK ITEM is one `(GEMM, warp-tile)` pair with a WHOLE `k` extent (design D1.1).** Full-`k`
is a BIT-IDENTITY requirement and not a convenience: an item fixes its own accumulation order
internally, so WHICH warp runs it is arithmetically irrelevant, while splitting a `k` extent
would be split-K INSIDE the tile and would move `y`'s summation order. Nothing here splits one.

**The item TABLE IS DERIVED, NOT STORED (design D1.2):** the fold's item count is
`(m-tiles in the window) * NT_DV` and a claimed index decodes to `(m-tile, n-tile)` by the same
strided arithmetic a warp loop walks. Cost is one `atomicAdd` per claim and ZERO shared memory
beyond the counters.

```
kCtlReadDone   warps that have retired the READOUT phase, i.e. finished every read of S this
               tile makes. Gates the fold, which WRITES S. (This is B6, a counter and not a
               barrier.)
kCtlGramDone   warps that have retired the GRAM phase. Gates phase C, which READS Gm. (B5.)
kCtlFold0 + w  window w's fold item-claim cursor -- the pool's whole state. One per window.
```

Both PHASE counters count WARPS and every warp retires both UNCONDITIONALLY, which is what makes
the deadlock argument ([below](#ctl-reset)) a two-line one.

The fold's claim cursors are ONE PER WINDOW, because a narrowed `WT` makes the fold's item pool
per-window: window `w`'s pool must start at zero, and resetting a shared cursor mid-tile would
need a rendezvous to do it safely — which is the thing this whole rung is removing. `QPO` of
them is the compile-time MAXIMUM (`kNWin <= kQPO`), it is `BC / kMmaKQuantum <= 4`, and the
unused tail costs 4 bytes each on the arms that do not window. I1-generic: an expression in `BC`
and `kMmaKQuantum`, never a literal.

### <a id="release-acquire"></a>HAZARD `release-acquire` — the fence order in `retire_item` / `eligible`

**RELEASE (`retire_item`).** The fence comes FIRST, so every shared store the warp made for the
phase it is leaving is visible CTA-wide before the counter that advertises them moves. The
`atomicAdd` is executed by lane 0 ALONE because the counters count WARPS — a per-lane count would
make the target a lane count and the phase fact would stop being a phase fact.

**ACQUIRE (`eligible`).** The load is `volatile` because the counter is written by OTHER warps
and a cached load would spin forever; the fence comes AFTER a successful test, so no subsequent
shared load of the published object can be hoisted above the observation that published it.
WARP-UNIFORM by construction — every lane loads the same address, so the predicate is identical
in all 32 lanes with no shuffle.

`await_count` is the spin, written in terms of `eligible` so that the release/acquire discipline
lives in exactly TWO functions and can be audited there rather than at every phase edge.
`__nanosleep` backs the polling warp off the issue port so a spinning warp does not compete with
the warps it is waiting for.

## <a id="ctl-reset"></a>37. The tile's control-block reset, and the deadlock argument

The reset needs no barrier of its own on EITHER side, and both halves of that are PROGRAM-ORDER
facts rather than hopes:

- BEHIND it stands the previous tile's end-of-tile `__syncthreads()` (or, for the first tile,
  the CTA-init one), so no warp can still be reading last tile's counters;
- AHEAD of it stands the staging publication `__syncthreads()`, which is unconditional and sits
  between this store and the first `retire_item`. It publishes the zeroed counters on the same
  trip it publishes `Rf`, `WT` and `Vw`.

### <a id="edge-dag"></a>Deadlock freedom, as the edge DAG and the always-runnable claim

NODES (per tile): `STAGE` (gather / clocks / row scales / the three panel builds), `A` (readout +
mass term 1), `G` (Gram + `Gm` publication), `F` (fold items + mass fold), `C` (intra + mass
term 2 + emission), `B7` (end of tile).

| edge | mechanism |
|---|---|
| `STAGE -> A`, `STAGE -> G_0` | the staging `__syncthreads()` (hardware) |
| `G_w -> F_w`, `F_w -> G_{w+1}` | the window's SLOT-READY / SLOT-FREE `__syncthreads()` pair (`!kWtPlane` only; window `w+1`'s build overwrites `WT`'s rows) |
| `A -> F` | `kCtlReadDone` reaching `kWarps` (readout READS `S`, the fold WRITES it). Taken ONCE, in window 0 |
| `G -> C` | `kCtlGramDone` (`G` writes `Gm`, `C` reads it). `G`'s publication rides the LAST window. The mechanism FORKS on `kNarrowGate` and both forms carry the SAME edge ([`consumer_kernel.md#gram-gate`](consumer_kernel.md#gram-gate)) |
| `F -> B7`, `C -> B7` | program order into the end-of-tile barrier |

The relation is a TOTAL ORDER on `{STAGE, {A, G_0}, F_0, G_1, F_1, ..., C, B7}` and therefore
ACYCLIC. (`A` is unordered with respect to every `G_w` — the Gram reads `Rf` and `WT` and touches
neither `S` nor `Gm`'s readers — and so are `F` and `C`: `F` writes `S`, `C` reads `Gm`, `Vw`,
`sr`, `den`, and the sets are disjoint. That is the overlap this rung exists to expose.)

**ALWAYS-RUNNABLE**, and it is two facts:

1. **Every increment a waiter needs is issued, in program order, before any warp can reach the
   wait that needs it.** For `kCtlReadDone` that is the unconditional retirement every warp
   executes above the window loop, target `kWarps`. For `kCtlGramDone` it holds on BOTH ARMS of
   the `kNarrowGate` fork, because on each the retiring set and the target are derived from ONE
   expression rather than chosen separately — wide: every warp retires, target `kWarps`; narrow:
   `mixed && warp < kGramWarps` retires, target `kGramWarps`. Taking the narrow arm, which is
   the one with a guard to discharge:
   - both guards are CTA-UNIFORM in `mixed` (it is `is_read && is_write`, the tile's schedule
     bits) and compile-time-comparable in `warp`, so either the whole CTA takes the counter or
     none of it does. A tile cannot have waiters without retirers;
   - on a `mixed` tile warps `0..kGramWarps-1` execute the retirement inside `if (w + 1 ==
     kNWin)`, which every warp reaches because `kNWin` is CTA-uniform, and that is strictly
     before the wait in program order. The counter is monotone with target `kGramWarps`, so the
     wait terminates after exactly `kGramWarps` retirements;
   - on a `!mixed` tile nobody retires, nobody waits, and the word stays at the reset value —
     which is re-established next tile by the same unconditional store, on a path every warp
     reaches whichever branch it took.
2. **No `__syncthreads` is ever reached while another warp spins.** Every warp executes the same
   `kNWin` iterations and the same barriers within them (`is_write`, `w` and `kNWin` are all
   CTA-uniform), and the only wait inside the loop — `kCtlReadDone` in window 0 — is on a counter
   every warp has already retired, ABOVE the loop. So a warp sitting at a SLOT-READY or SLOT-FREE
   barrier is never waiting on a warp that is spinning on something that warp itself must supply.

## <a id="cp-async"></a>37b. The async fetch primitives (`cp.async`)

`cp_async16` / `cp_async_commit` / `cp_async_wait_all` are the sm_80+ async
global-to-shared copy, held here because an async copy's COMPLETION is a
phase-ordering fact exactly like the counters above: the kernel issues a tile's
amplitude fetch at one program point and consumes it behind a later barrier
([`consumer_kernel.md#amp-prefetch`](consumer_kernel.md#amp-prefetch)), and the
commit/wait pair is what makes that edge real rather than timed.

Three properties are load-bearing:

- **`cg`, not `ca`**: the copy bypasses L1. The slab is read once per tile, so
  an L1 line would be evicted work; L2 still serves any re-fetch across CTAs.
- **The zero-fill operand**: `cp.async.cg.shared.global [dst], [src], 16,
  src_bytes` always writes 16 bytes of shared memory, reads only `src_bytes`
  of global and zero-fills the rest — branch-free handling of the last line of
  a stream, with `src_bytes = 0` a pure zero write.
- **`wait_group` is per-thread**: it settles only the ISSUING thread's
  committed groups. Cross-thread visibility of the copied bytes still requires
  a barrier, which is why every wait in the kernel sits immediately in front of
  a `__syncthreads()` the tile already owned.
