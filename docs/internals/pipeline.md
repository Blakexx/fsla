# `csrc/rola/src/pipeline.cuh` — the consumer's item-stream pipeline

The intra-CTA phase protocol: three counters, two release/acquire primitives and one
spin, plus the edge DAG and the deadlock argument that rest on them. A warp that
reaches a phase edge early executes another WORK ITEM instead of stalling on a
rendezvous — that is the whole point, and it is why these are counters and not
`__syncthreads()`.

The phases themselves are [`consumer_kernel.md`](consumer_kernel.md); the shared
memory the control block occupies is priced by [`smem_layout.md`](smem_layout.md),
which includes this file for exactly `ctl_words`.

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

The tile's — the CODE calls one pass through the CTA's main loop a **visit**, one
group of `kGroupTokens == 16` gathered tokens — intra-CTA phase ordering is carried by
COUNTERS rather than by `__syncthreads()`. The point is not to spell a barrier
differently: a warp that reaches a phase edge early goes and executes another WORK
ITEM instead of stalling on a rendezvous, which is what makes the fold a fluid pool.

**A WORK ITEM is one `(GEMM, warp-tile)` pair with a WHOLE `k` extent.** Full-`k` is a
BIT-IDENTITY requirement and not a convenience: an item fixes its own accumulation
order internally, so WHICH warp runs it is arithmetically irrelevant, while splitting
a `k` extent would be split-K INSIDE the tile and would move the summation order.
Nothing here splits one.

**The item table is DERIVED, not stored.** The fold's item count is
`n_live * NT_DV` (`n_live` = live write atoms in the visit's write mask, `NT_DV` = the
`DV`-wide output's `n8` column count) and a claimed index decodes to `(atom, n-tile)`
by the same strided arithmetic a warp loop walks. Cost is one `atomicAdd` per claim and
ZERO shared memory beyond the counters.

```
kCtlReadDone   warps that have retired the READOUT phase, i.e. finished every read of S
               this visit makes. Gates the fold, which WRITES S.
kCtlGramDone   warps that have retired the GRAM phase. Gates the intra term, which
               READS Gm.
kCtlFold0      the fold's ONE item-claim cursor for the visit.
```

`ctl_words(BC) == kCtlFold0 + 1`: exactly three counters, always. There is no per-window
term — an earlier design windowed the write panel and gave the fold one claim cursor
PER WINDOW (`kCtlFold0 + w`); the window died (`ROLA_C_WT_WINDOW`, git history) and
took the per-window cursor with it, leaving one cursor for the visit's whole item pool.

Both PHASE counters count WARPS and every warp that participates in a phase retires it
UNCONDITIONALLY on that path, which is what makes the deadlock argument
([below](#ctl-reset)) a two-line one.

### <a id="release-acquire"></a>HAZARD `release-acquire` — the fence order in `retire_item` / `eligible`

**RELEASE (`retire_item`).** The fence comes FIRST, so every shared store the warp
made for the phase it is leaving is visible CTA-wide before the counter that
advertises them moves. The `atomicAdd` is executed by lane 0 ALONE because the
counters count WARPS — a per-lane count would make the target a lane count and the
phase fact would stop being a phase fact.

**ACQUIRE (`eligible`).** The load is `volatile` because the counter is written by
OTHER warps and a cached load would spin forever; the fence comes AFTER a successful
test, so no subsequent shared load of the published object can be hoisted above the
observation that published it. WARP-UNIFORM by construction — every lane loads the
same address, so the predicate is identical in all 32 lanes with no shuffle.

`await_count` is the spin, written in terms of `eligible` so that the release/acquire
discipline lives in exactly TWO functions and can be audited there rather than at
every phase edge. `__nanosleep` backs the polling warp off the issue port so a
spinning warp does not compete with the warps it is waiting for.

## 37. The visit's control-block reset, and the deadlock argument

The reset needs no barrier of its own on EITHER side, and both halves of that are
PROGRAM-ORDER facts rather than hopes:

- BEHIND it stands the previous visit's end-of-visit `__syncthreads()` (or, for the
  first visit, the CTA-init one), so no warp can still be reading last visit's
  counters;
- AHEAD of it stands the staging publication `__syncthreads()`, which is unconditional
  and sits between this store and the first `retire_item`. It publishes the zeroed
  counters on the same trip it publishes the visit's staged operands.

### <a id="ctl-reset"></a><a id="edge-dag"></a>Deadlock freedom, as the edge DAG and the always-runnable claim

NODES (per visit): `STAGE` (membership assembly / gather / row-scale and panel
builds), `A` (readout + mass term 1, read atoms only), `G` (Gram + `Gm` publication,
mixed atoms only), `F` (fold items + mass fold, write atoms only), `C` (intra + mass
term 2 + emission, mixed atoms only), `END` (split-K emission, the epilogue, the
end-of-visit barrier).

| edge | mechanism |
|---|---|
| `STAGE -> A`, `STAGE -> G` | the staging `__syncthreads()` (hardware), after the group's operands are staged |
| `A -> F` | `kCtlReadDone` reaching `kWarps` (readout READS `S`, the fold WRITES it). Every warp retires it unconditionally, whether or not it ran `A`'s work — retirement is not gated on `is_read` |
| `G -> C` | `kCtlGramDone` reaching `kGramWarps`. Both the retirement (`if (mixed && warp < kGramWarps) retire_item(...)`) and the wait (`if (mixed) await_count(...)`) are conditioned on the SAME CTA-uniform predicate `mixed = is_read && is_write`, so a visit either has both a Gram and an intra term or neither |
| `F -> END`, `C -> END` | program order into the end-of-visit `__syncthreads()` |

The relation is a TOTAL ORDER on `{STAGE, {A, G}, F, C, END}` and therefore ACYCLIC.
(`A` is unordered with respect to `G` — the Gram reads the fragment-ordered read panel
and `WT`, and touches neither `S` nor `Gm`'s readers — and so are `F` and `C`: `F`
writes `S`, `C` reads `Gm`, `Vw`, `sr`, `den`, and the sets are disjoint. That overlap
is what the counter design exists to exploit instead of serializing away.)

**ALWAYS-RUNNABLE**, and it is two facts:

1. **Every increment a waiter needs is issued, in program order, before any warp can
   reach the wait that needs it.** For `kCtlReadDone` that is the unconditional
   retirement every warp executes above the fold, target `kWarps`. For
   `kCtlGramDone`: `mixed` is CTA-uniform (`is_read`/`is_write` are derived from the
   visit's read/write atom masks, the SAME expression evaluated identically by every
   thread) and `warp < kGramWarps` is compile-time comparable, so either the whole CTA
   takes the counter — retirement happens, on the warps `0..kGramWarps-1`, strictly
   before the wait in program order — or none of it does, in which case nobody
   retires, nobody waits, and the word stays at the reset value, re-established next
   visit by the same unconditional store on a path every warp reaches whichever branch
   it took. A visit cannot have waiters without retirers.
2. **No `__syncthreads()` is ever reached while another warp spins.** Between the
   staging barrier and the end-of-visit barrier there is no further `__syncthreads()`
   in the visit body — only the two counter waits (`kCtlReadDone` before the fold's
   first claim, `kCtlGramDone` before the intra term), and both are on counters every
   contributing warp has already retired, in program order, before any warp can reach
   the corresponding wait. So no warp sitting at the end-of-visit barrier is waiting on
   a warp that is itself spinning on something only that first warp could supply.
