# `csrc/rola/src/summaries.cuh` + `csrc/rola/src/summaries.cu` — the producer-epilogue routing summary pass

the routing summary pass, as an ahead-of-time CUDA kernel.
This doc covers **two** source files, because they are one component: the header
carries the parameter block and the pass's declared bounds, the `.cu` carries the
kernel and the pybind entry. The mirror rule is one doc per source basename, so
they land here together. **Part A** is the header; **Part B** is the translation
unit.

Symbols: `D` is the level count; `b_l` level `l`'s width; `N` the leaf count;
`BC` the state block (owner size) and `owners_total = N / BC`; `BT` the token
tile; `T` the token count and `L` the sequence length; `BH = B * H`; `j` the
`first_spanned_level`; `c` the `slice_digits`; `QPO = BC / MMA_K_QUANTUM`;
`K_hat` the per-owner gathered k-extent. A *word* throughout means one `int32`
whose bit `t` is "token `t0 + t`", so `BT <= 32`.

---

# Part A — the header (`summaries.cuh`)

## <a id="why-a-kernel"></a>1. Why this is a kernel and not fused torch

An eager-torch derivation of this pass lives in `rola/routing/summaries.py` as the
ENUMERATED REFERENCE this kernel is gated against — a second derivation, not a
fallback: no runtime path reaches it, and it refuses to be called as one.

Three reasons the runtime pass is hand-written:

1. **No Inductor in the runtime.** One standing rule exists partly to
   remove `torch.compile` as a runtime dependency of the planner; a fused torch
   summary pass would put it straight back. Eager torch instead means ~30 kernel
   launches and a DRAM round trip per intermediate, on a pass whose whole
   justification is that it is microseconds.
2. **The representation is bits.** Every step of the pass is a word operation —
   per-tile union words over DIGITS, `__popc`, `__ballot_sync`, an
   atomicOr-and-count-new-bits. Torch has no vocabulary for that; it would spell
   each one as a boolean tensor and a reduction.
3. **One launch, one D2H, zero host syncs** is a property of the pass, and the
   only way to have it is to write the pass.

## <a id="hand-roll-rule"></a>2. The standing rule this follows

**HAND-ROLL SEMANTICS, USE LIBRARIES FOR COMMODITY PRIMITIVES.** The digit
algebra, the owner rectangle, the candidacy union and the leaf AND-chain are
SEMANTICS and are written out. The scalar folds are COMMODITY and are
`cub::BlockReduce`. CUB is header-only and ships with the pinned toolkit, so it is
covered by the closed-world codegen policy's toolchain pin rather than vendored
separately.

The one place CUB is NOT used is the histogram. `cub::BlockHistogram` needs a
compile-time bin count, and the bin count here is `QPO + 1 = BC/16 + 1` — which
invariant I1 forbids turning into a literal or a template ladder. SMEM atomics
over a runtime bin count are the standard alternative and cost one `atomicAdd` per
candidate pair on a block-local array.

## <a id="what-it-computes"></a>3. What the pass computes, and at what cost

Per `(bh, tile)` CTA:

| output | what it is | cost |
|---|---|---|
| `words` | one `int32` per `(side, level, digit)`, bit `t` = token `tile*BT + t` is live | `O(BT * sum_l b_l)` coalesced reads — ONE pass over the routing, which the CTA has to make anyway |
| `occupancy` | `__popc` of those words | free |
| `K_hat` | per owner, the product over spanned levels of the owner-slice popcount of the UNION word | `O(owners_total)` |
| `candidacy` | per owner, per side, the fixed-digit AND and the slice OR | `O(owners_total)` |
| `histogram` | `ceil(K_hat / MMA_K_QUANTUM)`, binned | `O(owners_total)` |
| `leaf bits` | §4.1's canonical per-leaf WRITE bitmap | `O(N)` — the irreducible term |

The leaf bitmap is irreducible because a per-leaf union over tokens is an OR of
ANDs and cannot be condensed from any per-owner statistic. It is warp-packed with
`__ballot_sync` so 32 leaves cost ONE `atomicOr`, and it is skipped entirely for
an owner that is not write-candidate in this tile — an in-kernel decision on data
the CTA has already fetched, which is invariant I3's own criterion and not a plan
field.

The owner-level condensation §4.2 reports rides the SAME `atomicOr`: the return
value says which bits were newly set, so `__popc` of them is the exact increment
for the active-owner counter and no second pass over the bitmap is needed. **That
is what keeps the whole thing to one launch.**

## <a id="quantum"></a>4. `kMmaKQuantum = 16`

§6's MMA k-extent, and therefore the quantum every gathered extent rounds up to. A
SHAPE of the instruction the consumer issues, never a tuning threshold.

**It is not defined here.** This pass's bin count and the consumer's atom
granularity are the same hardware number, so there is ONE definition — in
`arch_table.cuh`, which this header includes:
[arch_table.md#mma-quantum](arch_table.md#mma-quantum).
`rola.routing.summaries.MMA_K_QUANTUM` is the Python mirror; the host entry point
checks the derived bin count against the caller's, so a drift is a failure and not
a silent one.

## <a id="declared-bounds"></a>5. The declared bounds, and why each is a refusal

| constant | value | what it bounds |
|---|---|---|
| `kTokenBitsPerWord` | 32 | token bits carried by one support word |
| `kMaxTotalDigits` | 2048 | `sum_l b_l`, the digit-row extent one CTA stages |
| `kMaxLevels` | 16 | `D` |

`kTokenBitsPerWord`: a `BT` above this cannot be REPRESENTED and is REFUSED, never
split into a second code path.

`kMaxTotalDigits`: two `int32` arrays of this length live in shared memory (read
words and write words), so the budget is `8 * kMaxTotalDigits` bytes = 16 KiB. It
is a STRUCTURAL bound under the standing *"design for `nc ~ 65K`"* rule: at the
parity topology (`branches = 256, 256`) `sum_l b_l = 512`, so this admits 4x that,
with the union words derived on the fly rather than stored.

`kMaxLevels`: a declared ceiling on `D`, generous relative to the consumer's own
`D <= 4` (spec §1, `consumer.cuh`'s `kMaxLevels`) — the summary pass carries no
level-count restriction of its own to justify a tighter one.

## <a id="params"></a>6. `SummaryParams` — every argument the pass takes

Nothing derived that the host can derive. `level_amps` is the producer's FOLDED
per-level allocation, `[BH, T, b_l]` fp32 — exactly what `RoutingViews.read_folded`
/ `.write_folded` hold. Liveness is EXACT nonzero-ness of the amplitude (spec
§1.2: entmax's exact support zeros are the scheduling authority), never a
threshold, and never a magnitude test on anything downstream of it.

The per-level integer vectors are `level_width[l] = b_l`, `level_row_offset[l]`
(offset into the digit-row space), `level_leaf_stride[l] = prod(b_{l+1..D-1})` and
`level_fixed_radix[l] = prod(b_{l+1..j-1})`, which is `1` on spanned levels. The
scalars are `D`, `BH`, `T`, `BT`, `n_tiles`, `sum_l b_l`, `N`, `BC`,
`owners_total`, `n_slices`, `c`, `j`, `bins = QPO + 1`, `leaf_words = ceil(N/32)`
and `owner_words = ceil(owners_total/32)`. The outputs are `occupancy` `[2 * D]`,
`histogram` `[bins]`, the two scalar counters, `written_leaf_bitmap`
`[BH, leaf_words]` (§4.1's canonical truth) and `written_owner_bitmap`
`[BH, owner_words]` (the derived condensation).

<a id="params-by-value"></a>**The struct is PASSED BY VALUE, in the kernel's
parameter space, and that is deliberate.** The alternative is four small device
tensors staged with `from_blob(...).to(device)`, i.e. four pageable host-to-device
copies on a pass whose declared budget is ONE transfer and ZERO host
synchronizations. Kernel parameters land in constant memory with the launch, so
this costs no transfer at all: `kMaxLevels * (2*8 + 4*4) = 512` bytes, far inside
the 4 KiB parameter limit.

## <a id="entry-points"></a>7. The two entry points

`launch_summaries` is one launch, no host synchronization, nothing allocated.
Every output tensor is caller-owned and must be ZEROED beforehand — the kernel
folds into them with atomics, so it ACCUMULATES rather than assigns, and that is
what lets one launch serve every `(bh, tile)` pair.

`rola_routing_summaries_cuda` is the pybind surface. It validates shapes, builds
the device pointer arrays, and launches. It returns nothing: the outputs are the
caller's tensors, so the Python side owns the ONE D2H transfer and can decide what
rides it.

---

# Part B — the translation unit (`summaries.cu`)

## <a id="launch-shape"></a>8. The launch shape

`kThreads = 256`, `kWarps = 8`. `kMaxBlocks = 4096` is a BOUNDED grid: the kernel
walks `(bh, tile)` rows with a GRID-STRIDE loop, so the per-CTA constant work is
paid once per CTA instead of once per tile, and the grid does not grow with `L`. A
launch SHAPE, not a threshold on any data.

`kMaxSlices = 512` and `kMaxBins = 33` are the ceilings the static shared-memory
budget is sized from. `n_slices <= b_j`, and a level wider than `kMaxSlices` has
already failed `kMaxTotalDigits`. `kMaxBins` is `QPO + 1` at `BC = 512` and is the
ARRAY BOUND ONLY: every expression that uses the bin count uses
`p.bins = BC / kMmaKQuantum + 1` (invariant I1).

## <a id="shared-state"></a>9. `SharedState` — one struct, so the budget is a `sizeof`

The shared-memory layout is a single struct rather than a sum maintained by hand,
so the budget is a `sizeof` and the `static_assert` below it is meaningful:

```
static_assert(sizeof(SharedState) <= 48 * 1024, ...)
```

A CUB version bump that grew `TempStorage` past the histogram it shares bytes
with, or past the SMEM a CTA may hold, must fail the COMPILE and not the launch.

<a id="scratch-union"></a>`fold` (CUB's temp storage) and `hist` **share the same
bytes**, as an explicit `union`: they are used in DISJOINT phases separated by a
barrier, and stating that as a union is stronger than leaving a reader to infer it
from barrier placement. The barrier that separates them is the one inside
`fold_u64` — `TempStorage` is reused by the next fold AND by `hist`, so removing
it aliases live data.

`level_pc[l]` is the popcount of level `l`'s UNION digit set for this tile;
`slice_pc[sl]` the per-boundary-slice union digit count; `slice_any_r` /
`slice_any_w` the per-slice per-side liveness flags.

## <a id="fold-u64"></a>10. `fold_u64` is block-wide

Fold one per-thread partial into a global counter with ONE atomic per block.
BLOCK-WIDE: every thread must reach it, which is why it is only ever called from
block-uniform control flow.

## <a id="write-candidate"></a>11. `write_candidate` — exact, not conservative

Is this owner WRITE-CANDIDATE in the staged tile? A §4 product-set membership test
on data the CTA already holds — invariant I3's own criterion for an in-kernel
decision — and EXACT rather than conservative: the slice must have some live write
digit, and every fixed level's owner digit must have a nonzero write word.

## <a id="support-words"></a>12. Step 1 — the support words, and occupancy for free

One `int32` per `(side, level, digit)`; bit `t` is "this digit is live at token
`t0 + t`". Liveness is EXACT nonzero-ness of the amplitude (spec §1.2), never a
threshold.

Threads cover consecutive DIGITS with the token loop inside, so each read is
coalesced across the warp. The padded token tail (`t0 + t >= T`) contributes no
bit, which is the whole of what a ragged `L` needs.

§4.2's `occupancy` counts live `(token, digit)` PAIRS, not the union, so it is
exactly `__popc` of the word — free here, and a second pass over the routing
anywhere else. It is folded per level INSIDE this loop rather than accumulated in
a per-thread array, because a `[kMaxLevels]` register array indexed by a loop
variable spills to local memory.

## <a id="union-popcounts"></a>13. Step 2 — per-level and per-slice union popcounts

The union WORDS are `(read | write) != 0` — a bitmap over DIGITS, which is the form
the owner-slice restriction needs: **a popcount cannot be restricted to a
sub-range after the fact.**

Levels strictly BELOW the boundary are spanned whole by every owner, so their
restricted popcount is the level's own and is owner-INVARIANT; that product
(`inner`) is computed once per tile rather than once per owner.

## <a id="owner-loop"></a>14. Step 3 — the owner loop: `K_hat`, candidacy, the histogram

FIXED levels: the owner holds ONE digit of each, so each contributes a 0/1
token-validity factor to `K_hat` and an AND term to candidacy.

§4 **CANDIDACY is the union of the two SIDES' product sets, never the candidacy of
the union word.** ORing the sides first would admit an owner whose read digit and
write digit are live at DIFFERENT levels — an owner no side ever visits.

`K_hat <= BC` STRUCTURALLY: the boundary level contributes at most `slice_digits`,
every level below it at most its full width, and their product IS `BC` by the owner
rectangle. So `bins = QPO + 1` is exhaustive, and no clamp exists anywhere in this
pass.

## <a id="leaf-bitmap"></a>15. Step 4 — §4.1's canonical per-leaf write bitmap

A leaf is written iff at SOME token of this tile EVERY level's digit for it is live
on the write side: an OR of ANDs, which is why this term is `O(N)` and cannot be
condensed from anything computed above. The AND is over the token WORDS, so all
`BT` tokens are tested at once (and the AND-chain short-circuits on `word == 0`).

<a id="ballot-warp-uniform"></a>Threads take CONSECUTIVE leaves within a warp, so
`__ballot_sync` of the per-leaf bit IS the packed 32-bit word: ONE `atomicOr` per
32 leaves. **The whole warp takes the loop together** (`chunk` is warp-uniform),
which is what makes the ballot well-formed on the ragged tail as well — a lane
whose leaf is past `N` still reaches the ballot, contributing `false`.

## <a id="owner-condensation"></a>16. The owner condensation rides the same atomic

§4.1's DERIVED owner condensation is folded on the SAME data rather than
recomputed. This 32-leaf chunk spans `owner_lo .. owner_hi` — one owner when
`BC >= 32`, `32/BC` of them otherwise — and a nonzero sub-mask means that owner is
written.

The `atomicOr` RETURNS the old word, so `__popc` of the newly set bits is the exact
increment for the active-owner counter. That is what keeps the pass to ONE launch,
instead of needing a second sweep over a bitmap that is only complete after the
grid drains.

## <a id="host-checks"></a>17. The host entry's refusals

Every bound in [§5](#declared-bounds) is enforced at the pybind surface with a
message that says it is a BUDGET or a REPRESENTATION change, not a runtime
condition to branch on:

- `D` in `[1, kMaxLevels]`, one folded `[BH, T, b_l]` allocation per level per
  side, all CUDA, fp32 and contiguous, and the two sides identical in shape;
- `BT` in `[1, kTokenBitsPerWord]` — "*a wider `BT` is a representation change that
  must widen the word*";
- `BC` a whole number of `kMmaKQuantum` atoms (spec §1b), and `BC` divides `N`;
- `sum_l b_l <= kMaxTotalDigits` — "*that is a BUDGET change (two `int32` arrays of
  this length)*";
- `n_slices <= kMaxSlices`; `j` indexes a level; `n_slices * c == b_j`, i.e. the
  boundary level's slices partition it EXACTLY;
- the six output tensors' dtypes and extents, including `histogram.numel() == bins`
  — which is the drift check [§4](#quantum) refers to.

`level_fixed_radix[l] = prod(b_{l+1..j-1})` is built here: the mixed radix of the
owner id's FIXED part, level 0 most significant, and `1` on the spanned levels,
which never index an owner.

# Part C — the python module (`rola/routing/summaries.py`)

## What Is Frozen Here And What Is Not

:class:`RoutingSummaries` is a CONTACT STRUCT: the contract freezes its five fields and
neither track may edit them. Everything else in this module is implementation.

**The ``u`` words are INTERNAL and never exported** (amended 2026-08-02 on user review). They are the per-tile per-level union WORDS -- the OR of a tile's ``BT``
support words at one level, carried as a bitmap over DIGITS rather than as a popcount,
because the owner-slice restriction at the boundary level needs digit identity. They
are the factorized stepping-stone that avoids the ``O(n_tiles * N)`` leaf-granularity
product: the whole chain

    union words -> per-owner slice popcount -> K_hat product -> ceil -> histogram bin

reduces ON DEVICE inside this pass, and only the reduced histogram leaves it.

**``u`` is NOT a coherence or multiplicity statistic.** It counts DISTINCTNESS, which
is exactly what quanta cost depends on. No multiplicity statistic exists here because
no decision consumes one.

## Why ``K_Hat`` Is An Upper Bound, And Why That Is The Safe Direction

``K_hat(o,t) = prod_l (owner-slice popcount of the level-l union word)`` is the size of
the product of the per-level unions; the true ``K`` is the size of the union of the
per-token products. ``union of products SUBSET product of unions``, so
``K_hat >= K`` pairwise, hence ``ceil(K_hat/16) >= ceil(K/16)`` pairwise, hence the
histogram's mass sits at or above the true quantum count, which is the safe
direction: a rule reading it can only under-state how sparse a launch is, never
over-state it. ``K_hat <= BC`` always, because the boundary level contributes at most
``slice_digits`` and every level below it at most its full width, whose product is
``BC`` by the owner rectangle -- so the histogram's ``QPO + 1`` bins are exhaustive
and no clamp is needed.

The residual is the cross-level COLLISION correction, passed in as the ``gamma``
argument, whose conservative default ``1.0`` leaves the bound uncorrected. It is an
argument rather than a constant because it has to be applied per ``(owner, tile)``
BEFORE the quantization and this is the only place that granularity still exists. At
the default the multiply is skipped outright, so an uncorrected run is bit-identical to
one with no hook at all.

## Cost, Stated Rather Than Discovered

Two terms, and they are different orders:

* the histogram/candidacy pass is ``O(BH * n_tiles * owners_total)``;
* :attr:`RoutingSummaries.written_leaf_bitmap` is ``O(BH * n_tiles * N)``, because
  the PER-LEAF write bitmap is the canonical truth (nine named consumers,
  including exact page-pool sizing) and a per-leaf union over tokens cannot be
  read off a per-owner statistic. It is the same order as the retired census's leaf
  plane, and it is a bit-AND chain rather than that pass's six reductions. Both passes
  are CHUNKED so the transient is bounded independently of ``N`` and of ``L`` -- a
  structural bound under the standing "design for nc ~ 65K" rule, not a tuning knob.

## Where The Pass Actually Runs

**THE RUNTIME PASS IS A HAND-WRITTEN AOT CUDA KERNEL**, ``csrc/rola/src/summaries.cu``
(user-ruled 2026-08-02). :func:`summarize_routing` calls it and nothing
else. The three reasons are in that file's header; the one that governs this module is
that the planner's whole retirement exists partly to remove ``torch.compile`` as
a RUNTIME dependency, and a fused-torch summary pass would put it straight back.

**THE TORCH IMPLEMENTATION SURVIVES AS THE ENUMERATED REFERENCE**
(:func:`summarize_routing_reference`), which is a second-derivation role and NOT a
fallback. The distinction is enforced rather than described: no runtime path calls it,
:func:`summarize_routing` does not degrade to it when the extension is missing, and the
gate that compares the two is the only caller. That is the same relationship
``_tile_union_stats_multi_enumerated`` has to the bit-word census, and it is what makes
"the kernel is right" a checkable claim instead of a self-consistency one.

## Synchronization Budget

The device work -- the kernel launch, or
:func:`summarize_routing_reference_device` on the reference path -- completes with ZERO
host synchronizations and leaves everything on device; the wrapper adds exactly ONE D2H
transfer, of two scalars, to fill the contract's two ``float`` fields. That split is what makes
the zero-sync property gateable, and it is why the two halves are separate functions.
