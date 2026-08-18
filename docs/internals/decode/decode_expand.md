# `csrc/rola/src/decode/decode_expand.{cuh,cu}` — the `T = 1` support expansion

`supp(R)` and `supp(W)` as EXACT leaf bitmaps, built by Kronecker broadcast. The RECURSION
is the header (`decode_expand.cuh`), called by the step for both sides and by the growth
probe for the write side; the TU (`decode_expand.cu`) is the growth probe and nothing
else.

Why the support is a Cartesian product at `T = 1`, and therefore why a bitmap is
the natural object at all, is [`decode.md`
§2](decode.md#kronecker-support). This doc covers what the expansion kernel does
with that fact.

Symbols: `N` leaves; `D` levels; `b_l` level `l`'s width;
`S_l = prod(b_{l..D-1})` its span, so `S_0 = N` and `S_D = 1`;
`n_words = ceil(N/32)`; `bh` the batch-head; `M` the touched-row count.

---

## <a id="why-a-bitmap"></a>1. Why a bitmap and not a candidate list

The alternative to building the support is BOUNDING it: a mixed-radix odometer
over the per-level UNION lists, skipping candidates outside
`supp(R) UNION supp(W)` ahead of the state load. That is correct and costs no
bandwidth, but the union is not a Cartesian product, so the odometer has to
enumerate the containing product `prod_l U_l` — whereas the exact set is an
`N`-bit map that is 8 KiB at `N = 65,536` against a state stream of 130 MiB.

**The bound is more expensive to reason about than the truth is to materialize.**

So none of the machinery a bound needs exists on this path:

- per-row `in_R` / `in_W` running-AND chains;
- a skip test;
- a compacted list with digit ids and membership flags;
- an enumeration-factor statistic, and the bench observable that would have to
  report it.

## <a id="the-cost"></a>2. The cost, and what an estimate must not leave out

A bitmap's DEAD ranges are written as zeros too, so the irreducible cost of
publishing an `N`-bit map is `ceil(N/32)` words per side REGARDLESS of sparsity —
8 KiB per side per `bh` at `N = 65,536`, 16 KiB for both. Counting only the
broadcast COPIES prices a sparse at-scale expansion at ~544 B per `bh` and
undercounts by exactly that difference.

That is a FLOOR, not an inefficiency. What the SMEM arm ([§5](#smem-arm)) buys is
that the `D - 1` INTERMEDIATE levels cost SMEM traffic rather than HBM traffic, so
the HBM figure IS the floor. The global arm pays the intermediates to HBM and is
`D` times the floor.

Both are noise against the dense-read state stream (0.012 % at at-scale), and both
are visible against a very sparse high-batch step; see the findings doc.

## <a id="the-recursion"></a>3. `expand_side` — the recursion over levels, innermost first

The expansion, for ONE side; `amp` is that side's flat `[sum_l b_l]` staged row set, the
one the fold left in shared memory ([`decode_fold.md`](decode_fold.md)).

Level `D-1` seeds one inner block of `b_{D-1}` bits from its own amplitudes. Each
outer level `l` then broadcasts the `S_{l+1}`-bit block it has into every LIVE
digit's leaf range, and leaves every DEAD digit's range zero. After level 0 the
buffer holds `S_0 = N` bits, which is the exact support.

**Membership is the exact-nonzero test**, and that is not a threshold: it is the
DEFINING EXPRESSION of `schedule.py`'s support bitmap
(`read_mask = tuple(level != 0 ...)`, and §2386's "exact nonzero-ness of the
amplitude, never a threshold"). Evaluating it here rather than reading a
materialized bitmap is the same predicate, one step earlier.

The destination-word loop stops at `span_out`, so bits at or beyond the level's
span stay zero — which is what lets the GEMV walk with no leaf bound test
([`decode.md` §26](decode.md#aligned-subranges)).

## <a id="destination-ownership"></a>4. Destination-word ownership, and the ping-pong

**DESTINATION-WORD OWNERSHIP is what makes this race-free without atomics.** One
thread owns one destination word and PULLS every bit it needs, rather than each
source digit PUSHING into ranges that two threads could share at a boundary word.
The pull needs at most `ceil(32 / S_{l+1}) + 1` source digits per destination
word, so the integer division that finds the digit is paid once or twice per WORD
and never once per BIT. The gather itself is
[`decode.md` §14](decode.md#extract-bits)'s funnel shift.

<a id="ping-pong"></a>**PING-PONG, not in place.** Destination word 0 is read by
the thread writing destination word 100 — every digit broadcasts the SAME source
block — so writing into the source would be a read-write race that no barrier
placement can fix. The two buffers are the fix, and they are why every caller pays
`2 * n_words` words on top of whatever it publishes into.

## <a id="smem-arm"></a>5. `IN_SMEM` is a structure-driven arm, not a threshold

`n_words` is a TOPOLOGY fact (`ceil(N/32)`) and the SMEM budget is a HARDWARE fact, and
spec I2 admits both as branch inputs. The two arms compute the IDENTICAL function — only
where the maps live moves.

It is a TEMPLATE axis rather than a runtime pointer because it decides the ADDRESS SPACE
the walk's hottest index is in: `IN_SMEM ? smem : work` folds at compile time, so the true
arm's bitmap accesses are shared-memory instructions rather than generic ones.

The two callers hold different counts:

| caller | words per CTA | why |
|---|---|---|
| the step | `4 * n_words` | both published side maps plus the ping-pong pair |
| the growth probe | `2 * n_words` | the ping-pong pair; its ONE map is its output (`probe_bits`) |

`decode_work_words_per_cta` is the single authority over whether either overflows, taken
over both so ONE global buffer serves both, and it answers `0` at every shipped topology.
The host asks it (`rola_decode_work_words`) rather than mirroring its arithmetic in Python,
because a mirror is the thing that drifts. The global arm's slices are indexed per CTA at
`(bh * n_split + seg)` — the step's split-K CTAs each expand their own copy
([`decode.md` §21b](decode.md#per-cta-prologue)) — so no two CTAs ever write the same word.

## <a id="coarse-index"></a>6. The coarse popcount index

The index the walk partitions and skips with belongs to the STEP, which is where it is
built and where it is documented: [`decode.md` §12](decode.md#coarse-index). It is not part
of the expansion — the growth probe never forms it, because it partitions nothing.

## <a id="growth-probe"></a>7. The growth probe — the same expansion, asked one step earlier

`decode_growth_probe_kernel` is this TU's whole content, and it exists so a paged decode
step can skip its admission preamble. It answers one question — does this step
deposit into an atom the page table does not map? — and PUBLISHES the object it answered
that question from: `atom_bits`, `[BH, atoms_per_bh]` bool, the step's exact write-atom
set. A step that does grow admits from that buffer, so the set is derived ONCE, here,
instead of once here and again on the host.

**It is here, and not in `paging.py`, for one reason.** It calls `fold_side` and then
`expand_side` — the same two device functions the step calls, over the same raw write views
— so the write support it tests IS the write support the step will expand. Any other placement would
mean a second derivation of the same rule, and a residency answer derived from a support
that could drift is worse than no answer: a missed atom is a deposit written into a `slot <
0` row, which is silently dropped.

It runs BEFORE the step rather than inside it, which is the one thing the placement cannot
be talked out of. The step's own expansion is followed, in the same kernel, by the walk that
deposits; the walk is what needs the corrected page table, so a flag the step itself raised
would be a flag raised one phase too late.

<a id="published-map"></a>**Its expansion is PUBLISHED, and the step's is not.** The probe
writes its `N`-bit write map to `probe_bits` (`[BH, n_words]`) and condenses the atoms out
of it, so the object the residency answer was reached from is a real buffer rather than an
internal — which is also what lets `tests/integration/test_decode_growth_trigger.py` hold
it to `_leaf_support`, the definition written in torch. The step publishes nothing: its two
maps live and die in the shared memory of the CTA that built them, and the gate on THAT
expansion reads it off the state diff instead
(`tests/oracle/test_decode_vs_oracle.py::test_g3_expansion_is_the_exact_leaf_support`).

<a id="growth-reduction"></a>**THE REDUCTION, and why it is all-or-nothing.** Every CTA
writes its own `growth[bh]` unconditionally, so that buffer needs no zero-fill; the
LAST-ARRIVING CTA then ORs the whole vector into `growth_any`, a `[1]` int32, and resets
the arrival counter it used — the same self-resetting pattern the GEMV's `ctr` uses, and
for the same reason: no `memset` may stand between two decode steps if the step is to be
capturable. `growth_any` is what the STEP reads as its gate
([`decode.md` §4c](decode.md#the-gate)), and the skip it drives is all-or-nothing across
batch-heads on purpose — a per-`bh` skip would need a per-`bh` replay, and a replay that
re-ran an already-deposited `bh` would double its write.

The condensation is two half-word tests and no loop over leaves: a bitmap WORD is exactly
TWO atoms, because an atom is 16 leaves and leaf order is the bitmap's bit order. Word `w`
covers atoms `2w` (its low half) and `2w + 1` (its high half), each guarded against
`atoms_per_bh` for a leaf space whose last word is half used. Both bytes are written
UNCONDITIONALLY: a dead atom's `false` is as much of the published set as a live atom's
`true`, and writing only the live ones would leave the previous step's answer standing in
the buffer the next admission plans from.

The reduction is a `__shared__ int` written by any thread that found something and read by
thread 0 — no atomic, because every writer writes the same value — and every CTA writes its
own `growth[bh]` entry unconditionally, so the buffer needs no zero-fill between steps. The
host reduces `[BH]` to a verdict after one pinned copy; the cost ledger and the measured
alternative are [`decode.md` §4b](decode.md#growth-probe).

It shares the step's `bits_w` and `scratch` workspace. Nothing reads either between the two
launches, and the step's own expansion overwrites both.
