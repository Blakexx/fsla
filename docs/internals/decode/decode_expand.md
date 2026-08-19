# `csrc/rola/src/decode/decode_expand.cuh` — the `T = 1` support expansion

`supp(R)` and `supp(W)` as EXACT leaf bitmaps, built by Kronecker broadcast. The whole of
this file is the recursion, `expand_sides` — a header, not a translation unit, because
its only caller is the step kernel: both sides' expansion is one phase of `decode.cu`'s
per-CTA prologue, called once per CTA over the amplitudes that CTA's own fold staged.

Why the support is a Cartesian product at `T = 1`, and therefore why a bitmap is
the natural object at all, is [`decode.md`
§2](decode.md#kronecker-support). This doc covers what the expansion does with that fact.

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

## <a id="the-recursion"></a>3. `expand_sides` — the recursion over levels, innermost first

The expansion, for `N_SIDES` sides at once. `amp0`/`amp1` are the sides' flat
`[sum_l b_l]` staged row sets, the ones the step's own prologue left in shared memory
([`decode_fold.md`](decode_fold.md)); `buf` is `2 * N_SIDES * n_words` words, side `i`'s
ping-pong pair at `2 * i * n_words`. It returns the recursion's final PARITY, which is
the same for every side, so side `i`'s published map is `buf + 2*i*n_words + parity *
n_words` and no publishing pass exists. Parity rather than a pointer pair because an
indexable local array of pointers is a stack frame, and this kernel has none.

Level `D-1` seeds one inner block of `b_{D-1}` bits from its own amplitudes, for every
side at once. Each outer level `l` then broadcasts the `S_{l+1}`-bit block it has into
every LIVE digit's leaf range, and leaves every DEAD digit's range zero. After level 0
the buffer holds `S_0 = N` bits, which is the exact support.

**THE SIDES SHARE THE BARRIERS AND THE DIGIT ARITHMETIC**, which is the whole reason they
are one loop: the destination-word decomposition depends on the LEVEL and the word, never
on the side, and a second pass over it would double both the barrier count and the
integer divisions for no new answer.

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
placement can fix. The two buffers are the fix, and they are why the step pays
`2 * n_words` words per side on top of whatever it publishes into.

## <a id="smem-arm"></a>5. `IN_SMEM` is a structure-driven arm, not a threshold

`n_words` is a TOPOLOGY fact (`ceil(N/32)`) and the SMEM budget is a HARDWARE fact, and
spec I2 admits both as branch inputs. The two arms compute the IDENTICAL function — only
where the maps live moves.

It is a TEMPLATE axis rather than a runtime pointer because it decides the ADDRESS SPACE
the walk's hottest index is in: `IN_SMEM ? smem : work` folds at compile time, so the true
arm's bitmap accesses are shared-memory instructions rather than generic ones.

The step's CTAs need `4 * n_words` words each: both published side maps plus the
ping-pong pair. `decode_work_words_per_cta` is the single authority over whether that
overflows the device's shared-memory budget, and it answers `0` at every shipped
topology — the ordinary arm, where the whole bitmap lives in SMEM and never reaches DRAM
at all. The host asks it (`rola_decode_work_words`) rather than mirroring its arithmetic
in Python, because a mirror is the thing that drifts. The global arm's slices are indexed
per CTA at `(bh * n_split + seg)` — the step's split-K CTAs each expand their own copy
([`decode.md` §21b](decode.md#per-cta-prologue)) — so no two CTAs ever write the same word.

## <a id="coarse-index"></a>6. The coarse popcount index

The index the walk partitions and skips with belongs to the STEP, which is where it is
built and where it is documented: [`decode.md` §12](decode.md#coarse-index). It is not
part of the expansion — this header builds only the two leaf bitmaps, not the index over
them.
