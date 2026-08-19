# `csrc/rola/src/decode/decode_expand.cuh` — the `T = 1` support expansion

`supp(R)` and `supp(W)` as EXACT leaf bitmaps. The file is two things: `expand_inner`,
the recursion that materializes the INNER BLOCK, and `SideView`, which applies level 0's
broadcast to that block at the word a caller asks for. It is a header, not a translation
unit, because its only caller is the step kernel: the expansion is one phase of
`decode.cu`'s per-CTA prologue, over the amplitudes that CTA's own fold staged.

Why the support is a Cartesian product at `T = 1`, and therefore why a bitmap is
the natural object at all, is [`decode.md`
§2](decode.md#kronecker-support). This doc covers what the expansion does with that fact.

Symbols: `N` leaves; `D` levels; `b_l` level `l`'s width;
`S_l = prod(b_{l..D-1})` its span, so `S_0 = N` and `S_D = 1`;
`n_words = ceil(N/32)`; `inner_words = ceil(S_1/32)`; `bh` the batch-head;
`M` the touched-row count.

---

## <a id="why-a-bitmap"></a>1. Why a bitmap and not a candidate list

The alternative to building the support is BOUNDING it: a mixed-radix odometer
over the per-level UNION lists, skipping candidates outside
`supp(R) UNION supp(W)` ahead of the state load. That is correct and costs no
bandwidth, but the union is not a Cartesian product, so the odometer has to
enumerate the containing product `prod_l U_l` — whereas the exact set is a bitmap.

**The bound is more expensive to reason about than the truth is to materialize.**

So none of the machinery a bound needs exists on this path:

- per-row `in_R` / `in_W` running-AND chains;
- a skip test;
- a compacted list with digit ids and membership flags;
- an enumeration-factor statistic, and the bench observable that would have to
  report it.

## <a id="the-cost"></a>2. What is materialized, and what is not

A bitmap's DEAD ranges have to be written as zeros too, so materializing an `N`-bit map
costs `ceil(N/32)` words per side REGARDLESS of sparsity — and the step's grid is
`(n_split, BH)` with `n_split` itself growing with `N`, so a materialized level 0 makes
the step's fixed cost QUADRATIC in the leaf space. At `N = 65,536` that was 2048 words
per side per CTA across 256 CTAs, and 32 KiB of shared memory that capped residency at
2 CTAs/SM against the declared 5.

**None of it is necessary, because level 0 is a BROADCAST and not data.**
`supp = prod_l supp_l` means the level-0 expansion is one inner block of `S_1` bits
repeated into every live outer digit's leaf range and zero elsewhere. So any bitmap word
is a pure function of

- the inner block (`inner_words` words — 8 at `(256, 256)` against 2048), and
- level 0's staged amplitude row (already in shared memory, `b_0 <= 256` floats).

What is materialized is the inner block. What is computed on demand is every word of the
leaf map. The step's fixed cost per CTA is then `O(S_1/32 + b_0)`, which does not grow
with `N` at fixed level widths.

## <a id="the-recursion"></a>3. `expand_inner<D>` — the recursion over levels, innermost first

Levels `1..D-1`, for both sides at once. `amp_r`/`amp_w` are the sides' flat
`[sum_l b_l]` staged row sets, the ones the step's own prologue left in shared memory
([`decode_fold.md`](decode_fold.md)); `buf` is `4 * inner_words` words, side `i`'s
ping-pong pair at `2 * i * inner_words`. It returns the recursion's final PARITY, which is
the same for both sides, so side `i`'s block is `buf + 2*i*inner_words + parity *
inner_words` and no publishing pass exists. Parity rather than a pointer pair because an
indexable local array of pointers is a stack frame, and this kernel has none.

Level `D-1` seeds one block of `b_{D-1}` bits from its own amplitudes. Each outer level
`l` down to level 1 then broadcasts the `S_{l+1}`-bit block it has into every LIVE digit's
leaf range, and leaves every DEAD digit's range zero. After level 1 the buffer holds `S_1`
bits.

`D == 1` has no inner level: `S_1` is the empty product `1` and the block is one set bit,
which level 0 then broadcasts into each of its `b_0 = N` leaves. The degenerate case runs
the SAME view, not a second path.

**THE SIDES SHARE THE BARRIERS AND THE DIGIT ARITHMETIC**, which is the whole reason they
are one loop: the destination-word decomposition depends on the LEVEL and the word, never
on the side, and a second pass over it would double both the barrier count and the
integer divisions for no new answer.

**Membership is the exact-nonzero test**, and that is not a threshold: it is the
DEFINING EXPRESSION of `schedule.py`'s support bitmap
(`read_mask = tuple(level != 0 ...)`). Evaluating it here rather than reading a
materialized bitmap is the same predicate, one step earlier.

The destination-word loop stops at `span_out`, so bits at or beyond the level's
span stay zero — which is what lets the GEMV walk with no leaf bound test
([`decode.md` §26](decode.md#aligned-subranges)).

## <a id="the-view"></a>3b. `SideView` — level 0 as a function

A view is `{inner block, level 0's amplitude row, inner_words, S_1, N}`: two SHARED-memory
pointers (32-bit addresses) and three integers, which is why one survives into the walk
where a materialized `N`-bit map could not.

`view_word_at(v, base, g, i)` returns the bitmap word starting at leaf `base`, given that
leaf's position as `(outer digit g, offset i within the digit)`. It is the level-0
broadcast body evaluated for one destination word: for each aligned sub-range of the word
it takes `extract_bits` of the inner block when `amp[g] != 0`, and nothing when it is
zero. Bits at or beyond `N` stay zero, exactly as a materialized map would have them.

**THE POSITION IS AN ARGUMENT BECAUSE EVERY CALLER ALREADY HAS IT.** The walk decomposes
`leaf_base` into digits anyway ([`decode.md` §25](decode.md#divisions-per-word)) and that
decomposition's first digit and remainder ARE `(g, i)` — which is why the decomposition
now stands above the union test. `locate_row` scans contiguous words, so it opens with one
division and carries the pair across them. Level 0's broadcast therefore costs no integer
division on either path. `view_word(v, w)` is the convenience for the atom tests, which
jump rather than walk, and it pays the one division.

## <a id="destination-ownership"></a>4. Destination-word ownership, and the ping-pong

**DESTINATION-WORD OWNERSHIP is what makes the recursion race-free without atomics.** One
thread owns one destination word and PULLS every bit it needs, rather than each
source digit PUSHING into ranges that two threads could share at a boundary word.
The pull needs at most `ceil(32 / S_{l+1}) + 1` source digits per destination
word, so the integer division that finds the digit is paid once or twice per WORD
and never once per BIT. The gather itself is
[`decode.md` §14](decode.md#extract-bits)'s funnel shift.

<a id="ping-pong"></a>**PING-PONG, not in place.** Destination word 0 is read by
the thread writing the last destination word — every digit broadcasts the SAME source
block — so writing into the source would be a read-write race that no barrier
placement can fix. The two buffers are the fix, and they are why the recursion pays
`2 * inner_words` words per side.

## <a id="smem-arm"></a>5. `IN_SMEM` is a structure-driven arm, not a threshold

`inner_words` is a TOPOLOGY fact (`ceil(S_1/32)`) and the SMEM budget is a HARDWARE fact,
and spec I2 admits both as branch inputs. The two arms compute the IDENTICAL function —
only where the inner blocks live moves.

It is a TEMPLATE axis rather than a runtime pointer because it decides the ADDRESS SPACE
of an index the prologue and the walk both read: `IN_SMEM ? smem : work` folds at compile
time, so the true arm's accesses are shared-memory instructions rather than generic ones.

The step's CTAs need `4 * inner_words` words each: both sides' blocks plus the
ping-pong pair. `decode_work_words_per_cta` is the single authority over whether that
overflows the device's shared-memory budget, and it answers `0` at every shipped
topology. It stays a real arm rather than dead weight because `S_1 = N / b_0` is small
only when `b_0` is not: a topology whose outer level is one digit wide materializes the
whole leaf space, and that is the case the arm exists for. The host asks it
(`rola_decode_work_words`) rather than mirroring its arithmetic in Python, because a
mirror is the thing that drifts. The global arm's slices are indexed per CTA at
`(bh * n_split + seg)` — the step's split-K CTAs each expand their own copy
([`decode.md` §21b](decode.md#per-cta-prologue)) — so no two CTAs ever write the same word.

## <a id="coarse-index"></a>6. The coarse popcount index

The index the walk partitions and skips with belongs to the STEP, which is where it is
built and where it is documented: [`decode.md` §12](decode.md#coarse-index). It is not
part of the expansion — this header builds the inner block and serves words over it, not
an index across them.
