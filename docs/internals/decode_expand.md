# `csrc/rola/src/decode_expand.cu` — the `T = 1` support expansion

`supp(R)` and `supp(W)` as EXACT leaf bitmaps. One CTA per batch-head: it builds
two `N`-bit maps by Kronecker broadcast and emits the coarse popcount index the
GEMV partitions and skips with.

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

The expansion, for ONE side; `amp` is that side's per-level amplitude row set.

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
placement can fix. The two buffers are the fix, and they are why the SMEM arm
costs `2 * n_words` words.

## <a id="smem-arm"></a>5. `IN_SMEM` is a structure-driven arm, not a threshold

The ping-pong pair is `2 * n_words` words; `n_words` is a TOPOLOGY fact
(`ceil(N/32)`) and the SMEM budget is a HARDWARE fact, and spec I2 admits both as
branch inputs. The two arms compute the IDENTICAL function — only where the
intermediate levels live moves — which is why the G3 gate runs both arms against
the same expected bitmaps.

`decode_expand_smem_bytes` is the ping-pong pair and nothing else: the
coarse-index scan's storage is static `__shared__`, sized by the block rather than
by the topology. It is host-visible so the launcher picks the arm from the
device's real budget rather than from a literal.

## <a id="coarse-index"></a>6. The coarse popcount index

`pc_coarse[g]` is the number of set bits of `bits_r | bits_w` in groups strictly
before `g`; `pc_coarse[n_groups]` is `M`, the exact touched-row count. The GEMV
binary-searches it to resolve a ROW range into a WORD range
([`decode.md` §19](decode.md#locate-row)), and consults it again DURING the walk
to skip a wholly empty group without reading its 32 words
([`decode.md` §24](decode.md#coarse-consult)).

The count is taken from the PUBLISHED words rather than accumulated per side
during publication, and the reason is arithmetic rather than tidiness: the two
sides' popcounts SUM to something that is not `popcount(r | w)` — a leaf in both
supports would be counted twice, and `M` has to be the row count the GEMV will
actually visit.

This is the ONE place the union is formed outside a register, and it is formed as
a COUNT. No third bitmap is ever stored.

The block-wide exclusive sum is `cub::BlockScan`, carried across the outer
`base`-strided loop by a `__shared__ int running`. That is a commodity primitive
on a scalar fold; the digit algebra above it is semantics and is written out —
the same standing rule `summaries.cuh` states in full
([`summaries.md` §2](summaries.md#hand-roll-rule)).

## <a id="growth-probe"></a>7. The growth probe — the same expansion, asked one step earlier

`decode_growth_probe_kernel` is the second kernel in this TU and it exists so a paged
decode step can skip its admission preamble. It answers one question — does this step
deposit into an atom the page table does not map? — and PUBLISHES the object it answered
that question from: `atom_bits`, `[BH, atoms_per_bh]` bool, the step's exact write-atom
set. A step that does grow admits from that buffer, so the set is derived ONCE, here,
instead of once here and again on the host.

**It is here, and not in `paging.py`, for one reason.** It calls `expand_side` — the same
device function `decode_expand_kernel` calls, over the same `amp_write` pointers — so the
write support it tests IS the write support the step will expand. Any other placement would
mean a second derivation of the same rule, and a residency answer derived from a support
that could drift is worse than no answer: a missed atom is a deposit written into a `slot <
0` row, which is silently dropped.

It runs BEFORE the step rather than inside it, which is the one thing the placement cannot
be talked out of. The expansion in the step's own launch is followed immediately by the
GEMV, and the GEMV is what needs the corrected page table; a flag read after that launch is
a flag read one step too late.

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
