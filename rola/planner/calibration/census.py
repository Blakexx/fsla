"""The EXACT tile-union census, as packed-bit-word arithmetic.

``ROLA_V3_SPEC.md`` §3 step 2's gather is priced by §6 from one object: per
``(owner, tile)``, the size ``K`` of the UNION of the read and write supports over
the tile's tokens (``GRADIENT_PROOFS_WOLFRAM.md`` P5.21 item 1 -- the backward
iterates ``support(W) UNION support(R)``, so a write-only count would price a
gather the kernel is forbidden to build), together with §5's causal-intra count
``KI`` and the MMA quanta of both.

**This module is the EXACT census and nothing else.** It is deliberately free of
any dependency on :mod:`rola.planner.schedule` -- it takes the folded per-level
masks, the widths and ``BC``, and returns integer counters. That is not stylistic:
``BC`` is DERIVED from launch-static facts (:mod:`rola.planner.derive`), so the exact
census is a BENCH instrument -- the collision-correction fit and the one-time
validation of the derivation against measured argmins -- and an instrument must not
reach back into the planner for geometry it can derive from ``(widths, BC)`` itself.

WHAT REPLACED WHAT (fix brief #2, 2026-08-01). The previous implementation
expanded the live leaf set of every ``(bh, token)`` pair -- a mixed-radix decode of
a flat offset whose radices are the per-level support COUNTS -- and
``scatter_reduce_``'d two token extremes per ``(finest tile, leaf)`` cell. It was
``O(support)`` in the expansion and ``O(BH * n_tiles * N)`` in the reduction, and
it paid for the ragged expansion in host synchronizations: one ``nonzero()`` per
level per side, one ``.cpu()`` for the chunk split and one ``int(m.sum())`` per
chunk per side (measured 198 syncs / 3,557 launches per call at ``L = 16384``,
`planner-glue-attribution.md` Task A, where the decode and its
``repeat_interleave`` were 51 % + 15 % of self time). This module keeps the second
term, deletes the first outright, and is device-resident end to end: it issues no
host synchronization at all, and the ONE transfer of the whole call lives in
:func:`resolve_census`.

THE REPRESENTATION -- a TOKEN bitset, the same currency
``rola.planner.schedule._pack_token_bits`` hands the consumer. For one tile of
``BT <= 32`` tokens, one ``(side, level, digit)`` cell is a single ``int32`` whose
bit ``t`` is "digit ``d`` of level ``l`` is live at token ``tile*BT + t``". Two
identities make the whole census fall out of it:

1. **The leaf axis is a BROADCAST, not a gather.** ``leaf = sum_l d_l *
   prod(widths[l+1:])`` is exactly the row-major flattening of a
   ``[w_0, ..., w_{D-1}]`` grid, so the per-leaf word
   ``lw[i] = AND_l tw[l, digit_l(i)]`` is the AND of the ``D`` per-level word
   tensors each viewed with its width on its own axis. No index is ever
   materialized. Both roles §3 gives the levels are covered by the same AND: the
   SPANNED levels (``l >= j``) distinguish leafs, and the FIXED levels (``l < j``)
   cannot, but a zero there kills the token's row scale, so they shrink every
   leaf's word identically (:func:`rola.planner.schedule.tile_union_stats`).
2. **Both counts are bit tests on that word.** ``K`` counts leafs with
   ``(lw_r | lw_w) != 0``, and §5's causal-intra bit ``first_write <= last_read``
   is::

       ki_bit = ( lw_r & -( lw_w & -lw_w ) ) != 0

   ``lw_w & -lw_w`` isolates the LOWEST set bit of the write word, i.e.
   ``2^first_write``; negating a power of two in two's complement gives
   ``~(2^f - 1)``, the mask of every bit position ``>= f``; ANDing it with the read
   word asks "is some read at a token index at or after the first write", which is
   the causal condition verbatim. The no-write case needs no branch (``-0 == 0``
   masks nothing, so the bit is false), and ``BT = 32`` needs no UNSIGNED
   comparison -- which is why this form is used rather than the obvious
   ``lowbit(w) <= r``, whose comparison would be signed on torch's ``int32``. The
   negation relies on two's-complement wraparound at ``2^31`` (torch integer
   arithmetic is C-like), and ``test_the_causal_intra_bit_is_a_bit_test_that_survives_bit_31``
   pins exactly that word.

WHY THERE IS NO ``BT`` FOLD ANY MORE. The previous pass measured at the finest
candidate and folded coarser ones by ``min``/``max`` on the extremes; the
enumerated reference folds bitmaps and carries the cross-boundary
``Rany_B AND Wany_A`` term. Neither is needed here: ``BT`` is pinned to
``_SUPPORTED_BT = (16, 32)`` one layer up, every candidate therefore fits in ONE
32-bit token word, and the words are rebuilt directly at each candidate from the
masks -- which is the DEFINITION, so exactness is by construction rather than by a
fold identity. The rebuild is ``O(BH * T * sum(widths))``, i.e. negligible beside
the ``O(BH * n_tiles * N)`` leaf pass it feeds. ``BT > 32`` is refused loudly (see
:func:`_token_support_words`) rather than silently taking a second code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from rola.routing.exact_summaries import _leaf_words, _token_support_words

#: Token bits carried by one support word. A candidate ``BT`` must fit one word;
#: ``_SUPPORTED_BT = (16, 32)`` in ``rola.planner.schedule`` is what guarantees it,
#: exactly as it was what guaranteed the retired pass's ``int16`` token offsets.
_TOKEN_BITS_PER_WORD = 32

#: Element budget for one chunk of the leaf-word plane, in ``int32`` leaf-words per
#: side. The pass's ONE transient is the ``[rows, N]`` leaf-word tensor (``rows`` a
#: run of ``(bh, tile)`` pairs), so chunking over rows is what keeps the footprint
#: independent of ``N * BH * n_tiles`` -- the ``O(L*N)`` object §4 forbids
#: materializing. A STRUCTURAL bound under the standing "design for nc ~ 65K" rule,
#: not a tuning knob: at ``widths=(256,256)``, ``BH=8``, ``L=16384`` the full plane
#: is 537 M words and this admits it in 32 chunks.
_CENSUS_CELL_CHUNK = 1 << 24

#: The counter names the census produces per candidate ``BT``, in the ONE order the
#: single host transfer is packed in. ``sum_K``/``sum_sat``/``sum_ks`` are absent
#: from the result when either side is dense by configuration -- the caller
#: substitutes the closed form it provably equals, and no reduction is issued.
_PER_BT_COUNTERS = ("sum_K", "sum_sat", "sum_ks", "sum_KI", "sum_kis")

#: §6's MMA k-extent. Mirrored from ``rola.planner.schedule.MMA_K_QUANTUM`` rather
#: than imported, because importing it would couple this module to the planner it
#: is scheduled to leave; :func:`rola.planner.schedule._tile_union_stats_multi`
#: asserts the two agree, so a drift is a test failure and not a silent one.
MMA_K_QUANTUM = 16


@dataclass(frozen=True, slots=True)
class CensusResult:
    """Device-resident integer counters, plus the host-side denominators.

    Everything is a 0-dim ``int64`` CUDA/CPU tensor and NOTHING has been read back:
    the caller adds whatever counters of its own it has (the planner adds §4's
    candidacy popcount) and resolves the whole set in ONE transfer via
    :func:`resolve_census`. That is the entire synchronization budget of the pass.
    """

    #: ``{(BT, counter_name): 0-dim int64}`` for the names in ``_PER_BT_COUNTERS``.
    per_bt: dict[tuple[int, str], torch.Tensor]
    #: The SUMMARY scalars -- the cheap statistics of a realization, computed from the
    #: same words in the same pass as the exact counters so the two can be compared
    #: without a second walk:
    #:
    #: * ``("occupancy", side, level)`` -- live ``(token, digit)`` pairs of that
    #:   ``(side, level)`` cell. Divided by ``denominators[("occupancy", side,
    #:   level)] = BH * T * width_l`` it is the per-level support OCCUPANCY, i.e.
    #:   the per-token expected fraction of that level's digits that are live.
    #: * ``("active_owners", BT)`` -- ``(bh, tile, owner)`` triples with ``K > 0``.
    #:   Divided by ``denominators[("active_owners", BT)] = BH * n_tiles *
    #:   owners_total`` it is the per-owner ACTIVE FRACTION at that ``BT``. Absent
    #:   when a side is dense by configuration, where it is identically ``1.0``.
    summaries: dict[tuple, torch.Tensor]
    #: Host-side denominators for ``summaries``; see above.
    denominators: dict[tuple, int]


def resolve_census(counters: dict, extra: dict | None = None) -> dict:
    """Read a whole census back in ONE host transfer, in a deterministic order.

    ``counters`` maps any hashable key to a 0-dim integer tensor; ``extra`` is
    merged in first so a caller's own accumulators (the planner's candidacy
    popcount) ride the same transfer rather than adding a second synchronization.
    Keys are ordered by ``repr`` so the packing is stable across calls and across
    Python's dict insertion order -- the transfer is one ``torch.stack`` and one
    ``.cpu()``, and an unstable order would make the pass's launch sequence
    unstable for no reason.
    """
    merged = dict(counters)
    if extra:
        merged.update(extra)
    if not merged:
        return {}
    keys = sorted(merged, key=repr)
    flat = torch.stack([merged[key].reshape(()) for key in keys]).to(torch.int64).cpu().tolist()
    return dict(zip(keys, flat))


def _census_chunk(read_words: tuple[torch.Tensor, ...] | None,
                  write_words: tuple[torch.Tensor, ...] | None,
                  widths: tuple[int, ...], owners_total: int, BC: int,
                  ) -> tuple[torch.Tensor, ...]:
    """One chunk's six counters, as ONE pure function of the chunk's token words.

    Returns ``(sum_K, sum_sat, sum_ks, sum_KI, sum_kis, active)``, every one a 0-dim
    ``int64``. ``read_words``/``write_words`` are ``[rows, width]`` per level, or
    ``None`` for a side declared dense by configuration -- in which case the
    ``K``-family entries come back as exact zeros, and the CALLER drops them (it
    substitutes the closed form they provably equal) rather than reading a
    fabricated number.

    **This is the whole numerical content of the census, and it is deliberately one
    side-effect-free function of tensors.** Everything above it is bookkeeping
    (packing, chunking, key naming). Two consequences that are the point rather than
    a coincidence: the §2.7 relocation to the offline bench moves this function
    unchanged, and the ~13x whole-pass fusion lever measured in
    `planner-bitword-census.md` §4 is a wrapper on exactly this signature -- the
    ``[rows, N]`` leaf plane it forms is an INTERMEDIATE of a reduction whose inputs
    are ``O(rows * sum(widths))`` and whose outputs are scalars, so nothing about the
    answer requires that plane to reach DRAM.
    """
    zero = torch.zeros((), dtype=torch.int64,
                       device=(read_words or write_words)[0].device)
    sum_K = sum_sat = sum_ks = active = zero
    if read_words is not None and write_words is not None:
        read_lw = _leaf_words(read_words, widths)
        write_lw = _leaf_words(write_words, widths)
        k_count = (read_lw | write_lw).ne(0).view(-1, owners_total, BC).sum(dim=-1)
        sum_K = k_count.sum()
        sum_sat = (k_count == BC).sum()
        # The QUANTUM, summed BEFORE any mean: the kernel moves
        # `ceil(K / MMA_K_QUANTUM)` k-steps, which is 0 at K = 0 and 1 for every K in
        # [1, 16], and `ceil(mean) != mean(ceil)` (`TileUnionStats`' note).
        sum_ks = ((k_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()
        # SUMMARY (b) of `CensusResult`: the per-owner ACTIVE fraction's numerator.
        # Computed unconditionally because it is one comparison-reduction over an
        # array that is already live and already reduced three other ways; making it
        # optional would buy nothing and would fork the graph this function is.
        active = (k_count > 0).sum()
        # Identity 2: the causal-intra bit, with no prefix scan and no extremes
        # buffer. See the module docstring for the derivation.
        ki_bit = (read_lw & -(write_lw & -write_lw)).ne(0)
    elif read_words is None:
        # WRITE side live, READ side dense: `last_read` is the tile's own last token,
        # so the causal bit degenerates to "written somewhere in this tile".
        ki_bit = _leaf_words(write_words, widths).ne(0)
    else:
        # READ side live, WRITE side dense: `first_write == 0`, so it degenerates to
        # "read somewhere in this tile" -- symmetric.
        ki_bit = _leaf_words(read_words, widths).ne(0)
    ki_count = ki_bit.view(-1, owners_total, BC).sum(dim=-1)
    sum_KI = ki_count.sum()
    sum_kis = ((ki_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()
    return sum_K, sum_sat, sum_ks, sum_KI, sum_kis, active


#: THE SHIPPED CALL. :func:`_census_chunk` is a pure function whose inputs are
#: ``O(rows * sum(widths))`` words and whose outputs are six scalars; the ``[rows, N]``
#: leaf plane between them is an INTERMEDIATE of a reduction and nothing about the
#: answer requires it to reach DRAM. Eager torch has to write it and re-read it once
#: per elementwise step, which made the pass purely memory-bound -- 112 ms of device
#: time at the at-scale cell, 30 % of it in ``bitwise_and`` alone. Fusing it is
#: therefore not a tuning knob but the removal of the pass's only remaining term, and
#: it is measured at **6-9x on whole-pass wall, bit-identical on every
#: field of every cell class** (`planner-bitword-census.md` §4).
#:
#: ``dynamic=True`` from the first call, so the row extent -- the one axis that varies
#: within a run (the last chunk is short, and ``L`` moves between calls) -- never
#: triggers a recompile; the graph specializes on ``(D, widths, owners_total, BC,
#: device, which sides are live)``, which is a per-CONFIGURATION set and therefore
#: bounded. Measured cost of that specialization: +25 s on the whole
#: ``tests/unit/test_planner.py`` run (1032 tests, 24 s -> 49 s), which is the price
#: of compiling every toy geometry the gates sweep, not a per-call cost.
#:
#: **This makes `torch.compile`/inductor a RUNTIME requirement of the planner**, on
#: CPU as well as CUDA. That is a real new dependency edge and it is stated here
#: rather than discovered: there is no eager path kept alongside, by the no-fallback
#: rule, so an environment without a working inductor backend fails loudly here.
_census_chunk_fused = torch.compile(_census_chunk, dynamic=True)


def tile_union_census(read_mask: tuple[torch.Tensor, ...],
                      write_mask: tuple[torch.Tensor, ...],
                      widths: tuple[int, ...], BC: int,
                      bt_candidates: tuple[int, ...], *,
                      dense_side: tuple[bool, bool] = (False, False),
                      summaries: bool = True) -> CensusResult:
    """The exact per-``(owner, tile)`` census, as word arithmetic. See the module docstring.

    ``read_mask``/``write_mask`` are the folded ``[BH, T, width]`` boolean supports,
    one per level (``RoutingViews.read_mask``/``write_mask``). ``dense_side`` is the
    caller's DECLARATION that a side is full-support by configuration
    (:class:`rola.routing.density.StaticDensity`): the declared side's words are
    never built and its ``K``-family counters are absent from the result, because
    the caller substitutes the closed form they provably equal. Both sides dense is
    refused -- there is then nothing to census at all, and calling anyway would be
    a caller bug that silently costs a launch.

    Owners are a RESHAPE of the leaf axis (``owner == leaf // BC``,
    :meth:`OwnerGeometry.leaf_base`), so there is no owner loop and no per-owner
    digit gather: ``K`` for every owner at once is one
    ``[rows, owners, BC] -> sum`` over the union bit.

    ``summaries`` controls whether the SUMMARY scalars are PUBLISHED (see
    :class:`CensusResult`). It defaults to on because the collision-correction fit
    wants them; a caller that only needs the exact counters passes
    ``False``, which drops the per-level occupancy reductions -- the only part with a
    cost of its own, measured at 0.2 ms of the at-scale cell's pass. The per-owner
    active count is NOT conditional: it is one more comparison-reduction over an
    array :func:`_census_chunk` has already reduced three other ways, so making it
    optional would buy nothing and would fork that function's graph.
    """
    if len(read_mask) != len(widths) or len(write_mask) != len(widths):
        raise ValueError(f"one mask per level per side required (D={len(widths)})")
    if all(dense_side):
        raise ValueError(
            "tile_union_census called with BOTH sides declared dense: every statistic is "
            "then the closed form in (BC, BT, T, BH, owners_total) and nothing is measured, "
            "so the caller must not issue this pass at all.")
    N = prod(widths)
    if N % BC:
        raise ValueError(f"BC={BC} must divide N={N} (owners partition the leaf space)")
    owners_total = N // BC
    BH, T, _ = read_mask[0].shape
    device = read_mask[0].device
    live_sides = tuple(side for side in (0, 1) if not dense_side[side])
    masks_by_side = (read_mask, write_mask)

    per_bt: dict[tuple[int, str], torch.Tensor] = {}
    summary: dict[tuple, torch.Tensor] = {}
    denominators: dict[tuple, int] = {}

    # SUMMARY (a): per-level support occupancy per side -- a single reduction over
    # the same masks the words are packed from. It is
    # measured on the live sides only; a side declared dense has occupancy 1 by the
    # declaration, and measuring it would contradict the declaration rather than
    # check it (the check belongs to the gate, not to the hot path).
    if summaries:
        for side in live_sides:
            for level, mask in enumerate(masks_by_side[side]):
                summary[("occupancy", side, level)] = mask.sum(dtype=torch.int64)
                denominators[("occupancy", side, level)] = BH * T * widths[level]

    for BT in bt_candidates:
        n_tiles = (T + BT - 1) // BT
        rows_total = BH * n_tiles
        words = {side: tuple(w.reshape(rows_total, -1)
                             for w in _token_support_words(masks_by_side[side], BT, n_tiles))
                 for side in live_sides}
        measures_k = len(live_sides) == 2

        acc = [torch.zeros((), dtype=torch.int64, device=device) for _ in range(6)]
        rows_per_chunk = max(1, _CENSUS_CELL_CHUNK // N)
        for lo in range(0, rows_total, rows_per_chunk):
            hi = min(lo + rows_per_chunk, rows_total)
            chunk = tuple(None if dense_side[side]
                          else tuple(w[lo:hi] for w in words[side])
                          for side in (0, 1))
            for slot, value in enumerate(_census_chunk_fused(*chunk, widths, owners_total, BC)):
                acc[slot] = acc[slot] + value

        for slot, name in enumerate(_PER_BT_COUNTERS):
            if not measures_k and name in ("sum_K", "sum_sat", "sum_ks"):
                continue                       # the caller substitutes the closed form
            per_bt[(BT, name)] = acc[slot]
        # SUMMARY (b): the per-owner active fraction at this `BT`.
        if measures_k and summaries:
            summary[("active_owners", BT)] = acc[5]
            denominators[("active_owners", BT)] = rows_total * owners_total

    return CensusResult(per_bt=per_bt, summaries=summary, denominators=denominators)


def summary_fractions(result: CensusResult, resolved: dict) -> dict[tuple, float]:
    """The summary counters, as fractions in ``[0, 1]``.

    ``resolved`` is the output of :func:`resolve_census` over (at least)
    ``result.summaries``. Kept separate from the census itself for the same reason
    the transfer is: the counters are exact integers on the device and the division
    is a host-side presentation step, so nothing here forces a synchronization of
    its own.
    """
    return {key: resolved[key] / result.denominators[key] for key in result.summaries}
