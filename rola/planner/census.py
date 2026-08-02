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
the 2026-08-01 planner end-state ruling (``FORWARD_ROADMAP.md`` §2.7) makes the
runtime selector closed-form and RETIRES the exact census to the offline bench
(calibration, the union/collision-correction fit, and a regression-gated auditor
that the closed-form selector agrees with the exact argmin). It is wired as the
interim runtime census by :func:`rola.planner.schedule._tile_union_stats_multi`
today; it must stay relocatable tomorrow, so it does not reach back into the
planner for geometry it can derive from ``(widths, BC)`` itself.

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
   arithmetic is C-like), and ``test_the_causal_bit_identity_holds_at_bit_31``
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
    #: The SUMMARY scalars the closed-form selector of §2.7's end-state ruling will
    #: read instead of the exact census. Computed from the same words in the same
    #: pass so the follow-up batch has its inputs already measured:
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


def _token_support_words(masks: tuple[torch.Tensor, ...], BT: int,
                         n_tiles: int) -> tuple[torch.Tensor, ...]:
    """Per level, ``[BH, n_tiles, width]`` ``int32``; bit ``t`` = token ``tile*BT+t`` live.

    Liveness is EXACT nonzero-ness of the amplitude (the masks are
    ``RoutingViews.read_mask``/``write_mask``, spec §1.2) -- never a threshold. The
    padded token tail (``n_tiles*BT - T`` tokens) contributes no bit, exactly as it
    contributes none to the tile bitmap, so a ragged ``L`` needs no term of its own.

    The weights carry bit 31 as ``-2**31`` and the summed powers of two are
    DISJOINT, so every partial sum lies in ``[-2**31, 2**31 - 1]`` and the ``int32``
    accumulation is exact with no wraparound to argue about -- the same argument
    ``rola.planner.schedule._pack_token_bits`` makes for the same reason.
    """
    if not 1 <= BT <= _TOKEN_BITS_PER_WORD:
        raise ValueError(
            f"BT must be in [1, {_TOKEN_BITS_PER_WORD}] to carry one token bit each, got {BT}. "
            "The census represents a tile as ONE token word; spec §6 pins BT to "
            "_SUPPORTED_BT = (16, 32), which is what makes that sound. A BT above 32 is a "
            "spec change that must widen the word (a token-word axis), not a runtime "
            "condition to branch on.")
    BH, T, _ = masks[0].shape
    device = masks[0].device
    pad = n_tiles * BT - T
    weights = torch.ones((), dtype=torch.int64, device=device) << torch.arange(
        BT, device=device, dtype=torch.int64)
    weights = torch.where(weights >= (1 << 31), weights - (1 << 32), weights)
    weights = weights.to(torch.int32).view(1, 1, BT, 1)
    words = []
    for mask in masks:
        if pad:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
        tiled = mask.reshape(BH, n_tiles, BT, mask.shape[-1])
        words.append((tiled * weights).sum(dim=2, dtype=torch.int32))
    return tuple(words)


def _leaf_words(level_words: tuple[torch.Tensor, ...], widths: tuple[int, ...],
                lo: int, hi: int) -> torch.Tensor:
    """``[hi-lo, N]`` ``int32`` leaf words for a run of ``(bh, tile)`` rows.

    Identity 1 of the module docstring: the AND over levels is a pure BROADCAST,
    because the leaf id is the row-major flattening of the ``[w_0, ..., w_{D-1}]``
    digit grid. This is the ONE place the pass touches an ``N``-sized object, and
    it is the answer's own granularity -- the retired pass materialized the same
    plane as an ``int16`` extremes buffer, so the term is not new.
    """
    D = len(widths)
    out = None
    for level, words in enumerate(level_words):
        shape = [1] * D
        shape[level] = widths[level]
        block = words.reshape(-1, widths[level])[lo:hi].view(hi - lo, *shape)
        out = block if out is None else out & block
    return out.reshape(hi - lo, -1)


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
    (:class:`rola.planner.schedule.StaticDensity`): the declared side's words are
    never built and its ``K``-family counters are absent from the result, because
    the caller substitutes the closed form they provably equal. Both sides dense is
    refused -- there is then nothing to census at all, and calling anyway would be
    a caller bug that silently costs a launch.

    Owners are a RESHAPE of the leaf axis (``owner == leaf // BC``,
    :meth:`OwnerGeometry.leaf_base`), so there is no owner loop and no per-owner
    digit gather: ``K`` for every owner at once is one
    ``[rows, owners, BC] -> sum`` over the union bit.

    ``summaries`` controls whether the §2.7 end-state selector's SUMMARY scalars
    are computed alongside (see :class:`CensusResult`). It defaults to on because
    the offline auditor and the calibration fit both want them; the interim runtime
    caller passes ``False``, since it has nothing to read them with yet and a
    device reduction whose result is discarded is waste, not readiness.
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
    # the same masks the words are packed from, which is the input the closed-form
    # selector of §2.7's end-state ruling reads INSTEAD of this census. It is
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
        words = {side: _token_support_words(masks_by_side[side], BT, n_tiles)
                 for side in live_sides}

        acc = {name: torch.zeros((), dtype=torch.int64, device=device)
               for name in _PER_BT_COUNTERS}
        measures_k = len(live_sides) == 2
        wants_active = measures_k and summaries
        active = torch.zeros((), dtype=torch.int64, device=device)

        rows_per_chunk = max(1, _CENSUS_CELL_CHUNK // N)
        for lo in range(0, rows_total, rows_per_chunk):
            hi = min(lo + rows_per_chunk, rows_total)
            leaf = {side: _leaf_words(words[side], widths, lo, hi) for side in live_sides}
            if measures_k:
                read_lw, write_lw = leaf[0], leaf[1]
                k_count = (read_lw | write_lw).ne(0).view(hi - lo, owners_total, BC).sum(dim=-1)
                acc["sum_K"] += k_count.sum()
                acc["sum_sat"] += (k_count == BC).sum()
                # The QUANTUM, summed BEFORE any mean: the kernel moves
                # `ceil(K / MMA_K_QUANTUM)` k-steps, which is 0 at K = 0 and 1 for
                # every K in [1, 16], and `ceil(mean) != mean(ceil)` -- the
                # difference IS the compact saving (`TileUnionStats`' note).
                acc["sum_ks"] += ((k_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()
                if wants_active:
                    active = active + (k_count > 0).sum()
                # Identity 2: the causal-intra bit, with no prefix scan and no
                # extremes buffer. See the module docstring for the derivation.
                ki_bit = (read_lw & -(write_lw & -write_lw)).ne(0)
            elif dense_side[0]:
                # READ dense: `last_read` is the tile's own last token, so the
                # causal bit degenerates to "written somewhere in this tile".
                ki_bit = leaf[1].ne(0)
            else:
                # WRITE dense: `first_write == 0`, so it degenerates to
                # "read somewhere in this tile" -- symmetric.
                ki_bit = leaf[0].ne(0)
            ki_count = ki_bit.view(hi - lo, owners_total, BC).sum(dim=-1)
            acc["sum_KI"] += ki_count.sum()
            acc["sum_kis"] += ((ki_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()

        for name in _PER_BT_COUNTERS:
            if not measures_k and name in ("sum_K", "sum_sat", "sum_ks"):
                continue
            per_bt[(BT, name)] = acc[name]
        # SUMMARY (b): the per-owner active fraction at this `BT`.
        if wants_active:
            summary[("active_owners", BT)] = active
            denominators[("active_owners", BT)] = rows_total * owners_total

    return CensusResult(per_bt=per_bt, summaries=summary, denominators=denominators)


def summary_fractions(result: CensusResult, resolved: dict) -> dict[tuple, float]:
    """The §2.7 closed-form selector's inputs, as fractions in ``[0, 1]``.

    ``resolved`` is the output of :func:`resolve_census` over (at least)
    ``result.summaries``. Kept separate from the census itself for the same reason
    the transfer is: the counters are exact integers on the device and the division
    is a host-side presentation step, so nothing here forces a synchronization of
    its own.
    """
    return {key: resolved[key] / result.denominators[key] for key in result.summaries}
