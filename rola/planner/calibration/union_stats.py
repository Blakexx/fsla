# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE EXACT TILE-UNION STATISTICS -- what a routing realization actually gathers.

For one realized routing and one ``BC``, this module answers the question the schedule
alone cannot: over every ``(owner, token-group)`` pair, how many leaves does the union
of that group's tokens touch, how many of those pairs are CANDIDATES at all, and how
many MMA k-quanta the gathered extent costs. It is EXACT -- an enumeration over the
realized masks, not a model of them -- and it is a MEASUREMENT INSTRUMENT: nothing in
the dispatch path calls it, because ``BC`` is derived from launch-static facts
(:mod:`rola.planner.derive`) and no runtime rule reads a routing statistic.

What calls it: the bench cells, which report these numbers beside a measured time, and
the one-time validation of the derivation's ceilings against measured argmins.

**IT LIVES OUTSIDE THE SHIPPED IMPORT GRAPH ON PURPOSE.**
:mod:`rola.planner.calibration.census` makes ``torch.compile``/inductor a dependency,
on CPU as well as CUDA, with no eager path beside it. Nothing under ``rola/`` imports
this package at module scope, so ``import rola`` never reaches inductor.

Symbols: ``L``/``T`` tokens; ``N`` leaves; ``D`` levels; ``BC`` leaves per owner;
``BT`` the statistics granularity (the group quantum); ``BH`` = batch x heads;
``MMA_K_QUANTUM`` the MMA k-extent; ``K`` the gathered leaf extent of one pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from rola.planner.calibration.census import resolve_census, tile_union_census
from rola.planner.schedule import (
    MMA_K_QUANTUM,
    RoutingViews,
    owner_geometry,
    routing_views,
)
from rola.routing.density import StaticDensity
from rola.routing.types import Topology

#: The ONE granularity these statistics are taken at: the group quantum
#: (MMA_K_QUANTUM gathered tokens). Aligned-16-tile statistics are the group-shaped
#: over-approximation of member statistics -- a measurement, never able to mis-execute
#: a plan.
_SUPPORTED_BT = (16,)



@dataclass(frozen=True, slots=True)
class TileUnionStats:
    """Exact per-``(owner, tile)`` statistics of §3 step 2's gather.

    Every field is computed from the REALIZED routing tensors over the realized
    schedule -- not from a distributional model, and not from the per-level
    marginals :class:`LevelSupportStats` carries. That distinction is the whole
    point: ``K`` is a per-TILE UNION, and a union over ``BT`` tokens saturates far
    faster than any per-token marginal predicts.

    ``mean_K``/``saturated_frac`` are over ALL ``(owner, tile)`` pairs;
    ``mean_K_candidate``/``mean_KI_candidate`` are over the pairs the kernel
    actually walks (the §4 candidate bitmap), which is the population that pays
    the gather. They differ: a tile can be a candidate on the AND of per-level
    TILE bitmaps and still have ``K == 0``, because tile liveness is an OR over
    the tile's tokens while the leaf test is an AND at a common token.

    **``mean_ksteps_candidate``/``mean_kisteps_candidate`` are the QUANTA, and they
    are what §6 prices.** The kernel does not move ``K`` leafs, it moves
    ``ceil(K / MMA_K_QUANTUM)`` MMA k-steps: a candidate tile with ``K = 1`` costs
    the same k-step as one with ``K = 16``, and a candidate tile with ``K = 0``
    costs NONE. ``mean_K_candidate`` cannot express either end. Measured at
    ``atscale-L2048``: ``mean_K_candidate / BC = 0.9724`` against
    ``mean_ksteps_candidate / (BC/16) = 0.606`` -- 39 % of that cell's candidate
    pairs realize ``K = 0``, which ``mean_K_candidate`` cannot express at either
    end. Both are means over the CANDIDATE population, like the two means
    above them, so a term may mix them freely.

    ``n_cohorts`` is ``owners_total * BH`` -- the CTA grid the launch requests, one
    CTA per ``(bh, owner)``. It is an UPPER BOUND on the live grid rather than the
    grid: an owner with no candidate tile at all still gets a CTA that exits early.
    Measured tightness (§2.32): ``atscale-L2048`` has 8180 live owners of 8192
    (99.9 %) and every ``topo`` cell is exact, so no separate liveness pass is
    warranted -- but it is a bound and §6's wave term states it as one.
    """

    BC: int
    BT: int
    mean_K: float
    saturated_frac: float
    mean_K_candidate: float
    mean_KI_candidate: float
    #: Mean ``ceil(K / MMA_K_QUANTUM)`` over CANDIDATE pairs -- the readout's and the
    #: fold's k-step count, which is what the kernel pays.
    mean_ksteps_candidate: float
    #: Mean ``ceil(KI / MMA_K_QUANTUM)`` over CANDIDATE pairs -- the intra Gram's.
    mean_kisteps_candidate: float
    candidate_frac: float
    n_pairs: int
    #: ``owners_total * BH``: the requested CTA grid, an upper bound on the live one.
    n_cohorts: int


def tile_union_stats(read_levels: tuple[torch.Tensor, ...],
                     write_levels: tuple[torch.Tensor, ...],
                     topology: Topology, BC: int, BT: int, *,
                     views: RoutingViews | None = None,
                     static_density: StaticDensity | None = None) -> TileUnionStats:
    """Compute :class:`TileUnionStats` exactly, for one ``(BC, BT)`` candidate.

    ``K(owner, tile) = |{ i in [0,BC) : exists t in the tile, R[t, leaf(i)] != 0
    or W[t, leaf(i)] != 0 }|``, over the UNION of the two supports
    (``GRADIENT_PROOFS_WOLFRAM.md`` P5.21 item 1 -- the backward iterates
    ``support(W) UNION support(R)``, so a write-only count would price a gather
    the kernel is forbidden to build).

    **All ``D`` levels enter, in the two different roles §3 gives them.** The
    SPANNED levels (``l >= j``) distinguish leafs of one owner, so they are ANDed
    per leaf. The FIXED levels (``l < j``) cannot distinguish two leafs -- but a
    zero there kills the whole token's row scale (§3 step 3), so they AND into
    TOKEN VALIDITY, which shrinks every leaf's support word identically. Omitting
    them is not a conservative approximation; it is a different and strictly
    weaker gather, and it is what the kernel and every instrument did before
    2026-07-30: at ``branches=[8,16], BC=16`` (``j = 1``) it made the whole
    width-8 level invisible and reported the flagship entmax-gain-8 cell as
    ``K = 15.91/16, sat 0.908`` against the exact ``11.13/16, sat 0.017``.

    The side gains are excluded, exactly as the kernel excludes them: ``g_write``
    scales the deposit but NOT the clock (:func:`build_clock_stream`), so folding
    it in would drop leafs whose survival still evolves.

    ``KI`` is §5's causal intra sub-list length -- a leaf is included iff it is
    read at or after its first in-tile write -- which is what prices the
    ``realized_leaf`` intra term (§6).

    This is a MEASUREMENT function: no kernel path and no dispatch rule reads its
    results.

    ``views`` is an optional :class:`RoutingViews` carrying the folded tensors and
    the memoized per-``BT`` tile bitmaps, so a caller measuring and scheduling the
    same realization walks the routing once. Passing
    it changes NOTHING about what is computed -- it is the same reduction over the
    same tensors -- only how many times the shared prefix is recomputed.

    ``static_density`` declares which ``(level, side)`` cells are full-support BY
    CONFIGURATION (:class:`StaticDensity`). It changes NOTHING about what is
    computed either -- a dense side's statistics are substituted by the closed form
    they provably equal -- only how much of the data-dependent pass has to run to
    reach them.
    """
    if views is None:
        views = routing_views(read_levels, write_levels, topology)
    else:
        views.require_source(read_levels, write_levels)
    return _tile_union_stats_multi(views, topology, BC, (BT,), static_density)[BT]


def _bt_candidate_chain(bt_candidates: tuple[int, ...]) -> tuple[int, ...]:
    """Sorted unique candidates, validated as a power-of-two chain above the finest.

    The ENUMERATED reference requires it structurally: it measures once at the finest
    candidate and OR-pairs adjacent tiles, which only lands on the right tile count
    when each candidate is ``2^k`` times the finest. The bit-word production pass no
    longer needs the chain at all -- it re-packs the tokens at each candidate, which
    is the definition -- but the constraint is kept and validated in ONE place so the
    two implementations remain comparable candidate for candidate, which is what the
    equality gate compares.

    **Recorded, not enforced here (delta review R1, restated for fix brief #2):** the
    bit-word pass carries a tile's tokens as the BITS of one ``int32``, so it
    represents ``BT`` up to 32. Nothing in this function bounds ``BT``; what bounds it
    is ``_SUPPORTED_BT``, and
    :func:`rola.planner.calibration.census._token_support_words` refuses anything wider
    outright. The guarantee is therefore real but lives elsewhere, and a granularity
    outside that set must widen the word rather than become a runtime condition to
    branch on.
    """
    cands = tuple(sorted({int(BT) for BT in bt_candidates}))
    if not cands:
        raise ValueError("bt_candidates must be non-empty")
    BT0 = cands[0]
    for BT in cands:
        ratio, rem = divmod(BT, BT0)
        if rem or (ratio & (ratio - 1)):
            raise ValueError(
                f"BT candidates must be a power-of-two chain above the finest ({BT0}); "
                f"got {cands}. The enumerated reference's fold OR-pairs adjacent tiles, so a "
                "candidate that is not 2^k times the finest has no exact fold there and could "
                "no longer be compared against the production pass candidate for candidate.")
    return cands


def _tile_union_stats_multi(views: RoutingViews, topology: Topology, BC: int,
                            bt_candidates: tuple[int, ...],
                            static_density: StaticDensity | None = None,
                            ) -> dict[int, TileUnionStats]:
    """:class:`TileUnionStats` for EVERY candidate ``BT``, from PACKED BIT WORDS.

    Bitwise-identical output to :func:`_tile_union_stats_multi_enumerated` (gated on
    the full planner grid), at ``O(BH * n_tiles * N)`` instead of
    ``Theta(BH * T * N)``: the reference asks of every ``(token, owner, leaf)`` cell
    whether it is live, and this asks it of every ``(tile, owner, leaf)`` cell with
    the tile's ``BT`` tokens carried as BITS of one ``int32``. Three derivations make
    that possible; :mod:`rola.planner.census` owns the first two and this function
    owns the third.

    **1. Owners are contiguous leaf blocks, and ``owner == leaf // BC``.** By
    :meth:`OwnerGeometry.leaf_base` the owner's leaf base is exactly ``owner * BC``
    and its ``BC`` leafs are the next ``BC`` ids, so the owner axis is a RESHAPE of
    the leaf axis -- no owner loop, no per-owner digit gather, and ``K`` for every
    owner at once is one ``[..., owners, BC] -> sum`` over the union bit.

    **2. One token word per ``(tile, level, digit)`` carries both counts.** Bit ``t``
    of ``tw[l, d]`` is "digit ``d`` of level ``l`` is live at token ``tile*BT + t``";
    the leaf word is the AND over levels (a BROADCAST, because the leaf id is the
    row-major flattening of the digit grid), the union bit is ``(lw_r | lw_w) != 0``
    and §5's causal-intra bit is ``(lw_r & -(lw_w & -lw_w)) != 0``. See
    :mod:`rola.planner.census` for the derivation of that last expression and for why
    it needs neither a prefix scan nor an unsigned comparison. Because a candidate
    ``BT`` (``_SUPPORTED_BT``, at most 32) fits ONE word, each candidate's words are
    rebuilt from the masks directly -- which is the DEFINITION of the statistic, so
    no fold identity and no cross-boundary carry term is involved at all.

    **3. Candidacy is a PRODUCT SET, so its count is closed-form.** A candidate owner
    at a tile is one whose boundary slice is tile-live AND whose every fixed digit is
    tile-live (§4), i.e. the candidate set on one side is
    ``{live slices} x prod_{l<j} {live digits of level l}``, whose size is the
    product of the §4 TILE BITMAPS' popcounts. ``cand = cand_r OR cand_w`` is a union
    of two product sets and the intersection of two product sets is the product of
    the per-axis intersections, so inclusion-exclusion gives ``|A| + |B| - |A and B|``
    exactly, from popcounts alone -- ``O(BH * n_tiles * sum(widths))``, independent of
    ``N`` and of the support. This one stays HERE rather than moving to the census,
    because it reads :meth:`RoutingViews.tile_live`, which the schedule builder shares.

    **``sum_K_candidate == sum_K`` and ``sum_KI_candidate == sum_KI``, proved not
    assumed.** If ``K(owner, tile) > 0`` then some leaf of that owner is live on some
    side at some token of the tile, so EVERY level's digit for that leaf is live at
    that token, so every level's tile bitmap bit is set -- the boundary level's
    through its slice OR -- and the owner is a candidate on that side. Every pair with
    ``K > 0`` is therefore a candidate, and restricting the sum to candidates removes
    only zero terms; ``KI > 0`` implies ``K > 0`` a fortiori. The equal-to-the-
    reference gate exercises this on the sparse and zero-level scenarios, where the
    two populations genuinely differ.

    **HOST SYNCHRONIZATIONS: exactly one, and it is the read-back.** The retired
    ``O(support)`` scatter implementation (fix brief #2 deleted it) had to know the
    length of a ragged expansion, which cost one ``nonzero()`` per level per side, one
    ``.cpu()`` for the chunk split and one ``int(m.sum())`` per chunk per side -- 198
    syncs and 3,557 launches per call at ``L = 16384``
    (`planner-glue-attribution.md` Task A). The word pass is device-resident end to
    end, and every accumulator -- the census's and this function's candidacy popcount
    alike -- rides ONE :func:`~rola.planner.census.resolve_census` transfer.

    **What is not counted at all: a side that is dense BY CONFIGURATION**
    (``static_density``, §2.7's ruling). Its statistics are a closed form in
    ``(BC, BT, T, BH, owners_total)`` alone, derived below, and the word pass runs
    only for the genuinely-routed side. Passing no ``static_density`` measures both.

    **THE CLOSED FORM, derived (fix brief #1, 2026-08-01).** Let one side be dense,
    i.e. every level's digit live at every token. Then for every ``(owner, tile)``
    pair whose tile holds at least one token -- and there are exactly
    ``ceil(T / BT)`` such tiles per ``bh``, which is ``n_tiles`` -- every one of the
    owner's ``BC`` leafs is live on that side at every token of the tile. Hence:

    * ``K = BC`` on every pair, so ``sum_K = n_pairs * BC``, ``mean_K = BC`` and
      ``saturated_frac = 1``. (The word pass tiles exactly ``ceil(T/BT)`` tiles, so
      unlike the retired one it carries no padded tiles that have to cancel.)
    * ``ksteps = ceil(BC / MMA_K_QUANTUM)`` on every pair, by the same argument
      applied to the quantum (``sum_ks`` is summed BEFORE the mean, as always).
    * ``candidate_frac = 1``: derivation 3's ``p`` for the dense side is
      ``n_slices * prod_{l<j} w_l``, which is ``owners_total`` identically, and
      ``cand = p_read + p_write - p_both`` with ``p_both = p_sparse <= p_dense``
      collapses to ``owners_total``. The tile bitmaps are therefore not needed at
      all, which is why the candidacy pass below is skipped whole rather than
      simplified.
    * ``KI`` STAYS DATA-DEPENDENT, and it is the only thing that does. With the READ
      side dense the causal bit degenerates to "the leaf is written somewhere in this
      tile" (``lw_w != 0``) and with the WRITE side dense to "read somewhere in this
      tile" (``lw_r != 0``); with BOTH dense it is constant ``True`` and ``KI = BC``,
      so the census is not issued at all and the whole pass is arithmetic.
    """
    cands = _bt_candidate_chain(bt_candidates)
    geometry = owner_geometry(topology.widths, BC)
    j = geometry.first_spanned_level
    c = geometry.slice_digits
    n_slices = geometry.n_slices
    owners_total = geometry.owners_total
    BH, T = views.BH, views.T

    # THE STATIC/DYNAMIC SPLIT (fix brief #1). `dense_side[side]` is a statement
    # about the CONFIGURATION -- see the closed form in the docstring -- and every
    # skip below is keyed on it alone, never on a width, a density measurement or a
    # size threshold. With no declaration both sides are live and the census
    # measures everything.
    dense_side = (False, False) if static_density is None else (
        static_density.side_dense("read"), static_density.side_dense("write"))

    #: Totals that are arithmetic in the geometry, filled only where a side is
    #: dense by construction; they REPLACE the corresponding accumulator rather
    #: than being checked against it, so no reduction is issued for them at all.
    static_totals: dict[tuple[int, str], int] = {}
    #: Device accumulators; resolved in ONE transfer together with the census's.
    counters: dict[tuple[int, str], torch.Tensor] = {}

    n_tiles_at = {BT: (T + BT - 1) // BT for BT in cands}
    if any(dense_side):
        # Closed form, per the docstring's derivation. `ceil(BC / MMA_K_QUANTUM)`
        # is the per-pair k-step count at `K = BC`, summed (never averaged) first.
        bc_ksteps = -(-BC // MMA_K_QUANTUM)
        for BT in cands:
            n_pairs = owners_total * BH * n_tiles_at[BT]
            static_totals[(BT, "sum_K")] = n_pairs * BC
            static_totals[(BT, "sum_sat")] = n_pairs
            static_totals[(BT, "sum_cand")] = n_pairs
            static_totals[(BT, "sum_ks")] = n_pairs * bc_ksteps
            if all(dense_side):
                # Both sides dense: the causal intra bit is constant `True`.
                static_totals[(BT, "sum_KI")] = n_pairs * BC
                static_totals[(BT, "sum_kis")] = n_pairs * bc_ksteps
    else:
        # Derivation 3: candidacy, closed-form from the tile bitmaps' popcounts.
        for BT in cands:
            r_tile, w_tile = views.tile_live(BT)
            n_tiles = r_tile[0].shape[1]
            slice_r, slice_w = (side[j].reshape(BH, n_tiles, n_slices, c).any(dim=-1)
                                for side in (r_tile, w_tile))
            p_read = slice_r.sum(dim=-1)
            p_write = slice_w.sum(dim=-1)
            p_both = (slice_r & slice_w).sum(dim=-1)
            for level in range(j):
                p_read = p_read * r_tile[level].sum(dim=-1)
                p_write = p_write * w_tile[level].sum(dim=-1)
                p_both = p_both * (r_tile[level] & w_tile[level]).sum(dim=-1)
            counters[(BT, "sum_cand")] = (p_read + p_write - p_both).sum()

    if not all(dense_side):
        # `summaries=False`: nothing here reads the per-level occupancy scalars, and
        # issuing a device reduction nothing reads is waste. A caller that wants them
        # calls the census directly, where `summaries` defaults to True.
        census = tile_union_census(views.read_mask, views.write_mask, topology.widths,
                                   BC, tuple(cands), dense_side=dense_side,
                                   summaries=False)
        counters.update(census.per_bt)

    totals: dict[tuple[int, str], int] = dict(static_totals)
    totals.update(resolve_census(counters))

    out: dict[int, TileUnionStats] = {}
    for BT in cands:
        n_pairs = owners_total * BH * n_tiles_at[BT]
        n_cand = totals[(BT, "sum_cand")]
        out[BT] = TileUnionStats(
            BC=BC, BT=BT,
            mean_K=totals[(BT, "sum_K")] / n_pairs if n_pairs else 0.0,
            saturated_frac=totals[(BT, "sum_sat")] / n_pairs if n_pairs else 0.0,
            mean_K_candidate=totals[(BT, "sum_K")] / n_cand if n_cand else 0.0,
            mean_KI_candidate=totals[(BT, "sum_KI")] / n_cand if n_cand else 0.0,
            # `sum_ks`/`sum_kis` are over ALL pairs, divided by the CANDIDATE count
            # -- valid for the same reason `sum_K_candidate == sum_K` is (docstring
            # above): `K > 0` implies candidacy, and `ceil(0/16) == 0`, so the
            # non-candidate pairs contribute exactly zero to both numerators.
            mean_ksteps_candidate=totals[(BT, "sum_ks")] / n_cand if n_cand else 0.0,
            mean_kisteps_candidate=totals[(BT, "sum_kis")] / n_cand if n_cand else 0.0,
            candidate_frac=n_cand / n_pairs if n_pairs else 0.0,
            n_pairs=n_pairs,
            n_cohorts=owners_total * BH)
    return out


#: Element budget for one owner CHUNK of the statistics pass, in booleans per side.
#: The pass is vectorized OVER OWNERS, so the transient leaf tensor is
#: ``[BH, n_tiles, BT, chunk, BC]`` -- i.e. ``BH * T * chunk * BC`` booleans. The
#: chunk is sized from this budget so the transient never grows with ``N``: at
#: ``chunk == owners_total`` the tensor would be ``BH * T * N``, which is exactly
#: the ``O(L*N)`` object §4 forbids materializing. Chunking is therefore a
#: STRUCTURAL bound, not a tuning knob -- the standing "design for nc ~ 65K" rule
#: is what it exists for, and the flagship cell (``BH*T*BC = 262144``) fits a
#: 64-owner chunk inside it, so the flagship pays exactly one chunk.
_STATS_OWNER_CHUNK_ELEMS = 1 << 24


def _tile_union_stats_multi_enumerated(views: RoutingViews, topology: Topology, BC: int,
                                      bt_candidates: tuple[int, ...]) -> dict[int, TileUnionStats]:
    """:class:`TileUnionStats` for EVERY candidate ``BT``, from ONE pass at the finest.

    **A TEST-ONLY REFERENCE IMPLEMENTATION. It must never feed a production
    decision**. It enumerates
    the FULL ``[BH, n_tiles, BT, owners, BC]`` leaf rectangle in owner chunks, so
    it costs ``Theta(BH * T * N)`` boolean work no matter how sparse the routing
    is -- 59-74 % of a layer step at scale when it was on one. The
    fast path is :func:`_tile_union_stats_multi`, which counts the same
    statistics from packed token words (:mod:`rola.planner.calibration.census`). This one
    survives because it is written from the fold identity below and is therefore an
    independent second derivation to gate the production pass against.

    **The fold is an identity, not an approximation, and this is the derivation.**
    Fix an owner and a coarse tile of ``2*BT`` tokens, split into halves ``A``
    (first ``BT``) and ``B`` (second). Write ``live_r[t, i]`` / ``live_w[t, i]``
    for the per-token, per-leaf read/write liveness (the AND over the spanned
    levels' digits AND the fixed levels' token validity -- see
    :func:`tile_union_stats`), and per half define the four leaf BITMAPS

        U    [i] = OR_t (live_r | live_w)        popcount -> K
        Rany [i] = OR_t live_r
        Wany [i] = OR_t live_w
        I    [i] = OR_t (live_r[t,i] AND OR_{t' <= t} live_w[t',i])   popcount -> KI

    Then, over the coarse tile,

        U    = U_A | U_B                                   (an OR over tokens is associative)
        Rany = Rany_A | Rany_B
        Wany = Wany_A | Wany_B
        I    = I_A | I_B | (Rany_B AND Wany_A)

    ``U``/``Rany``/``Wany`` are plain token-ORs and fold trivially. ``I`` does not,
    because §5's causal prefix ``OR_{t' <= t} live_w`` CROSSES the half boundary:
    a leaf read in ``B`` counts if it was written anywhere in ``A``, which neither
    half's own ``I`` records. That carry is exactly one term, ``Rany_B AND Wany_A``,
    and adding it makes the fold exact -- which is why the fold has to be over
    bitmaps and cannot be over the counts ``K``/``KI`` (a popcount cannot be
    un-ORed).

    Tile CANDIDACY does not fold from the per-owner candidate flags either: it is
    an AND (over fixed levels) of ORs (over the boundary slice's digits and over
    the tile's tokens), and AND-of-OR does not commute with a further OR over
    tokens. It IS exact when folded one level down, on the per-``(level, digit)``
    TILE bitmaps -- ``r_tile_2BT[l, d] = r_tile_A[l, d] | r_tile_B[l, d]`` -- and
    that is what :meth:`RoutingViews.tile_live` does, forming the candidacy AND
    fresh at each candidate ``BT`` from the folded bitmaps.

    The padded token tail is inert on both counts: a padded token is exactly zero
    in every level, so it is dead in ``live_r``/``live_w`` and contributes no bit
    anywhere. ``ceil(ceil(T/BT)/2) == ceil(T/(2*BT))`` makes the tile count agree
    with a direct computation at the coarse ``BT``, so an odd fine tile count folds
    against an all-dead partner and lands on the right number of coarse tiles.

    Everything is accumulated as EXACT int64 counts and reduced to floats once, on
    the host, after a SINGLE device transfer -- the per-owner boolean-mask indexing
    the previous form used (``k.flatten()[cand]``) forced a synchronization per
    owner per side.
    """
    cands = _bt_candidate_chain(bt_candidates)
    BT0 = cands[0]

    geometry = owner_geometry(topology.widths, BC)
    j = geometry.first_spanned_level
    widths = topology.widths
    D = topology.D
    c = geometry.slice_digits
    n_slices = geometry.n_slices
    owners_total = geometry.owners_total
    BH, T = views.BH, views.T
    device = views.read_mask[0].device
    n_tiles0 = (T + BT0 - 1) // BT0

    def _tiled(mask: torch.Tensor) -> torch.Tensor:
        pad = n_tiles0 * BT0 - T
        if pad:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
        return mask.view(BH, n_tiles0, BT0, mask.shape[-1])

    r = [_tiled(mask) for mask in views.read_mask]      # [BH, n_tiles0, BT0, width]
    w = [_tiled(mask) for mask in views.write_mask]

    # Owner decomposition, vectorized: the inverse of `OwnerGeometry.owner_digits`
    # evaluated for every owner at once, on device, so no Python loop is O(N/BC).
    owner_ids = torch.arange(owners_total, device=device)
    slice_ids = owner_ids % n_slices                                   # [owners]
    rest = owner_ids // n_slices
    fixed_digit = [(rest // max(1, prod(widths[level + 1:j]))) % widths[level]
                   for level in range(j)]                              # each [owners]

    # Leaf-local digit of level `l >= j`: `base_l + (i / stride_l) % m_l`, the same
    # (base, stride, modulus) triple `level_descriptors` hands the kernel -- now
    # broadcast over owners rather than recomputed per owner.
    local = torch.arange(BC, device=device)
    span_index = []
    for level in range(j, D):
        stride = max(1, prod(widths[level + 1:]))
        modulus = c if level == j else widths[level]
        digits = (local // stride) % modulus                           # [BC]
        if level == j:
            digits = slice_ids.unsqueeze(1) * c + digits.unsqueeze(0)  # [owners, BC]
        else:
            digits = digits.unsqueeze(0).expand(owners_total, BC)
        span_index.append(digits)

    # §4 candidacy is formed per candidate `BT` from that candidate's folded
    # per-(level, digit) tile bitmaps -- see the docstring on why it cannot be
    # folded from the per-owner flags.
    slice_any: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tile_bitmaps: dict[int, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}
    n_tiles_at: dict[int, int] = {}
    for BT in cands:
        r_tile, w_tile = views.tile_live(BT)
        tile_bitmaps[BT] = (r_tile, w_tile)
        n_tiles_at[BT] = r_tile[0].shape[1]
        slice_any[BT] = tuple(
            side[j].view(BH, n_tiles_at[BT], n_slices, c).any(dim=-1)  # [BH, n_tiles, n_slices]
            for side in (r_tile, w_tile))

    # Six exact int64 accumulators per candidate; ONE host transfer at the end.
    zero = lambda: torch.zeros((), dtype=torch.int64, device=device)  # noqa: E731
    acc = {BT: {name: zero() for name in
                ("sum_K", "sum_sat", "sum_cand", "sum_K_cand", "sum_KI_cand",
                 "sum_ks_cand", "sum_kis_cand")}
           for BT in cands}

    per_owner = max(1, BH * n_tiles0 * BT0 * BC)
    chunk = max(1, min(owners_total, _STATS_OWNER_CHUNK_ELEMS // per_owner))

    for lo in range(0, owners_total, chunk):
        hi = min(lo + chunk, owners_total)
        C = hi - lo

        live_r = live_w = None
        for k_level, level in enumerate(range(j, D)):
            index = span_index[k_level][lo:hi].reshape(-1)             # [C*BC]
            rl = r[level][..., index].view(BH, n_tiles0, BT0, C, BC)
            wl = w[level][..., index].view(BH, n_tiles0, BT0, C, BC)
            live_r = rl if live_r is None else (live_r & rl)
            live_w = wl if live_w is None else (live_w & wl)
        # FIXED levels -> per-token validity, broadcast across the BC leaf axis.
        for level in range(j):
            index = fixed_digit[level][lo:hi]                          # [C]
            live_r = live_r & r[level][..., index].unsqueeze(-1)
            live_w = live_w & w[level][..., index].unsqueeze(-1)

        # The four leaf bitmaps at the FINEST granularity: [BH, n_tiles0, C, BC].
        u = (live_r | live_w).any(dim=2)
        r_any = live_r.any(dim=2)
        w_any = live_w.any(dim=2)
        i_bits = (live_r & live_w.cummax(dim=2).values).any(dim=2)
        del live_r, live_w

        for step, BT in enumerate(cands):
            if step:
                factor = BT // cands[step - 1]
                for _ in range(factor.bit_length() - 1):
                    if u.shape[1] % 2:                                 # pad with a dead tile
                        u, r_any, w_any, i_bits = (
                            torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 1))
                            for t in (u, r_any, w_any, i_bits))
                    a, b = slice(0, None, 2), slice(1, None, 2)
                    i_bits = (i_bits[:, a] | i_bits[:, b]) | (r_any[:, b] & w_any[:, a])
                    u = u[:, a] | u[:, b]
                    r_any = r_any[:, a] | r_any[:, b]
                    w_any = w_any[:, a] | w_any[:, b]
            assert u.shape[1] == n_tiles_at[BT], (
                f"fold landed on {u.shape[1]} tiles at BT={BT}, expected {n_tiles_at[BT]}")

            k_count = u.sum(dim=-1)                                    # [BH, n_tiles, C] int64
            ki_count = i_bits.sum(dim=-1)

            r_tile, w_tile = tile_bitmaps[BT]
            slice_r, slice_w = slice_any[BT]
            cand_r = slice_r[..., slice_ids[lo:hi]]                    # [BH, n_tiles, C]
            cand_w = slice_w[..., slice_ids[lo:hi]]
            for level in range(j):
                index = fixed_digit[level][lo:hi]
                cand_r = cand_r & r_tile[level][..., index]
                cand_w = cand_w & w_tile[level][..., index]
            cand = cand_r | cand_w

            bucket = acc[BT]
            bucket["sum_K"] += k_count.sum()
            bucket["sum_sat"] += (k_count == BC).sum()
            bucket["sum_cand"] += cand.sum()
            bucket["sum_K_cand"] += (k_count * cand).sum()
            bucket["sum_KI_cand"] += (ki_count * cand).sum()
            q = MMA_K_QUANTUM
            bucket["sum_ks_cand"] += (((k_count + q - 1) // q) * cand).sum()
            bucket["sum_kis_cand"] += (((ki_count + q - 1) // q) * cand).sum()

    order = [(BT, name) for BT in cands for name in
             ("sum_K", "sum_sat", "sum_cand", "sum_K_cand", "sum_KI_cand",
              "sum_ks_cand", "sum_kis_cand")]
    flat = torch.stack([acc[BT][name] for BT, name in order]).cpu().tolist()
    totals = {key: value for key, value in zip(order, flat)}

    out: dict[int, TileUnionStats] = {}
    for BT in cands:
        n_pairs = owners_total * BH * n_tiles_at[BT]
        n_cand = totals[(BT, "sum_cand")]
        out[BT] = TileUnionStats(
            BC=BC, BT=BT,
            mean_K=totals[(BT, "sum_K")] / n_pairs if n_pairs else 0.0,
            saturated_frac=totals[(BT, "sum_sat")] / n_pairs if n_pairs else 0.0,
            mean_K_candidate=totals[(BT, "sum_K_cand")] / n_cand if n_cand else 0.0,
            mean_KI_candidate=totals[(BT, "sum_KI_cand")] / n_cand if n_cand else 0.0,
            mean_ksteps_candidate=totals[(BT, "sum_ks_cand")] / n_cand if n_cand else 0.0,
            mean_kisteps_candidate=totals[(BT, "sum_kis_cand")] / n_cand if n_cand else 0.0,
            candidate_frac=totals[(BT, "sum_cand")] / n_pairs if n_pairs else 0.0,
            n_pairs=n_pairs,
            n_cohorts=owners_total * BH)
    return out


__all__ = ["TileUnionStats", "tile_union_stats"]
