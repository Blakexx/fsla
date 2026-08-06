"""The RETIRED runtime cost model, relocated (``THUNDER_RUN_SPEC.md`` §6 P4).

Everything here used to run inside ``rola.planner.schedule`` on every forward. Under
the end-state ruling it is CALIBRATION and AUDIT machinery, not a runtime decider, so
it lives with the bench:

* the exact per-``(owner, tile)`` census (:mod:`rola.planner.calibration.census`)
  and the ``TileUnionStats`` it produces;
* ``plan_cost_per_token`` and its four measured constants, which are the FIT whose
  compiled output ships as ``rola.planner.select``'s rule table;
* the ``BT`` ARGMIN itself -- ``select_plan_axes``, ``select_plan``,
  ``plan_and_schedule`` -- which is what the auditor runs against the closed form.

**THREE THINGS THIS MOVE BUYS, and they are the reason it is not cosmetic.**

1. **The ``torch.compile``/inductor RUNTIME dependency leaves the shipped package.**
   ``census`` states in its own docstring that it makes inductor a runtime requirement
   of the planner on CPU as well as CUDA, with no eager path kept beside it. That edge
   is now a BENCH dependency.
2. **Selection stops auditing itself.** A runtime shadow would keep alive exactly the
   object the ruling deletes, and its regret metric would be denominated in the model
   being retired. Here the regret is denominated in a bench artefact, which is what it
   is.
3. **The constants become re-fittable without touching the shipped package.** §9 step 2
   re-calibrates here and recompiles the rule table; nothing in ``rola/`` moves.

**WHAT STILL IMPORTS THIS, and how.** ``rola.planner.schedule`` keeps PEP-562 lazy
CALL PROXIES for the moved names (``select_plan``, ``plan_and_schedule``,
``tile_union_stats``, ...), because they are documented public API
(``rola/__init__.py``: "STABLE construction, ADVISORY contents") and because five bench
instruments and two out-of-track test suites call them. The proxies resolve on CALL and
not on import, so ``import rola`` never reaches this package and a shipped install
without the bench fails loudly at the call site rather than at import.

Symbols are the planner's throughout: ``L``/``T`` tokens; ``N`` leaves; ``D`` levels;
``BC`` leaves per owner; ``BT`` token tile; ``BH`` = batch x heads; ``MMA_K_QUANTUM``
the MMA k-extent; ``ds = ceil(BC/MMA_K_QUANTUM)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from rola.planner.calibration.census import resolve_census, tile_union_census
from rola.planner.schedule import (
    _SUPPORTED_BT,
    COMPILED_RING_DEPTH,
    DEFAULT_SM_COUNT,
    DEFAULT_SM_SHARED_BYTES,
    MMA_K_QUANTUM,
    OccupancyOracle,
    Plan,
    RoutingViews,
    Schedule,
    StaticDensity,
    build_schedule,
    owner_geometry,
    resolve_occupancy,
    routing_views,
    spanned_moduli,
)
from rola.routing.types import Topology

__all__ = [
    "COST_MODEL_PROVENANCE",
    "DECAY_KSTEP_WORK",
    "INTRA_GRAM_KSTEP_WORK",
    "READOUT_FOLD_KSTEP_WORK",
    "TILE_FIXED_WORK",
    "PlanAxes",
    "TileUnionStats",
    "plan_and_schedule",
    "plan_cost_per_token",
    "plan_waves",
    "select_plan",
    "select_plan_axes",
    "tile_union_stats",
]


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

    This is a PLANNER function. Its results are reduced to compile-time template
    parameters before launch; no kernel path reads them (§3's branch rule).

    ``views`` is an optional :class:`RoutingViews` carrying the folded tensors and
    the memoized per-``BT`` tile bitmaps, so the selector and the schedule builder
    can share one pass over the routing (see :func:`plan_and_schedule`). Passing
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
    is ``_SUPPORTED_BT = (16, 32)``, checked one layer up in
    :func:`select_plan_axes`, and spec §6's pinning of ``BT`` to that set --
    :func:`rola.planner.census._token_support_words` refuses anything wider outright.
    The guarantee is therefore real but lives elsewhere, and a future ``BT`` outside
    that set is a spec change that must widen the word rather than a runtime condition
    to branch on.
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
        # `summaries=False`: the §2.7 closed-form selector that will read them does
        # not exist yet, and issuing a device reduction nothing reads is waste rather
        # than readiness. The offline auditor calls the census directly, where
        # `summaries` defaults to True.
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
    is -- which made ``select_plan`` 59-74 % of the layer at scale. The
    production path is :func:`_tile_union_stats_multi`, which counts the same
    statistics from packed token words (:mod:`rola.planner.census`). This one
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
# ---------------------------------------------------------------------------
# §6 cost model -- ONE formula prices the intra lowering and BT
# ---------------------------------------------------------------------------

# `MMA_ISSUE_WORK` IS DELETED (2026-07-30, adversarial review §4.2), and the
# readout term it scaled with it. It priced the readout's MMA issues per active
# leaf column -- `d_v * w / (128 * k_unit)`, i.e. 0.375 work-units at `w = 6`,
# `d_v = 64`, `k_unit = 8` -- against an in-tile term of 136-264 work-units per
# token. NO DECISION ANYWHERE IN THIS MODEL DEPENDS ON IT: it is absent from the
# intra crossover entirely, and at the deepest measured deficit it would need
# `w > 2.4e4` -- 4000x its own derivation -- to outweigh a per-tile term. A
# constant that cannot influence a
# decision at any value within three orders of magnitude of its derivation is not
# a plan parameter; carrying it as one asserted a tunability the calibration gate
# could not gate (the review mutated it 60x and nothing moved). The work it named
# is real and is now un-modelled EXPLICITLY, which is checkable, rather than
# modelled by a number no gate can bind. `k_unit`, whose only use was that term,
# goes with it: `mma_precision` is a NUMERIC FORMAT axis of the kernel and is no
# longer an input to the cost model.
#
# The two constants that remain each have a value at which the calibration table
# fails -- `test_the_cost_model_reproduces_its_calibration` asserts exactly that,
# per constant, so "binding" is a gated property rather than a claim.

#: Per-CANDIDATE-TILE fixed work that does not scale with ``K``: the epoch
#: bookkeeping, the row scales, the amplitude staging, the block barriers, the
#: write-conditioned publish. Same units. This is the term that makes a LARGER
#: ``BT`` cheaper, and it is the only one that does.
#:
#: **RE-MEASURED 2026-08-01 (M-1), 120.0 -> 3510.35, AND IT ABSORBS THE DELETED
#: `STAGE_COLUMN_WORK`.** The old value was never isolated: it was the residue of an
#: old-consumer calibration, and the work it names was really being carried by the
#: staging term this repository has now retired (see the deletion note below).
#:
#: Instrument: `ncu --clock-control base`, `smsp__inst_executed.sum`, ONE profiled
#: launch per arm (`--launch-skip 1 --launch-count 1`), RTX 3080 Ti, 2026-08-01, on
#: the M-1 isolating cells in `benchmarks/bench/cells.py` (all dense/dense, so the
#: unions saturate and NO routing statistic moves between them).
#:
#: **TWO INDEPENDENT ISOLATING SOLVES, EACH OF THEM TWO-STAGE.** A per-candidate-
#: tile constant cannot be read off a single cell, because a launch's instructions
#: are `n_cohorts * (F + n_tiles * P)` -- a per-COHORT part `F` (CTA prologue and
#: epilogue, the owner's state load and store) and a per-TILE part `P`. So each
#: geometry is measured at three (or two) values of `n_tiles = T / BT` at fixed
#: everything else, `F` and `P` are separated, and only then is `P` split across
#: `BC` into a constant and a per-dense-k-quantum slope:
#:
#:   `iso-tile-w16x4x64-BC64-T{128,256,512}` -> P = 8771.0, F = 9041.0
#:   `iso-tile-w16x4x64-BC16-T{128,256,512}` -> P = 4823.0, F = 4535.0
#:       => slope 1316.0 / dense k-quantum, INTERCEPT **3507.0**
#:   `topo-D2-w64x64-BC64` + `iso-tile-w64x64-BC64-T256` -> P = 8135.0, F = 8434.0
#:   `topo-D2-w64x16-BC16` + `iso-tile-w64x16-BC16-T256` -> P = 4669.0, F = 4036.0
#:       => slope 1155.3 / dense k-quantum, INTERCEPT **3513.7**
#:
#: The two intercepts are 0.19 % apart across a different `D`, a different width
#: multiset and a 4x different owner count, and the three-point `T` sweep solves `F`
#: to the same value from both of its consecutive pairs (9041.0 / 9041.0), so the
#: `F + n_tiles * P` form is CHECKED rather than assumed. The landed value is the
#: mean of the two intercepts and
#: `test_the_cost_model_reproduces_its_calibration` re-runs every argmin across the
#: whole measured range `[3507.0, 3513.7]`.
#:
#: **ARM-INVARIANT, MEASURED.** On `iso-tile-w16x4x64-BC64-T{128,256}` the per-tile
#: solve is 8771.0 without decay against 13345.0 with it, a delta of 4574.0 against
#: `DECAY_KSTEP_WORK * 4 = 4132.6` (+10.7 %) -- so this constant is the part the
#: decay arm ADDS TO, and that arm re-derives on a geometry it was not measured on.
#:
#: **THE PER-COHORT REMAINDER `F` IS MEASURED AND DELIBERATELY NOT MODELLED**, which
#: is the `MMA_ISSUE_WORK` rule applied to a term that is real: `F` = 7999 (D=1),
#: 8434 (D=2), 9041 (D=3) at `BC = 64` and 4036 / 4535 at `BC = 16`, essentially
#: arm-invariant (9041 without decay, 9506 with). It is a per-COHORT cost, so
#: per token it is `F / T` -- exactly `BT`-INVARIANT and arm-invariant, hence unable
#: to move the `BT` argmin at any value -- and it is 12 % of the
#: T=128 gate cells but 0.9 % at `atscale-L2048` and below 0.1 % at the parity cell,
#: i.e. it vanishes in exactly the regime the planner is for. Carrying it would add
#: a constant no gate can bind and an `n_pairs / n_cohorts` input the currency does
#: not otherwise need. It is recorded here instead, which is checkable.
#:
#: FALSIFIABLE: an isolating `T` sweep whose intercept falls outside
#: `[3507.0, 3513.7]`, or a geometry on which `F + n_tiles * P` does not solve
#: consistently from two disjoint pairs.
TILE_FIXED_WORK = (3507.0 + 3513.7) / 2.0

#: Per-K-STEP cost of the readout, the fold and the panel builds -- the three
#: extents that ride the k-quantum. Same units, per candidate tile, at the
#: `BT = 16` the constants were measured at.
#:
#: Instrument: the same counters at `atscale-L2048`, over a 3.39 k-step difference
#: in extent: 1035.0 M against 1812.1 M over 208836 candidate pairs, a delta of
#: 3721 per tile = **1755** per k-step.
#:
#: **IT IS PRICED PER QUANTUM, NEVER PER LEAF.** `ceil(K/16)` is what the kernel
#: issues: a tile with `K = 1` pays a whole k-step and a tile with `K = 0` pays
#: none.
READOUT_FOLD_KSTEP_WORK = 1755.0

#: Per-K-STEP cost of the intra Gram's own contraction, same units and same `BT`.
#:
#: Instrument: `wk1-D2-w64x64-BC64` against `topo-D2-w64x64-BC64` -- the same
#: widths, the same `BC`, the same saturated READ union, and a write of `k = 1` per
#: token that cuts `KI` from 64.00 to 0.33 while leaving `K = BC`. The delta is
#: therefore the intra extent and nothing else: **144** per k-step, a 12x demotion
#: of the term that used to dominate the `BT` decision.
INTRA_GRAM_KSTEP_WORK = 144.0

#: Per-K-STEP cost of the LeafMassDecay arm on top of the same tile: the clock
#: prefix, the two `expf` survival panels, the two coefficient folds and the `s0`
#: entry/exit sweeps, plus the `gram`/`realized_leaf` -> `panel` lowering the arm
#: forces. Same units, same `BT = 16`.
#:
#: **PINNED BY TWO ISOLATING CELL PAIRS, AND IT HAS NO PER-TILE FIXED COMPONENT.**
#: `topo-D2-w64x64-BC64` and `topo-D2-w64x16-BC16` are the same topology, the same
#: support, the same `BT`, and saturate identically -- so they differ ONLY in the
#: k-step count (4 against 1) and a two-point solve separates a per-tile constant
#: from a per-k-step one. Counters (`ncu --clock-control base`,
#: `smsp__inst_executed.sum`, one profiled launch per arm, 2026-08-01, over 1024
#: candidate pairs each):
#:
#:   4372.9 and 1054.1 per tile ->  slope **1106.3**, intercept  -52.3
#:   3948.8 and 1069.8 per tile ->  slope   **959.7**, intercept +110.1
#:
#: Both intercepts are within 10 % of the smaller delta of ZERO, so decay carries no
#: per-candidate-tile fixed cost: it rides the `ceil(K/16)` quantum exactly like the
#: readout. The landed value is the mean of the two slopes, and
#: `test_the_cost_model_reproduces_its_calibration` re-runs every ordering across
#: the WHOLE measured range `[959.7, 1106.3]`, so the residual spread is a gated
#: property rather than a rounding choice.
#:
#: **THE `BT * BC` ALTERNATIVE STRUCTURE IS REFUTED BY 15.8x**, and by a comparison
#: with no free parameter: two `atscale-L2048` arms with IDENTICAL `BT` and `BC`
#: differing only in realized extent measure 4345.5 against 275.8 per tile, so a
#: `BT * BC`-scaled decay term -- which predicts the same delta on both -- cannot
#: move there at all. Only an extent-scaled term can.
#:
#: FALSIFIABLE: a saturated-extent cell whose per-k-step decay delta falls outside
#: `[959.7, 1106.3]`, or any cell whose delta needs a per-tile constant term.
DECAY_KSTEP_WORK = (1106.3 + 959.7) / 2.0

# ---------------------------------------------------------------------------
# `STAGE_COLUMN_WORK` IS DELETED (M-1, RULING A, 2026-08-01) -- and it is deleted
# ON A MEASUREMENT, not on the source read that suspected it.
# ---------------------------------------------------------------------------
# It priced `2 * gram_cols * W / ring_slots` per candidate tile at `W = 85`, named
# for a staging ring, bracketed `(40.5, 143.3)` by three cells, and it was known to
# be load-bearing: setting it to 0 flipped FOUR of the eight measured argmins. The
# source read had already established that the surviving consumer has no staging
# ring (`p.ring_slots`, `p.gram_cols` and `p.level_gram_offset` were written by the
# host and read by ZERO sites in `consumer_kernel.cuh`; all three are since REMOVED
# end-to-end -- `ring_slots` with H3's agreement gate, the two gram fields in the
# terminal cycle), but a name being wrong is
# not permission to delete a term that carries a decision, so the disposition
# waited for the isolating cells.
#
# **WHY NO CALIBRATION CELL COULD SEE THE VARIABLE.** `gram_cols = sum(m_l)` over
# the spanned levels, and `sum(m_l) <= prod(m_l) = BC` with equality exactly when
# ONE level is spanned. All four calibration cells span one level, so `gram_cols`
# was IDENTICALLY `BC` on all eight argmin rows and all twelve absolute-total rows:
# the term was algebraically indistinguishable from `16 * W` per dense k-quantum,
# and its `gram_cols` shape -- the only thing making it a STAGING term rather than
# an extent term -- had never been observed at all.
#
# **THE MEASUREMENT, and it is an EXACT EQUALITY rather than a small residual.**
# Four cells at `BC = 64`, `D = 3`, `N = 4096`, the same width MULTISET permuted so
# that only the owner rectangle's level decomposition moves -- same grid, same 1024
# candidate tiles, same saturated unions (`ncu --clock-control base`,
# `smsp__inst_executed.sum`, one profiled launch each, RTX 3080 Ti, 2026-08-01):
#
#     `iso-gc-w32x16x8-BC64`         gram_cols = 16   10,062,336
#     `iso-gc-w64x16x4-BC64`         gram_cols = 20   10,062,336
#     `iso-gc-w64x32x2-BC64`         gram_cols = 34   10,062,336
#     `iso-tile-w16x4x64-BC64-T128`  gram_cols = 64   10,138,752
#     `iso-gc-w4x16x64-BC64`         gram_cols = 64   10,138,752
#
# The first three are BYTE-IDENTICAL across a 2.1x range of `gram_cols`. The
# per-staged-column price is therefore **exactly 0.000**, not 85: at
# `gram_cols = 34` the term was charging 2890 work-units per tile of a mechanism
# that retires no instruction. What the counts DO separate is the number of FIXED
# levels `j` (the two `gram_cols = 64` cells have `j = 2`, the other three
# `j = 1`): 76,416 instructions per launch = **74.6 per candidate tile per fixed
# level**, reproduced at the other `BC` by `iso-tile-w16x4x64-BC16-T128` (j=2)
# against `iso-gc-w64x16x4-BC16` (j=1) as 73.6 per tile -- the SAME delta at
# `d(gram_cols) = 8` that the `BC = 64` pair shows at `d(gram_cols) = 44`, which is
# the second, independent refutation of the column scaling.
#
# **WHERE THE WORK WENT.** The magnitude the term carried was real; its variable was
# not. It is now inside `TILE_FIXED_WORK`, re-measured by isolating cells from 120
# to 3510.35 -- a per-candidate-tile constant, `gram_cols`-invariant, which is the
# shape the eight argmins actually need (see that constant's own docstring). The
# 74.6-per-fixed-level geometry term is MEASURED AND NOT MODELLED: it is 2 % of the
# constant it would ride on and decision-inert, and this repository does not carry
# constants no gate can bind (`MMA_ISSUE_WORK`'s deletion, 2026-07-30).
#
# **THE RING AXIS IS NOW UNPRICED, AND THAT IS STATED RATHER THAN HIDDEN.**
# `ring_slots` was priced by this term and by nothing else; the occupancy oracle is
# ring-INVARIANT (the consumer's shared memory is entirely compile-time) and the
# kernel reads the field at zero sites, so after this deletion the depth is inert in
# every currency the planner has. `select_plan_axes` therefore STATES the depth
# instead of arbitrating it. If §2.3's item buffers ever build a real ring, its
# price is a new measurement on the kernel that has one -- not this constant
# revived.

#: Where the constants above come from, carried next to them so a number cannot be
#: quoted without its provenance (spec §8's standing warning).
#:
#: **RE-BASED 2026-08-01 ONTO THE NEW CONSUMER, and the old basis is retired rather
#: than amended.** The 48-cell `compaction2.md` §B2 grid measured the OLD consumer,
#: a different algorithm. Re-using its constants to bind constants measured on this
#: kernel would have been the two-kernel conflict §2.28 stopped on.
COST_MODEL_PROVENANCE = (
    "Re-based 2026-08-01 on the NEW consumer. RTX 3080 Ti, `ncu --clock-control "
    "base`, `smsp__inst_executed.sum`, ONE profiled launch per arm "
    "(`--launch-skip 1 --launch-count 1`), every run `flock`-serialized. Cells are "
    "the frozen registry in `benchmarks/ops/benchmark_rola_v3_new_consumer.py`, and "
    "each constant has an ISOLATING cell or cell pair rather than a fit: "
    "READOUT_FOLD_KSTEP_WORK from `atscale-L2048`; "
    "INTRA_GRAM_KSTEP_WORK from `wk1-D2-w64x64-BC64` against "
    "`topo-D2-w64x64-BC64`, which differ only in KI; DECAY_KSTEP_WORK from the "
    "`BC=64`/`BC=16` pair of one topology, which differ only "
    "in the k-step count; and TILE_FIXED_WORK (M-1, 2026-08-01) from TWO "
    "independent isolating solves over the `iso-tile-*` cells, each separating a "
    "per-COHORT cost from a per-TILE one with a three-point `T` sweep before "
    "splitting the per-tile part across `BC`. The `iso-gc-*` cells belong to the "
    "same batch and are the one that removed a term: they hold `BC`, `D`, `N` and "
    "the grid fixed while permuting the owner rectangle, and measure the price of a "
    "staged Gram column as EXACTLY zero. "
    "KNOWN AND MEASURED, NOT MODELLED, so the next reader does not rediscover it: "
    "(1) a per-COHORT remainder of 4036-9506 instructions (CTA prologue/epilogue, "
    "state load/store) -- `BT`- and arm-invariant per token, hence unable to move "
    "the argmin, and below 1 % of the at-scale cells; (2) 74.6 per candidate tile "
    "per FIXED level, the only geometry dependence that survives measurement; "
    "(3) the dense k-quantum is OVER-priced by this model at 1899 (readout+intra) "
    "against 1033 (D=1) / 1155 (D=2) / 1316 (D=3) measured, which is now the "
    "LEADING absolute-total residual and grows ~141 per level per quantum -- an "
    "isolating re-measurement of READOUT_FOLD_KSTEP_WORK, queued, never a re-fit "
    "against the rows it must reproduce. ONE machine, ONE toolchain: the constants are plan "
    "parameters, not machine truths, and spec §8's measurement basis replaces them "
    "without a code change. WHAT BINDS THEM is "
    "`test_the_cost_model_reproduces_its_calibration`, which asserts the `BT` "
    "ARGMIN -- the decision `select_plan_axes` actually makes -- "
    "against the measured argmins of the §2.32 matrix and the dense-toll signs. "
    "KNOWN LIMITATION, PROVED RATHER THAN SUSPECTED: on a SATURATED-DENSE cell "
    "(`K == BC` on every candidate pair) with decay the per-slice `BT` preference "
    "INVERTS against this model, with BT=32 retiring 14 % FEWER instructions in "
    "5.7-27.2 % MORE time: that is a STALL, and a work model does not model stalls. "
    "A spill-byte term is FORBIDDEN as the fix: its constant could only be set by "
    "the two cells it would then have to reproduce."
    "WHAT BOUNDS THIS CLAIM, now that the arm it used to be stated against is gone: the saturated-dense cells of the measured-argmin table are `topo-D2-w64x64-BC64` and `topo-D2-w64x16-BC16` (candidate_frac 1.0, K = ds), and the shipped closed form reproduces the MEASURED argmin BT=32 on all four of their rows, decay off and on -- the two decay rows being exactly the cells the inversion was measured on. The inversion is therefore a residual that does not move the decision at any cell it was measured at; that is a measurement, asserted by `tests/unit/test_planner.py`'s calibration gates, and not a structural exclusion."
)


def plan_cost_per_token(topology: Topology, BC: int, stats: TileUnionStats, *,
                        decay: bool = False,
                        tile_fixed_work: float = TILE_FIXED_WORK,
                        readout_fold_kstep_work: float = READOUT_FOLD_KSTEP_WORK,
                        intra_gram_kstep_work: float = INTRA_GRAM_KSTEP_WORK,
                        decay_kstep_work: float = DECAY_KSTEP_WORK) -> float:
    """§6's crossover pricing, in work-units per token, over the QUANTA the kernel
    issues rather than the leafs it names.

    Per CANDIDATE tile, and then divided by ``BT`` so the result is comparable
    across ``BT``::

        W_tile = tile_fixed_work
               + readout_fold_kstep_work * ksteps  * (BT / 16)
               + intra_gram_kstep_work   * kisteps * (BT * (BT + 1) / 272)
               + decay_kstep_work        * ksteps  * (BT / 16)    (if decay)

        cost_per_token = candidate_frac * W_tile / BT

    where ``candidate_frac`` is the realized share of ``(owner, tile)`` pairs the §4
    bitmap admits -- see the derivation at the return statement, which shows it is
    the DEFINITION of "per token" rather than an added term, carries no constant,
    and is identically ``1.0`` on every saturated-dense cell.

    where ``ksteps`` and ``kisteps`` are both the compiled-in extent
    ``ceil(BC / 16)``: the consumer walks the full ``BC`` whatever the union
    realizes, so the extents are the template's and not the routing's.

    **WHY QUANTA AND NOT ``mean_K``.** The kernel issues ``ceil(K / 16)`` MMA
    k-steps, so a tile with ``K = 1`` costs a whole one and a tile with ``K = 0``
    costs none -- a distinction ``mean_K`` structurally cannot express, and the
    reason ``TileUnionStats`` accumulates the ceilings per pair.

    **WHAT REPLACED WHAT, AND WHY IT IS A REPLACEMENT.** The pre-2026-08-01 form
    priced a ``K_eff * (D - j)`` panel assembly and a ``BT * I_eff`` in-tile term
    against an OLD-CONSUMER calibration. The replacement constants are each measured
    on THIS kernel at an isolating cell (see their own docstrings), and the panel
    assembly is inside ``readout_fold_kstep_work`` because that is what the
    isolating delta contains.

    **THE TWO ``BT`` SCALINGS ARE STRUCTURAL, and they are now GATED rather than
    posited.** The readout, the fold and the panel builds are ``[BT x K] x [K x DV]``
    and linear in ``BT``; the intra Gram is causal, ``BT(BT+1)/2`` entries. Both are
    normalized to the ``BT = 16`` cell the constants were measured at
    (``16 * 17 = 272``). §2.28 could only posit them; the §2.32 matrix measured the
    ``BT`` axis at both values on both ``BC`` and the calibration gate now binds the
    ``BT`` argmin against it.

    **THE DECAY TERM RIDES THE SAME QUANTUM**, which is measured rather than assumed:
    two isolating cell pairs differing only in k-step count give slopes 1106.3 and
    959.7 with intercepts within 10 % of zero, and the ``BT * BC`` alternative is
    refuted by 15.8x at ``atscale-L2048`` (identical ``BT`` and ``BC``, 15.8x
    different deltas).

    **WHAT IS STILL NOT MODELLED, EXPLICITLY.** Residency and the WAVE REGIME are
    not here at all -- they are physical facts and ``select_plan_axes`` reads them
    from the occupancy oracle, per §6.5.

    **THE `intra_lowering` AXIS IS GONE** (item 6b, 2026-08-01), and with it the
    branch that priced ``gram``/``panel`` on the spanned moduli instead of the leaf
    list. The surviving consumer includes ``intra_panel.cuh`` unconditionally and
    does ONE intra lowering, so there is no second extent to price.
    """
    if stats.BC != BC:
        raise ValueError(f"stats were computed at BC={stats.BC}, priced at BC={BC}")
    # `topology` enters this function as a PRECONDITION and no longer as a price
    # (M-1, 2026-08-01): the owner rectangle must be admissible at this `BC` --
    # `owner_geometry` raises otherwise, and pricing an unaligned rectangle is
    # pricing a plan that cannot be built -- but its SHAPE costs nothing, which the
    # `iso-gc-*` cells measure as an EXACT equality across a 2.1x range of
    # `gram_cols`. Keeping the check is what makes that a stated finding rather than
    # a silently dropped argument.
    owner_geometry(topology.widths, BC)
    BT = stats.BT
    quantize = lambda n: float(-(-int(n) // MMA_K_QUANTUM))  # noqa: E731

    # The kernel walks the full `BC` whatever the union realizes, so the extents are
    # the template's, not the routing's.
    ksteps = kisteps = quantize(BC)

    w_tile = tile_fixed_work
    w_tile += readout_fold_kstep_work * ksteps * (BT / MMA_K_QUANTUM)
    w_tile += intra_gram_kstep_work * kisteps * (
        BT * (BT + 1) / float(MMA_K_QUANTUM * (MMA_K_QUANTUM + 1)))
    if decay:
        w_tile += decay_kstep_work * ksteps * (BT / MMA_K_QUANTUM)
    # ---- THE CANDIDATE FRACTION, and it is a DEFINITION, not a term ----------
    # `w_tile` is the cost of ONE CANDIDATE tile. The tiles an owner actually
    # visits are the CANDIDATE ones -- a `(owner, tile)` pair outside the §4 tile
    # bitmap is never enumerated by the CTA, so it costs nothing at all -- and there
    # are `candidate_frac * owners_total / BT` of them per token:
    #
    #   work / token = n_candidate_pairs * w_tile / (BH * T)
    #                = candidate_frac * owners_total * BH * (T / BT) * w_tile / (BH * T)
    #                = candidate_frac * owners_total * w_tile / BT
    #
    # `owners_total` is `BT`-INVARIANT and cell-constant, so it cancels out of every
    # comparison this function feeds and is not carried; `candidate_frac` is NOT
    # `BT`-invariant, and dropping it was a definitional error rather than a missing
    # effect. Omitting it asserted `candidate_frac == 1` -- that every owner is
    # candidate on every tile -- which is true on a dense cell and wrong by 1.8x on
    # a routed one.
    #
    # ZERO CONSTANTS: `candidate_frac` is an exact statistic of the realized
    # schedule, from the same reduction that produces `mean_ksteps_candidate` and in
    # the same class as every other input here. And it is IDENTICALLY 1.0 on all
    # four SATURATED-DENSE calibration cells, so it cannot have been fitted to the
    # rows it must reproduce: it moves only the two routed cells.
    #
    # MEASURED, 2026-08-01 (`ncu --clock-control base`, `smsp__inst_executed.sum`,
    # ONE profiled launch per arm), which is what identified it. At
    # `atscale-L2048-BC16` dense, the union coarsens from `candidate_frac = 0.1936`
    # at `BT = 16` to `0.3574` at `BT = 32` -- a 1.846x ratio, because a 32-token
    # tile union catches owners a 16-token one misses. The per-candidate-tile model
    # predicts `BT = 32` retires 0.821x the instructions; the machine retires
    # **1.481x**. `0.821 * 1.846 = 1.516` against 1.481, a 2.4 % residual. That 1.8x
    # WAS §2.33's "unexplained 2.3x", and it is a WORK MISCOUNT rather than the
    # stall it was filed as; the stall is the separate 1.42x issue-rate factor
    # recorded on `plan_waves`.
    #
    # FALSIFIABLE: a cell whose measured `BT = 32 / BT = 16` instruction ratio
    # departs from `candidate_frac`-corrected `w_tile / BT` by more than the ~2-3 %
    # the two at-scale cells show.
    return stats.candidate_frac * w_tile / BT


def plan_waves(n_cohorts: int, blocks_per_sm: int, sm_count: int) -> float:
    """The launch's WAVE COUNT, and the one regime distinction §6 is allowed to make.

    ``n_cohorts / (sm_count * blocks_per_sm)``, floored at 1. Every input is a
    RECORDED fact rather than a modelled one: the cohort count comes from the
    statistics pass, the CTAs/SM from the driver's occupancy oracle, the SM count
    from the device. Nothing here is calibrated and there is no constant to tune.

    **WHY IT IS A MULTIPLIER ON THE PER-CTA COST, AND WHY IT IS FLOORED AT 1.**
    Below one wave the whole grid is resident at once, so elapsed time is one CTA's
    latency and having fewer CTAs buys nothing -- residency cannot help a machine
    that is already holding everything. Above one wave the CTAs are serialized in
    batches of ``sm_count * blocks_per_sm``, so elapsed time is total work divided
    by that capacity and a candidate that fits FEWER CTAs per SM pays for it
    proportionally. ``max(1, waves)`` is the single expression that degenerates to
    both, and the ``1`` is structural (a wave count cannot be less than one wave of
    latency), not a threshold.

    **THIS IS THE STRUCTURE THE MEASUREMENT FOUND, and it is why
    ``_CALIBRATION_BT``'s KEY was wrong.** The §2.32 matrix's eight joint argmins
    partition perfectly on the wave regime -- every many-wave cell wants
    ``BT = 16``, every one-wave cell wants ``BT = 32`` -- and
    NEITHER ``BC`` NOR ``decay``, the two axes that table was keyed on, appears in
    the partition at all: both ``BC`` agree on every row and both decay arms agree
    on every row. ``waves_per_multiprocessor`` was 34-51 on the many-wave cells and
    0.40-0.80 on the one-wave cells.

    **AND IT IS NOT `harvest.md` §19.2's REJECTED GLOBAL DIVISOR.** That rejection
    was a measured counterexample: dividing by ``blocks_per_sm`` flipped the
    flagship decay cell to ``BT = 16`` where `compaction2.md` §B2 had measured
    ``BT = 32`` faster at 23 of 24 cell groups. §B2 measured the OLD consumer. On
    THIS kernel §2.32 measures ``BT = 16`` faster at the flagship on every arm, by
    89-153 % against paired IQRs of 0.005-0.025 -- i.e. the counterexample has been
    re-measured and now points the other way, which is the only thing that could
    have licensed the divisor and is recorded here as the licence.
    """
    if sm_count < 1:
        raise ValueError(f"sm_count must be >= 1, got {sm_count}")
    if blocks_per_sm < 1:
        raise ValueError(f"blocks_per_sm must be >= 1, got {blocks_per_sm}")
    return max(1.0, n_cohorts / float(sm_count * blocks_per_sm))


@dataclass(frozen=True, slots=True)
class PlanAxes:
    """The axes §6 selects from one cost model, plus the evidence."""

    BT: int
    stats: TileUnionStats
    cost_per_token: float
    #: The residency the OCCUPANCY ORACLE reports for the selected instantiation
    #: (§6 residency, item 2(a)) -- the driver's number in production and the
    #: injected one in tests. There is no analytic mirror to confuse it with any
    #: more: `mirror_blocks_per_sm` and the SMEM footprint it was computed from were
    #: deleted with the consumer they modelled (item 6b), after the exhaustive driver
    #: probe measured the mirror INVERTING the bound it claimed.
    blocks_per_sm: int = 0
    #: Staging RING depth (§6 residency). STATED, not selected (M-1 RULING A,
    #: 2026-08-01, twin docstring at `Plan.ring_slots`), and what it states is the
    #: COMPILED depth (`ROLA_C_RING_DEPTH`, shipped 1; see
    #: `schedule.COMPILED_RING_DEPTH` and the H3 agreement gate). EVIDENCE carried
    #: alongside the candidate that was selected, not a per-candidate arbitration.
    ring_slots: int = COMPILED_RING_DEPTH
    #: The launch's WAVE COUNT for the selected candidate (`plan_waves`), and the
    #: SM count it was computed against. EVIDENCE, and the evidence that matters
    #: most: the wave regime is what the §2.32 argmins partition on, so a plan that
    #: did not record it could not be checked against them.
    waves: float = 1.0
    sm_count: int = DEFAULT_SM_COUNT
    #: `cost_per_token * waves` -- what selection actually MINIMIZED. Carried
    #: separately from `cost_per_token` so the two halves of the decision (the work
    #: model and the physical regime) stay individually readable.
    effective_cost_per_token: float = 0.0


def select_plan_axes(read_levels: tuple[torch.Tensor, ...],
                     write_levels: tuple[torch.Tensor, ...],
                     topology: Topology, BC: int, *, decay: bool,
                     bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                     views: RoutingViews | None = None,
                     static_density: StaticDensity | None = None,
                     normalization: str = "global",
                     sm_count: int | None = None,
                     mma_precision: str = "bf16",
                     occupancy: OccupancyOracle | None = None,
                     **cost_kwargs) -> PlanAxes:
    """Choose ``BT`` and the ring depth (spec §6, rev .5).

    The realized routing tensors are measured ONCE at the finest candidate ``BT``
    and folded exactly to the coarser candidates
    (:func:`_tile_union_stats_multi` -- exact, over the realized schedule, not a
    distributional guess), every candidate ``BT`` is priced by
    :func:`plan_cost_per_token`, weighted by :func:`plan_waves`, and the cheapest is
    returned.

    **``intra_lowering`` IS NO LONGER AN AXIS** (item 6b, 2026-08-01). The consumer
    includes ``intra_panel.cuh`` unconditionally and does one intra lowering, so
    there is nothing to select; the whole crossover -- ``INTRA_LOWERINGS``,
    ``select_intra_lowering``, ``crossover_intra_lowering``,
    ``implemented_intra_lowerings`` and ``Plan.intra_lowering`` -- was
    old-consumer machinery and died with it.

    **This is the sanctioned §7.4.1 pattern and NOT a runtime density branch.** The
    result is compile-time template parameters stamped into ``Plan`` before the
    launch. Device code never sees ``K``, a saturation, or a fill; the kernel it
    runs is chosen, not steered.
    """
    for BT in bt_candidates:
        if BT not in _SUPPORTED_BT:
            raise ValueError(f"BT candidates must lie in {_SUPPORTED_BT} (spec §6), got {BT}")
    if views is None:
        views = routing_views(read_levels, write_levels, topology)
    else:
        views.require_source(read_levels, write_levels)
    # ONE stats pass over every candidate granularity: each candidate `BT` fits one
    # 32-bit token word, so each is measured from the definition rather than folded
    # from a finer one (see `_tile_union_stats_multi` derivation 2).
    # `static_density` (§2.7's ruling, fix brief #1) substitutes the closed form for
    # any side the CONFIGURATION makes full-support, so only genuinely-routed sides
    # are censused at all. It is a structural declaration, not a threshold: nothing
    # here compares a width, a density or a size.
    stats_by_bt = _tile_union_stats_multi(views, topology, BC, tuple(bt_candidates),
                                          static_density)

    if sm_count is None:
        sm_count = DEFAULT_SM_COUNT
    gram_cols = sum(spanned_moduli(topology, BC))

    # The occupancy oracle is an INPUT (coordinator ruling 2026-07-31), resolved at
    # the call site and consulted here as data. Selection therefore stays a pure
    # function of its recorded inputs -- the machine-independence §10 needs -- while
    # the ring's residency decision still uses the physical number rather than a
    # proxy the Gates proved lies.
    occupancy = resolve_occupancy(occupancy, BC=BC, d_v=topology.d_v)

    candidates: list[PlanAxes] = []
    for BT in bt_candidates:
        stats = stats_by_bt[BT]
        # ---- THE RING DEPTH IS STATED, NOT ARBITRATED (M-1, 2026-08-01) --
        # It used to be arbitrated in the cost currency, because the ring's
        # BENEFIT was a term in `plan_cost_per_token` (`STAGE_COLUMN_WORK`,
        # divided by the depth) and its COST was the residency the deeper ring
        # might give up, read from the driver's occupancy oracle. RULING A
        # deleted the term on a measurement -- the per-staged-column price is
        # EXACTLY zero, see the deletion note beside `TILE_FIXED_WORK` -- and the
        # oracle is ring-INVARIANT because the consumer's shared memory is
        # entirely compile-time. Both sides of that arbitration are therefore
        # identically equal at every depth, and a `min` over equal keys is not a
        # decision: it is a tie-break pretending to be one.
        #
        # So the depth is STATED as `COMPILED_RING_DEPTH` -- the depth the
        # binary actually compiles (`ROLA_C_RING_DEPTH`, shipped 1; the H3
        # agreement gate pins it to the ratification manifest). The earlier
        # "deepest built" statement (2) described a runtime carriage the kernel
        # read at zero sites; that plumbing is deleted, and the compile-time
        # `consumer_ring_depth()` is the sole authority on the depth run.
        # The FEASIBILITY check survives unchanged,
        # because it never was about the ring: a candidate the oracle prices at
        # 0 CTAs/SM does not fit the machine and must be refused rather than
        # divided by.
        #
        # REOPENS ON A MEASUREMENT: if §2.3's item buffers build a real ring,
        # its price is measured on the kernel that has one and this becomes an
        # arbitration again -- with a constant isolated on that kernel, never
        # this one's revived.
        #
        # THE ORACLE IS THE SOLE AUTHORITY FOR PHYSICAL FACTS (user ruling
        # extension, 2026-07-31): the cost model owns WORK -- the routing
        # statistics and the measured per-tile quanta -- and the driver owns
        # every PHYSICAL constant. There is no analytic mirror any more: it
        # modelled the deleted consumer, and the exhaustive probe measured it
        # inverting the bound it claimed.
        ring_slots = COMPILED_RING_DEPTH
        blocks = occupancy(BT, BC, topology.d_v, normalization == "global",
                           decay, mma_precision, gram_cols, ring_slots)
        if blocks == 0:
            raise ValueError(
                f"the candidate BT={BT}, BC={BC} does not "
                f"fit an SM: the occupancy oracle reports 0 CTAs/SM, so its "
                "footprint exceeds the shared memory and the launch would fail")
        cost = plan_cost_per_token(topology, BC, stats,
                                   decay=decay, **cost_kwargs)
        # ---- THE WAVE REGIME, THE ONLY PHYSICAL TERM IN THE ARGMIN --------
        # `plan_cost_per_token` is per-CTA work. What a launch takes is that
        # work times the number of WAVES the grid is serialized into, and the
        # wave count is `n_cohorts / (sm_count * blocks_per_sm)` floored at 1 --
        # every input a recorded fact, no constant. Below one wave the term is
        # identically 1 and the argmin is the pure work model (which prefers the
        # LARGER tile, because the per-tile fixed cost amortizes); above it, a
        # candidate that fits fewer CTAs per SM pays proportionally (which
        # prefers the SMALLER tile, because `BT = 32` costs a CTA at both `BC`).
        # That is the partition the §2.32 argmins exhibit, and it is why
        # `_CALIBRATION_BT`'s `(BC, decay)` key could not express the answer.
        #
        waves = plan_waves(stats.n_cohorts, blocks, sm_count)
        candidates.append(PlanAxes(
            BT=BT, stats=stats,
            cost_per_token=cost, blocks_per_sm=blocks, ring_slots=ring_slots,
            waves=waves, sm_count=sm_count,
            effective_cost_per_token=cost * waves))

    assert candidates  # bt_candidates is non-empty by construction
    # ---- §6's RESIDENCY RULE IS DELETED (item 6b, 2026-08-01) ---------------
    # It fired on `cheapest.blocks_per_sm == 1` -- the 1-CTA/SM cliff, where there is
    # no latency hiding at all. That cliff was a fact about the DELETED
    # register-resident consumer, not this one (72448 B at `BT = 32, BC = 64, d_v = 64`). An EXHAUSTIVE query of this
    # consumer's occupancy oracle over the entire launchable surface --
    # `BT(2) x BC(2) x d_v(2) x decay(2)` = 16 keys -- measures
    # `BT = 16: 3 or 4` and `BT = 32: 2`, MINIMUM OVER ALL 32 = 2. The guard can
    # never be true, so the rule was dead code and `PlanAxes.residency_override` was
    # permanently `None`. Re-pointing its fixtures could not fix that: no cell
    # exists, in the frozen set or outside it, because no INSTANTIATION exists.
    #
    # The wave term is what carries residency into the argmin now, and it does so on
    # every candidate rather than at one threshold: `cost * max(1, n_cohorts /
    # (SMs * blocks_per_sm))`. Below one wave residency correctly drops out (the
    # whole grid is resident); above it, a candidate that fits fewer CTAs per SM pays
    # proportionally.
    #
    # FALSIFIABLE IN ONE LINE, and the falsification is a MEASUREMENT rather than an
    # argument: an instantiation this consumer can launch at 1 CTA/SM.
    # `test_no_launchable_instantiation_reaches_the_one_cta_cliff` is that probe.
    return min(candidates, key=lambda c: c.effective_cost_per_token)


def select_plan(read_levels: tuple[torch.Tensor, ...],
                   write_levels: tuple[torch.Tensor, ...],
                   topology: Topology, BC: int, *, normalization: str, decay: bool,
                   mma_precision: str = "bf16",
                   amp_dtype: torch.dtype | None = None,
                   bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                   views: RoutingViews | None = None,
                   static_density: StaticDensity | None = None,
                   sm_shared_bytes: int | None = None,
                   sm_count: int | None = None,
                   occupancy: OccupancyOracle | None = None,
                   **cost_kwargs) -> tuple[Plan, PlanAxes]:
    """THE plan build: a ``Plan`` whose §6 axes are SELECTED, not hand-written.

    Returns the plan and the :class:`PlanAxes` evidence it was built from (the
    realized :class:`TileUnionStats` and the winning cost), so a caller can record
    WHY a plan is what it is next to the number it produced. Everything §6 does not
    select is passed in: ``BC`` and ``normalization`` are contract, ``mma_precision``
    is a numeric-format axis (not a cost axis, though it IS part of the
    instantiation identity and is therefore forwarded to the occupancy oracle --
    `MmaPrec` has its own register pressure), and ``warp_tile`` is the realized
    ``(1, d_v/32)`` the consumer implements -- stated here rather than defaulted so
    the driver's check has something to check.

    **Selection is per ROUTING REALIZATION, and this function does not cache.** The
    statistics are exact over the realized schedule, so a plan is only as valid as
    the routing it was measured on; reusing one across steps is a policy decision
    with a correctness-free but performance-visible consequence, and it belongs to
    the caller that knows the step structure. Nothing here hides it.
    """
    if sm_shared_bytes is None:
        sm_shared_bytes = DEFAULT_SM_SHARED_BYTES
    axes = select_plan_axes(read_levels, write_levels, topology, BC, decay=decay,
                            bt_candidates=bt_candidates, views=views,
                            static_density=static_density,
                            normalization=normalization, sm_count=sm_count,
                            mma_precision=mma_precision, occupancy=occupancy,
                            **cost_kwargs)
    plan = Plan(topology=topology, normalization=normalization, BT=axes.BT, BC=BC,
                  warp_tile=(1, max(1, topology.d_v // 32)), decay=decay,
                  mma_precision=mma_precision, amp_dtype=amp_dtype,
                  sm_shared_bytes=sm_shared_bytes, ring_slots=axes.ring_slots,
                  occupancy_blocks_per_sm=axes.blocks_per_sm)
    return plan, axes


def plan_and_schedule(read_levels: tuple[torch.Tensor, ...],
                         write_levels: tuple[torch.Tensor, ...],
                         g_read: torch.Tensor | None,
                         g_write: torch.Tensor,
                         topology: Topology, BC: int, *, normalization: str, decay: bool,
                         mma_precision: str = "bf16",
                         amp_dtype: torch.dtype | None = None,
                         bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                         static_density: StaticDensity | None = None,
                         sm_shared_bytes: int | None = None,
                         sm_count: int | None = None,
                         occupancy: OccupancyOracle | None = None,
                         **cost_kwargs) -> tuple[Plan, PlanAxes, Schedule]:
    """THE planner plane: select the §6 axes and build the §4 schedule in ONE pass.

    :func:`select_plan` and :func:`build_schedule` walk the same routing
    tensors and independently recompute the same prefix -- the ``[B,T,H,w] ->
    [BH,T,w]`` fold (a real copy), the exact-nonzero mask, and the per-``(level,
    digit)`` §4 tile bitmap. This entry point builds one :class:`RoutingViews` and
    hands it to both, so the prefix is computed once and the builder's bitmaps at
    the selected ``BT`` are the ones the selection already formed (folded exactly
    from the finest candidate, never re-reduced).

    **Nothing about the SEMANTICS moves.** The returned plan is the plan
    :func:`select_plan` returns on the same inputs, bit for bit, and the
    schedule is the schedule :func:`build_schedule` builds from it; both remain
    callable standalone and construct their own view when none is supplied. Per-step
    selection stands -- a ``RoutingViews`` lives inside one forward and is not a
    cross-step cache.
    """
    views = routing_views(read_levels, write_levels, topology)
    plan, axes = select_plan(read_levels, write_levels, topology, BC,
                                normalization=normalization, decay=decay,
                                mma_precision=mma_precision, amp_dtype=amp_dtype,
                                bt_candidates=bt_candidates, views=views,
                                static_density=static_density,
                                sm_shared_bytes=sm_shared_bytes,
                                sm_count=sm_count, occupancy=occupancy,
                                **cost_kwargs)
    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan, views=views)
    return plan, axes, schedule
