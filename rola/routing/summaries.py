"""The producer-epilogue ROUTING SUMMARY pass (docs/internals/summaries.md).

This is the ONE statistic the runtime reads about the realized routing. It replaces the exact
per-``(owner, tile)`` census in the hot path: the census asks its question of every
``(tile, owner, leaf)`` cell, and this asks it of every ``(tile, owner)`` cell --
``O(BH * n_tiles * owners_total)``, which is the census's cost divided by ``BC``.

Symbols, defined once and used unqualified: ``L``/``T`` tokens; ``N`` leaves; ``D``
levels; ``b_l`` level widths; ``BC`` leaves per owner block; ``BT`` token tile; ``BH``
= batch x heads; ``MMA_K_QUANTUM = 16``; ``QPO = BC / MMA_K_QUANTUM`` quanta per
owner; ``K(o,t)`` the union leaf count of owner ``o`` over tile ``t``.

The contract (frozen fields), the K_hat upper-bound argument, the cost
statement, the kernel-vs-reference relationship and the synchronization budget
live in docs/internals/summaries.md (Part C).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod

import torch

#: §6's MMA k-extent, and therefore the quantum every gathered extent is rounded up
#: to. Mirrored from ``rola.ops.paging.MMA_K_QUANTUM`` rather than imported: this
#: module is deliberately free of any consumer-side import (see
#: :func:`summarize_routing`'s ``geometry`` argument for why), and it asserts the two agree
#: against the geometry it is handed, so a drift is a failure and not a silent one.
MMA_K_QUANTUM = 16

#: Token bits carried by one support word. A ``BT`` above this cannot be represented
#: and is refused rather than silently taking a second code path.
_TOKEN_BITS_PER_WORD = 32

#: Bits packed per ``int32`` of :attr:`RoutingSummaries.written_leaf_bitmap`.
_BITS_PER_WORD = 32

#: Element budget for one chunk of the ``[rows, owners_total]`` quanta plane, in
#: elements. STRUCTURAL: the pass's one transient is that plane (``rows`` a run of
#: ``(bh, tile)`` pairs), so chunking over rows keeps the footprint independent of
#: ``owners_total * BH * n_tiles``.
_QUANTA_CHUNK_ELEMS = 1 << 24

#: Element budget for one chunk of the ``[rows, N]`` leaf-word plane. Same rule, one
#: granularity finer -- this is the ``O(L*N)`` object §4 forbids materializing whole.
_LEAF_CHUNK_ELEMS = 1 << 24

#: Environment switch for the §4.2 underflow assertion. Off by default: it costs a
#: host read (the only one this module can issue outside its single transfer), and a
#: declaration that has gone false mis-PRICES a plan rather than mis-executing one
#: today -- but under §5.4's factored feeder the same declaration becomes
#: execution-affecting, which is why the check exists at all.
DEBUG_ASSERTS_ENV = "ROLA_DEBUG_ASSERTS"


@dataclass(frozen=True, slots=True)
class RoutingSummaries:
    """§4.2's contact struct, verbatim. Produced by the producer epilogue.

    Field-by-field, with the definition each one is measured to:

    ``occupancy``
        ``[2, D]`` float. ``occupancy[side, l]`` is the mean over ``(bh, token)`` of
        ``popcount(support word of level l) / b_l`` -- the per-token expected fraction
        of that level's digits that are live. Side 0 is READ, side 1 is WRITE. This is
        the input to §4.4's saturation trigger, and the ONLY place a "dense" claim can
        be checked against the data.

    ``quanta_hist``
        ``[QPO + 1]`` float, summing to 1 over CANDIDATE ``(owner, tile)`` pairs (or to
        0 when there are none). Bin ``q`` is the fraction of candidate pairs with
        ``ceil(K_hat / MMA_K_QUANTUM) == q``. A HISTOGRAM and not a mean, because
        ``ceil`` is nonlinear and the ``q = 0`` mass -- the pairs whose union is
        empty -- is a distribution fact the mean cannot carry.

    ``active_owner_frac``
        Mean over ``bh`` of the fraction of owners with ANY write in the sequence,
        condensed from ``written_leaf_bitmap`` (§4.1's derived owner granularity), so
        the two cannot disagree.

    ``candidate_frac``
        The realized share of ``(owner, tile)`` pairs the §4 tile bitmap admits, at the
        ``BT`` this object was measured at. A first-class term of the per-token cost --
        not a correction to one -- because a pair outside the bitmap is never
        enumerated by any CTA and costs nothing at all.

    ``written_leaf_bitmap``
        §4.1's CANONICAL TRUTH: ``[BH, ceil(N/32)]`` ``int32``, bit ``leaf`` of ``bh``
        set iff some token of the sequence writes that leaf, in canonical leaf order,
        as the union over tokens of the EXACT write support. Every other granularity
        (atom-level for §4.5's paging, owner-level for reporting) is CONDENSED from
        this and never stored as truth.
    """

    occupancy: torch.Tensor
    quanta_hist: torch.Tensor
    active_owner_frac: float
    candidate_frac: float
    written_leaf_bitmap: torch.Tensor

    def atom_bitmap(self, N: int, quantum: int = MMA_K_QUANTUM) -> torch.Tensor:
        """§4.1's ATOM-LEVEL condensation: OR over each ``quantum``-leaf group.

        ``[BH, N // quantum]`` bool. This is what §4.5's paging admission and its
        planned-exact pool sizing key on, and it is COMPUTED here rather than stored
        as a second truth -- §4.1 is explicit that the per-leaf bitmap is the only
        thing that is canonical.
        """
        if N % quantum:
            raise ValueError(
                f"the atom condensation groups leaves in runs of {quantum} (the MMA "
                f"k-quantum), so it needs {quantum} | N; got N={N}")
        bits = unpack_leaf_bitmap(self.written_leaf_bitmap, N)
        return bits.view(bits.shape[0], N // quantum, quantum).any(dim=-1)


def unpack_leaf_bitmap(words: torch.Tensor, N: int) -> torch.Tensor:
    """``[BH, n_words]`` int32 -> ``[BH, N]`` bool. Inverse of the packing below."""
    as_u64 = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(_BITS_PER_WORD, device=words.device, dtype=torch.int64)
    bits = ((as_u64.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
    return bits.reshape(*words.shape[:-1], -1)[..., :N]


def _pack_leaf_bitmap(flags: torch.Tensor) -> torch.Tensor:
    """Pack a trailing boolean axis into little-endian ``int32`` bit patterns.

    This module is the producer epilogue and must not import a consumer (see
    :func:`summarize_routing`'s ``geometry`` argument). Ten lines of bit arithmetic
    with a gate on each side is the cheaper of the two couplings.
    """
    *lead, n = flags.shape
    n_words = (n + _BITS_PER_WORD - 1) // _BITS_PER_WORD
    padded = torch.zeros((*lead, n_words * _BITS_PER_WORD), dtype=torch.bool,
                         device=flags.device)
    padded[..., :n] = flags
    grouped = padded.reshape(*lead, n_words, _BITS_PER_WORD).to(torch.int64)
    weights = torch.ones((), dtype=torch.int64, device=flags.device) << torch.arange(
        _BITS_PER_WORD, device=flags.device, dtype=torch.int64)
    words = (grouped * weights).sum(dim=-1)
    words = torch.where(words >= (1 << 31), words - (1 << 32), words)
    return words.to(torch.int32)


def _token_support_words(masks: tuple[torch.Tensor, ...], BT: int,
                         n_tiles: int) -> tuple[torch.Tensor, ...]:
    """Per level, ``[BH, n_tiles, b_l]`` ``int32``; bit ``t`` = token ``tile*BT + t`` live.

    Liveness is EXACT nonzero-ness of the amplitude, never a threshold: entmax's exact
    support zeros are the scheduling authority (spec §1.2). The padded token tail
    contributes no bit, so a ragged ``L`` needs no term of its own.

    The weights carry bit 31 as ``-2**31`` and the summed powers of two are DISJOINT,
    so every partial sum lies in ``[-2**31, 2**31 - 1]`` and the ``int32`` accumulation
    is exact with no wraparound to argue about.
    """
    if not 1 <= BT <= _TOKEN_BITS_PER_WORD:
        raise ValueError(
            f"BT must be in [1, {_TOKEN_BITS_PER_WORD}] to carry one token bit each, got "
            f"{BT}. The summary pass represents a tile as ONE token word; a wider BT is a "
            "spec change that must widen the word, not a runtime condition to branch on.")
    BH, T, _ = masks[0].shape
    device = masks[0].device
    pad = n_tiles * BT - T
    weights = torch.ones((), dtype=torch.int64, device=device) << torch.arange(
        BT, device=device, dtype=torch.int64)
    weights = torch.where(weights >= (1 << 31), weights - (1 << 32), weights)
    weights = weights.to(torch.int32).view(1, 1, BT, 1)
    out = []
    for mask in masks:
        if pad:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
        tiled = mask.reshape(BH, n_tiles, BT, mask.shape[-1])
        out.append((tiled * weights).sum(dim=2, dtype=torch.int32))
    return tuple(out)


def _fixed_level_product(per_level: list[torch.Tensor]) -> torch.Tensor:
    """``[rows, prod(widths[:j])]`` from one ``[rows, b_l]`` factor per FIXED level.

    The owner id's fixed part is ``rest = sum_{l<j} d_l * prod(widths[l+1:j])`` -- a
    row-major mixed radix with level 0 most significant -- so the per-owner factor is
    the successive outer product taken in level order. Level 0 first is what makes the
    flattened index that radix and not its reverse.
    """
    out = None
    for factor in per_level:
        if out is None:
            out = factor
        else:
            out = (out.unsqueeze(-1) * factor.unsqueeze(1)).reshape(factor.shape[0], -1)
    return out


def _leaf_words(level_words: tuple[torch.Tensor, ...],
                widths: tuple[int, ...]) -> torch.Tensor:
    """``[rows, N]`` ``int32`` leaf words from one chunk's ``[rows, b_l]`` token words.

    The AND over levels is a pure BROADCAST, because the leaf id is the row-major
    flattening of the ``[b_0, ..., b_{D-1}]`` digit grid: no index is materialized.
    Bit ``t`` of the leaf word is "every level's digit for this leaf is live at token
    ``tile*BT + t``", which is exactly the per-token write support.
    """
    D = len(widths)
    rows = level_words[0].shape[0]
    out = None
    for level, words in enumerate(level_words):
        shape = [1] * D
        shape[level] = widths[level]
        block = words.view(rows, *shape)
        out = block if out is None else out & block
    return out.reshape(rows, -1)


@dataclass(frozen=True, slots=True)
class _DeviceSummaries:
    """Everything §4.2 needs, still on device. See :func:`summarize_routing_device`."""

    occupancy: torch.Tensor
    quanta_hist: torch.Tensor
    written_leaf_bitmap: torch.Tensor
    #: 0-dim int64: candidate ``(owner, tile)`` pairs, and owners with any write.
    n_candidate: torch.Tensor
    n_written_owners: torch.Tensor
    #: Host-side denominators (pure arithmetic in the geometry; no measurement).
    n_pairs: int
    n_owner_slots: int


def summarize_routing_reference_device(read_mask: tuple[torch.Tensor, ...],
                             write_mask: tuple[torch.Tensor, ...],
                             geometry, BT: int, *,
                             gamma: float = 1.0) -> _DeviceSummaries:
    """THE fused pass. Device-resident end to end; issues ZERO host synchronizations.

    ``read_mask``/``write_mask`` are the folded ``[BH, T, b_l]`` boolean supports, one
    per level -- exact nonzero-ness of the routing amplitude, which is what
    ``RoutingViews`` already holds and what the producer has in hand at its epilogue.

    ``geometry`` is the §3 owner rectangle
    (:class:`rola.routing.geometry.OwnerGeometry`), passed as an ARGUMENT rather than
    derived here. That is deliberate: the rectangle derivation has exactly one owner,
    and a second derivation of it would be a second authority.

    ``gamma`` is the cross-level collision correction, an argument for the
    same one-owner reason. **It is applied HERE, per ``(owner, tile)``, BEFORE the
    quantization, and that placement is the whole content of the constant.**
    ``ceil`` is nonlinear and the ``q = 0`` mass is a distribution fact the mean
    cannot carry, so scaling the MEAN afterwards is a different and strictly less
    conservative correction. At the shipped default ``1.0`` this is bit-identically inert -- the
    multiply is skipped, not merely a no-op -- so a run with no fit behaves exactly as
    if the hook did not exist.
    """
    widths = tuple(geometry.widths)
    BC = int(geometry.BC)
    if len(read_mask) != len(widths) or len(write_mask) != len(widths):
        raise ValueError(f"one allocation per level per side required (D={len(widths)})")
    # The reference takes the same fp32 ALLOCATIONS the kernel does and derives liveness
    # itself, so the two are gated on identical inputs: a fixture that handed one side a
    # pre-computed boolean mask would be making the `!= 0` decision for it. Exact
    # nonzero-ness, never a threshold (spec §1.2).
    read_mask = tuple(t if t.dtype == torch.bool else t != 0 for t in read_mask)
    write_mask = tuple(t if t.dtype == torch.bool else t != 0 for t in write_mask)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            f"the collision correction gamma must lie in (0, 1]: K_hat is an UPPER bound on K and the "
            f"correction can only shrink it toward the truth, never past it. Got {gamma!r}.")
    if BC % MMA_K_QUANTUM:
        raise ValueError(
            f"BC={BC} must be a whole number of MMA_K_QUANTUM={MMA_K_QUANTUM} atoms "
            "(spec §1b: every legal BC is a whole number of atoms by construction)")
    j = geometry.first_spanned_level
    c = geometry.slice_digits
    n_slices = geometry.n_slices
    owners_total = geometry.owners_total
    N = prod(widths)
    QPO = BC // MMA_K_QUANTUM

    BH, T, _ = read_mask[0].shape
    device = read_mask[0].device
    n_tiles = (T + BT - 1) // BT
    rows_total = BH * n_tiles

    # ---- (a) per-(side, level) occupancy -----------------------------------
    # One reduction per cell over the SAME masks the words are packed from. The
    # denominator is host arithmetic, so no measurement rides on it.
    occupancy = torch.stack([
        torch.stack([mask.sum(dtype=torch.int64).to(torch.float64)
                     / float(BH * T * widths[level])
                     for level, mask in enumerate(side)])
        for side in (read_mask, write_mask)]).to(torch.float32)

    read_words = tuple(w.reshape(rows_total, -1)
                       for w in _token_support_words(read_mask, BT, n_tiles))
    write_words = tuple(w.reshape(rows_total, -1)
                        for w in _token_support_words(write_mask, BT, n_tiles))

    # ---- (b) the quanta histogram and the candidate fraction ----------------
    # The u WORDS are `words != 0`: a bitmap over DIGITS, which is the form the
    # owner-slice restriction needs (a popcount cannot be restricted to a sub-range).
    # They never leave this loop.
    hist = torch.zeros(QPO + 2, dtype=torch.int64, device=device)
    n_candidate = torch.zeros((), dtype=torch.int64, device=device)
    rows_per_chunk = max(1, _QUANTA_CHUNK_ELEMS // max(1, owners_total))
    for lo in range(0, rows_total, rows_per_chunk):
        hi = min(lo + rows_per_chunk, rows_total)
        rows = hi - lo
        u_read = [w[lo:hi] != 0 for w in read_words]
        u_write = [w[lo:hi] != 0 for w in write_words]
        u_union = [r | w for r, w in zip(u_read, u_write)]

        # Levels BELOW the boundary are spanned whole by every owner, so their
        # restricted popcount is the level's own and is owner-INVARIANT.
        inner = torch.ones(rows, dtype=torch.int64, device=device)
        for level in range(j + 1, len(widths)):
            inner = inner * u_union[level].sum(dim=-1, dtype=torch.int64)
        # The BOUNDARY level: each owner holds one aligned run of `c` digits.
        slice_pc = u_union[j].view(rows, n_slices, c).sum(dim=-1, dtype=torch.int64)
        # FIXED levels: the owner holds ONE digit of each, so the factor is 0 or 1 and
        # multiplies K_hat by exactly the token-validity of that digit.
        fixed = _fixed_level_product(
            [u_union[level].to(torch.int64) for level in range(j)])
        k_hat = slice_pc * inner.unsqueeze(-1)                     # [rows, n_slices]
        if fixed is not None:
            k_hat = fixed.unsqueeze(-1) * k_hat.unsqueeze(1)       # [rows, R, n_slices]
        k_hat = k_hat.reshape(rows, owners_total)

        # §4 CANDIDACY, per side and then unioned. It is NOT the union word's
        # candidacy: `cand = cand_read OR cand_write`, and taking the OR of the two
        # sides' words first would admit an owner whose read digit and write digit are
        # live at DIFFERENT levels, which no side ever visits.
        cand = None
        for u_side in (u_read, u_write):
            side_cand = u_side[j].view(rows, n_slices, c).any(dim=-1)
            fixed_side = _fixed_level_product([u_side[level] for level in range(j)])
            if fixed_side is not None:
                side_cand = fixed_side.unsqueeze(-1) & side_cand.unsqueeze(1)
            side_cand = side_cand.reshape(rows, owners_total)
            cand = side_cand if cand is None else (cand | side_cand)

        if gamma != 1.0:
            # Per (owner, tile), before the ceil. `floor(gamma * K_hat)` keeps the
            # result an integer count of LEAVES, which is what the quantum divides.
            k_hat = (k_hat.to(torch.float64) * gamma).floor().to(torch.int64)
        q = (k_hat + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM         # in [0, QPO]
        # Non-candidates are parked in the sentinel bin `QPO + 1` so they cannot
        # collide with the genuinely-empty candidates in bin 0. `minlength` keeps the
        # shape static, so this
        # is sync-free.
        binned = torch.where(cand, q, torch.full_like(q, QPO + 1))
        # `torch.bincount` is NOT usable here: its CUDA path reads the input's max to
        # size its output, which is a host synchronization per chunk -- exactly what
        # this pass's budget forbids. `QPO + 2 <= 6` equality-reductions over a tensor
        # that is already resident cost less than that sync would, and the shape is
        # static by construction.
        hist = hist + torch.stack(
            [(binned == b).sum(dtype=torch.int64) for b in range(QPO + 2)])
        n_candidate = n_candidate + cand.sum(dtype=torch.int64)

    # ---- (c) §4.1's canonical per-leaf WRITE bitmap -------------------------
    # The union over tokens of the EXACT write support: a leaf is written iff at some
    # token EVERY level's digit for it is live on the write side. That is an OR of
    # ANDs and cannot be read off any per-owner statistic, which is why this term is
    # `O(BH * n_tiles * N)` while everything above it is `O(BH * n_tiles * owners)`.
    written = torch.zeros((BH, N), dtype=torch.bool, device=device)
    tiles_per_chunk = max(1, _LEAF_CHUNK_ELEMS // max(1, N))
    for bh in range(BH):
        base = bh * n_tiles
        for lo in range(0, n_tiles, tiles_per_chunk):
            hi = min(lo + tiles_per_chunk, n_tiles)
            chunk = tuple(w[base + lo:base + hi] for w in write_words)
            written[bh] |= (_leaf_words(chunk, widths) != 0).any(dim=0)

    # §4.1's derived OWNER granularity, condensed from the canonical bitmap rather
    # than measured a second time -- so the two cannot disagree.
    n_written_owners = written.view(BH, owners_total, BC).any(dim=-1).sum(dtype=torch.int64)

    # The histogram is published as FRACTIONS of the candidate population, on device:
    # the division by a device counter is what keeps this side of the pass sync-free.
    # The sentinel bin is dropped here, which is also what makes the published bins
    # sum to 1 rather than to the candidate fraction.
    denom = n_candidate.clamp(min=1).to(torch.float32)
    quanta_hist = hist[:QPO + 1].to(torch.float32) / denom

    return _DeviceSummaries(
        occupancy=occupancy, quanta_hist=quanta_hist,
        written_leaf_bitmap=_pack_leaf_bitmap(written),
        n_candidate=n_candidate, n_written_owners=n_written_owners,
        n_pairs=rows_total * owners_total, n_owner_slots=BH * owners_total)


def summarize_routing_reference(read_mask: tuple[torch.Tensor, ...],
                                write_mask: tuple[torch.Tensor, ...],
                                geometry, BT: int, *,
                                density=None, gamma: float = 1.0,
                                debug: bool | None = None) -> RoutingSummaries:
    """§4.2's struct, with ONE D2H transfer and no other host synchronization.

    ``density`` is the configuration's :class:`~rola.routing.density.StaticDensity`
    declaration; when it is supplied AND debug assertions are on, the declaration is
    checked against the realized occupancy this pass has already reduced (see below).
    ``debug`` defaults to the ``ROLA_DEBUG_ASSERTS`` environment switch.

    **THE MEASURED ``BT`` IS THE CALLER'S CHOICE AND IT IS NOT FREE.** Every field
    below except ``occupancy`` and ``written_leaf_bitmap`` is a statistic OF A TILING,
    so it is measured at one ``BT``. §4.2 specifies the layer's call at the finest
    candidate; a caller that needs the statistic at another granularity calls again,
    which costs a second pass and is stated rather than hidden. See the findings note
    on the interaction between this and §4.3's joint ``(BC, BT)`` argmin.
    """
    device_summaries = summarize_routing_reference_device(
        read_mask, write_mask, geometry, BT, gamma=gamma)
    # THE ONE TRANSFER. Both host-side scalars ride it; reading them separately would
    # be two synchronizations for one answer.
    n_candidate, n_written_owners = torch.stack(
        [device_summaries.n_candidate, device_summaries.n_written_owners]).cpu().tolist()

    summaries = RoutingSummaries(
        occupancy=device_summaries.occupancy,
        quanta_hist=device_summaries.quanta_hist,
        active_owner_frac=(n_written_owners / device_summaries.n_owner_slots
                           if device_summaries.n_owner_slots else 0.0),
        candidate_frac=(n_candidate / device_summaries.n_pairs
                        if device_summaries.n_pairs else 0.0),
        written_leaf_bitmap=device_summaries.written_leaf_bitmap)

    if density is not None and (debug if debug is not None
                                else os.environ.get(DEBUG_ASSERTS_ENV, "") not in ("", "0")):
        _assert_declared_density_is_realized(summaries, density)
    return summaries


def summarize_routing(read_amps: tuple[torch.Tensor, ...],
                      write_amps: tuple[torch.Tensor, ...],
                      geometry, BT: int, *,
                      density=None, gamma: float = 1.0,
                      debug: bool | None = None) -> RoutingSummaries:
    """THE runtime pass: one CUDA launch, one D2H transfer, zero host synchronizations.

    ``read_amps``/``write_amps`` are the producer's FOLDED per-level allocations,
    ``[BH, T, b_l]`` fp32 and contiguous -- exactly what ``RoutingViews.read_folded``
    holds. Liveness is EXACT nonzero-ness of the amplitude (spec §1.2), computed inside
    the kernel, so nothing materializes a boolean mask on the way in.

    ``geometry`` is the §3 owner rectangle, passed rather than derived for the reason
    :func:`summarize_routing_reference_device` gives: the rectangle has exactly one
    owner, and a second derivation of it would be a second authority.

    **THERE IS NO CPU PATH AND NO FALLBACK.** A call without the extension, or on CPU
    tensors, raises. :func:`summarize_routing_reference` exists and computes the same
    thing, but it is the GATE's reference and not this function's understudy -- reaching
    for it here would be exactly the silent-degradation this codebase refuses.
    """
    from rola.ops._ext import extension

    widths = tuple(geometry.widths)
    BC = int(geometry.BC)
    _validate(widths, BC, BT, gamma)
    if not read_amps[0].is_cuda:
        raise RuntimeError(
            "the routing summary pass is a CUDA kernel (csrc/rola/src/summaries.cu) and "
            "has no CPU path by design. `summarize_routing_reference` computes the same "
            "five fields in torch, but it is the enumerated REFERENCE the kernel is "
            "gated against, not a fallback: calling it here would make an unbuilt "
            "extension silently change which code decides a plan.")

    BH, T = read_amps[0].shape[0], read_amps[0].shape[1]
    N = prod(widths)
    device = read_amps[0].device
    QPO = BC // MMA_K_QUANTUM
    zeros = lambda n, dt: torch.zeros(n, dtype=dt, device=device)   # noqa: E731
    occupancy = zeros(2 * len(widths), torch.int64)
    histogram = zeros(QPO + 1, torch.int64)
    candidate_count = zeros(1, torch.int64)
    owner_count = zeros(1, torch.int64)
    owners_total = N // BC
    leaf_bitmap = zeros((BH, (N + 31) // 32), torch.int32)
    owner_bitmap = zeros((BH, (owners_total + 31) // 32), torch.int32)

    extension().rola_routing_summaries(
        [t.contiguous() for t in read_amps], [t.contiguous() for t in write_amps],
        list(widths), BT, BC, geometry.first_spanned_level, geometry.slice_digits,
        geometry.n_slices,
        occupancy, histogram, candidate_count, owner_count,
        leaf_bitmap, owner_bitmap)

    # THE ONE TRANSFER. Both host-side scalars ride it; reading them separately would be
    # two synchronizations for one answer.
    n_candidate, n_written_owners = torch.stack(
        [candidate_count[0], owner_count[0]]).cpu().tolist()
    n_pairs = BH * ((T + BT - 1) // BT) * owners_total
    n_owner_slots = BH * owners_total

    summaries = RoutingSummaries(
        occupancy=(occupancy.view(2, len(widths)).to(torch.float32)
                   / torch.tensor([float(BH * T * w) for w in widths], device=device)),
        quanta_hist=histogram.to(torch.float32) / max(1, n_candidate),
        active_owner_frac=n_written_owners / n_owner_slots if n_owner_slots else 0.0,
        candidate_frac=n_candidate / n_pairs if n_pairs else 0.0,
        written_leaf_bitmap=leaf_bitmap)

    if density is not None and _debug_enabled(debug):
        _assert_declared_density_is_realized(summaries, density)
    return summaries


def _debug_enabled(debug: bool | None) -> bool:
    return (debug if debug is not None
            else os.environ.get(DEBUG_ASSERTS_ENV, "") not in ("", "0"))


def _validate(widths: tuple[int, ...], BC: int, BT: int, gamma: float) -> None:
    """The three refusals both paths share. Each is a SPEC change, not a runtime branch."""
    if not 1 <= BT <= _TOKEN_BITS_PER_WORD:
        raise ValueError(
            f"BT must be in [1, {_TOKEN_BITS_PER_WORD}] to carry one token bit each, got "
            f"{BT}. The summary pass represents a tile as ONE token word; a wider BT is a "
            "spec change that must widen the word, not a runtime condition to branch on.")
    if BC % MMA_K_QUANTUM:
        raise ValueError(
            f"BC={BC} must be a whole number of MMA_K_QUANTUM={MMA_K_QUANTUM} atoms "
            "(spec §1b: every legal BC is a whole number of atoms by construction)")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            f"the collision correction gamma must lie in (0, 1]: K_hat is an UPPER bound on K and the "
            f"correction can only shrink it toward the truth, never past it. Got {gamma!r}.")
    del widths


def _assert_declared_density_is_realized(summaries: RoutingSummaries, density) -> None:
    """§4.2's UNDERFLOW assertion: a declared-dense cell whose realization has zeros.

    **This is the one place the check can live**, because it is the one place the masks
    are already reduced to the quantity it needs. Everywhere else it would be a second
    pass over the routing.

    **What it catches, MEASURED rather than hypothetical.** A ``softmax`` read side is
    declared dense from the activation registry's ``exact_zeros`` column -- softmax has
    no exact structural zero, mathematically. Float underflow does not care: a logit
    far enough below the row max makes ``exp`` round to exactly ``0``. Gate 2
    (2026-08-01, item 3e) measured a parity-shaped layer staying true through
    ``logit_gain in {1.0, 8.0, 16.0}`` and going FALSE at ``logit_gain = 32.0``, with
    realized read-side zero fractions of 0.002056 and 0.001883 across the two levels.

    **Why it is debug-only, and why it exists anyway.** TODAY a false declaration can
    only MIS-PRICE: nothing the kernel executes reads it, so the launch runs on the
    realized routing. Under §5.4's factored feeder the same declaration selects the DENSE
    feeder variant, which collapses the intersection -- at which point a false
    declaration is a correctness defect, not a rounding one. The assertion therefore
    changes character with the feeder and is written before it, not after.

    It costs a host read, which is why it is off by default and why it is here rather
    than folded into the single transfer: a check nobody enabled must not lengthen the
    critical path of every forward.
    """
    realized = summaries.occupancy.cpu()
    for side, duty, declared in ((0, "read", density.read), (1, "write", density.write)):
        for level, is_dense in enumerate(declared):
            if not is_dense:
                continue
            fraction = float(realized[side, level])
            if fraction != 1.0:
                raise AssertionError(
                    f"the configuration declares the {duty} side of level {level} DENSE "
                    f"(the activation registry's `exact_zeros` column says its activation "
                    f"cannot produce an exact structural zero), but the realized routing has "
                    f"occupancy {fraction!r} -- a realized zero fraction of "
                    f"{1.0 - fraction!r}. The usual cause is FLOAT UNDERFLOW, not a wrong "
                    "registry entry: a softmax logit far enough below the row max makes exp "
                    "round to exactly zero (MEASURED to begin at logit_gain = 32.0 on a "
                    "parity-shaped layer, Gate 2 item 3e). Today this mis-prices a plan; "
                    "under the factored feeder the dense variant collapses the intersection "
                    "and the same declaration becomes execution-affecting, which is what "
                    "this assertion is written for.")


__all__ = [
    "DEBUG_ASSERTS_ENV",
    "MMA_K_QUANTUM",
    "RoutingSummaries",
    "summarize_routing",
    "summarize_routing_reference",
    "summarize_routing_reference_device",
    "unpack_leaf_bitmap",
]
