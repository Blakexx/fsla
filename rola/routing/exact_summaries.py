# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The §4.2 routing summaries, derived EXACTLY from the leaf plane.

AN ENUMERATED REFERENCE -- ITS ONLY CALLER IS A GATE
(``tests/unit/test_routing_summaries.py``), and it is not meant to acquire a
production one. :func:`rola.routing.summaries.summarize_routing` measures the same
five fields from the FACTORIZED bound ``K_hat = prod_l (owner-slice popcount)`` at
``O(BH * n_tiles * owners_total)``; this walks the exact leaf plane at
``O(BH * n_tiles * N)`` -- the cost the shipped pass exists to avoid -- so the two can
be compared field for field. It survives under the pattern
:func:`rola.ops.decode._write_atom_bitmap` states: a SECOND, independent derivation of
an object the kernel computes, written from the definition rather than from the
kernel's structure. Deleting it would delete the gate's reference; calling it on a
shipped path would delete the gate's independence.

**INDEPENDENCE IS WHY THIS IS ITS OWN MODULE.** ``rola.routing.summaries`` carries its
own ``_token_support_words`` / ``_leaf_words`` with DIFFERENT bodies, and a shared
helper would make the gate prove only self-consistency -- the failure mode that let the
shared-router bug through. The two derivations stay textually apart.

THE REPRESENTATION is a TOKEN bitset: for one tile of ``BT <= 32`` tokens, one
``(side, level, digit)`` cell is a single ``int32`` whose bit ``t`` is "digit ``d`` of
level ``l`` is live at token ``tile*BT + t``". The leaf axis is then a BROADCAST rather
than a gather -- ``leaf = sum_l d_l * prod(widths[l+1:])`` is the row-major flattening
of the ``[w_0, ..., w_{D-1}]`` digit grid, so the per-leaf word is the AND of the ``D``
per-level word tensors each viewed with its width on its own axis, and no index is ever
materialized.
"""

from __future__ import annotations

from math import prod

import torch

#: Token bits carried by one support word. A candidate ``BT`` must fit one word.
_TOKEN_BITS_PER_WORD = 32

#: Element budget for one chunk of the leaf-word plane, in ``int32`` leaf-words per
#: side. The pass's ONE transient is the ``[rows, N]`` leaf-word tensor (``rows`` a run
#: of ``(bh, tile)`` pairs), so chunking over rows is what keeps the footprint
#: independent of ``N * BH * n_tiles`` -- the ``O(L*N)`` object §4 forbids
#: materializing. A STRUCTURAL bound under the standing "design for nc ~ 65K" rule, not
#: a tuning knob: at ``widths=(256,256)``, ``BH=8``, ``L=16384`` the full plane is 537 M
#: words and this admits it in 32 chunks.
_CENSUS_CELL_CHUNK = 1 << 24

#: §6's MMA k-extent. Mirrored from ``rola.ops.paging.MMA_K_QUANTUM`` rather than
#: imported, because importing it would give this reference a dependency on the plane
#: it is written to be independent of.
MMA_K_QUANTUM = 16


def _token_support_words(masks: tuple[torch.Tensor, ...], BT: int,
                         n_tiles: int) -> tuple[torch.Tensor, ...]:
    """Per level, ``[BH, n_tiles, width]`` ``int32``; bit ``t`` = token ``tile*BT+t`` live.

    Liveness is EXACT nonzero-ness of the amplitude (the masks are
    ``RoutingViews.read_mask``/``write_mask``, spec §1.2) -- never a threshold. The
    padded token tail (``n_tiles*BT - T`` tokens) contributes no bit, exactly as it
    contributes none to the tile bitmap, so a ragged ``L`` needs no term of its own.

    The weights carry bit 31 as ``-2**31`` and the summed powers of two are
    DISJOINT, so every partial sum lies in ``[-2**31, 2**31 - 1]`` and the ``int32``
    accumulation is exact with no wraparound to argue about.
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


def _leaf_words(level_words: tuple[torch.Tensor, ...], widths: tuple[int, ...]) -> torch.Tensor:
    """``[rows, N]`` ``int32`` leaf words from one chunk's per-level ``[rows, width]`` words.

    Identity 1 of the module docstring: the AND over levels is a pure BROADCAST,
    because the leaf id is the row-major flattening of the ``[w_0, ..., w_{D-1}]``
    digit grid. This is the ONE place the pass touches an ``N``-sized object, and it
    is the answer's own granularity -- the retired scatter pass materialized the same
    plane as an ``int16`` extremes buffer, so the term is not new.
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


def exact_routing_summaries(read_mask: tuple[torch.Tensor, ...],
                            write_mask: tuple[torch.Tensor, ...],
                            geometry, BT: int):
    """The §4.2 :class:`RoutingSummaries` struct, computed EXACTLY. The AUDITOR's twin.

    ``rola.routing.summaries.summarize_routing`` measures the same five fields from the
    FACTORIZED bound ``K_hat = prod_l (owner-slice popcount)``, at
    ``O(BH * n_tiles * owners_total)``. This computes them from the exact leaf plane at
    ``O(BH * n_tiles * N)`` -- the cost the runtime pass exists to avoid -- so the two
    can be compared field for field. That comparison is what the collision-correction
    fit regresses, and its upper-bound property is:
    ``sum_q q * hist_bound[q] >= sum_q q * hist_exact[q]`` on every cell, because
    ``K_hat >= K`` pairwise.

    **INDEPENDENCE IS THE POINT, so nothing here is shared with the pass it audits.**
    Candidacy below is formed by GATHERING each owner's fixed digits and boundary slice
    out of the per-level tile bitmaps; the production pass forms it as a successive
    OUTER PRODUCT over the fixed levels. Those are two derivations of the same set, and
    a shared helper would make the gate prove only self-consistency -- the failure mode
    that let the shared-router bug through.

    ``geometry`` is the §3 owner rectangle, passed for the same reason
    :func:`rola.routing.summaries.summarize_routing_device` takes it: this module does
    not derive it, and a second derivation would be a second authority.
    """
    from rola.routing.summaries import RoutingSummaries, _pack_leaf_bitmap

    widths = tuple(geometry.widths)
    BC = int(geometry.BC)
    # Same input contract as the production pass and the kernel: the fp32 ALLOCATIONS,
    # with liveness derived here as exact nonzero-ness (spec §1.2). A gate that fed one
    # side a pre-computed boolean mask would be making that decision for it.
    read_mask = tuple(t if t.dtype == torch.bool else t != 0 for t in read_mask)
    write_mask = tuple(t if t.dtype == torch.bool else t != 0 for t in write_mask)
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

    # The owner id decomposed by GATHER: `owner = rest * n_slices + slice`, with `rest`
    # the mixed radix of the fixed digits (level 0 most significant).
    owner_ids = torch.arange(owners_total, device=device)
    slice_of = owner_ids % n_slices
    rest = owner_ids // n_slices
    fixed_of = [(rest // max(1, prod(widths[level + 1:j]))) % widths[level]
                for level in range(j)]

    read_words = tuple(w.reshape(rows_total, -1)
                       for w in _token_support_words(read_mask, BT, n_tiles))
    write_words = tuple(w.reshape(rows_total, -1)
                        for w in _token_support_words(write_mask, BT, n_tiles))

    occupancy = torch.stack([
        torch.stack([mask.sum(dtype=torch.int64).to(torch.float64)
                     / float(BH * T * widths[level])
                     for level, mask in enumerate(side)])
        for side in (read_mask, write_mask)]).to(torch.float32)

    hist = torch.zeros(QPO + 1, dtype=torch.int64, device=device)
    n_candidate = torch.zeros((), dtype=torch.int64, device=device)
    rows_per_chunk = max(1, _CENSUS_CELL_CHUNK // N)
    for lo in range(0, rows_total, rows_per_chunk):
        hi = min(lo + rows_per_chunk, rows_total)
        rows = hi - lo
        read_lw = _leaf_words(tuple(w[lo:hi] for w in read_words), widths)
        write_lw = _leaf_words(tuple(w[lo:hi] for w in write_words), widths)
        k_count = (read_lw | write_lw).ne(0).view(rows, owners_total, BC).sum(dim=-1)

        cand = None
        for words in (read_words, write_words):
            tile_bits = [words[level][lo:hi] != 0 for level in range(len(widths))]
            side_cand = tile_bits[j].view(rows, n_slices, c).any(dim=-1)[:, slice_of]
            for level in range(j):
                side_cand = side_cand & tile_bits[level][:, fixed_of[level]]
            cand = side_cand if cand is None else (cand | side_cand)

        q = (k_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM
        for bin_index in range(QPO + 1):
            hist[bin_index] += ((q == bin_index) & cand).sum(dtype=torch.int64)
        n_candidate = n_candidate + cand.sum(dtype=torch.int64)

    # §4.1's canonical per-leaf write bitmap, in its own `bh`-outer loop so the OR is
    # a whole-slab reduction rather than a per-row scatter.
    written = torch.zeros((BH, N), dtype=torch.bool, device=device)
    tiles_per_chunk = max(1, _CENSUS_CELL_CHUNK // N)
    for bh in range(BH):
        base = bh * n_tiles
        for lo in range(0, n_tiles, tiles_per_chunk):
            hi = min(lo + tiles_per_chunk, n_tiles)
            chunk = tuple(w[base + lo:base + hi] for w in write_words)
            written[bh] |= (_leaf_words(chunk, widths) != 0).any(dim=0)

    n_pairs = rows_total * owners_total
    n_owner_slots = BH * owners_total
    return RoutingSummaries(
        occupancy=occupancy,
        quanta_hist=hist.to(torch.float32) / n_candidate.clamp(min=1).to(torch.float32),
        active_owner_frac=(
            int(written.view(BH, owners_total, BC).any(dim=-1).sum()) / n_owner_slots
            if n_owner_slots else 0.0),
        candidate_frac=int(n_candidate) / n_pairs if n_pairs else 0.0,
        written_leaf_bitmap=_pack_leaf_bitmap(written))


__all__ = ["exact_routing_summaries"]
