"""General per-token routing statistics primitives, exact and gradient-free.

Both functions here are consumed by :class:`~rola.routing.factors.RouteFactors`'s
``support_counts()``/``touched_bits()`` -- exact, detached facts about which digits
a routing allocation touched, derived on demand from the stored tensors rather than
computed eagerly every step (`docs/internals/DELETIONS.md`, the "eager auxiliary
statistics" row).
"""

from __future__ import annotations

import torch


def exact_support_count(values: torch.Tensor) -> torch.Tensor:
    """Exact nonzero count along the last (digit) axis: ``s_l(t)`` (spec §4.5 Tier 1).

    ``values``: ``[..., width]`` a level-side's post-activation allocation for
    one or more tokens. Returns ``[...]`` counts as the input dtype's real
    (not necessarily integer) type -- callers needing an integer count should
    ``.round().long()`` after the (only mathematically necessary) float cast
    disappears; kept float here so it composes without an extra cast when fed
    straight into ``target_count``/``slack`` arithmetic.
    """
    return (values != 0).sum(dim=-1).to(values.dtype)


def touched_bitset_update(touched: torch.Tensor, chunk_write_values: torch.Tensor) -> torch.Tensor:
    """Update an exact, detached per-digit touched-bit mask.

    ``touched``: ``[..., width]`` bool, digits touched by any write so far this
    sequence. ``chunk_write_values``: ``[..., T_chunk, width]`` this chunk's
    post-activation write allocations. A digit is "touched" the first token any
    write allocates nonzero mass to it -- this is an OR-reduction, not a sum, and is
    always detached: membership is a mechanical count, no gradient flows through it.
    """
    chunk_touched = (chunk_write_values.detach() != 0).any(dim=-2)
    return touched | chunk_touched


__all__ = [
    "exact_support_count",
    "touched_bitset_update",
]
