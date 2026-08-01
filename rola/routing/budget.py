"""Budget-conditioning arithmetic primitives (architecture §4.5, Tier 1 + usage conditioning).

Pure, chunk-granular functions with no autograd surprises: every quantity that
must be *exact* (support counts, touched bits) is computed without gradient;
every quantity that must be *differentiable* (soft usage, the surrogate
consumed by ``losses.py``) is left attached to the graph and the caller
decides when to ``.detach()``.

Scope note (recorded per the task brief's "simplest form" instruction): the
architecture text describes a conditioning vector ``(r, slack,
usage-summary)`` and a per-digit usage-bias head ``g`` without pinning an
exact functional form for either the temperature head or ``g``. This module
implements the simplest form that satisfies the stated words:

- ``r`` (target support fraction) is a per-(batch) scalar in ``(0, 1]``.
- ``target_count(t)`` is the cumulative *expected* nonzero-count budget through
  token ``t``: ``r * width * t``.
- ``usage_count(t)`` is the cumulative *exact* nonzero count through token
  ``t`` (detached -- it is a conditioning input, not a loss term itself;
  ``losses.py`` owns the differentiable surrogate that trains toward it).
- ``slack(t) = target_count(t) - usage_count(t)``, normalized by
  ``target_count`` to keep the conditioning feature O(1) regardless of
  sequence position (spec: "slack -> 0" should be a trained-on regime, which
  requires the fed value to be dimensionless, not raw counts that grow with
  ``t``).

All functions operate at **chunk granularity** (``BT``, default 32, spec
§4.5 "Operational granularity"): callers feed already-aggregated per-chunk
tensors, never per-token ones, so no semantic quantity here is silently
denominated per-chunk in a way that would make it plan-dependent -- ``r`` and
the *target fraction* are per-token throughout; only the *evaluation
schedule* (how often slack is recomputed and fed back) is chunk-granular.
"""

from __future__ import annotations

import torch

DEFAULT_CHUNK_SIZE = 32


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


def chunk_target_count(r: torch.Tensor, width: int, token_count: int) -> torch.Tensor:
    """Cumulative expected support-count budget through ``token_count`` tokens.

    ``r``: broadcastable target fraction in ``(0, 1]``. ``target_count = r *
    width * token_count`` -- the mean per-token support size ``r * width``
    accumulated over the tokens seen so far (spec §4.5: "enforces the sequence
    MEAN of per-token support size ... against a conditioned target fraction").
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if token_count < 0:
        raise ValueError(f"token_count must be >= 0, got {token_count}")
    if torch.any((r <= 0) | (r > 1)):
        raise ValueError("r must lie in (0, 1] (target support fraction)")
    return r * float(width) * float(token_count)


def chunk_slack(target_count: torch.Tensor, usage_count: torch.Tensor) -> torch.Tensor:
    """``slack(t) = target_count(t) - usage_count(t)`` (spec §4.5, raw units)."""
    return target_count - usage_count


def normalized_slack(target_count: torch.Tensor, usage_count: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Slack as a dimensionless fraction of the target, clamped to ``[-1, 1]``.

    Feeding raw (unbounded, growing-with-``t``) slack counts to a learned head
    would make its input distribution nonstationary across sequence positions;
    normalizing by the target keeps it O(1) so "slack -> 0" is a trainable
    regime rather than a value the head only ever sees at one specific ``t``.
    """
    slack = chunk_slack(target_count, usage_count)
    return (slack / (target_count.clamp(min=eps))).clamp(-1.0, 1.0)


def update_soft_usage(running_usage: torch.Tensor, chunk_write_values: torch.Tensor) -> torch.Tensor:
    """Accumulate soft per-digit usage: a differentiable prefix sum of route mass.

    ``running_usage``: ``[..., width]`` cumulative soft usage through the
    previous chunk. ``chunk_write_values``: ``[..., T_chunk, width]`` this
    chunk's post-activation write allocations. Returns the updated running
    usage (``running_usage + sum_t chunk_write_values[t]``), differentiable
    end-to-end -- this is legal under the no-``O(L*N)``-state-snapshot
    invariant because it is a route-stream prefix sum, the same shape family
    as the decay clocks already carried across chunks (architecture §4.5,
    "Usage conditioning").
    """
    return running_usage + chunk_write_values.sum(dim=-2)


def touched_bitset_update(touched: torch.Tensor, chunk_write_values: torch.Tensor) -> torch.Tensor:
    """Update an exact, detached per-digit touched-bit mask (Tier 2 bitset primitive).

    ``touched``: ``[..., width]`` bool, digits touched by any write so far this
    sequence. ``chunk_write_values``: ``[..., T_chunk, width]`` this chunk's
    post-activation write allocations. A digit is "touched" the first token any
    write allocates nonzero mass to it -- this is an OR-reduction, not a sum,
    and is always detached (spec §4.5: "exact touched bits ... as an auxiliary
    DETACHED feature").
    """
    chunk_touched = (chunk_write_values.detach() != 0).any(dim=-2)
    return touched | chunk_touched


def chunk_boundaries(token_count: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[tuple[int, int]]:
    """Split ``[0, token_count)`` into contiguous ``chunk_size``-token slices.

    The last chunk may be shorter (``token_count`` need not be a multiple of
    ``chunk_size``) -- this is a training-time conditioning granularity, not a
    kernel tiling constant, so it need not divide evenly.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [(start, min(start + chunk_size, token_count)) for start in range(0, token_count, chunk_size)]


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "chunk_boundaries",
    "chunk_slack",
    "chunk_target_count",
    "exact_support_count",
    "normalized_slack",
    "touched_bitset_update",
    "update_soft_usage",
]
