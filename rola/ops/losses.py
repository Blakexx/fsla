"""Training-side budget-conditioning losses (architecture §4.5; NOT layer/producer scope).

This module owns the two loss tiers the producer's conditioning surface
(`routing/producer_v3.py`) actuates against, but does not itself produce
routing or own any nn.Parameter -- it consumes a `V3RoutingOutput` (or its
constituent tensors) and returns loss terms plus updated dual variables, which
a training loop is responsible for adding into the total loss and carrying
forward across steps.

Tier 1 (per-token support budget, compute dial): a GECO-style dual-ascent
Lagrangian against the EXACT measured mean support size, actuated through a
differentiable Tsallis/participation-ratio surrogate (spec §4.5: "A per-token
differentiable surrogate supplies the descent direction the exact count
lacks").

Tier 2 (locality-weighted recruitment tax, memory/locality dial): a streaming
first-touch tax over touched-prefix bitsets, evaluated per realized active
write tuple (spec §4.5, Tier 2).

**Flagged simplification** (recorded per the task brief): Tier 2's exact
economics ride the deposit's own O(A_w*D) sparse Cartesian enumeration of
active write tuples, which requires kernel-level machinery this branch does
not yet have (V3 kernel Stages 3-4 are still pending, `ROLA_V3_SPEC.md` §10).
This module's `tier2_prefix_novelty_tax` is therefore a reference-grade
implementation: it enumerates active write digit tuples explicitly per
token via nonzero-mask Cartesian products (correct, auditable, and identical
in result to the eventual sparse form) rather than the production O(A_w*D)
traversal. It is appropriate for the small D/width configurations Stage 1-2
tests exercise and is NOT the kernel-side implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from rola.routing.budget import exact_support_count


@dataclass(frozen=True, slots=True)
class DualAscentLambda:
    """One Lagrange multiplier per budgeted level-side (Tier 1, GECO-style, spec §4.5)."""

    value: torch.Tensor  # scalar or [H]

    def step(self, exact_mean: torch.Tensor, target: float, *, lr: float) -> "DualAscentLambda":
        """``lambda <- relu(lambda + lr * (exact_mean - target))`` (dual ascent, detached)."""
        updated = torch.relu(self.value.detach() + lr * (exact_mean.detach() - target))
        return DualAscentLambda(updated)


def tsallis_participation_surrogate(values: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable per-token support-size surrogate: the participation ratio.

    ``values``: ``[..., width]`` a level-side's post-activation simplex
    (``sum_d values[...,d] = 1``). Returns ``[...]`` in ``[1, width]``,
    ``1 / sum_d p_d^2`` -- exactly 1 at a one-hot allocation and ``width`` at
    the uniform allocation, so it tracks the exact nonzero count's two
    endpoints while remaining differentiable everywhere entmax itself is
    (spec §4.5: "Tsallis entropy or the participation ratio ... is the
    natural choice here since entmax is itself a Tsallis-regularized
    activation, so the surrogate and the activation it shapes share one
    family"). The participation ratio is the alpha=2 Tsallis case
    (``S_2 = 1 - sum p^2``, and ``1/sum p^2`` is its reciprocal form, chosen
    over the raw entropy because it is already scaled to the same ``[1,
    width]`` units as the exact count it approximates -- no separate rescale
    is needed to compare surrogate and target).
    """
    return 1.0 / (values.pow(2).sum(dim=-1) + eps)


def tier1_support_loss(
    values: torch.Tensor,
    target_fraction: float,
    lambda_state: DualAscentLambda,
    *,
    lr: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, DualAscentLambda]:
    """One level-side's Tier 1 loss term (spec §4.5).

    ``values``: ``[B,T,H,width]`` post-activation allocation (must be
    zero-capable -- calling this on a softmax level's output is a config
    error the caller, not this function, is responsible for rejecting per
    spec's "loud config validation error, not a no-op").
    ``target_fraction``: the conditioned target ``r`` (or a fixed
    hyperparameter target when not sampled per-sequence), a fraction of
    ``width``.

    Returns ``(loss, exact_mean_support, updated_lambda_state)``:
    - ``loss = lambda.detach() * surrogate.mean()`` -- the GGECO-style term a
      training loop adds to the total loss; the multiplier is detached so the
      surrogate's own gradient, not the multiplier's, trains the router.
    - ``exact_mean_support``: the exact (detached) per-sequence mean nonzero
      count, ``O(L*width)`` (spec: "free") -- this is what should be reported
      /plotted against the target, not the surrogate.
    - ``updated_lambda_state``: the dual-ascent step against the EXACT mean,
      to carry into the next training step.
    """
    width = values.shape[-1]
    target = target_fraction * width
    surrogate = tsallis_participation_surrogate(values)
    exact_mean = exact_support_count(values).mean()
    loss = lambda_state.value.detach() * surrogate.mean()
    updated = lambda_state.step(exact_mean, target, lr=lr)
    return loss, exact_mean, updated


@dataclass(frozen=True, slots=True)
class TouchedPrefixState:
    """Per-sequence touched-prefix bitsets across every routing depth (Tier 2, spec §4.5).

    ``bits[l]``: ``[B,H,prefix_count_l]`` bool, where ``prefix_count_l =
    prod(widths[0..l])`` -- one bit per distinct prefix ``(d_0..d_l)``, not
    per full leaf. Always detached (spec: exact touched bits are a mechanical
    count, no gradient flows through membership itself).
    """

    bits: tuple[torch.Tensor, ...]

    @staticmethod
    def init(widths: tuple[int, ...], *, B: int, H: int, device) -> "TouchedPrefixState":
        prefix_counts = []
        running = 1
        for width in widths:
            running *= width
            prefix_counts.append(running)
        return TouchedPrefixState(
            bits=tuple(torch.zeros(B, H, count, dtype=torch.bool, device=device) for count in prefix_counts))


def _active_write_tuples(write_levels: tuple[torch.Tensor, ...], b: int, h: int, t: int) -> list[tuple[int, ...]]:
    """Enumerate one token's active write digit tuples via a nonzero-mask Cartesian product.

    Reference-grade (module docstring): correct and auditable, not the
    production O(A_w*D) sparse traversal.
    """
    per_level_digits = [
        torch.nonzero(write_levels[level][b, t, h], as_tuple=False).flatten().tolist()
        for level in range(len(write_levels))
    ]
    return list(itertools.product(*per_level_digits))


def tier2_prefix_novelty_tax(
    write_levels: tuple[torch.Tensor, ...],
    touched: TouchedPrefixState,
    weights: tuple[float, ...],
) -> tuple[torch.Tensor, TouchedPrefixState]:
    """Streaming first-touch novelty tax over one chunk (spec §4.5, Tier 2).

    ``write_levels``: ``tuple[[B,T,H,width_l], ...]``, one full chunk's
    post-activation write allocations (same contract as the producer's
    ``write_levels`` output, sliced to a chunk).
    ``weights``: ``w_l`` per depth, config (default uniform recovers a flat
    first-touch tax; bytes-weighted is the memory-slope control -- spec).

    Returns ``(tax, updated_touched)`` where ``tax`` is ``[B,H]``, summed over
    the chunk's tokens: for each active write tuple, ``sum_l w_l *
    1[prefix untouched] * (realized amplitude product through depth l)``.
    """
    D = len(write_levels)
    if len(weights) != D:
        raise ValueError(f"weights must have D={D} entries, got {len(weights)}")
    B, T, H, _ = write_levels[0].shape
    widths = tuple(level.shape[-1] for level in write_levels)

    bits = [b.clone() for b in touched.bits]
    tax = torch.zeros(B, H, device=write_levels[0].device, dtype=write_levels[0].dtype)

    for b in range(B):
        for h in range(H):
            for t in range(T):
                for digits in _active_write_tuples(write_levels, b, h, t):
                    amplitude = write_levels[0].new_ones(())
                    prefix_index = 0
                    for level, digit in enumerate(digits):
                        amplitude = amplitude * write_levels[level][b, t, h, digit]
                        prefix_index = prefix_index * widths[level] + digit
                        if not bool(bits[level][b, h, prefix_index]):
                            tax[b, h] = tax[b, h] + weights[level] * amplitude
                            bits[level][b, h, prefix_index] = True

    return tax, TouchedPrefixState(bits=tuple(bits))


__all__ = [
    "DualAscentLambda",
    "TouchedPrefixState",
    "tier1_support_loss",
    "tier2_prefix_novelty_tax",
    "tsallis_participation_surrogate",
]
