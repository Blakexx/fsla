"""Canonical fp64 routing reference helpers."""

from __future__ import annotations

import torch

from fsla.routing.entmax.reference import entmax, entmax_forward
from fsla.routing.projection import RouterLayout
from fsla.routing.types import EntmaxActivation, RoutingActivation, SoftmaxActivation


def _pack_token_support(support: torch.Tensor) -> torch.Tensor:
    """Pack semantic ``[..., L, branch]`` support into signed uint32-bit words."""
    tokens, branches = support.shape[-2:]
    n_words = (tokens + 31) // 32
    padding = n_words * 32 - tokens
    if padding:
        tail = torch.zeros(
            (*support.shape[:-2], padding, branches), dtype=torch.bool, device=support.device)
        support = torch.cat((support, tail), dim=-2)
    bits = (1 << torch.arange(32, device=support.device, dtype=torch.int64)).view(
        *((1,) * (support.ndim - 2)), 1, 32, 1)
    return (support.to(torch.int64).reshape(*support.shape[:-2], n_words, 32, branches) * bits).sum(
        dim=-2).to(torch.int32).contiguous()


def _activation_factors(
    logits: torch.Tensor,
    activation: RoutingActivation,
    *,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(activation, SoftmaxActivation):
        masked = logits if mask is None else logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(masked, dim=-1), None
    if isinstance(activation, EntmaxActivation):
        values = entmax(logits, activation.alpha, dim=-1, mask=mask)
        _, support, _ = entmax_forward(logits, activation.alpha, dim=-1, mask=mask)
        return values, _pack_token_support(support)
    raise TypeError(f"activation must be a routing activation, got {activation!r}")


def routing_factor_levels(
    logits: torch.Tensor,
    layout: RouterLayout,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Solve reference semantic read/write factors from the projection's flat logits ``[S, T, packed_router_width]``:
    ``[S, T, width_l]`` a write level and ``[S * G, T, width_l]`` a read level, read head ``g`` at rows ``s * G + g``,
    a tied entry its write level's."""
    if type(layout) is not RouterLayout:
        raise TypeError(f"layout must be a RouterLayout, got {type(layout).__name__}")
    if logits.ndim != 3 or logits.shape[-1] != layout.packed_router_width:
        raise ValueError(
            "routing logits must be [S, T, packed_router_width] with packed_router_width="
            f"{layout.packed_router_width}, got {tuple(logits.shape)}")

    reads, writes = [], []

    for index in range(layout.D):
        write_factor, _ = _activation_factors(
            logits[..., layout.write_columns(index)], layout.topology.write[index], mask=None)
        heads = []
        for g, entries in enumerate(layout.heads):
            columns = layout.read_columns(g, index)
            heads.append(write_factor if columns is None
                         else _activation_factors(logits[..., columns], entries[index], mask=None)[0])
        reads.append(heads[0] if len(heads) == 1 else torch.stack(heads, dim=1).flatten(0, 1))
        writes.append(write_factor)

    return tuple(reads), tuple(writes)


__all__ = ["routing_factor_levels"]
