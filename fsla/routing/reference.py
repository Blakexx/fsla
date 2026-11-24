"""Canonical fp64 routing reference helpers."""

from __future__ import annotations

import torch

from fsla.routing.entmax.reference import entmax, entmax_forward
from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    RoutingActivation,
    SoftmaxActivation,
    TiedRouting,
)


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
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Solve reference semantic read/write factors without materializing a descriptor."""
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if read_logits.ndim != 3 or read_logits.shape[-1] != routing.packed_logit_width:
        raise ValueError(
            "routing logits must be [BH, T, packed_logit_width] with packed_logit_width="
            f"{routing.packed_logit_width}, got {tuple(read_logits.shape)}")

    reads, writes = [], []

    for index, level in enumerate(routing.levels):
        span = routing.level_logit_slice(index)
        read_level = read_logits[..., span]
        write_level = write_logits[..., span]
        assert isinstance(level, (IndependentRouting, TiedRouting))
        if isinstance(level, TiedRouting):
            write_factor, _ = _activation_factors(
                write_level, level.activation("write"), mask=None)
            read_factor = write_factor
        else:
            read_factor, _ = _activation_factors(
                read_level, level.read, mask=None)
            write_factor, _ = _activation_factors(
                write_level, level.write, mask=None)
        reads.append(read_factor)
        writes.append(write_factor)

    return tuple(reads), tuple(writes)


__all__ = ["routing_factor_levels"]
