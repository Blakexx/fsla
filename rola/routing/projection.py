"""Packed router projection for a :class:`ResolvedRouting` plan."""

from __future__ import annotations

import torch

from rola.routing.types import ResolvedRouting


def _packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
    projection_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project packed router weights in one explicitly selected arithmetic domain."""
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    heads = routing.validate_packed_router(route_W, route_bias)

    def validate_hidden(hidden: torch.Tensor, name: str) -> None:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim not in (3, 4):
            shape = getattr(hidden, "shape", None)
            raise ValueError(f"{name} must have rank 3 or 4, got {shape}")
        hidden_width = hidden.shape[-1]
        if hidden_width != route_W.shape[2]:
            raise ValueError(
                f"{name} hidden width must match route_W hidden width {route_W.shape[2]}, "
                f"got {hidden_width}")
        if hidden.ndim == 4 and hidden.shape[2] != heads:
            raise ValueError(
                f"{name} rank-4 head count must match route_W heads {heads}, "
                f"got {hidden.shape[2]}")

    validate_hidden(h, "h")
    if h_w is not None:
        validate_hidden(h_w, "h_w")
        if h_w.shape != h.shape or h_w.device != h.device:
            raise ValueError("h_w must match h shape and device")

    def project(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        equation = "bld,hsdc->bhlsc" if hidden.ndim == 3 else "blhd,hsdc->bhlsc"
        logits = torch.einsum(equation, hidden.to(projection_dtype), weight.to(projection_dtype))
        if bias is not None:
            logits = logits + bias.to(projection_dtype)[None, :, None]
        return logits.reshape(hidden.shape[0] * heads, hidden.shape[1], weight.shape[1], weight.shape[3])

    if h_w is None:
        packed = project(h, route_W, route_bias)
        write_logits = packed[:, :, :routing.D]
        read_partial = packed[:, :, routing.D:]
    else:
        write_bias = None if route_bias is None else route_bias[:, :routing.D]
        write_logits = project(h_w, route_W[:, :routing.D], write_bias)
        if not routing.untied_levels:
            return write_logits, write_logits
        read_bias = None if route_bias is None else route_bias[:, routing.D:]
        read_partial = project(h, route_W[:, routing.D:], read_bias)

    if not routing.untied_levels:
        return write_logits, write_logits
    if len(routing.untied_levels) == routing.D:
        return read_partial, write_logits
    untied_read_slots = {level: slot for slot, level in enumerate(routing.untied_levels)}
    read_logits = torch.stack(
        [
            read_partial[:, :, untied_read_slots[level]]
            if level in untied_read_slots
            else write_logits[:, :, level]
            for level in range(routing.D)
        ],
        dim=2,
    )
    return read_logits, write_logits


def packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project production router logits in fp32."""
    return _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float32)


def packed_router_logits_reference(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project canonical reference router logits in fp64."""
    return _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float64)
