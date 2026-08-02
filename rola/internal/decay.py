"""Compact mixed-radix decay semantics shared by the public RoLA paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PUBLIC_DECAY_KERNELS = ("rla", "gla_static", "gla_mass_detached")


def validate_decay_kernel(kernel: str) -> str:
    """Validate the complete public decay vocabulary and return its exact spelling."""
    if kernel not in PUBLIC_DECAY_KERNELS:
        raise ValueError(
            "kernel must be one of 'rla', 'gla_static', or 'gla_mass_detached', "
            f"got {kernel!r}"
        )
    return kernel


def decay_parameter_offsets(branches: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return the flat offset of each level in ``[*, sum(branches)]`` storage."""
    offsets = []
    offset = 0
    for branch in branches:
        offsets.append(offset)
        offset += branch
    return tuple(offsets)


def decay_parameter_width(branches: tuple[int, ...] | list[int]) -> int:
    """Return the exact compact width for per-level decay parameters."""
    return sum(branches)


def validate_compact_decay_theta(
    decay_theta: torch.Tensor,
    *,
    leading_dim: int,
    branches: tuple[int, ...] | list[int],
    device: torch.device,
    name: str = "decay_theta",
) -> None:
    """Validate compact per-level decay logits ``[leading_dim, sum(branches)]``."""
    if not isinstance(decay_theta, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(decay_theta).__name__}")
    expected = (leading_dim, decay_parameter_width(branches))
    if tuple(decay_theta.shape) != expected:
        raise ValueError(f"{name} must have compact shape {expected}, got {tuple(decay_theta.shape)}")
    if decay_theta.device != device:
        raise ValueError(f"{name} must be on the q device")


def compact_log_retention_from_public(
    kernel: str,
    decay_theta: torch.Tensor | None,
    *,
    leading_dim: int,
    branches: tuple[int, ...] | list[int],
    device: torch.device,
) -> torch.Tensor | None:
    """Convert public decay logits to the private compact log-retention ABI."""
    kernel = validate_decay_kernel(kernel)
    if kernel == "rla":
        if decay_theta is not None:
            raise ValueError("rla does not accept decay_theta")
        return None
    if decay_theta is None:
        raise ValueError(f"{kernel} requires decay_theta")
    validate_compact_decay_theta(
        decay_theta, leading_dim=leading_dim, branches=branches, device=device)
    # Preserve fp64 in the canonical naive oracle. Native callers normalize
    # this compact tensor to fp32 at the private CUDA ABI boundary.
    return -F.softplus(decay_theta).contiguous()


def _log_union_probability(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``log(exp(left) + exp(right) - exp(left + right))`` stably."""
    high = torch.maximum(left, right)
    low = torch.minimum(left, right)
    return high + torch.log1p(torch.exp(low - high) * (-torch.expm1(high)))


def leaf_log_retention_from_strides(
    compact_log_retention: torch.Tensor,
    radix_strides: tuple[int, ...] | list[int],
    branches: tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Compose the leaf log-retention without materializing rounded decay rates."""
    state_count = 1
    for branch in branches:
        state_count *= branch
    state_ids = torch.arange(
        state_count, device=compact_log_retention.device, dtype=torch.long)
    offsets = decay_parameter_offsets(branches)
    leaf_log_retention = None
    for level, (stride, branch) in enumerate(zip(radix_strides, branches)):
        index = (state_ids // stride) % branch
        selected = compact_log_retention[
            ..., offsets[level]:offsets[level] + branch
        ].index_select(-1, index)
        leaf_log_retention = (
            selected
            if leaf_log_retention is None
            else _log_union_probability(leaf_log_retention, selected)
        )
    if leaf_log_retention is None:
        raise ValueError("decay requires at least one routing level")
    return leaf_log_retention


def decay_retention_from_strides(
    write_mass: torch.Tensor,
    write_levels: tuple[torch.Tensor, ...] | list[torch.Tensor],
    routing,
    radix_strides: tuple[int, ...] | list[int],
    compact_log_retention: torch.Tensor | None,
    kernel: str,
) -> torch.Tensor:
    """Build the canonical naive clock without exposing a ``[D,nc]`` digit table."""
    kernel = validate_decay_kernel(kernel)
    if kernel == "rla":
        if compact_log_retention is not None:
            raise ValueError("rla does not accept compact log retention")
        return torch.ones_like(write_mass)
    if compact_log_retention is None:
        raise ValueError(f"{kernel} requires compact log retention")
    log_retention = leaf_log_retention_from_strides(
        compact_log_retention, radix_strides, routing.branches).unsqueeze(1)
    if kernel == "gla_static":
        if len(write_levels) != len(routing.branches):
            raise ValueError("write_levels must contain one factor tensor per routing level")
        state_ids = torch.arange(
            routing.state_count, device=write_mass.device, dtype=torch.long)
        event = torch.ones_like(write_mass, dtype=torch.bool)
        for level, (factors, stride, width, can_zero) in enumerate(zip(
            write_levels,
            radix_strides,
            routing.branches,
            routing.write_level_can_zero,
        )):
            if factors.shape[:-1] != write_mass.shape[:-1] or factors.shape[-1] != width:
                raise ValueError(
                    f"write_levels[{level}] must have shape "
                    f"{(*write_mass.shape[:-1], width)}, got {tuple(factors.shape)}"
                )
            if can_zero:
                digits = (state_ids // stride) % width
                event &= factors.index_select(-1, digits) != 0
        clock = event.to(log_retention.dtype)
        return torch.where(
            event,
            torch.exp(log_retention * clock),
            torch.ones_like(log_retention * clock),
        )

    clock = write_mass.detach()
    return torch.where(
        clock != 0,
        torch.exp(log_retention * clock),
        torch.ones_like(clock),
    )


def validate_compact_log_gamma(
    log_gamma: torch.Tensor,
    *,
    leading_dim: int,
    branches: tuple[int, ...] | list[int],
    device: torch.device,
    name: str = "log_gamma",
) -> None:
    """Validate a compact ``[leading_dim, sum(branches)]`` log-decay tensor."""
    if not isinstance(log_gamma, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(log_gamma).__name__}")
    expected = (leading_dim, decay_parameter_width(branches))
    if tuple(log_gamma.shape) != expected:
        raise ValueError(f"{name} must have compact shape {expected}, got {tuple(log_gamma.shape)}")
    if log_gamma.device != device:
        raise ValueError(f"{name} must be on the q device")


def batch_compact_log_gamma(
    log_gamma: torch.Tensor,
    *,
    batch_size: int,
    heads: int,
    branches: tuple[int, ...] | list[int],
    device: torch.device,
) -> torch.Tensor:
    """Expand compact per-head logs across batch without creating leaf parameters."""
    validate_compact_log_gamma(
        log_gamma, leading_dim=heads, branches=branches, device=device)
    return log_gamma.float().unsqueeze(0).expand(batch_size, -1, -1).reshape(
        batch_size * heads, decay_parameter_width(branches)
    ).contiguous()


__all__ = [
    "PUBLIC_DECAY_KERNELS",
    "batch_compact_log_gamma",
    "compact_log_retention_from_public",
    "decay_parameter_offsets",
    "decay_parameter_width",
    "decay_retention_from_strides",
    "leaf_log_retention_from_strides",
    "validate_compact_decay_theta",
    "validate_compact_log_gamma",
    "validate_decay_kernel",
]
