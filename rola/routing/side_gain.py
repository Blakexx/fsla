"""Side-gain projection and positive parameterization.

``ROLA_V3_SPEC.md`` §1.1 and architecture §4.2: routing allocations are
normalized per level per side, so without a magnitude the model can only ever
write total mass 1 per token. The per-side learned magnitude supplies it:

    raw:     hidden -> [value, packed route logits, read gain logit, write gain logit]
    global:  hidden -> [value, packed route logits, write gain logit]   # g_read := 1

This module owns only the gain half of that contract (value/route-logit
projection is unchanged packed-router machinery in ``projection.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

WRITE_GAIN = 0
READ_GAIN = 1


def softplus_side_gain(logit: torch.Tensor, bias: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Baseline positive parameterization: ``softplus(logit + bias)`` (spec §1.1).

    **Flagged model decision** (spec §1.1, architecture §4.2): unit gain at
    zero logits/bias is *not* migration-equivalent -- the exact ``ELU+1``
    decomposition gives ``g_read = sqrt(K)``, ``g_write = K`` for the historical
    width-``K`` feature map. This function supplies only the positive
    parameterization's functional form; the caller's choice of ``bias`` is a
    new model-level init decision requiring its own quality/stability gate
    before large training, and must not be presented as equivalent to any V1/V2
    checkpoint.
    """
    return F.softplus(logit + bias)


def init_side_gain_bias(target_gain: float) -> float:
    """Invert ``softplus`` to place an all-zero-logit gain at ``target_gain``.

    Convenience for the flagged init decision above: ``bias = softplus^-1(g)``
    so a caller who *does* want to start near a particular gain (e.g. unit gain,
    or an approximate ``sqrt(K)``/``K`` migration anchor for a specific fixed
    ``K``) can compute the bias explicitly and record the choice, rather than
    leaving it implicit at zero.
    """
    if target_gain <= 0:
        raise ValueError(f"target_gain must be > 0, got {target_gain}")
    import math

    return math.log(math.expm1(target_gain))


def validate_gain_projection(
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    *,
    heads: int,
    hidden_size: int,
    normalization: str,
) -> int:
    """Validate ``gain_W``/``gain_bias`` shapes and return the side count (1 or 2)."""
    if not isinstance(gain_W, torch.Tensor) or gain_W.ndim != 3:
        shape = getattr(gain_W, "shape", None)
        raise ValueError(f"gain_W must have shape [H, sides, hidden_size], got {shape}")
    h, sides, dm = gain_W.shape
    expected_sides = 1 if normalization == "global" else 2
    if h != heads or dm != hidden_size or sides != expected_sides:
        raise ValueError(
            f"gain_W must have shape [{heads}, {expected_sides}, {hidden_size}] "
            f"for normalization={normalization!r}, got {tuple(gain_W.shape)}")
    if gain_bias is not None:
        if not isinstance(gain_bias, torch.Tensor) or tuple(gain_bias.shape) != (heads, sides):
            shape = getattr(gain_bias, "shape", None)
            raise ValueError(f"gain_bias must have shape [{heads}, {sides}], got {shape}")
    return sides


def project_side_gains(
    h: torch.Tensor,
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    *,
    normalization: str,
    projection_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Project per-head side-gain logits and apply the positive parameterization.

    ``h``: ``[B, L, hidden_size]`` (shared source stream, matching the packed
    router's projection convention -- architecture §8 does not require a
    separate per-head hidden stream for gains).
    ``gain_W``: ``[H, sides, hidden_size]``, ``sides`` = 2 for raw
    (write-then-read, matching ``WRITE_GAIN``/``READ_GAIN``), 1 for global
    (write only; architecture §4.2 fixes ``g_read := 1``, no projection).

    Returns ``{'g_write': [B,L,H], 'g_read': [B,L,H] | None}``.
    """
    if normalization not in ("raw", "global"):
        raise ValueError(f"normalization must be 'raw' or 'global', got {normalization!r}")
    if not isinstance(h, torch.Tensor) or h.ndim != 3:
        shape = getattr(h, "shape", None)
        raise ValueError(f"h must have shape [B, L, hidden_size], got {shape}")
    heads = gain_W.shape[0]
    sides = validate_gain_projection(
        gain_W, gain_bias, heads=heads, hidden_size=h.shape[-1], normalization=normalization)
    logits = torch.einsum(
        "bld,hsd->bhls", h.to(projection_dtype), gain_W.to(projection_dtype))
    if gain_bias is not None:
        # logits is [B,H,L,sides]; gain_bias is [H,sides] -> broadcast into the H and
        # sides axes only. `[:, None, None, :]` (the naive transcription of the packed
        # router's own bias-broadcast idiom) is wrong here: that idiom's bias has an
        # explicit head axis in position 0, matching a [B,H,...] tensor whose *own* dim
        # 0 is B, not H -- placing gain_bias's H there instead silently broadcasts B and
        # H against each other whenever B == H (as it commonly does), corrupting the
        # batch dimension instead of raising. Caught by `tests/layers/test_rola_v3_layer.py`.
        logits = logits + gain_bias.to(projection_dtype)[None, :, None, :]
    logits = logits.permute(0, 2, 1, 3)  # [B,L,H,sides]
    write_gain = softplus_side_gain(logits[..., WRITE_GAIN])
    read_gain = softplus_side_gain(logits[..., READ_GAIN]) if sides == 2 else None
    return {"g_write": write_gain.to(h.dtype), "g_read": None if read_gain is None else read_gain.to(h.dtype)}


__all__ = [
    "READ_GAIN",
    "WRITE_GAIN",
    "init_side_gain_bias",
    "project_side_gains",
    "softplus_side_gain",
    "validate_gain_projection",
]
