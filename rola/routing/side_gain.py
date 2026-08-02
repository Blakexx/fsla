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

from dataclasses import dataclass

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


READ_GAIN_UNDER_GLOBAL = (
    "under normalization='global' the read gain is a parameter with an identically "
    "zero gradient: the readout is a ratio, so a per-token uniform read scale cancels "
    "identically.")


@dataclass(frozen=True, slots=True)
class GainConfig:
    """The INTERNAL gain surface: two independent booleans.

    The public layer parameter is a single `gain: bool`, mapped by
    :meth:`from_public`. This object is the research surface behind it -- a
    fine-grained combination such as `raw` with a write gain and no read gain is
    reachable by constructing one directly, and is deliberately not a public
    parameter, because the public surface is opinionated on purpose.

    A `False` side REMOVES its projection column rather than pinning it to 1: a
    parameter whose value is fixed is a parameter whose gradient is wasted, and a
    missing column is a constant, which has no gradient to waste.
    """

    write_gain: bool = True
    read_gain: bool = True

    @classmethod
    def from_public(cls, gain: bool, normalization: str) -> GainConfig:
        """Map the public `gain: bool` onto the two internal booleans.

        `gain=True` enables the write gain under both normalizations and the read
        gain only under `raw` (under `global` a read gain is a dead parameter), and
        `gain=False` disables both -- salience through routing only.
        """
        if type(gain) is not bool:
            raise TypeError(f"gain must be a bool, got {gain!r}")
        if normalization not in ("raw", "global"):
            raise ValueError(f"normalization must be 'raw' or 'global', got {normalization!r}")
        return cls(write_gain=gain, read_gain=gain and normalization == "raw")

    def validate(self, normalization: str) -> GainConfig:
        """The ONE refusal: a read gain under `global` is an identically-zero-gradient
        parameter, and it is rejected at construction rather than optimized."""
        if normalization not in ("raw", "global"):
            raise ValueError(f"normalization must be 'raw' or 'global', got {normalization!r}")
        if self.read_gain and normalization == "global":
            raise ValueError(
                "GainConfig(read_gain=True) with normalization='global': "
                + READ_GAIN_UNDER_GLOBAL
                + " Use normalization='raw', where the read gain is expressive, or "
                  "leave the read gain off.")
        return self

    @property
    def layout(self) -> tuple[str, ...]:
        """The projection's columns, in order. A column is present iff its gain is."""
        return tuple(
            side for side, enabled in (("write", self.write_gain), ("read", self.read_gain))
            if enabled)


def gain_layout(
    normalization: str, *, write_gain: bool = True, read_gain: bool | None = None,
) -> tuple[str, ...]:
    """The side-gain projection's columns, validated. THE ONE PLACE it is decided.

    `read_gain=None` means "whatever the normalization admits", which is the mapping
    `GainConfig.from_public` applies to an enabled public `gain`.
    """
    if read_gain is None:
        read_gain = normalization == "raw"
    return GainConfig(write_gain=write_gain, read_gain=read_gain).validate(normalization).layout


def validate_gain_projection(
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    *,
    heads: int,
    hidden_size: int,
    normalization: str,
    write_gain: bool = True,
    read_gain: bool | None = None,
) -> tuple[str, ...]:
    """Validate ``gain_W``/``gain_bias`` shapes and return the projection's layout."""
    layout = gain_layout(normalization, write_gain=write_gain, read_gain=read_gain)
    if not isinstance(gain_W, torch.Tensor) or gain_W.ndim != 3:
        shape = getattr(gain_W, "shape", None)
        raise ValueError(f"gain_W must have shape [H, sides, hidden_size], got {shape}")
    h, sides, dm = gain_W.shape
    if h != heads or dm != hidden_size or sides != len(layout):
        raise ValueError(
            f"gain_W must have shape [{heads}, {len(layout)}, {hidden_size}] for "
            f"normalization={normalization!r}, write_gain={write_gain}, "
            f"read_gain={read_gain!r} (layout {layout}), "
            f"got {tuple(gain_W.shape)}")
    if gain_bias is not None:
        if not isinstance(gain_bias, torch.Tensor) or tuple(gain_bias.shape) != (heads, sides):
            shape = getattr(gain_bias, "shape", None)
            raise ValueError(f"gain_bias must have shape [{heads}, {sides}], got {shape}")
    return layout


def project_side_gains(
    h: torch.Tensor,
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    *,
    normalization: str,
    write_gain: bool = True,
    read_gain: bool | None = None,
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
    layout = validate_gain_projection(
        gain_W, gain_bias, heads=heads, hidden_size=h.shape[-1],
        normalization=normalization, write_gain=write_gain, read_gain=read_gain)
    def _ones() -> torch.Tensor:
        return torch.ones(h.shape[0], h.shape[1], heads, dtype=h.dtype, device=h.device)

    if not layout:
        # No projected column at all. The gains are CONSTANTS, not pinned parameters:
        # there is no column, so there is nothing to take a gradient of. `g_read` is
        # `None` under `global` because the engine's contract there is that there IS
        # no read gain, and `1` under `raw` because the read channel still exists.
        return {"g_write": _ones(), "g_read": None if normalization == "global" else _ones()}
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
    logits = logits.permute(0, 2, 1, 3)  # [B,L,H,sides], columns ordered by `layout`
    gains: dict[str, torch.Tensor | None] = {"g_write": None, "g_read": None}
    for column, side in enumerate(layout):
        gains[f"g_{side}"] = softplus_side_gain(logits[..., column]).to(h.dtype)
    if gains["g_write"] is None:
        gains["g_write"] = _ones()
    if gains["g_read"] is None and normalization == "raw":
        gains["g_read"] = _ones()
    return gains


__all__ = [
    "GainConfig",
    "gain_layout",
    "READ_GAIN_UNDER_GLOBAL",
    "READ_GAIN",
    "WRITE_GAIN",
    "init_side_gain_bias",
    "project_side_gains",
    "softplus_side_gain",
    "validate_gain_projection",
]
