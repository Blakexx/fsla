"""Side-gain projection and positive parameterization.

Routing allocations are
normalized per level per side, so without a magnitude the model can only ever
write total mass 1 per token. The learned WRITE magnitude supplies it:

    hidden -> [value, packed route logits, write gain logit]     # g_read := 1

There is no read gain: the readout is a ratio, so ``g_read := 1`` under the
self-normalizing readout, where a per-token uniform read scale cancels identically
in the ratio. This module owns only the gain half of that contract (value/route-logit
projection is unchanged packed-router machinery in ``projection.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

WRITE_GAIN = 0


def softplus_side_gain(logit: torch.Tensor, bias: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Baseline positive parameterization: ``softplus(logit + bias)``.

    **Flagged model decision**: unit gain at
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


@dataclass(frozen=True, slots=True)
class GainConfig:
    """The INTERNAL gain surface: one boolean, behind the public ``gain: bool``.

    ``False`` REMOVES the projection column rather than pinning it to 1: a parameter
    whose value is fixed is a parameter whose gradient is wasted, and a missing column
    is a constant, which has no gradient to waste.
    """

    write_gain: bool = True

    @classmethod
    def from_public(cls, gain: bool) -> GainConfig:
        if type(gain) is not bool:
            raise TypeError(f"gain must be a bool, got {gain!r}")
        return cls(write_gain=gain)

    @property
    def layout(self) -> tuple[str, ...]:
        """The projection's columns, in order. A column is present iff its gain is."""
        return ("write",) if self.write_gain else ()


def gain_layout(*, write_gain: bool = True) -> tuple[str, ...]:
    """The side-gain projection's columns. THE ONE PLACE it is decided."""
    return GainConfig(write_gain=write_gain).layout


def validate_gain_projection(
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    *,
    heads: int,
    hidden_size: int,
    write_gain: bool = True,
) -> tuple[str, ...]:
    """Validate ``gain_W``/``gain_bias`` shapes and return the projection's layout."""
    layout = gain_layout(write_gain=write_gain)
    if not isinstance(gain_W, torch.Tensor) or gain_W.ndim != 3:
        shape = getattr(gain_W, "shape", None)
        raise ValueError(f"gain_W must have shape [H, sides, hidden_size], got {shape}")
    h, sides, dm = gain_W.shape
    if h != heads or dm != hidden_size or sides != len(layout):
        raise ValueError(
            f"gain_W must have shape [{heads}, {len(layout)}, {hidden_size}] for "
            f"write_gain={write_gain} (layout {layout}), got {tuple(gain_W.shape)}")
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
    write_gain: bool = True,
    projection_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Project per-head write-gain logits and apply the positive parameterization.

    ``h``: ``[B, L, hidden_size]`` (shared source stream, matching the packed
    router's projection convention -- the architecture does not require a
    separate per-head hidden stream for gains).
    ``gain_W``: ``[H, sides, hidden_size]``, ``sides`` = 1 (write only).

    Returns ``{'g_write': [B,L,H]}``.
    """
    if not isinstance(h, torch.Tensor) or h.ndim != 3:
        shape = getattr(h, "shape", None)
        raise ValueError(f"h must have shape [B, L, hidden_size], got {shape}")
    heads = gain_W.shape[0]
    layout = validate_gain_projection(
        gain_W, gain_bias, heads=heads, hidden_size=h.shape[-1], write_gain=write_gain)
    if not layout:
        #: No projected column at all. The gain is a CONSTANT, not a pinned parameter:
        #: there is no column, so there is nothing to take a gradient of.
        return {"g_write": torch.ones(h.shape[0], h.shape[1], heads, dtype=h.dtype,
                                      device=h.device)}
    logits = torch.einsum(
        "bld,hsd->bhls", h.to(projection_dtype), gain_W.to(projection_dtype))
    if gain_bias is not None:
        # logits is [B,H,L,sides]; gain_bias is [H,sides] -> broadcast into the H and
        # sides axes only. `[:, None, None, :]` (the naive transcription of the packed
        # router's own bias-broadcast idiom) is wrong here: that idiom's bias has an
        # explicit head axis in position 0, matching a [B,H,...] tensor whose *own* dim
        # 0 is B, not H -- placing gain_bias's H there instead silently broadcasts B and
        # H against each other whenever B == H (as it commonly does), corrupting the
        # batch dimension instead of raising.
        logits = logits + gain_bias.to(projection_dtype)[None, :, None, :]
    logits = logits.permute(0, 2, 1, 3)  # [B,L,H,sides], columns ordered by `layout`
    return {"g_write": softplus_side_gain(logits[..., 0]).to(h.dtype)}


class SideGain(nn.Module):
    """The side-gain producer: one learned per-head projection, positively parameterized.

    ``bias_init`` places an all-zero-logit gain at that value (``None`` leaves the bias
    at zero). Why the gain belongs to routing rather than to the layer:
    ``docs/api.md`` §2.1.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        hidden_size: int,
        config: GainConfig,
        bias_init: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)
        self.config = config
        self.layout = self.config.layout
        sides = len(self.layout)
        self.gain_W = nn.Parameter(torch.empty(self.num_heads, sides, self.hidden_size))
        if sides:
            nn.init.kaiming_uniform_(
                self.gain_W.view(self.num_heads * sides, self.hidden_size), a=5 ** 0.5)
        bias_value = 0.0 if bias_init is None else init_side_gain_bias(bias_init)
        self.gain_bias = nn.Parameter(torch.full((self.num_heads, sides), bias_value))

    def forward(self, stream: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{'g_write': [B,T,H]}`` over ``[B,T,hidden]``.

        fp64 in, fp64 out; anything else projects in fp32, the packed router's rule.
        """
        return project_side_gains(
            stream, self.gain_W, self.gain_bias, write_gain=self.config.write_gain,
            projection_dtype=(torch.float64 if stream.dtype == torch.float64
                              else torch.float32))


__all__ = [
    "WRITE_GAIN",
    "GainConfig",
    "SideGain",
    "gain_layout",
    "init_side_gain_bias",
    "project_side_gains",
    "softplus_side_gain",
    "validate_gain_projection",
]
