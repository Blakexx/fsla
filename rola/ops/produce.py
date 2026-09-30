# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRODUCER: one call from a stretch of tokens to the bundle -- the routes and the values, in the internal format.

    routes, values = produce(x, plan, weights)

The layer declares WHAT it routes (:class:`rola.routing.producer.Routing`), owns the weights, and compiles a
:class:`ProducerPlan` once; this is the one primitive that computes the bundle from them. It is the whole job --
the projection, every level's solve, the mass fold, the write gain, the quantization into unit-max rows and the
value digits -- because every step reads the same declaration, and it is one call so that one fused kernel can
stand behind it (docs/api.md section 2). The bundles it returns are the ones the ops take; a caller with its own
levels builds them with :meth:`RouteFactors.from_levels` and :meth:`Values.from_projection`, the same format
utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.factors import RouteFactors
from rola.routing.producer import Routing
from rola.routing.projection import _packed_router_logits
from rola.routing.side_gain import WRITE_GAIN, softplus_side_gain
from rola.values import DEFAULT_VALUE_DIGITS, Values

__all__ = ["ProducerPlan", "ProducerWeights", "produce", "projection_dtype", "route_logits"]


@dataclass(frozen=True)
class ProducerPlan:
    """What one layer's producer computes, compiled once from its declaration: the routing, the heads, the widths
    in and out, and the value digits. Every per-call decision is read off this, never off the data."""

    routing: Routing
    num_heads: int
    hidden_size: int
    d_v: int
    value_digits: int = DEFAULT_VALUE_DIGITS
    #: per level, whether a side's fold is an identity -- the topology's DECLARED simplex outputs, never a runtime
    #: `sum == 1` test, which would be the reduction the skip exists to delete (docs/api.md section 2.2)
    read_skip: tuple[bool, ...] = field(init=False)
    write_skip: tuple[bool, ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in ("num_heads", "hidden_size", "d_v"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact int, got {value!r}")
        topology = self.routing.topology
        object.__setattr__(self, "read_skip", tuple(topology.duty_simplex_output(level, "read")
                                                    for level in range(topology.D)))
        object.__setattr__(self, "write_skip", tuple(topology.duty_simplex_output(level, "write")
                                                     for level in range(topology.D)))


@dataclass(frozen=True)
class ProducerWeights:
    """The tensors a producer call reads, as the call reads them: ``route_W [H, hidden, columns]`` in the projection's
    dtype (``projection_dtype``), ``route_bias [H, columns]`` or ``None``, ``gain_bias [H, gain columns]``, and
    ``v_W [H * d_v, hidden]`` (a linear's weight) in bf16."""

    route_W: torch.Tensor
    route_bias: torch.Tensor | None
    gain_bias: torch.Tensor
    v_W: torch.Tensor


def projection_dtype(x: torch.Tensor) -> torch.dtype:
    """The routing projection's arithmetic: fp64 in, fp64 out (the oracle-equivalence gate runs there); on CUDA bf16
    operands with fp32 accumulate, the tensor-core class; fp32 on the host, which has no tensor cores to reach."""
    if x.dtype == torch.float64:
        return torch.float64
    return torch.bfloat16 if x.is_cuda else torch.float32


def produce(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """``x [B, T, hidden]`` -> ``(RouteFactors, Values)``, the bundle the ops take."""
    if x.ndim != 3 or x.shape[-1] != plan.hidden_size:
        raise ValueError(f"x must be [B, T, {plan.hidden_size}], got {tuple(x.shape)}")
    B, T, _ = x.shape
    H = plan.num_heads
    routing = plan.routing
    read_logits, write_logits, gain_logits = route_logits(x, plan, weights)
    read_bh, write_bh = production_routing_factor_levels(read_logits, write_logits, routing.resolved)

    #: THE READ MASS IS CARRIED, NOT APPLIED: the readout is a ratio, so a per-token scale common to every leaf
    #: cancels between numerator and denominator. The WRITE mass is APPLIED, because the decay clock is defined on
    #: the stored level; it rides the write gain.
    read_levels, read_mass = _fold(read_bh, B=B, H=H, skip=plan.read_skip, carry_mass=True)
    write_levels, write_mass = _fold(write_bh, B=B, H=H, skip=plan.write_skip, carry_mass=False)
    g_write = _write_gain(gain_logits, weights.gain_bias, B=B, T=T, H=H, dtype=read_levels[0].dtype,
                          device=x.device)
    routes = RouteFactors.from_levels(routing.topology, read_levels, write_levels,
                                      g_write if write_mass is None else g_write * write_mass,
                                      read_mass=read_mass)

    #: `v` FLOWS bf16 into the format (docs/api.md section 4.1)
    v = F.linear(x.to(torch.bfloat16), weights.v_W)
    values = Values.from_projection(rearrange(v, "b l (h d) -> b l h d", d=plan.d_v), digits=plan.value_digits)
    return routes, values


def route_logits(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights):
    """The projection step alone: ``(read_logits, write_logits, gain_logits)`` as the solve reads them.

    TOKEN-MAJOR ``[B, T, H, W]`` on the bf16 (CUDA) path -- the GEMM's own output order, so the head axis is free
    (docs/internals/entmax/entmax.md#stream-split); the fp64 and host paths read the flat plane. Named so a gate that
    needs the logits the producer computes reads this step rather than reconstructing it.
    """
    dtype = projection_dtype(x)
    return _packed_router_logits(
        x, weights.route_W, plan.routing.resolved, route_bias=weights.route_bias, projection_dtype=dtype,
        token_major=dtype is torch.bfloat16, gain_columns=plan.routing.gain_columns)


def _write_gain(gain_logits, gain_bias, *, B, T, H, dtype, device) -> torch.Tensor:
    """``g_write [B, T, H]``: ``softplus(logit + bias)``, or the constant 1 when the gain column is ablated."""
    if gain_logits is None or gain_logits.shape[-1] == 0:
        return torch.ones(B, T, H, dtype=dtype, device=device)
    biased = gain_logits + gain_bias.to(gain_logits.dtype)[None, None, :, :]
    return softplus_side_gain(biased[..., WRITE_GAIN]).to(dtype)


def _fold(levels: tuple[torch.Tensor, ...], *, B: int, H: int, skip: tuple[bool, ...], carry_mass: bool):
    """Factor every level into ``(simplex, mass)`` and return the mass product, or ``None``.

    ``skip[l]`` is the topology's DECLARED ``simplex_output`` for that cell. ``carry_mass`` leaves the mass ON the
    levels and returns the product for the caller to apply -- legal only where the consumer's use of the levels is
    homogeneous of degree zero (docs/api.md section 2.2). The divide runs in the solve's ``[BH, T, width]`` domain,
    because the public ``[B, T, H, width]`` layout is a permute of it.
    """
    normalized: list[torch.Tensor] = []
    mass_product: torch.Tensor | None = None
    for folded, unit_sum in zip(levels, skip, strict=True):
        if not unit_sum:
            #: THE SUM IS AT LEAST fp32, whatever the level is stored in: a width-256 sum of bf16 addends in bf16
            #: loses three digits of the mass the readout's floor is scaled by
            mass = folded.sum(dim=-1, dtype=torch.promote_types(folded.dtype, torch.float32))
            if not carry_mass:
                folded = folded / mass.unsqueeze(-1).to(folded.dtype)
            mass_product = mass if mass_product is None else mass_product * mass
        width = folded.shape[-1]
        normalized.append(folded.reshape(B, H, -1, width).permute(0, 2, 1, 3))
    if mass_product is None:
        return tuple(normalized), None
    return tuple(normalized), mass_product.reshape(B, H, -1).permute(0, 2, 1)
