# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRODUCER: one call from a stretch of tokens to the bundle -- the routes and the values, in the internal format.

    routes, values, masks = produce_prefill(x, routing, weights)     # or produce_decode: the decode step's layout

The layer declares WHAT it routes (:class:`fsla.routing.producer.Routing`) and owns the weights; these two are the one primitive per
consumer that computes the bundle from them, and the masks beside it in the layout of that consumer (:mod:`fsla.routing.masks`). The heads, the
hidden width and the value width are read off the arguments, never declared twice. It is the whole job --
the projection, every level's solve, the write gain (folded into write level 0's scale), the quantization into
unit-max rows and the value digits -- because every step reads the same declaration, and it is one call so that one fused kernel stands
behind it (docs/api.md section 2): on CUDA with no gradient, the producer kernel's one launch, which is not built on
this line. The torch composition is the reference -- fp64, the host -- and, until the
native backward has the kernel write the levels a gradient reads, the path a gradient takes. The bundles it
returns are the ones the ops take; a caller with its own levels builds them with :meth:`RouteFactors.from_levels` and
:meth:`Values.from_projection`, the same format utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
from einops import rearrange

from fsla.engine.facts.operands import requires_backward
from fsla.ops._ext import not_built
from fsla.ops.descriptor import Descriptor, call_facts, descriptor
from fsla.routing.entmax.production import production_routing_levels
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import DECODE, PREFILL, Masks, pack_digits, pack_tokens
from fsla.routing.producer import Routing
from fsla.routing.projection import _packed_router_logits
from fsla.routing.side_gain import WRITE_GAIN, softplus_side_gain
from fsla.routing.types import READ_SUPPORT, WRITE_SUPPORT, UnionRouting
from fsla.values import Values

__all__ = ["ProducerWeights", "activation_walks", "hidden_contiguous", "level_facts",
           "produce_decode", "produce_prefill", "projection_dtype", "route_logits"]


@dataclass(frozen=True)
class ProducerWeights:
    """The tensors a producer call reads, as the call reads them: ``route_W [H, hidden, columns]`` in the projection's
    dtype (``projection_dtype``), ``route_bias [H, columns]`` or ``None``, ``gain_bias [H, gain columns]``, and
    ``v_W [H * d_v, hidden]`` (a linear's weight) in bf16. The kernel reads both weights along the hidden axis, so it
    takes them with that axis contiguous (:func:`hidden_contiguous`; a linear's weight already is)."""

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


def hidden_contiguous(route_W: torch.Tensor) -> torch.Tensor:
    """``route_W [H, hidden, columns]`` stored with its hidden axis contiguous and viewed back: the kernel loads a
    column's whole hidden run in 16-byte chunks, straight into its MMA fragments."""
    return route_W.transpose(1, 2).contiguous().transpose(1, 2)


def produce_prefill(x: torch.Tensor, routing: Routing, weights: ProducerWeights) -> tuple[RouteFactors, Values, Masks]:
    """``x [B, T, hidden]`` -> ``(RouteFactors, Values, Masks)``, the operands the prefill takes, the masks token-packed (:data:`fsla.routing.masks.PREFILL`):
    the fused kernel's on CUDA with no gradient -- not built on this line (:func:`fsla.ops._ext.not_built`) -- the composition's otherwise."""
    return _produce(x, routing, weights, PREFILL)


def produce_decode(x: torch.Tensor, routing: Routing, weights: ProducerWeights) -> tuple[RouteFactors, Values, Masks]:
    """``x [B, 1, hidden]`` -> ``(RouteFactors, Values, Masks)``, the operands the decode step takes, the masks digit-packed (:data:`fsla.routing.masks.DECODE`):
    the fused kernel's on CUDA with no gradient -- not built on this line (:func:`fsla.ops._ext.not_built`) -- the composition's otherwise."""
    return _produce(x, routing, weights, DECODE)


def _produce(x: torch.Tensor, routing: Routing, weights: ProducerWeights, consumer: str) -> tuple[RouteFactors, Values, Masks]:
    _geometry(x, weights)
    operands = (x, weights.route_W, weights.route_bias, weights.gain_bias, weights.v_W)
    if x.is_cuda and x.dtype != torch.float64 and not requires_backward(*(t for t in operands if t is not None)):
        not_built("produce")
    return _composition(x, routing, weights, consumer)


def _geometry(x: torch.Tensor, weights: ProducerWeights) -> tuple[int, int]:
    """``(num_heads, d_v)`` as the arguments state them: ``num_heads`` is ``route_W``'s leading extent, ``hidden`` is ``x``'s last (and ``route_W``'s
    middle), ``d_v`` is ``v_W``'s rows over the heads. A shape that does not state them is refused naming the shapes."""
    route_W, v_W = weights.route_W, weights.v_W
    if route_W.ndim != 3:
        raise ValueError(f"route_W must be [H, hidden, columns], got {tuple(route_W.shape)}")
    num_heads, hidden = route_W.shape[0], route_W.shape[1]
    if x.ndim != 3 or x.shape[-1] != hidden or hidden < 1:
        raise ValueError(f"x must be [B, T, hidden] with hidden = route_W's {hidden} (route_W {tuple(route_W.shape)}), got {tuple(x.shape)}")
    if v_W.ndim != 2 or num_heads < 1 or v_W.shape[0] < num_heads or v_W.shape[0] % num_heads:
        raise ValueError(f"v_W must be [H * d_v, hidden] with H = route_W's {num_heads}, so a whole d_v of at least 1 a head; "
                         f"got v_W {tuple(v_W.shape)} against route_W {tuple(route_W.shape)}")
    return num_heads, v_W.shape[0] // num_heads


def _composition(x: torch.Tensor, routing: Routing, weights: ProducerWeights, consumer: str) -> tuple[RouteFactors, Values, Masks]:
    """The producer in torch: the fp64 and host reference, and the path a gradient takes."""
    B, T, _ = x.shape
    H, d_v = _geometry(x, weights)
    read_logits, write_logits, gain_logits = route_logits(x, routing, weights)
    read_bh, write_bh, support = production_routing_levels(read_logits, write_logits, routing.resolved)

    #: THE LEVELS AS THEY ARE: no level need sum to 1 (docs/CONSTITUTION.md#r23). The producer's write gain is its
    #: own, folded into write level 0, so its scale carries it as the kernel's level-0 job stores it.
    read_levels, write_levels = _public(read_bh, B=B, H=H), _public(write_bh, B=B, H=H)
    g_write = _write_gain(gain_logits, weights.gain_bias, B=B, T=T, H=H, dtype=read_levels[0].dtype,
                          device=x.device)
    write_levels = (write_levels[0] * g_write.unsqueeze(-1), *write_levels[1:])
    routes = RouteFactors.from_levels(routing.topology, read_levels, write_levels)
    masks = _masks(routing, consumer, read_levels, write_levels, support, B=B, H=H)

    #: `v` FLOWS bf16 into the format (docs/api.md section 4.1); the fp64 reference computes it in fp64
    v_dtype = torch.float64 if x.dtype == torch.float64 else torch.bfloat16
    v = F.linear(x.to(v_dtype), weights.v_W.to(v_dtype))
    values = Values.from_projection(rearrange(v, "b l (h d) -> b l h d", d=d_v))
    return routes, values, masks


def route_logits(x: torch.Tensor, routing: Routing, weights: ProducerWeights):
    """The projection step alone: ``(read_logits, write_logits, gain_logits)`` as the solve reads them.

    TOKEN-MAJOR ``[B, T, H, W]`` on the bf16 (CUDA) path -- the GEMM's own output order, so the head axis is free
    (docs/internals/entmax/entmax.md#stream-split); the fp64 and host paths read the flat plane. Named so a gate that
    needs the logits the producer computes reads this step rather than reconstructing it.
    """
    dtype = projection_dtype(x)
    return _packed_router_logits(
        x, weights.route_W, routing.resolved, route_bias=weights.route_bias, projection_dtype=dtype,
        token_major=dtype is torch.bfloat16, gain_columns=routing.gain_columns)


def _write_gain(gain_logits, gain_bias, *, B, T, H, dtype, device) -> torch.Tensor:
    """``g_write [B, T, H]``: ``softplus(logit + bias)``, or the constant 1 when the gain column is ablated."""
    if gain_logits is None or gain_logits.shape[-1] == 0:
        return torch.ones(B, T, H, dtype=dtype, device=device)
    biased = gain_logits + gain_bias.to(gain_logits.dtype)[None, None, :, :]
    return softplus_side_gain(biased[..., WRITE_GAIN]).to(dtype)


def _public(levels: tuple[torch.Tensor, ...], *, B: int, H: int) -> tuple[torch.Tensor, ...]:
    """Each level from the solve's ``[BH, T, width]`` to the public ``[B, T, H, width]`` (a permute of it)."""
    return tuple(level.reshape(B, H, -1, level.shape[-1]).permute(0, 2, 1, 3) for level in levels)


def _masks(routing: Routing, consumer: str, read_levels, write_levels, support, *, B: int, H: int) -> Masks:
    """THE MASKS the producer emits, in the layout of ``consumer``: a bit where the producer's own level is
    nonzero (the producer knows what it gives). For the prefill, a level the CUDA solve wrote support words for takes
    them as the kernel wrote them (:func:`_solve_words`); the host composition packs the floating level."""
    def packed(level: torch.Tensor) -> torch.Tensor:
        return pack_tokens(level != 0) if consumer == PREFILL else pack_digits(level != 0)

    resolved = routing.resolved
    sides = []
    for side, levels in enumerate((read_levels, write_levels)):
        out = []
        for index, level in enumerate(levels):
            words = None if support is None or consumer != PREFILL else _solve_words(
                resolved, index, side, support, B=B, H=H)
            out.append(packed(level) if words is None else words)
        sides.append(tuple(out))
    return Masks(consumer, *sides)


def _solve_words(resolved, index: int, side: int, support: torch.Tensor, *, B: int, H: int) -> torch.Tensor | None:
    """The CUDA solve's own ``support_words`` of ``side``'s level ``index`` as ``[B, H, ceil(T / 32), width]`` (a free
    view: the stream axis is ``B * H``, ``H`` inner), or ``None`` where the solve wrote none for this side -- a level with
    no support slice, a union level (its one slice is the union's, not either side's), a side the level's support
    flags do not name."""
    level, offset, width = resolved.levels[index], resolved.support_offsets[index], resolved.branches[index]
    if offset < 0 or isinstance(level, UnionRouting):
        return None
    if not resolved.support_side_flags[index] & (READ_SUPPORT if side == 0 else WRITE_SUPPORT):
        return None
    return support.view(B, H, support.shape[1], -1)[..., offset:offset + width]


@cache
def activation_walks(topology) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """THE PRODUCER'S WALK RULE, ``(read, write)`` a bool a level: the levels a side WALKS BY ITS LISTS (the rest are walked digit by digit) -- the
    levels whose activation has exact zeros (an opaque level, which may have zeros, included). An action chosen once from the
    declaration's activation metadata and stated by :func:`level_facts` as decode's ``write_listed_k`` facts; no consumer asks the activations again."""
    return tuple(tuple(topology.activation_properties(level, side).exact_zeros for level in range(topology.D))
                 for side in ("read", "write"))


def level_facts(kernel: str, topology, widths: tuple[int, ...]) -> dict[str, int]:
    """The facts a call of ``kernel`` states per level: each width as its log2 (a level's width is a power of two,
    ``fsla._state.StateFormat``'s law), the write side's walk of each level (:func:`activation_walks`) where the kernel's call facts name it (decode's ``write_listed_k``, 1 where it
    walks the level by its lists; the prefill's facts are the widths and ``levels`` alone, its weights are the masks' and no fact of the call) and ``levels``;
    a level beyond the call's depth is 0. The kernel's own call facts name how many levels its contract holds."""
    for level, width in enumerate(widths):
        if width < 1 or width & (width - 1):
            raise ValueError(f"level {level + 1}'s width is {width}: a level's width is a power of two (the contract states it as "
                             f"width_bits_{level + 1}, its log2)")
    names = call_facts(kernel)
    depth = sum(name.startswith("width_bits_") for name in names)
    write_walk = activation_walks(topology)[1]
    facts = {"levels": len(widths)}
    for level in range(depth):
        have = level < len(widths)
        facts[f"width_bits_{level + 1}"] = widths[level].bit_length() - 1 if have else 0
        if f"write_listed_{level + 1}" in names:
            facts[f"write_listed_{level + 1}"] = int(write_walk[level]) if have else 0
    return facts


@cache
def _decode_descriptor(topology, d_v: int, decay: bool, batch_heads: int, device: int) -> Descriptor:
    return descriptor(kernel="decode", device=device, d_v=d_v, decay=int(decay), batch_heads=batch_heads,
                      **level_facts("decode", topology, tuple(topology.widths)))


@cache
def _prefill_descriptor(topology, d_v: int, decay: bool, tokens: int, batch_heads: int, stateful: bool, device: int) -> Descriptor:
    return descriptor(kernel="prefill", device=device, d_v=d_v, decay=int(decay), tokens=tokens, batch_heads=batch_heads, stateful=int(stateful),
                      **level_facts("prefill", topology, tuple(topology.padded_widths)))
