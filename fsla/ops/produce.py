# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRODUCER: one call from a stretch of tokens to the bundle -- the routes and the values, in the internal format.

    routes, values, masks, density = produce_prefill(x, routing, weights)     # or produce_decode: the decode step's layout, no density

The layer declares WHAT it routes (:class:`fsla.routing.producer.Routing`) and owns the weights; these two are the one primitive per
consumer that computes the bundle from them, and the masks beside it in the layout of that consumer (:mod:`fsla.routing.masks`). The heads, the
hidden width and the value width are read off the arguments, never declared twice. It is the whole job --
the projection, every level's solve, the write gain (folded into write level 0's scale), the quantization into
unit-max rows and the value digits -- because every step reads the same declaration, and it is one call so that one fused kernel stands
behind it (docs/api.md section 2): on CUDA with no gradient, the producer kernel's one launch, which is not built on
this line. The torch composition is the reference -- fp64, the host -- and, until the
native backward has the kernel write the levels a gradient reads, the path a gradient takes. The bundles it
returns are the ones the ops take; a caller with its own levels builds them with :meth:`RouteFactors.from_levels` and
``ScaledBlock.quantize(v, quant.VALUES)`` (:class:`fsla.routing.quant.ScaledBlock`), the same format utilities.
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
from fsla.routing import quant
from fsla.routing.entmax.production import production_routing_levels
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import DECODE, PREFILL, Masks, density_of_masks, pack_digits, pack_tokens
from fsla.routing.producer import Routing
from fsla.routing.projection import RouterLayout, router_logits
from fsla.routing.quant import ScaledBlock
from fsla.routing.side_gain import WRITE_GAIN, softplus_side_gain
from fsla.routing.types import Tied, activation_walks

__all__ = ["ProducerWeights", "hidden_contiguous", "level_facts",
           "produce_decode", "produce_prefill", "projection_dtype", "route_logits"]


@dataclass(frozen=True)
class ProducerWeights:
    """The tensors a producer call reads, as the call reads them, a slice a WRITE head: ``route_W [H_w, hidden,
    columns]`` in the projection's dtype (``projection_dtype``; its columns the routing's at the call's group,
    :meth:`~fsla.routing.producer.Routing.columns`), ``route_bias [H_w, columns]`` or ``None``, ``gain_bias [H_w, gain
    columns]``, and ``v_W [H_w * d_v, hidden]`` (a linear's weight) in bf16. The kernel reads both weights along the
    hidden axis, so it takes them with that axis contiguous (:func:`hidden_contiguous`; a linear's weight already
    is)."""

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


def produce_prefill(x: torch.Tensor, routing: Routing, weights: ProducerWeights, *,
                    group: int = 1) -> tuple[RouteFactors, ScaledBlock, Masks, torch.Tensor]:
    """``x [B, T, hidden]`` -> ``(RouteFactors, values, Masks, density)``, the operands the prefill takes, the values a
    :class:`~fsla.routing.quant.ScaledBlock` in the ``VALUES`` form, the masks token-packed (:data:`fsla.routing.masks.PREFILL`)
    and the density ``[2, D]`` of those masks (:func:`fsla.routing.masks.density_of_masks`, after the solve; a level-side whose activation
    cannot make exact zeros -- on the read side, no read head's -- has NO mask: its entry is ``None``, dense, and the byte 255).
    ``group`` is ``G``, the query heads reading each write head (the layer's): the read side is ``H_w * G`` heads.
    The fused kernel's on CUDA with no gradient -- not built on this line (:func:`fsla.ops._ext.not_built`) -- the composition's otherwise."""
    routes, values, masks = _produce(x, routing, weights, PREFILL, group)
    return routes, values, masks, density_of_masks(routes, masks)


def produce_decode(x: torch.Tensor, routing: Routing, weights: ProducerWeights, *,
                   group: int = 1) -> tuple[RouteFactors, ScaledBlock, Masks]:
    """``x [B, 1, hidden]`` -> ``(RouteFactors, values, Masks)``, the operands the decode step takes, the masks digit-packed
    (:data:`fsla.routing.masks.DECODE`), ``group`` as :func:`produce_prefill` takes it:
    the fused kernel's on CUDA with no gradient -- not built on this line (:func:`fsla.ops._ext.not_built`) -- the composition's otherwise."""
    return _produce(x, routing, weights, DECODE, group)


def _produce(x: torch.Tensor, routing: Routing, weights: ProducerWeights, consumer: str,
             group: int) -> tuple[RouteFactors, ScaledBlock, Masks]:
    layout = routing.layout(group)
    _geometry(x, weights)
    operands = (x, weights.route_W, weights.route_bias, weights.gain_bias, weights.v_W)
    if x.is_cuda and x.dtype != torch.float64 and not requires_backward(*(t for t in operands if t is not None)):
        not_built("produce")
    return _composition(x, routing, layout, weights, consumer)


def _geometry(x: torch.Tensor, weights: ProducerWeights) -> tuple[int, int]:
    """``(write heads, d_v)`` as the arguments state them: the write heads are ``route_W``'s leading extent, ``hidden`` is ``x``'s last (and
    ``route_W``'s middle), ``d_v`` is ``v_W``'s rows over the heads. A shape that does not state them is refused naming the shapes."""
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


def _composition(x: torch.Tensor, routing: Routing, layout: RouterLayout, weights: ProducerWeights,
                 consumer: str) -> tuple[RouteFactors, ScaledBlock, Masks]:
    """The producer in torch: the fp64 and host reference, and the path a gradient takes."""
    B, T, _ = x.shape
    H_w, d_v = _geometry(x, weights)
    logits, gain_logits = route_logits(x, routing, weights, group=layout.group)
    read_bh, write_bh, support = production_routing_levels(logits, layout)

    #: THE LEVELS AS THEY ARE: no level need sum to 1 (docs/CONSTITUTION.md#r23). The producer's write gain is its
    #: own, folded into write level 0, so its scale carries it as the kernel's level-0 job stores it.
    read_levels, write_levels = _public(read_bh, B=B, H=H_w * layout.group), _public(write_bh, B=B, H=H_w)
    g_write = _write_gain(gain_logits, weights.gain_bias, B=B, T=T, H=H_w, dtype=read_levels[0].dtype,
                          device=x.device)
    write_levels = (write_levels[0] * g_write.unsqueeze(-1), *write_levels[1:])
    routes = RouteFactors.from_levels(routing.topology, read_levels, write_levels)
    masks = _masks(layout, consumer, read_levels, write_levels, support, B=B, H_w=H_w)

    #: `v` FLOWS bf16 into the format (docs/api.md section 4.1); the fp64 reference computes it in fp64
    v_dtype = torch.float64 if x.dtype == torch.float64 else torch.bfloat16
    v = F.linear(x.to(v_dtype), weights.v_W.to(v_dtype))
    values = ScaledBlock.quantize(rearrange(v, "b l (h d) -> b l h d", d=d_v), quant.VALUES)
    return routes, values, masks


def route_logits(x: torch.Tensor, routing: Routing, weights: ProducerWeights, *, group: int = 1):
    """The projection step alone: ``(logits, gain_logits)`` as the solve reads them
    (:func:`fsla.routing.projection.router_logits`), at ``group`` read heads a write head.

    TOKEN-MAJOR ``[B, T, H_w, W]`` on the bf16 (CUDA) path -- the GEMM's own output order, so the head axis is free
    (docs/internals/entmax/entmax.md#stream-split); the fp64 and host paths read the flat plane. Named so a gate that
    needs the logits the producer computes reads this step rather than reconstructing it.
    """
    dtype = projection_dtype(x)
    return router_logits(
        x, weights.route_W, routing.layout(group), route_bias=weights.route_bias, projection_dtype=dtype,
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


def _masks(layout: RouterLayout, consumer: str, read_levels, write_levels, support, *, B: int, H_w: int) -> Masks:
    """THE MASKS the producer emits, in the layout of ``consumer``: a bit where the producer's own level is
    nonzero (the producer knows what it gives). For the prefill, a level the CUDA solve wrote support words for takes
    them as the kernel wrote them (:func:`_solve_words`); the host composition packs the floating level. A prefill level-side whose
    declared activation cannot make exact zeros -- on the read side, no read head's -- gets no mask (``None``: dense); a read level
    some heads of which can make zeros has one mask over every query head, a dense head's bits its rows'; the decode's masks are
    made for every level-side."""
    def packed(level: torch.Tensor) -> torch.Tensor:
        return pack_tokens(level != 0) if consumer == PREFILL else pack_digits(level != 0)

    walks = activation_walks(layout.topology)
    sides = []
    for side, levels in enumerate((read_levels, write_levels)):
        out = []
        for index, level in enumerate(levels):
            if consumer == PREFILL and not walks[side][index]:
                out.append(None)
                continue
            words = None if support is None or consumer != PREFILL else _solve_words(
                layout, index, side, support, B=B, H_w=H_w)
            out.append(packed(level) if words is None else words)
        sides.append(tuple(out))
    return Masks(consumer, *sides)


def _solve_words(layout: RouterLayout, index: int, side: int, support, *, B: int, H_w: int) -> torch.Tensor | None:
    """The CUDA solve's own support words of ``side``'s level ``index`` as ``[B, H, ceil(T / 32), width]`` (``H`` the
    side's heads; a free view of the write side's, whose stream axis is ``B * H_w``, ``H_w`` inner), or ``None`` where
    the solve's words do not cover the side's every head: a level no solve wrote words for, a read level whose heads
    are not all tied (the write side's words, each write head's for its ``G`` query heads) or all untied with words."""
    write_support, read_support = support
    width, group = layout.widths[index], layout.group

    def words(plane: torch.Tensor, heads: int, offset: int) -> torch.Tensor:
        return plane.view(B, heads, plane.shape[1], -1)[..., offset:offset + width]

    write_offset = layout.write_support_offsets[index]
    if side == 1:
        return None if write_offset < 0 else words(write_support, H_w, write_offset)
    entries = tuple(heads[index] for heads in layout.heads)
    if all(type(entry) is Tied for entry in entries):
        if write_offset < 0:
            return None
        written = words(write_support, H_w, write_offset)
        return written.unsqueeze(2).expand(B, H_w, group, *written.shape[2:]).reshape(B, H_w * group, *written.shape[2:])
    read_offset = layout.read_support_offsets[index]
    if read_offset >= 0 and all(type(entry) is not Tied and entry.can_zero for entry in entries):
        return words(read_support, H_w * group, read_offset)
    return None


def level_facts(kernel: str, topology, widths: tuple[int, ...]) -> dict[str, int]:
    """The facts a call of ``kernel`` states per level: each width as its log2 (a level's width is a power of two,
    ``fsla._state.StateFormat``'s law), the write side's walk of each level (:func:`activation_walks`) where the kernel's call facts name it (decode's ``write_listed_k``, 1 where it
    walks the level by its lists; the prefill's facts are the widths and ``levels`` alone, its densities are the call's own input and no fact of it) and ``levels``;
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
