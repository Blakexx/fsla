# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRODUCER: one call from a stretch of tokens to the bundle -- the routes and the values, in the internal format.

    routes, values = produce(x, plan, weights)

The layer declares WHAT it routes (:class:`rola.routing.producer.Routing`), owns the weights, and compiles a
:class:`ProducerPlan` once; this is the one primitive that computes the bundle from them. It is the whole job --
the projection, every level's solve, the mass fold, the write gain, the quantization into unit-max rows and the
value digits -- because every step reads the same declaration, and it is one call so that one fused kernel stands
behind it (docs/api.md section 2): on CUDA with no gradient, the producer kernel's one launch
(docs/internals/produce/produce.md). The torch composition is the reference -- fp64, the host -- and, until the
native backward has the kernel write the levels a gradient reads, the path a gradient takes. The bundles it
returns are the ones the ops take; a caller with its own levels builds them with :meth:`RouteFactors.from_levels` and
:meth:`Values.from_projection`, the same format utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange

from rola.engine.facts.operands import requires_backward
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.factors import U8_MAX_DEPTH, RouteFactors
from rola.routing.producer import Routing
from rola.routing.projection import _packed_router_logits
from rola.routing.side_gain import WRITE_GAIN, softplus_side_gain
from rola.routing.sparse import SparseLevel, slot_of, word_count
from rola.routing.types import IndependentRouting, TiedRouting, UnionRouting
from rola.values import DEFAULT_VALUE_DIGITS, Values

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "ProducerPlan", "ProducerWeights", "SideLayout", "hidden_contiguous",
           "job_table", "kernel_bundle", "kernel_outputs", "launch", "ledger_bind", "produce", "projection_dtype",
           "route_logits", "run_kernel", "side_layouts"]


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


#: THE JOB TABLE's kinds and solves, mirrored from csrc/rola/src/produce/produce.cuh (`JobKind`, `SolveKind`).
_ROUTE, _VALUE = 0, 1
_SOLVES = {("softmax", None): 0, ("entmax", 1.5): 1, ("entmax", 2.0): 2, ("union", 1.5): 3, ("union", 2.0): 4}
#: the kernel's widest job (a union level's two rows, or a value job's channels) and widest solve row
_MAX_COLUMNS = 512
_MAX_WIDTH = 256


def _solve(activation) -> int:
    """The kernel's solve for an activation, by its registry tag."""
    if activation.tag not in _SOLVES:
        raise ValueError(f"the producer kernel has no solve for the activation {activation.tag}")
    return _SOLVES[activation.tag]


@dataclass(frozen=True)
class SideLayout:
    """Where one side's levels land in the kernel's outputs: a dense level at its column ``out`` in the side's packed
    rows (``dense`` wide), a sparse level at its first row ``out`` of the side's sparse buffers (``sparse`` rows).
    docs/internals/produce/produce.md#sparse"""

    is_sparse: tuple[bool, ...]
    out: tuple[int, ...]
    dense: int
    sparse: int


def side_layouts(plan: ProducerPlan, tokens: int) -> tuple[SideLayout, SideLayout]:
    """``(read, write)``: each side's levels packed by form, in level order. The producer stores a side sparse where
    its activation has exact zeros -- over more than one token; a one-token call is all rows."""
    topology, widths = plan.routing.topology, plan.routing.resolved.branches
    layouts = []
    for side in ("read", "write"):
        is_sparse = tuple(tokens > 1 and topology.activation_properties(level, side).exact_zeros
                          for level in range(topology.D))
        out, dense, sparse = [], 0, 0
        for level, width in enumerate(widths[:topology.D]):
            out.append(sparse if is_sparse[level] else dense)
            if is_sparse[level]:
                sparse += width
            else:
                dense += width
        layouts.append(SideLayout(is_sparse, tuple(out), dense, sparse))
    return layouts[0], layouts[1]


def job_table(plan: ProducerPlan, tokens: int = 2) -> tuple[list[tuple[int, ...]], int]:
    """``(jobs, route jobs a head)``: the plan compiled into the kernel's job table (docs/internals/produce/produce.md
    #jobs). A route job is one head's one solve -- a union level's two logit rows, a tied level's one row filling both
    planes, or one side of an independent level; a value job is the heads whose channels fit one job. A job row is
    ``(kind, head, heads, solve, width, in0, in1, out_read, out_write, level, fold_read, fold_write, sparse_read,
    sparse_write)``, its sides' outputs placed by :func:`side_layouts` for a call of ``tokens`` tokens."""
    resolved = plan.routing.resolved
    slots, read_slot = resolved.packed_slot_offsets, resolved.packed_read_slot_lookup
    rl, wl = side_layouts(plan, tokens)
    per_head: list[tuple[int, ...]] = []
    for level, spec in enumerate(resolved.levels):
        width = resolved.branches[level]
        if width > _MAX_WIDTH:
            raise ValueError(f"the producer solves rows of at most {_MAX_WIDTH} leaves; level {level} is {width} wide")
        write_col, read_col = slots[level], slots[read_slot[level]]
        fold_read, fold_write = int(not plan.read_skip[level]), int(not plan.write_skip[level])
        ro, wo = rl.out[level], wl.out[level]
        rs, ws = int(rl.is_sparse[level]), int(wl.is_sparse[level])
        if isinstance(spec, UnionRouting):
            per_head.append((_ROUTE, 0, 1, _SOLVES[("union", spec.alpha)], width, read_col, write_col, ro, wo, level,
                             fold_read, fold_write, rs, ws))
        elif isinstance(spec, TiedRouting):
            per_head.append((_ROUTE, 0, 1, _solve(spec.op), width, write_col, -1, ro, wo, level,
                             fold_read, fold_write, rs, ws))
        elif isinstance(spec, IndependentRouting):
            per_head.append((_ROUTE, 0, 1, _solve(spec.read), width, read_col, -1, ro, -1, level,
                             fold_read, 0, rs, 0))
            per_head.append((_ROUTE, 0, 1, _solve(spec.write), width, write_col, -1, -1, wo, level,
                             0, fold_write, 0, ws))
    jobs = [(job[0], head, *job[2:]) for head in range(plan.num_heads) for job in per_head]
    if plan.d_v > _MAX_COLUMNS:
        raise ValueError(f"a value job holds at most {_MAX_COLUMNS} channels, one head's d_v={plan.d_v} does not fit")
    heads_per = max(1, _MAX_COLUMNS // 2 // plan.d_v)
    for head in range(0, plan.num_heads, heads_per):
        jobs.append((_VALUE, head, min(heads_per, plan.num_heads - head), 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0))
    return jobs, len(per_head)


def _shape(job: tuple[int, ...]) -> int:
    """A job's control as the kernel reads it, one int16 word in the launch's parameters: ``sparse_write << 11 |
    sparse_read << 10 | fold_write << 9 | fold_read << 8 | writes << 7 | reads << 6 | kind << 5 | rows << 3 | solve``,
    ``rows`` a solve row's elements a lane as a power of two's exponent, ``reads``/``writes`` the sides a route job
    writes, ``fold_read``/``fold_write`` whose fold applies and ``sparse_read``/``sparse_write`` which side is a
    sparse level (produce.cuh `Job`)."""
    (kind, _head, _heads, solve, width, _in0, _in1, out_read, out_write, _level, fold_read, fold_write, sparse_read,
     sparse_write) = job
    rows = 0 if width <= 32 else 1 if width <= 64 else 2 if width <= 128 else 3
    return (sparse_write << 11 | sparse_read << 10 | fold_write << 9 | fold_read << 8 | int(out_write >= 0) << 7
            | int(out_read >= 0) << 6 | kind << 5 | rows << 3 | solve)


@lru_cache(maxsize=64)
def _job_tensor(plan: ProducerPlan, device: str, tokens: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(table, shapes, route jobs a head)`` for a call of ``tokens`` tokens (one, or more: the forms differ): the job
    table's addressing on the device (the head, the heads, the width, the columns in and out, the level) and its
    control words on the host."""
    jobs, per_head = job_table(plan, tokens)
    table = torch.tensor([job[1:3] + job[4:10] for job in jobs], dtype=torch.int32, device=device)
    return table, torch.tensor([_shape(job) for job in jobs], dtype=torch.int16), per_head


#: the per-(token tile, head) arrival counters and the per-(job, side, sequence, window) look-back flags, zero between
#: calls (the kernel resets its own), one buffer each a device, grown as a call needs
_COUNTERS: dict[tuple[str, str], torch.Tensor] = {}
_TOKEN_TILE = 32


def _counters(kind: str, device: torch.device, count: int) -> torch.Tensor:
    key = (kind, str(device))
    if key not in _COUNTERS or _COUNTERS[key].numel() < count:
        _COUNTERS[key] = torch.zeros(max(count, 1), dtype=torch.int32, device=device)
    return _COUNTERS[key]


#: THE PHASE LEDGER'S SHAPE: ``[jobs][warps][phases]``, the kernel's own `ProducePhase` order, booked over the tiles.
LEDGER_PHASES = ("job", "project", "barrier", "row_load", "threshold", "levels", "bundle", "finish", "values")
LEDGER_WARPS = 8


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/produce/produce.md#phase-ledger)
    """Bind the phase ledger the next launches book into, ``[jobs, 8, 9]`` int64 cycles, or ``None``."""
    from rola.ops._ext import extension

    extension().rola_produce_ledger_bind(ledger)


def kernel_outputs(plan: ProducerPlan, B: int, T: int, device: torch.device) -> dict[str, torch.Tensor | None]:
    """Every buffer the producer kernel writes for ``B`` sequences of ``T`` tokens, unwritten: each side's dense
    levels as one packed plane head-major ``[B, H, T, rows]`` (every dense output is head-major -- a head's tokens
    contiguous, which every prefill reader walks, and the same bytes as token-major at one token,
    docs/internals/produce/produce.md#bundle), and each side's sparse buffers ``words_*``, ``terms_*``
    (docs/internals/produce/produce.md#sparse)."""
    H, D = plan.num_heads, plan.routing.topology.D
    row_dtype = torch.uint8 if D <= U8_MAX_DEPTH else torch.float16

    def empty(*shape, dtype=torch.float32):
        return torch.empty(B, H, T, *shape, dtype=dtype, device=device)

    out = {}
    slot = slot_of(T)
    for side, layout in zip(("read", "write"), side_layouts(plan, T)):
        out[f"amp_{side}"] = empty(layout.dense, dtype=row_dtype)
        out[f"words_{side}"] = torch.empty(B, H, layout.sparse, word_count(T), dtype=torch.int32, device=device)
        out[f"terms_{side}"] = torch.empty(B, H, layout.sparse, -(-T // slot), slot, dtype=row_dtype, device=device)
    out.update(scale_read=empty(D), scale_write=empty(D), mass_read=empty(D), mass_write=empty(D),
               read_mass=empty(), g_write=empty(), v_hat=empty(plan.d_v, dtype=torch.int8),
               v_lo=empty(plan.d_v, dtype=torch.uint8) if plan.value_digits == 2 else None, v_scale=empty())
    return out


def launch(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights, out: dict[str, torch.Tensor | None],
           probes: tuple) -> dict[str, torch.Tensor | None]:
    """The fused producer's ONE launch over bf16 CUDA operands, into ``out`` (:func:`kernel_outputs`'s buffers);
    ``probes`` the logit and level probes, ``(None, None)`` in production. Returns ``out`` with every dense output
    viewed ``[B, T, H, ...]`` and the sparse buffers as allocated."""
    from rola.ops._ext import extension

    B, T, hidden = x.shape
    if hidden % 8:
        raise ValueError(f"the producer kernel reads hidden runs in 16-byte chunks: the hidden size must be a multiple "
                         f"of 8, got {hidden}")
    H, D = plan.num_heads, plan.routing.topology.D
    device = x.device
    jobs, shapes, per_head = _job_tensor(plan, str(device), min(T, 2))
    tiles = -(-(B * T) // _TOKEN_TILE) if T == 1 else B * -(-T // _TOKEN_TILE)
    windows = 1 if T == 1 else -(-T // slot_of(T))
    route_bias = None if weights.route_bias is None else weights.route_bias.float().contiguous()
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).to(torch.bfloat16).contiguous(), weights.route_W, route_bias,
        weights.gain_bias.float().contiguous(), weights.v_W, jobs,
        *(out[name] for name in ("amp_read", "amp_write", "scale_read", "scale_write", "mass_read", "mass_write",
                                 "read_mass", "g_write", "v_hat", "v_lo", "v_scale")),
        _counters("arrivals", device, tiles * H), out["words_read"], out["words_write"], out["terms_read"],
        out["terms_write"], _counters("flags", device, jobs.shape[0] * 2 * B * windows * 2), *probes, shapes, D,
        per_head)
    return {name: t if t is None or name.startswith(("words_", "terms_")) else t.transpose(1, 2)
            for name, t in out.items()}


def run_kernel(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> dict[str, torch.Tensor | None]:
    """The producer kernel over fresh buffers: :func:`launch` into :func:`kernel_outputs`."""
    B, T, _ = x.shape
    return launch(x, plan, weights, kernel_outputs(plan, B, T, x.device), (None, None))


def produce(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """``x [B, T, hidden]`` -> ``(RouteFactors, Values)``, the bundle the ops take: the kernel's on CUDA with no
    gradient, the composition's otherwise."""
    if x.ndim != 3 or x.shape[-1] != plan.hidden_size:
        raise ValueError(f"x must be [B, T, {plan.hidden_size}], got {tuple(x.shape)}")
    operands = (x, weights.route_W, weights.route_bias, weights.gain_bias, weights.v_W)
    if x.is_cuda and x.dtype != torch.float64 and not requires_backward(*(t for t in operands if t is not None)):
        return _kernel_bundle(x, plan, weights)
    return _composition(x, plan, weights)


def _kernel_bundle(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    return kernel_bundle(run_kernel(x, plan, weights), plan)


def kernel_bundle(out: dict[str, torch.Tensor], plan: ProducerPlan) -> tuple[RouteFactors, Values]:
    """:func:`run_kernel`'s outputs as the bundle: a dense level's rows and every scale a view of the packed planes
    the kernel writes, a sparse level a view of its side's buffers (the ops read them by address), and the read mass
    only where a read level folds."""
    D, widths = plan.routing.topology.D, plan.routing.resolved.branches

    def entries(side: str, layout: SideLayout) -> tuple:
        plane, words, terms = out[f"amp_{side}"], out[f"words_{side}"], out[f"terms_{side}"]
        levels = []
        for level in range(D):
            o, w = layout.out[level], widths[level]
            if not layout.is_sparse[level]:
                levels.append(plane[..., o:o + w])
            else:
                levels.append(SparseLevel(words=words[:, :, o:o + w], terms=terms[:, :, o:o + w]))
        return tuple(levels)

    def scales(plane: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(plane[..., level] for level in range(D))

    rl, wl = side_layouts(plan, out["scale_read"].shape[1])
    routes = RouteFactors(topology=plan.routing.topology, amp_read=entries("read", rl),
                          amp_write=entries("write", wl), scale_read=scales(out["scale_read"]),
                          scale_write=scales(out["scale_write"]), g_write=out["g_write"],
                          read_mass=None if all(plan.read_skip) else out["read_mass"])
    return routes, Values(v_hat=out["v_hat"], v_lo=out["v_lo"], scale=out["v_scale"])


def _composition(x: torch.Tensor, plan: ProducerPlan, weights: ProducerWeights) -> tuple[RouteFactors, Values]:
    """The producer in torch: the fp64 and host reference, and the path a gradient takes."""
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

    #: `v` FLOWS bf16 into the format (docs/api.md section 4.1); the fp64 reference computes it in fp64
    v_dtype = torch.float64 if x.dtype == torch.float64 else torch.bfloat16
    v = F.linear(x.to(v_dtype), weights.v_W.to(v_dtype))
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
