# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The producer interface, and the simplex producer this package ships.

RoLA is linear attention under a structured feature map, so the feature map is the
one slot the public surface has. A PRODUCER fills that slot for one or more routing
levels: it turns a source stream into the read-side and write-side simplices those
levels route with. :class:`LevelProducer` is the whole contract, and it is DUCK-TYPED
-- the library never asks what a producer is, only that its outputs are well formed
(``docs/api.md``). Correctness is producer-blind by construction: support bits are
derived from the returned tensors, never from anything a producer says about
itself.

**The interface, in full.**

``widths``      one branch width per level the producer covers, in its own order.
``forward``     one call producing EVERY level the producer covers.
``dense_sides`` optional structural declaration, per covered level (see below).

**One call per producer.** The gateway calls each producer once, with every level
that producer covers, and places the results in the slots the list assigned them. A
producer covering several levels is therefore how a model gets ONE launch across them:
:class:`SimplexProducer` takes a width tuple and runs the whole group through one
packed projection and one batched solve, which is the shape every shipped model uses.
Cross-producer batching -- a `fuse` protocol letting the gateway merge two producers'
launches -- is deliberately NOT built: the prize is launch overhead across at most four
levels, nobody has measured it to matter, and the machinery (merge, scatter, a
bit-identity gate) is real. Revisit if a measurement asks for it; the docket entry for
this span records the design that was worked out and set down.

**The gateway owns packing.** A producer returns amplitudes in the plain documented
layout, one ``[B, T, H, width_l]`` tensor per level, and knows nothing about how the
kernel wants them arranged. Turning those into the folded, masked view the schedule and
the summary pass consume is the gateway's job and lives in one place.

**Stat-expose is optional and never load-bearing.** A producer that already knows a
summary statistic as a byproduct of its solve may return it on
:class:`ProducedRoutes`; the gateway measures whatever is missing. A supplied statistic
feeds PRICING only -- membership and support bits are always recomputed from the
tensors -- so a wrong one costs its author performance and can never cost anyone
correctness.

**The shipped producer.** :class:`SimplexProducer` learns a packed per-head router
projection and solves it with the exact production entmax/softmax closed form
(``entmax/production.py``). It is a duck like any other: it implements this interface
and gets no privileged path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from rola.routing.budget import exact_support_count
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.projection import _packed_router_logits
from rola.routing.types import (
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    SoftmaxActivation,
    UnionRouting,
)

__all__ = [
    "LevelProducer",
    "ProducedRoutes",
    "RouteSource",
    "SimplexProducer",
    "dense_routing",
    "sparse_routing",
]


def dense_routing() -> IndependentRouting:
    """Softmax on both duties, untied -- the full-support level."""
    return IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                              write_activation=SoftmaxActivation())


def sparse_routing(alpha: float = 1.5) -> UnionRouting:
    """One exact entmax solve determining a shared support, split across duties."""
    return UnionRouting(alpha)


@dataclass(frozen=True, slots=True)
class RouteSource:
    """Everything a producer is given, for every level it covers.

    ``read``/``write`` are ``[B, T, hidden_size]`` source streams; they are the SAME
    tensor object whenever the model does not split them, which a producer may use to
    take a single-projection path.
    """

    read: torch.Tensor
    write: torch.Tensor

    @property
    def batch(self) -> int:
        return self.read.shape[0]

    @property
    def tokens(self) -> int:
        return self.read.shape[1]

    @property
    def shared_stream(self) -> bool:
        return self.write is self.read


@dataclass(frozen=True, slots=True)
class ProducedRoutes:
    """One producer's answer, in ITS OWN level order.

    ``read``/``write``: one ``[B, T, H, width_l]`` tensor per covered level. They need
    not be unit-sum -- the gateway factors each level into ``(mass, simplex)`` and
    folds the mass into the side gain, which is where architecture §4.2 puts a
    per-level magnitude.

    ``support_counts``/``touched_bits``: OPTIONAL stat-expose, keyed by the producer's
    own level index. Anything absent the gateway measures from the tensors; anything
    present is used for PRICING only.
    """

    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    support_counts: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    touched_bits: dict[int, torch.Tensor] = field(default_factory=dict)


class LevelProducer(nn.Module):
    """The feature-map slot's contract. Subclass it, or duck-type it.

    Subclassing buys nothing but the name: a module that declares ``widths`` and
    returns a well-formed :class:`ProducedRoutes` satisfies the gateway without
    inheriting anything.
    """

    #: One branch width per covered level, in the order ``forward`` returns them.
    widths: tuple[int, ...] = ()

    def forward(self, source: RouteSource) -> ProducedRoutes:  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(source) -> ProducedRoutes")


class SimplexProducer(LevelProducer):
    """The shipped producer: a learned packed router projection and an exact solve.

    ONE producer may cover SEVERAL levels, and that is how a model gets one launch
    across them: the covered levels share one packed projection and one batched solve,
    dispatched per level inside the kernel without a privileged level. A model whose
    levels all use this producer therefore pays one GEMM and one solve for the whole
    topology, which is what the shipped configurations do.

    Args:
        widths: the branch width ``b_l`` of each level this producer fills -- an int
            for one level, a tuple for several.
        routing: the routing tag for those levels -- :func:`dense_routing`,
            :func:`sparse_routing`, or any ``IndependentRouting``/``UnionRouting``. One
            tag broadcasts to every covered level; a sequence gives per-level control.
            The tag names activations from the closed registry
            (``rola/routing/activations.py``); users never supply property metadata.
        hidden_size / num_heads: the source stream's width and the head count.
        bias: a learned bias on the routing projection.
        expose_stats: return exact support counts as a solve byproduct instead of
            leaving the gateway to measure them (the fused-vs-separate arm; both arms
            produce identical numbers -- see ``docs/api.md`` for which one is faster
            and by how much).
    """

    def __init__(
        self,
        widths: int | tuple[int, ...],
        routing: LevelRouting | list | tuple | None = None,
        *,
        hidden_size: int,
        num_heads: int,
        bias: bool = False,
        expose_stats: bool = False,
    ) -> None:
        super().__init__()
        _widths = (widths,) if type(widths) is int else tuple(widths)
        if not _widths or any(type(width) is not int for width in _widths):
            raise TypeError(
                f"SimplexProducer widths must be an exact int or a tuple of them, got {widths!r}")
        if routing is None:
            routing = dense_routing()
        _levels = (tuple(routing) if isinstance(routing, (list, tuple))
                   else (routing,) * len(_widths))
        if len(_levels) != len(_widths):
            raise ValueError(
                f"SimplexProducer got {len(_levels)} routing tags for {len(_widths)} widths")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError(f"hidden_size must be a positive exact int, got {hidden_size!r}")
        if type(num_heads) is not int or num_heads < 1:
            raise ValueError(f"num_heads must be a positive exact int, got {num_heads!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bool(bias)
        self.expose_stats = bool(expose_stats)
        self.levels = tuple(_levels)
        self.widths = tuple(_widths)
        self.routing = ResolvedRouting(levels=self.levels, branches=self.widths)

        self.route_W = nn.Parameter(
            torch.empty(num_heads, hidden_size, self.routing.packed_router_width))
        _init_packed_router(self.route_W)
        if self.bias:
            self.route_bias = nn.Parameter(
                torch.zeros(num_heads, self.routing.packed_router_width))
        else:
            self.register_parameter("route_bias", None)

    # --- pricing columns -----------------------------------------------------

    @property
    def level_routing(self) -> tuple[LevelRouting, ...]:
        """The routing tag of each level this producer covers, for the pricing columns."""
        return self.levels

    # --- production ----------------------------------------------------------

    def forward(self, source: RouteSource) -> ProducedRoutes:
        x = source.read
        x_w = None if source.shared_stream else source.write
        B, _, hidden = x.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"SimplexProducer was built for hidden_size={self.hidden_size} but the source "
                f"stream is {hidden} wide")
        H = self.num_heads
        read_bh, write_bh = self._solve(x, x_w)

        read = tuple(_unfold_bth(level, B, H) for level in read_bh)
        write = tuple(_unfold_bth(level, B, H) for level in write_bh)
        counts: dict[tuple[int, str], torch.Tensor] = {}
        if self.expose_stats:
            # THE FUSED ARM. The counts ride the solve's own output while it is still
            # in the producer's hands, in the layout the solve produced. The gateway's
            # arm measures the same thing from the unfolded tensors; both are exact
            # nonzero counts of the same values, so this is a placement decision and
            # never a numerical one.
            for index in range(len(self.widths)):
                counts[(index, "read")] = _unfold_count(exact_support_count(read_bh[index]), B, H)
                counts[(index, "write")] = _unfold_count(exact_support_count(write_bh[index]), B, H)
        return ProducedRoutes(read=read, write=write, support_counts=counts)

    def _project(self, x: torch.Tensor, x_w: torch.Tensor | None):
        """Packed router logits in the domain the caller's dtype selects.

        fp64 in, fp64 out (the oracle-equivalence gate runs the whole layer there);
        anything else projects in fp32, which is the precision guard a bf16 training
        run must still get on its routing logits.
        """
        projection_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        return _packed_router_logits(
            x, self.route_W, self.routing, route_bias=self.route_bias, h_w=x_w,
            projection_dtype=projection_dtype)

    def _solve(self, x: torch.Tensor, x_w: torch.Tensor | None):
        read_logits, write_logits = self._project(x, x_w)
        return production_routing_factor_levels(read_logits, write_logits, self.routing)


# --- helpers ----------------------------------------------------------------


def _init_packed_router(W: nn.Parameter) -> None:
    """Init ``[H, hidden, packed_router_width]`` like an ``nn.Linear(hidden, cols)``.

    The routing logit is ``h @ W[head]``, so the init must be a TRANSPOSED linear
    weight with ``fan_in == hidden``. ``kaiming_uniform_(a=sqrt(5))`` on ``[out,
    hidden]`` draws ``U(-1/sqrt(hidden), +1/sqrt(hidden))`` per element -- the bound
    does not depend on ``out`` -- so every level gets exactly the per-level fan-in
    semantics the projection GEMM has, one dot product over the full hidden dim per
    column.
    """
    H, hidden, cols = W.shape
    wt = torch.empty(H * cols, hidden, dtype=W.dtype, device=W.device)
    nn.init.kaiming_uniform_(wt, a=5 ** 0.5)
    W.data.copy_(wt.view(H, cols, hidden).transpose(-1, -2))


def _unfold_bth(tensor: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """``[B*H, T, width] -> [B, T, H, width]``, the public factor layout, as a VIEW.

    Deliberately not made contiguous. The solve emits ``[BH, T, width]``, which is
    already the layout the kernel's view wants; materializing the plain layout here and
    folding it back downstream is an inverse copy pair that costs a measured 2.8% of the
    step at ``L = 4096``. Returning a view keeps the fold free -- the permute the gateway
    applies is exactly this one undone -- while the public tensors still read as
    ``[B, T, H, width_l]`` for every consumer that wants them that way.
    """
    T, width = tensor.shape[1], tensor.shape[2]
    return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)


def _unfold_count(counts: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """``[B*H, T] -> [B, T, H]``, matching the public statistic layout."""
    return counts.reshape(B, H, -1).permute(0, 2, 1).contiguous()
