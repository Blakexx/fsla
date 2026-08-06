# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The producer interface, and the simplex producer this package ships.

RoLA is linear attention under a structured feature map, so the feature map is the
one slot the public surface has. A PRODUCER fills that slot for one or more routing
levels: it turns a source stream into the read-side and write-side simplices those
levels route with. :class:`LevelProducer` is the whole contract, and it is DUCK-TYPED
-- the library never asks what a producer is, only that its outputs are well formed
(``docs/api.md``). Correctness is producer-blind by construction: support bits and
summaries are derived from the returned tensors, never from anything a producer says
about itself.

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
(``entmax/production.py``), including the architecture §4.5 conditioning surface
(Tier-1 temperature actuation, the budget embedding, the usage-conditioned per-digit
bias). It is a duck like any other: it implements this interface and gets no
privileged path.

Flagged model choices, unchanged from the pre-slot producer and recorded because
architecture §4.5 states the mechanism's words but not an exact functional form: the
temperature head and the usage-bias head ``g`` are single learned linear maps over
their stated inputs; side gains are projected from the read source stream only; ``g``
applies uniformly to every write level when its ablation gate is on; and both
actuators consume the previous chunk's slack/usage (one-chunk lag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from rola.routing.budget import (
    BudgetConditioning,
    chunk_boundaries,
    chunk_target_count,
    exact_support_count,
    normalized_slack,
    touched_bitset_update,
    update_soft_usage,
)
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.projection import _packed_router_logits
from rola.routing.types import (
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    SoftmaxActivation,
    UnionRouting,
)

_TEMPERATURE_FLOOR = 1e-2

__all__ = [
    "LevelProducer",
    "ProducedRoutes",
    "RouteSource",
    "SimplexProducer",
    "dense_routing",
    "init_temp_bias",
    "sparse_routing",
]


def init_temp_bias(target_c: float) -> float:
    """Invert the Tier-1 temperature actuator's positive parameterization.

    ``c = softplus(bias) + _TEMPERATURE_FLOOR``, so this places a zero-input-logit
    actuator at exactly ``target_c``. Configuring a budgeted level must not itself
    perturb an otherwise-baseline model's routing distribution before any training
    signal exists, which is why the default target is the neutral ``c = 1`` rather
    than the raw-zero-bias value (``c = 0.7031``).
    """
    if target_c <= _TEMPERATURE_FLOOR:
        raise ValueError(
            f"target_c must be > the temperature floor ({_TEMPERATURE_FLOOR}), got {target_c}")
    return math.log(math.expm1(target_c - _TEMPERATURE_FLOOR))


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
    take a single-projection path. ``budget`` is the per-sequence ``[B]`` target
    support fraction, or ``None`` for the unconditioned baseline.
    """

    read: torch.Tensor
    write: torch.Tensor
    budget: torch.Tensor | None = None

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

    ``support_counts``/``touched_bits``/``slack_history``: OPTIONAL stat-expose, keyed
    by the producer's own level index. Anything absent the gateway measures from the
    tensors; anything present is used for PRICING only.
    """

    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    support_counts: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    touched_bits: dict[int, torch.Tensor] = field(default_factory=dict)
    slack_history: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)


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
        budget: architecture §4.5 conditioning, whose level indices address THIS
            producer's own levels.
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
        budget: BudgetConditioning | None = None,
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
        if budget is not None and type(budget) is not BudgetConditioning:
            raise TypeError(f"budget must be None or BudgetConditioning, got {budget!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.bias = bool(bias)
        self.expose_stats = bool(expose_stats)
        self.levels = tuple(_levels)
        self.widths = tuple(_widths)
        self.routing = ResolvedRouting(levels=self.levels, branches=self.widths)
        self.budget = budget
        if budget is not None:
            budget.validate_against(self.routing)

        self.route_W = nn.Parameter(
            torch.empty(num_heads, hidden_size, self.routing.packed_router_width))
        _init_packed_router(self.route_W)
        if self.bias:
            self.route_bias = nn.Parameter(
                torch.zeros(num_heads, self.routing.packed_router_width))
        else:
            self.register_parameter("route_bias", None)
        self._allocate_conditioning()

    # --- conditioning parameters (architecture §4.5) -------------------------

    def _allocate_conditioning(self, temp_bias_init: float = 1.0) -> None:
        budget = self.budget
        if budget is None:
            for name in ("temp_W", "temp_b", "usage_bias_W", "usage_bias_b", "budget_embed_W"):
                self.register_parameter(name, None)
            self._temp_bias_target = None
            return
        pairs = budget.budgeted_pairs
        if pairs:
            self.temp_W = nn.Parameter(torch.zeros(self.num_heads, len(pairs), 2))
            self.temp_b = nn.Parameter(
                torch.full((self.num_heads, len(pairs)), init_temp_bias(temp_bias_init)))
            self._temp_bias_target = float(temp_bias_init)
            self._temp_b_init_snapshot = self.temp_b.data.clone()
        else:
            self.register_parameter("temp_W", None)
            self.register_parameter("temp_b", None)
            self._temp_bias_target = None
        if budget.usage_bias_enabled:
            self.usage_bias_W = nn.Parameter(torch.zeros(self.num_heads, len(self.widths), 4))
            self.usage_bias_b = nn.Parameter(torch.zeros(self.num_heads, len(self.widths)))
        else:
            self.register_parameter("usage_bias_W", None)
            self.register_parameter("usage_bias_b", None)
        self.budget_embed_W = nn.Parameter(torch.zeros(self.hidden_size))

    def _apply(self, fn, recurse=True):
        """Keep ``temp_b``'s init target exact under a dtype-widening transform.

        ``temp_b`` is materialized fp32-rounded, so a naive cast to fp64 would carry
        that rounding forward and never recover the exact :func:`init_temp_bias`
        target. Decompose into ``init_snapshot + drift``, cast the drift like any
        tensor, and RECOMPUTE the init term in the post-``fn`` dtype.
        """
        temp_b = self._parameters.get("temp_b")
        if temp_b is None or self._temp_bias_target is None:
            return super()._apply(fn, recurse=recurse)
        del self._parameters["temp_b"]
        try:
            result = super()._apply(fn, recurse=recurse)
            with torch.no_grad():
                cast_ref = fn(temp_b.data)
                exact_init = torch.full_like(cast_ref, init_temp_bias(self._temp_bias_target))
                temp_b.data = exact_init + fn(temp_b.data - self._temp_b_init_snapshot)
                if temp_b.grad is not None:
                    temp_b.grad = fn(temp_b.grad)
                self._temp_b_init_snapshot = exact_init.clone()
            return result
        finally:
            self._parameters["temp_b"] = temp_b

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
        r = source.budget
        if r is not None and self.budget_embed_W is not None:
            x = _embed_budget(x, r, self.budget_embed_W)
            if x_w is not None:
                x_w = _embed_budget(x_w, r, self.budget_embed_W)

        conditioned = self.budget is not None and (
            (r is not None and self.budget.budgeted_pairs) or self.budget.usage_bias_enabled)
        if conditioned:
            read_bh, write_bh, slack = self._conditioned_solve(x, x_w, r, B=B, H=H)
        else:
            read_bh, write_bh = self._solve(x, x_w)
            slack = {}

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
        return ProducedRoutes(read=read, write=write, support_counts=counts, slack_history=slack)

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

    def _conditioned_solve(self, x, x_w, r, *, B: int, H: int):
        """Tier-1 temperature actuation and the usage-bias head, chunk-recurrent.

        Both actuators consume the PREVIOUS chunk's slack/usage (one-chunk lag), so
        this is a small explicit recurrence over ``budget.chunk_size`` slices. The
        projection is hoisted out of it: it is a pure per-token function of the source
        stream, so projecting once and slicing is bitwise identical to projecting per
        chunk and removes one launch per chunk.

        ``r`` may be ``None``: Tier 1 structurally requires it (its target is defined
        in terms of it) and stays inert, while the usage-bias head keeps running with
        its ``r``/slack features clamped to a neutral zero -- two mechanisms with two
        ablation gates, not one.
        """
        budget = self.budget
        device, dtype = x.device, x.dtype
        widths = self.widths
        D = len(widths)
        pairs = budget.budgeted_pairs
        pair_index = {pair: i for i, pair in enumerate(pairs)}
        L = x.shape[1]
        read_logits_full, write_logits_full = self._project(x, x_w)
        actuators_active = (r is not None and bool(pair_index)) or budget.usage_bias_enabled

        def head_param(param_over_h: torch.Tensor) -> torch.Tensor:
            return param_over_h.unsqueeze(0).expand(B, *param_over_h.shape).reshape(
                B * H, *param_over_h.shape[1:])

        r_bh = None if r is None else r.to(device=device, dtype=dtype)[:, None].expand(
            B, H).reshape(B * H)
        running_slack = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
        running_count = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
        slack_history: dict[tuple[int, str], list[torch.Tensor]] = {pair: [] for pair in pairs}
        running_usage = {level: torch.zeros(B * H, widths[level], dtype=dtype, device=device)
                         for level in range(D)}
        running_touched = {level: torch.zeros(B * H, widths[level], dtype=torch.bool, device=device)
                           for level in range(D)}
        tokens_seen = 0
        read_chunks: list[list[torch.Tensor]] = [[] for _ in range(D)]
        write_chunks: list[list[torch.Tensor]] = [[] for _ in range(D)]

        for start, end in chunk_boundaries(L, budget.chunk_size):
            read_logits = read_logits_full[:, start:end]
            write_logits = write_logits_full[:, start:end]
            # The two logit tensors MAY be one object (the fully tied single-solve
            # path), so conditioning accumulates out-of-place buffers and applies them
            # with one multiply-add rather than mutating either in place.
            if actuators_active:
                write_scale = torch.ones_like(write_logits)
                write_bias = torch.zeros_like(write_logits)
                read_scale = torch.ones_like(read_logits)
                if r_bh is not None:
                    for (level, side), idx in pair_index.items():
                        cond = torch.stack([r_bh, running_slack[(level, side)]], dim=-1)
                        temp_logit = (cond * head_param(self.temp_W[:, idx, :])).sum(-1)
                        if self.temp_b is not None:
                            temp_logit = temp_logit + head_param(
                                self.temp_b[:, idx:idx + 1]).squeeze(-1)
                        c = F.softplus(temp_logit) + _TEMPERATURE_FLOOR
                        buffer = write_scale if side == "write" else read_scale
                        buffer[:, :, self.routing.level_logit_slice(level)] = c[:, None, None]
                if budget.usage_bias_enabled:
                    r_feature = r_bh if r_bh is not None else torch.zeros(
                        B * H, dtype=dtype, device=device)
                    for level in range(D):
                        usage_norm = running_usage[level] / max(tokens_seen, 1)
                        touched_f = running_touched[level].to(dtype)
                        write_slack = running_slack.get(
                            (level, "write"), torch.zeros(B * H, dtype=dtype, device=device))
                        w0, w1, w2, w3 = (head_param(self.usage_bias_W[:, level, i:i + 1])
                                          for i in range(4))
                        offset = (w0 * usage_norm + w1 * touched_f
                                  + w2 * r_feature[:, None] + w3 * write_slack[:, None]
                                  + head_param(self.usage_bias_b[:, level:level + 1]))
                        span = self.routing.level_logit_slice(level)
                        write_bias[:, :, span] = write_bias[:, :, span] + offset[:, None, :]
                write_logits = write_logits * write_scale + write_bias
                read_logits = read_logits * read_scale

            read_bh, write_bh = production_routing_factor_levels(
                read_logits, write_logits, self.routing)
            for level in range(D):
                read_chunks[level].append(read_bh[level])
                write_chunks[level].append(write_bh[level])
            tokens_seen += end - start
            # The bookkeeping below reads the NORMALIZED level, which is what the
            # gateway will store and what the recurrence's clock is defined on.
            for level in range(D):
                stored = _normalized(write_bh[level])
                running_usage[level] = update_soft_usage(running_usage[level], stored)
                running_touched[level] = touched_bitset_update(running_touched[level], stored)
            for (level, side) in pairs:
                values = write_bh[level] if side == "write" else read_bh[level]
                if r_bh is not None:
                    running_count[(level, side)] = running_count[(level, side)] + exact_support_count(
                        values).detach().sum(dim=-1)
                    running_slack[(level, side)] = normalized_slack(
                        chunk_target_count(r_bh, widths[level], tokens_seen),
                        running_count[(level, side)])
                slack_history[(level, side)].append(running_slack[(level, side)])

        read = tuple(torch.cat(read_chunks[level], dim=1) for level in range(D))
        write = tuple(torch.cat(write_chunks[level], dim=1) for level in range(D))
        slack = {pair: torch.stack(hist, dim=-1).reshape(B, H, -1)
                 for pair, hist in slack_history.items()}
        return read, write, slack


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


def _embed_budget(hidden: torch.Tensor, r: torch.Tensor, embed_W: torch.Tensor) -> torch.Tensor:
    """Add the per-sequence budget scalar into a router source stream (§4.5)."""
    if r.ndim != 1 or r.shape[0] != hidden.shape[0]:
        raise ValueError(f"budget must have shape [B]=[{hidden.shape[0]}], got {tuple(r.shape)}")
    return hidden + r.to(hidden.dtype)[:, None, None] * embed_W.to(hidden.dtype)


def _normalized(level: torch.Tensor) -> torch.Tensor:
    return level / level.sum(dim=-1, keepdim=True)


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
