"""V3 routing producer (``ROLA_V3_SPEC.md`` §10 Stage 2; architecture §4.5).

Every configured level is produced **uniformly** through the fused production
solve (`entmax/production.py:production_routing_factor_levels` -- it dispatches
on `IndependentRouting`/`UnionRouting` without a privileged level and never
materializes a leaf tensor). That solve is the exact sorted closed form for
alpha=1.5/2.0 with its analytic fixed-alpha VJP, lowered to Triton on CUDA fp32
and to the identical torch closed form on CPU/fp64. The fp64 bisection in
`entmax/reference.py` (80 iterations of small elementwise kernels per level, on
the backward too) is **not** on this path any more -- it is the test oracle the
producer is gated against, never a runtime fallback. This module adds exactly
three things on top of that solve:

1. the `Topology` -> `ResolvedRouting` adapter that lets the (unchanged)
   packed-router projection and the production solve consume a V3
   `Topology` (they already only need `IndependentRouting | UnionRouting`
   levels plus branch widths -- see `types.py`'s own docstring);
2. active side gains (`side_gain.py`);
3. the §4.5 conditioning surface: Tier 1's per-side temperature actuator,
   the budget embedding, chunk-boundary usage/slack bookkeeping, and the
   usage-conditioned per-digit logit bias `g`.

**Flagged simplifications** (recorded per the task brief: "if genuinely
ambiguous, implement the simplest form satisfying §4.5's words and record the
choice prominently" -- architecture §4.5 states the mechanism's words but not
an exact functional form for the temperature head or `g`):

- The temperature head and `g` are single learned linear maps over their
  stated inputs (`(r, slack)` and `(usage, touched, r, slack)`
  respectively), not deeper MLPs. Nothing in §4.5 requires more.
- Side gains are projected from the read source stream `x` only (not a
  separate `x_w`-conditioned write gain) -- architecture §8's projection
  contract lists gains as ordinary projection outputs alongside value/route
  logits without specifying a source-stream split for them the way it does
  for the optional write-route convolution; using one shared source is the
  simplest reading.
- `g` is applied to **every** write level uniformly when its ablation gate is
  on (spec: "one learned per-digit usage-bias head g ... with an off
  switch"); it is not further scoped per-level, since the spec's off switch
  is described as one mechanism-level gate, not a per-level one.
- The temperature actuator and `g` both consume the **previous chunk's**
  slack/usage (one-chunk lag), matching the "usage through the end of the
  PREVIOUS chunk only" wording given for usage conditioning; applying the
  same lag to Tier 1's own actuator keeps both mechanisms on one information
  cutoff rather than inventing a second one.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rola.routing.budget import (
    DEFAULT_CHUNK_SIZE,
    chunk_boundaries,
    chunk_target_count,
    exact_support_count,
    normalized_slack,
    touched_bitset_update,
    update_soft_usage,
)
from rola.routing.projection import _packed_router_logits
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.side_gain import project_side_gains
from rola.routing.types import ResolvedRouting, Topology

_TEMPERATURE_FLOOR = 1e-2


def init_temp_bias(target_c: float) -> float:
    """Invert the Tier 1 temperature actuator's positive parameterization
    (``c = softplus(bias) + _TEMPERATURE_FLOOR``) to place a zero-input-logit
    actuator at exactly ``target_c``.

    Mirrors ``side_gain.init_side_gain_bias``'s inversion for the analogous
    gain case (D5 fix): configuring a budgeted level must not itself perturb
    an otherwise-baseline model's routing distribution before any training
    signal exists, so the layer's default ``temp_bias_init`` is ``1.0`` (a
    no-op temperature) rather than the raw-zero-bias default that produced
    ``c=0.7031`` at init.
    """
    if target_c <= _TEMPERATURE_FLOOR:
        raise ValueError(
            f"target_c must be > the temperature floor ({_TEMPERATURE_FLOOR}), got {target_c}")
    return math.log(math.expm1(target_c - _TEMPERATURE_FLOOR))


def _project_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    resolved: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None,
    h_w: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project packed router logits, matching `routing/projection.py`'s own two-domain
    convention rather than reintroducing a third one.

    `packed_router_logits` (production) always projects in fp32 regardless of
    the caller's working dtype -- deliberately, the same precision guard V2
    used, so a bf16/fp16 training run still gets fp32 routing logits.
    `packed_router_logits_reference` always projects in fp64, for oracle runs.
    A V3 producer call must pick the SAME domain a caller running the whole
    layer at fp64 (task item 4's oracle-equivalence gate) actually gets --
    fp64 in, fp64 out -- while an ordinary bf16/fp16/fp32 training call still
    gets the fp32 guard. Anything other than exactly fp64 therefore maps to
    fp32, matching `packed_router_logits`'s existing behavior exactly.
    """
    projection_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
    return _packed_router_logits(
        h, route_W, resolved, route_bias=route_bias, h_w=h_w, projection_dtype=projection_dtype)


def resolved_routing_from_topology(topology: Topology) -> ResolvedRouting:
    """Adapt a V3 ``Topology`` into the ``ResolvedRouting`` the packed-router
    projection and ``production_routing_factor_levels`` already consume.

    Both share the same ``IndependentRouting | UnionRouting`` level tags
    (``types.py``'s own docstring: "these types ... already satisfy the V3 §1
    shapes"). No new dispatch is written here.
    """
    return ResolvedRouting(
        levels=tuple(level.routing for level in topology.levels),
        branches=topology.widths,
    )


@dataclass(frozen=True, slots=True)
class BudgetConditioning:
    """Tier 1 + usage-conditioning configuration for one V3 routing plan (architecture §4.5).

    ``budgeted_write_levels``/``budgeted_read_levels``: level indices whose
    activation is temperature-actuated against ``(r, slack)``. Every entry
    must name a zero-capable (entmax) level-side -- softmax sides cannot carry
    a budget (spec: "a budget on a softmax side is a loud config validation
    error, not a no-op"); validated against a concrete topology in
    ``produce_v3_routing``, not here (this dataclass has no topology to check
    against).
    ``usage_bias_enabled``: the ablation gate for the usage-conditioned
    per-digit logit bias ``g`` (applied to every write level when True).
    ``chunk_size``: the training-time conditioning granularity (default 32,
    matching the kernel's default ``BT`` -- spec: "BT is a plan-time kernel
    constant ... derived as r*BT, never defined natively"; this is a
    *training* granularity choice tracking that constant, not the constant
    itself).
    """

    budgeted_write_levels: tuple[int, ...] = ()
    budgeted_read_levels: tuple[int, ...] = ()
    usage_bias_enabled: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        for name, levels in (
            ("budgeted_write_levels", self.budgeted_write_levels),
            ("budgeted_read_levels", self.budgeted_read_levels),
        ):
            levels = tuple(levels)
            if len(set(levels)) != len(levels):
                raise ValueError(f"{name} must not repeat a level index, got {levels!r}")
            if any(type(level) is not int or level < 0 for level in levels):
                raise ValueError(f"{name} entries must be nonnegative exact ints, got {levels!r}")
            object.__setattr__(self, name, levels)
        if type(self.usage_bias_enabled) is not bool:
            raise TypeError("usage_bias_enabled must be bool")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not self.budgeted_pairs and not self.usage_bias_enabled:
            # D8: an entirely inert conditioning object is accepted (not a config error --
            # unlike the softmax-side budget case, which IS a loud ValueError, see
            # `validate_against` below) but must not silently pass as a no-op: a caller who
            # allocates `budget_embed_W`/`RoLA(budget=...)` from this object gets a module
            # that does nothing.
            warnings.warn(
                "BudgetConditioning has no budgeted pairs and usage_bias_enabled=False -- this "
                "conditioning object is fully inert (produce_v3_routing will always take the "
                "unconditioned baseline path for it). If this is intentional, pass budget=None "
                "instead.",
                stacklevel=2,
            )

    @property
    def budgeted_pairs(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (level, "write") for level in self.budgeted_write_levels
        ) + tuple((level, "read") for level in self.budgeted_read_levels)

    def validate_against(self, topology: Topology) -> None:
        D = topology.D
        for level in (*self.budgeted_write_levels, *self.budgeted_read_levels):
            if level >= D:
                raise ValueError(f"budgeted level {level} is out of range for D={D}")
        resolved = resolved_routing_from_topology(topology)
        for level in self.budgeted_write_levels:
            if not resolved.write_level_can_zero[level]:
                raise ValueError(
                    f"budget conditioning on level {level}'s write side requires a zero-capable "
                    "(entmax) activation; softmax sides cannot carry a budget (architecture §4.5)")
        for level in self.budgeted_read_levels:
            if not resolved.read_level_can_zero[level]:
                raise ValueError(
                    f"budget conditioning on level {level}'s read side requires a zero-capable "
                    "(entmax) activation; softmax sides cannot carry a budget (architecture §4.5)")


@dataclass(frozen=True, slots=True)
class V3RoutingOutput:
    """The producer's public contract: exactly ``naive_rola_v3``'s routing inputs, plus diagnostics.

    ``read_levels``/``write_levels``: ``tuple[Tensor, ...]``, one ``[B,T,H,width_l]``
    simplex per level -- the Stage 1 oracle's expected input contract verbatim.
    ``g_read``/``g_write``: ``[B,T,H]`` side gains (``g_read`` is ``None`` for
    ``normalization='global'``, architecture §4.2).
    ``exact_support_counts``: ``{(level, side): [B,T,H]}`` exact per-token
    nonzero counts for every level-side -- the Tier 1 loss's exact measurement
    (spec §4.5: "free", ``O(L*b_l)``), always populated regardless of whether
    budget conditioning is active.
    ``touched_bits``: ``{level: [B,H,width_l]}`` end-of-sequence touched-digit
    masks on the write side (Tier 2 bitset primitive), always populated.
    ``slack_history``: ``{(level, side): [B,H,num_chunks]}`` normalized slack
    at each chunk boundary, populated only for budgeted pairs.
    """

    read_levels: tuple[torch.Tensor, ...]
    write_levels: tuple[torch.Tensor, ...]
    g_read: torch.Tensor | None
    g_write: torch.Tensor
    exact_support_counts: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    touched_bits: dict[int, torch.Tensor] = field(default_factory=dict)
    slack_history: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)


def embed_budget(hidden: torch.Tensor, r: torch.Tensor | None, embed_W: torch.Tensor) -> torch.Tensor:
    """Add the per-sequence budget scalar into a router source stream (spec §4.5).

    ``r``: ``[B]`` or ``None`` (budget off -> ``hidden`` unchanged; this is the
    "sandwich sampling includes budget=off" branch, identical to the
    unconditioned baseline). ``embed_W``: ``[hidden_size]``.
    """
    if r is None:
        return hidden
    if r.ndim != 1 or r.shape[0] != hidden.shape[0]:
        raise ValueError(f"r must have shape [B]=[{hidden.shape[0]}], got {tuple(r.shape)}")
    return hidden + r.to(hidden.dtype)[:, None, None] * embed_W.to(hidden.dtype)


def _unfold_bth(tensor: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """Inverse of the BH-fold convention used throughout ``routing/*.py``."""
    T, width = tensor.shape[1], tensor.shape[2]
    return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3).contiguous()


def _split_level_mass(level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor one level's raw allocation into ``(mass, normalized simplex)``.

    ``production_routing_factor_levels``' output does not individually sum to 1 on
    ``UnionRouting``'s read side (``read = shared_support * sigmoid(delta)``,
    see ``entmax/production.py``'s union kernel) even though the V3 config
    contract (spec §4.1) requires ``sum_d p_read[l,t,d] = 1`` for every level.
    Split the mass out explicitly and fold it into the side gain instead of
    silently presenting an unnormalized level as a simplex -- this is exactly
    the pattern ``tests/ops/test_rola_v3_migration_equivalence.py``'s
    ``_split_level_mass`` already validates: ``mass * normalized == level``
    identically, so nothing is lost, only relocated to where spec §4.2 says a
    per-level magnitude belongs ("multiplicative per-level masses collapse
    into one side-wide product"). For ``IndependentRouting`` levels (already
    unit-sum by construction) ``mass`` is 1 up to floating-point rounding, so
    this is a no-op there.

    This is a loss-free identity for ``W`` (the value read back out of the
    fold always equals the original unnormalized level), but it is NOT
    automatically loss-free for the ``LeafMassDecay`` recurrence: the clock
    ``c[t,s] = prod_l p_write_stored[l,·]`` (architecture §2) is a function of
    the *normalized, stored* level, not of the fold-corrected level -- moving
    write-side mass out of the stored level and into ``g_write`` changes
    ``keep`` unless that write-side mass is identically 1. It happens to be
    exactly 1 today for every routing this module produces (union write is a
    softmax over the support, and independent-routing write activations are
    already unit-sum by construction), so the fold is harmless now, but a
    future write-side activation with mass != 1 would silently change decay
    semantics through this fold and would need its own accounting.
    """
    mass = level.sum(dim=-1)
    return mass, level / mass.unsqueeze(-1)


def _split_and_fold_levels(
    levels: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Apply ``_split_level_mass`` across every level and return the combined mass product."""
    normalized: list[torch.Tensor] = []
    mass_product: torch.Tensor | None = None
    for level in levels:
        mass, level_norm = _split_level_mass(level)
        normalized.append(level_norm)
        mass_product = mass if mass_product is None else mass_product * mass
    return tuple(normalized), mass_product


def _unconditioned_pass(
    x: torch.Tensor,
    x_w: torch.Tensor | None,
    route_W: torch.Tensor,
    route_bias: torch.Tensor | None,
    resolved: ResolvedRouting,
    *,
    B: int,
    H: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """The zero-budget-conditioning baseline path (spec §4.5, final "Implementation surface" line)."""
    h_w = x_w if (x_w is not None and x_w is not x) else None
    read_logits, write_logits = _project_logits(x, route_W, resolved, route_bias=route_bias, h_w=h_w)
    read_bh, write_bh = production_routing_factor_levels(read_logits, write_logits, resolved)
    read_norm, read_mass = _split_and_fold_levels(read_bh)
    write_norm, write_mass = _split_and_fold_levels(write_bh)
    read_levels = tuple(_unfold_bth(level, B, H) for level in read_norm)
    write_levels = tuple(_unfold_bth(level, B, H) for level in write_norm)
    return read_levels, write_levels, _unfold_bth(read_mass.unsqueeze(-1), B, H).squeeze(-1), \
        _unfold_bth(write_mass.unsqueeze(-1), B, H).squeeze(-1)


def produce_v3_routing(
    x: torch.Tensor,
    x_w: torch.Tensor | None,
    v: torch.Tensor,
    route_W: torch.Tensor,
    route_bias: torch.Tensor | None,
    gain_W: torch.Tensor,
    gain_bias: torch.Tensor | None,
    topology: Topology,
    normalization: str,
    *,
    budget: BudgetConditioning | None = None,
    r: torch.Tensor | None = None,
    budget_embed_W: torch.Tensor | None = None,
    temp_W: torch.Tensor | None = None,
    temp_b: torch.Tensor | None = None,
    usage_bias_W: torch.Tensor | None = None,
    usage_bias_b: torch.Tensor | None = None,
) -> V3RoutingOutput:
    """Produce every configured level's routing plus the §4.5 conditioning surface.

    ``x``: ``[B,L,hidden_size]`` read/shared router source stream.
    ``x_w``: ``[B,L,hidden_size]`` write router source stream, or ``None`` to
    share ``x`` (mirrors the layer's optional ``read_route_conv1d`` split).
    ``v``: the already-projected value stream, taken only for its ``[B,L,H]``
    shape (side-gain outputs must match it) -- never touched arithmetically.
    ``r``: ``[B]`` per-sequence budget (target support fraction), or ``None``
    for the unconditioned baseline.

    Usage conditioning (the ``g`` head) and Tier 1 (the temperature actuator)
    are separate mechanisms with independent ablation gates (architecture
    §4.5): Tier 1 structurally requires ``r`` (its target is defined in terms
    of ``r``), so it is inert whenever ``r is None`` -- exactly the "sandwich
    sampling includes budget=off" branch. Usage conditioning has no such
    dependency and must stay active on `usage_bias_enabled` alone; when ``r``
    is unavailable, its ``r``/slack input features are clamped to a neutral
    ``0`` (inert-flagged) rather than the whole mechanism being disabled (D3
    fix -- see ``_conditioned_pass``).
    """
    if normalization not in ("raw", "global"):
        raise ValueError(f"normalization must be 'raw' or 'global', got {normalization!r}")
    resolved = resolved_routing_from_topology(topology)
    B, L, dm = x.shape
    H = route_W.shape[0]
    D = topology.D
    widths = topology.widths
    if v.ndim != 4 or tuple(v.shape[:3]) != (B, L, H) or v.shape[3] != topology.d_v:
        raise ValueError(
            f"v must have shape [B,T,H,d_v]=({B},{L},{H},{topology.d_v}), got {tuple(v.shape)}")

    active_conditioning = budget is not None and (
        (r is not None and budget.budgeted_pairs) or budget.usage_bias_enabled)
    if budget is not None:
        budget.validate_against(topology)

    if r is not None and budget_embed_W is not None:
        x = embed_budget(x, r, budget_embed_W)
        if x_w is not None:
            x_w = embed_budget(x_w, r, budget_embed_W)

    if not active_conditioning:
        read_levels, write_levels, read_mass, write_mass = _unconditioned_pass(
            x, x_w, route_W, route_bias, resolved, B=B, H=H)
        slack_history: dict[tuple[int, str], torch.Tensor] = {}
    else:
        read_levels, write_levels, read_mass, write_mass, slack_history = _conditioned_pass(
            x, x_w, route_W, route_bias, resolved, topology, budget, r,
            temp_W=temp_W, temp_b=temp_b, usage_bias_W=usage_bias_W, usage_bias_b=usage_bias_b,
            B=B, H=H,
        )

    gain_projection_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    gains = project_side_gains(
        x, gain_W, gain_bias, normalization=normalization, projection_dtype=gain_projection_dtype)
    # Fold each level's mass (see `_split_level_mass`) into the side gain it belongs to
    # (spec §4.2: "multiplicative per-level masses collapse into one side-wide product").
    # `write_mass` always matters -- W directly scales the deposit under both
    # normalizations. `read_mass` matters only under `raw`: under `global`, R[t,s] enters
    # only as num[t]/den[t], and a per-token scale common to every leaf `s` cancels in that
    # ratio exactly (up to the eps term, the same reasoning
    # `test_rola_v3_migration_equivalence.py` uses at eps=0) -- production's global cells
    # have no read-gain projection to fold it into (architecture §4.2), so it is dropped.
    gains["g_write"] = gains["g_write"] * write_mass
    if normalization == "raw":
        gains["g_read"] = gains["g_read"] * read_mass

    exact_support_counts: dict[tuple[int, str], torch.Tensor] = {}
    for level in range(D):
        exact_support_counts[(level, "read")] = exact_support_count(read_levels[level])
        exact_support_counts[(level, "write")] = exact_support_count(write_levels[level])

    touched_bits: dict[int, torch.Tensor] = {}
    for level in range(D):
        touched = torch.zeros(B, H, widths[level], dtype=torch.bool, device=x.device)
        touched_bits[level] = touched_bitset_update(
            touched, write_levels[level].permute(0, 2, 1, 3))  # [B,H,T,width]

    return V3RoutingOutput(
        read_levels=read_levels,
        write_levels=write_levels,
        g_read=gains["g_read"],
        g_write=gains["g_write"],
        exact_support_counts=exact_support_counts,
        touched_bits=touched_bits,
        slack_history=slack_history if active_conditioning else {},
    )


def _conditioned_pass(
    x: torch.Tensor,
    x_w: torch.Tensor | None,
    route_W: torch.Tensor,
    route_bias: torch.Tensor | None,
    resolved: ResolvedRouting,
    topology: Topology,
    budget: BudgetConditioning,
    r: torch.Tensor | None,
    *,
    temp_W: torch.Tensor | None,
    temp_b: torch.Tensor | None,
    usage_bias_W: torch.Tensor | None,
    usage_bias_b: torch.Tensor | None,
    B: int,
    H: int,
) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor,
    dict[tuple[int, str], torch.Tensor],
]:
    """Chunk-recurrent conditioned path: Tier 1 temperature actuator + usage-bias head g.

    Both actuators consume the *previous* chunk's slack/usage (one-chunk lag,
    see module docstring), so this loop is a small, explicit chunk-recurrence
    over ``budget.chunk_size``-token slices -- the same granularity family the
    architecture already uses for chunkwise state (spec §4.5, "Usage
    conditioning": "the same chunkwise-recurrence primitive the architecture
    already uses for state").

    ``r`` may be ``None`` (D3 fix): Tier 1's temperature actuator structurally
    requires ``r`` (its target is defined in terms of it) and stays inert --
    ``pair_index`` is only ever populated with entries the temperature loop
    below actually visits when ``r_bh is not None``. Usage conditioning has no
    such dependency; when ``r`` is unavailable its ``r``/slack input features
    are clamped to ``0`` (inert-flagged, not disabled) rather than gating the
    whole ``g`` head on Tier 1's own on/off switch.
    """
    device, dtype = x.device, x.dtype
    D = topology.D
    widths = topology.widths
    pairs = budget.budgeted_pairs
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    L = x.shape[1]

    # Both actuators are chunk-invariant (Tier 1 needs `r`+a nonempty `pair_index`,
    # usage bias needs `budget.usage_bias_enabled`): if neither is live for this whole
    # call, the scale/bias buffers built below would stay at their identity on every
    # chunk. Skipping their construction/application entirely removes ~2.1 ms/call of
    # provable no-op arithmetic (adversarial review Claim 4) without touching the values
    # `production_routing_factor_levels` sees -- `read_logits`/`write_logits` pass
    # through the multiply-by-1/add-0 identity unchanged either way.
    actuators_active = (r is not None and bool(pair_index)) or budget.usage_bias_enabled

    # V6 fix: `_project_logits` is a pure per-token function of `x`/`x_w`, `route_W` and
    # `route_bias` -- it reads no conditioning state (unlike the temperature/usage-bias
    # actuators below, which genuinely depend on the previous chunk's output and cannot be
    # hoisted). Projecting once over the full sequence and slicing per chunk is bitwise
    # identical to projecting per chunk (the underlying einsum contracts only over the
    # hidden dim, independently per token) and removes 64 separate `[B, 32, ...]` einsum
    # launches at `L = 2048` in favor of one `[B, L, ...]` launch (producer-port.md §2's
    # named follow-up).
    shared_stream = x_w is None or x_w is x
    read_logits_full, write_logits_full = _project_logits(
        x, route_W, resolved, route_bias=route_bias, h_w=None if shared_stream else x_w)

    def _broadcast_head_param(param_over_h: torch.Tensor) -> torch.Tensor:
        """``[H, ...]`` learned-per-head params -> ``[BH, ...]`` (r is per-sequence, not per-head)."""
        return param_over_h.unsqueeze(0).expand(B, *param_over_h.shape).reshape(B * H, *param_over_h.shape[1:])

    # D4 fix: preserve the caller's dtype throughout the conditioning actuator path (this
    # already matches `running_usage`/`running_touched` below) instead of hard-casting to
    # fp32 -- an fp64 caller (the oracle-equivalence gate) must get fp64 conditioning
    # arithmetic all the way through, not just in the logit projection.
    r_bh = None if r is None else r.to(device=device, dtype=dtype)[:, None].expand(B, H).reshape(B * H)

    # Slack is normalized-dimensionless (budget.normalized_slack), so it composes directly
    # with r as a [-1,1]-ish feature; the running EXACT cumulative count is kept separately
    # in raw units (it is what target_count is compared against) and is always detached
    # (spec §4.5: it is a conditioning INPUT, not itself a loss term -- losses.py owns the
    # differentiable surrogate that trains toward it).
    running_slack = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
    running_exact_count = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
    slack_history: dict[tuple[int, str], list[torch.Tensor]] = {pair: [] for pair in pairs}
    running_usage = {level: torch.zeros(B * H, widths[level], dtype=dtype, device=device) for level in range(D)}
    running_touched = {
        level: torch.zeros(B * H, widths[level], dtype=torch.bool, device=device) for level in range(D)}
    tokens_seen = 0

    read_chunks: list[list[torch.Tensor]] = [[] for _ in range(D)]
    write_chunks: list[list[torch.Tensor]] = [[] for _ in range(D)]
    read_mass_chunks: list[torch.Tensor] = []
    write_mass_chunks: list[torch.Tensor] = []

    for start, end in chunk_boundaries(L, budget.chunk_size):
        T_chunk = end - start
        read_logits = read_logits_full[:, start:end]
        write_logits = write_logits_full[:, start:end]
        # [BH, T_chunk, D, real_width]. All conditioning below uses the PREVIOUS chunk's
        # slack/usage (one-chunk lag, module docstring) -- the running_* dicts have not yet
        # been updated with this chunk's own output at the point they are read here.
        #
        # `read_logits`/`write_logits` MAY be the same tensor object when every level is
        # tied with no untied read slot (packed_router_logits' single-solve fast path).
        # Conditioning must therefore never mutate them in place -- it accumulates
        # out-of-place scale/bias buffers instead and applies them with one final
        # multiply-add, so an aliased read/write pair is never double-counted and no
        # autograd version-counter hazard is introduced.
        if actuators_active:
            write_scale = torch.ones_like(write_logits)
            write_bias_t = torch.zeros_like(write_logits)
            read_scale = torch.ones_like(read_logits)

            # Tier 1's temperature actuator structurally requires r (D3): its target is
            # defined in terms of r, so it stays inert (scale left at 1) whenever r is
            # None, exactly the "sandwich sampling includes budget=off" branch.
            if r_bh is not None:
                for (level, side), idx in pair_index.items():
                    cond = torch.stack([r_bh, running_slack[(level, side)]], dim=-1)  # [BH,2]
                    weight = _broadcast_head_param(temp_W[:, idx, :])                 # [BH,2]
                    temp_logit = (cond * weight).sum(-1)                              # [BH]
                    if temp_b is not None:
                        temp_logit = temp_logit + _broadcast_head_param(temp_b[:, idx:idx + 1]).squeeze(-1)
                    c = F.softplus(temp_logit) + _TEMPERATURE_FLOOR                   # [BH]
                    scale_buffer = write_scale if side == "write" else read_scale
                    scale_buffer[:, :, level, :widths[level]] = c[:, None, None]

            if budget.usage_bias_enabled:
                # D3 fix: usage conditioning is a separate mechanism from Tier 1 with its
                # own ablation gate (`usage_bias_enabled`) and must stay active even when
                # `r` (and therefore Tier 1) is off -- clamp the r/slack input features to
                # a neutral 0 (inert-flagged) instead of skipping the head entirely.
                r_feature = r_bh if r_bh is not None else torch.zeros(B * H, dtype=dtype, device=device)
                for level in range(D):
                    width = widths[level]
                    usage_norm = running_usage[level] / max(tokens_seen, 1)          # [BH,width]
                    touched_f = running_touched[level].to(dtype)                     # [BH,width]
                    write_slack = running_slack.get((level, "write"), torch.zeros(B * H, dtype=dtype, device=device))
                    w0, w1, w2, w3 = (_broadcast_head_param(usage_bias_W[:, level, i:i + 1]) for i in range(4))
                    bias = _broadcast_head_param(usage_bias_b[:, level:level + 1])   # [BH,1]
                    offset = (
                        w0 * usage_norm + w1 * touched_f
                        + w2 * r_feature[:, None] + w3 * write_slack[:, None] + bias
                    )                                                                # [BH,width]
                    write_bias_t[:, :, level, :width] = write_bias_t[:, :, level, :width] + offset[:, None, :]

            write_logits = write_logits * write_scale + write_bias_t
            read_logits = read_logits * read_scale

        read_bh_raw, write_bh_raw = production_routing_factor_levels(
            read_logits, write_logits, resolved)
        read_bh, read_mass_chunk = _split_and_fold_levels(read_bh_raw)
        write_bh, write_mass_chunk = _split_and_fold_levels(write_bh_raw)
        for level in range(D):
            read_chunks[level].append(read_bh[level])
            write_chunks[level].append(write_bh[level])
        read_mass_chunks.append(read_mass_chunk)
        write_mass_chunks.append(write_mass_chunk)

        tokens_seen += T_chunk
        for level in range(D):
            running_usage[level] = update_soft_usage(running_usage[level], write_bh[level])
            running_touched[level] = touched_bitset_update(running_touched[level], write_bh[level])
        if r_bh is not None:
            for (level, side) in pairs:
                values = write_bh[level] if side == "write" else read_bh[level]
                running_exact_count[(level, side)] = (
                    running_exact_count[(level, side)] + exact_support_count(values).detach().sum(dim=-1))
                target_count = chunk_target_count(r_bh, widths[level], tokens_seen)
                running_slack[(level, side)] = normalized_slack(target_count, running_exact_count[(level, side)])
                slack_history[(level, side)].append(running_slack[(level, side)])
        else:
            # r is off: slack stays at its clamped/inert 0 (D3), but slack_history is only
            # populated for budgeted pairs when Tier 1 is actually active -- an empty history
            # entry for an off step is correct (nothing was actuated) and matches
            # `V3RoutingOutput.slack_history`'s "populated only for budgeted pairs" contract.
            for (level, side) in pairs:
                slack_history[(level, side)].append(running_slack[(level, side)])

    read_levels = tuple(_unfold_bth(torch.cat(read_chunks[level], dim=1), B, H) for level in range(D))
    write_levels = tuple(_unfold_bth(torch.cat(write_chunks[level], dim=1), B, H) for level in range(D))
    read_mass = _unfold_bth(torch.cat(read_mass_chunks, dim=1).unsqueeze(-1), B, H).squeeze(-1)
    write_mass = _unfold_bth(torch.cat(write_mass_chunks, dim=1).unsqueeze(-1), B, H).squeeze(-1)
    slack_out = {pair: torch.stack(hist, dim=-1).reshape(B, H, -1) for pair, hist in slack_history.items()}
    return read_levels, write_levels, read_mass, write_mass, slack_out


__all__ = [
    "BudgetConditioning",
    "V3RoutingOutput",
    "embed_budget",
    "init_temp_bias",
    "produce_v3_routing",
    "resolved_routing_from_topology",
]
