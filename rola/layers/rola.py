# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""RoLA — Routed (Value) Linear Attention, V3 layer.

``ROLA_V3_SPEC.md`` §10 Stage 2 / ``ROLA_HIERARCHICAL_VALUE_MEMORY_ARCHITECTURE.md``
§8, §9.4: this layer owns the value/route/gain/decay projection contract and
public routing configuration; it does not own, project, or reference Q or K in
any form. Architecture §9.3 ("Delete after parity"): this replaces the Q/K-era
layer in place, per the migration inventory row `fla_rola/layers/rola.py`.

**Execution backends, and the rule that picks between them.** This layer has TWO
execution backends and the choice between them is STRUCTURAL, decided by
:meth:`RoLA.select_execution_backend` before any of them runs:

* ``'oracle'`` -- ``naive_rola``, the Stage 1 canonical reference run in the
  layer's working dtype. It is fully differentiable and it is **the only
  implemented training path**, which is not the same thing as a fallback: nothing
  degrades into it after a failure, and no configuration reaches it because
  something else declined. It is selected when the module is in training mode,
  when the operation's own inputs carry a gradient, and when the tensors are not
  on a CUDA device.
* ``'cuda'`` -- the plan-selected V3 consumer (`ROLA_V3_SPEC.md` §3-§6):
  ``plan_and_schedule`` measures the realized routing, returns a stamped
  ``Plan`` and builds the §4 schedule from the same single pass over the routing
  tensors, then ``_consumer_forward`` launches the template instantiation that
  plan names. The consumer is **forward-only** -- there is no native backward for
  it yet -- so it requires eval mode AND no gradient reachability AND a CUDA device
  AND a launchable configuration; it is refused the moment any of those fails.

**The autograd trap this dispatch exists to close** (memory note
`layer-dispatch-autograd-gap`; the fleet-smoke ``x.grad=None`` failure was exactly
this class): a forward-only kernel reachable from a grad-requiring context does not
raise, it silently returns a tensor with no ``grad_fn`` and the graph is severed at
this layer. The predicate is therefore ``torch.is_grad_enabled() and any input to
the OPERATION requires grad`` -- evaluated on ``v``, the producer's own outputs, and
the decay dials (``LeafMassDecay.dials``, when a decay config is in play), which
are the tensors the operation actually consumes, not on the module's parameters (a
frozen-parameter, non-requiring-grad input under ``torch.no_grad`` has no graph to
sever and correctly takes the CUDA path). The decay dials are an OPERATION input --
``_consumer_forward`` is called with ``decay=decay_config`` -- and were missing
from the predicate before the fix pass recorded at `wiring-review.md` §2; a
dial-only-unfrozen finetune (the configuration ``decay_target_leaf_rate`` exists
for) reached the CUDA consumer and silently produced ``decay_theta.grad = None``.

**Training mode is also a dispatch condition, not just a gradient one** (fix pass,
`wiring-review.md` §3). ``self.training`` gates the CUDA path independently of
``torch.is_grad_enabled()``, because reentrant gradient checkpointing
(``torch.utils.checkpoint.checkpoint(..., use_reentrant=True)``, still PyTorch's
default when unset) runs its first pass under ``torch.no_grad()`` with
``module.training`` still ``True``. Without this condition that first pass selects
``'cuda'`` while backward recomputes the SAME step on the oracle -- the value that
flows downstream and the gradient that is taken would come from two different
implementations agreeing only to ``RTOL``, not from one graph. Gating on
``self.training`` closes this structurally: a module in training mode never takes
the CUDA path, under any wrapper, so a training-mode forward is graph-faithful
everywhere it is called, and eval mode is where the CUDA path's inference speed
is meant to matter.

**There is no silent degradation in the other direction either.** Once the CUDA
backend is selected, a configuration the consumer cannot run -- an unimplemented
intra lowering, an un-instantiated ``(d_v, BC, BT)``, an unrealized warp tile --
raises loudly out of ``plan_and_schedule``/``_consumer_forward``/the kernel's own
``TORCH_CHECK``s. It never falls back to the oracle.

**Deletion inventory applied here** (spec §1.2): `head_k_dim`, `qkv_proj`,
`qk_norm`/`q_norm`/`k_norm`, `q_conv1d`/`k_conv1d`, `_feature_map`, and all
Q/K-aware cache fields are gone. **Retained** (do not delete by association,
spec §1.2): `v_conv1d`, `read_route_conv1d`/`write_route_conv1d`, and the
`RMSNorm` module (unused by this layer today, kept importable for any future
non-Q/K route-source normalization -- none is wired here, since that would be
a new model justification per architecture §8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from rola.utils.cache import get_layer_cache, update_layer_cache

# OPTIONAL: the fused short causal convolution, used only when the caller asks for
# `use_short_conv=True` (default False). Guarded exactly as mamba-ssm guards
# `causal_conv1d`: the import never fails, and the assert that names the remedy is
# deferred to the point where the module is actually needed. This is what keeps
# `rola`'s hard dependency set at torch/einops/triton.
try:
    from fla.modules import ShortConvolution
except ImportError:  # noqa: BLE001
    ShortConvolution = None
from rola.interface import select_backend
from rola.ops.consumer import _consumer_forward, sm_shared_bytes
from rola.ops.naive import naive_rola
from rola.planner.schedule import (
    MMA_PRECISIONS,
    StaticDensity,
    owner_geometry,
    plan_and_schedule,
)
from rola.routing.producer import BudgetConditioning, init_temp_bias, produce_routing
from rola.routing.side_gain import GainConfig, init_side_gain_bias
from rola.routing.topology import resolve_b_spec
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
    UnionRouting,
)


@dataclass(frozen=True)
class LevelRoutingConfig:
    """Public read/write routing semantics for one factorization level.

    Unchanged from the Q/K-era layer -- this dataclass has never referenced
    Q/K, it only translates public read/write/tied/alpha choices into the
    ops-internal `IndependentRouting | UnionRouting` tags (`_resolve_level_routing`
    below), which is exactly `ROLA_V3_SPEC.md` §1's `LevelRouting` vocabulary.

    ``read``/``write`` are ``'dense'`` (softmax) or ``'sparse'`` (exact entmax
    with ``alpha`` equal to 1.5 or 2.0). An untied sparse/sparse level resolves
    to union routing: entmax determines shared support, then read and write
    mass split across that support. All other combinations resolve to
    independent routing. ``tied=True`` shares one projection; equal
    activations also share one solve.
    """
    read: str = 'dense'
    write: str = 'dense'
    tied: bool = False
    alpha: float = 1.5

    def __post_init__(self):
        if type(self.tied) is not bool:
            raise TypeError(f"LevelRoutingConfig.tied must be bool, got {self.tied!r}")
        for _side, _v in (('read', self.read), ('write', self.write)):
            if _v not in ('sparse', 'dense'):
                raise ValueError(f"LevelRoutingConfig.{_side} must be 'sparse' or 'dense', got {_v!r}")
        object.__setattr__(self, 'alpha', EntmaxActivation(self.alpha).alpha)


def resolve_level_routing_configs(
    level_routing_configs: LevelRoutingConfig | list[LevelRoutingConfig],
    D: int,
) -> list[LevelRoutingConfig]:
    """Resolve the layer's per-level routing config list of length D:
      - a single `LevelRoutingConfig`: broadcast to all D levels.
      - a `list[LevelRoutingConfig]` of length D: used as-is (explicit per-level control).
    Any other length raises — the list-length IS the structural check (branch on structure, not a
    magic threshold)."""
    if isinstance(level_routing_configs, LevelRoutingConfig):
        return [level_routing_configs] * D
    configs = list(level_routing_configs)
    if len(configs) != D:
        raise ValueError(
            f"level_routing_configs list must have exactly D={D} entries (one per routing level), "
            f"got {len(configs)}")
    if any(type(config) is not LevelRoutingConfig for config in configs):
        raise TypeError("level_routing_configs entries must be LevelRoutingConfig instances")
    return configs


def _resolve_level_routing(config: LevelRoutingConfig):
    """Translate one public config into the tagged operation plan exactly once."""
    if not config.tied and config.read == 'sparse' and config.write == 'sparse':
        return UnionRouting(config.alpha)

    def activation(side: str):
        return EntmaxActivation(config.alpha) if side == 'sparse' else SoftmaxActivation()

    return IndependentRouting(
        tied=config.tied,
        read_activation=activation(config.read),
        write_activation=activation(config.write),
    )


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from rola.state import RoLAState


class RoLA(nn.Module):
    r"""Routed (Value) Linear Attention layer -- V3.

    Args:
        hidden_size (int): model dim. Default 1024.
        num_heads (int): number of heads (H). Default 8.
        head_v_dim (int): per-head value dim (d_v). Default 32.
        levels_per_head (int): routing depth D. Default 1.
        states_per_level (int | list[int]): branching at every level (architecture §4.1).
            An int gives uniform branching b (N = b**D). A length-D list gives mixed-radix
            per-level branching. Every entry must be >= 2 (`LevelConfig` degenerate-width
            exception aside; the layer itself requires >= 2, matching V1/V2's routing
            convention -- a width-1 level carries no routing information).
        decay (str): 'none' | 'leaf_mass' (architecture §4.4). Default 'none'.
        decay_target_leaf_rate (float): the symmetric product-rate init target `rho_init`
            (architecture §4.4: `delta_l_init = rho_init ** (1/D)`); only used when
            `decay='leaf_mass'`. Default `2**-8`.
        state_norm (str): 'raw' | 'global'. Default 'global'.
        router_bias (bool): bias on the routing projections. Default False.
        gain_bias_init (float | None): if given, initializes the side-gain bias so that
            an all-zero-logit gain starts at this value (`side_gain.init_side_gain_bias`).
            **Flagged model decision** (spec §1.1): `None` (the default) leaves the gain
            bias at zero, i.e. unit gain at zero logits -- this is explicitly NOT
            migration-equivalent to any V1/V2 checkpoint (architecture §4.2 gives
            `g_read=sqrt(K)`, `g_write=K` for the exact decomposition at a specific K)
            and must not be presented as such; it is recorded here as the deliberate,
            simplest starting point, to be revisited by its own quality/stability gate
            before large training (spec §1.1 `TODO`).
        use_short_conv (bool): causal depthwise short conv on the value and routing-source
            streams. Default False.
        conv_size (int): short-conv kernel size. Default 4.
        conv_bias (bool): bias in the short conv. Default False.
        layer_idx (int): layer index (for the KV cache). Default None.
        level_routing_configs (LevelRoutingConfig | list[LevelRoutingConfig] | None): explicit
            per-level routing config. Default None, which broadcasts `LevelRoutingConfig()`
            (dense/dense, untied) to all levels.
        progressive_memory (bool): retained flag. Superseded for the CUDA path by
            `paging` below, which is the actual `ROLA_V3_SPEC.md` §7.1 mechanism.
            Default True.
        paging (bool): commit state pages on first write to an owner (spec §7.1).
            The resident state then scales with the owners the routing REALIZES
            rather than with the topology's provisioned `N`. One page is one owner
            (`page_states := BC`), pages are admitted by the write-side plan, and
            an owner without a page reads as zeros -- so paging is BITWISE
            INVISIBLE, not an approximation. CUDA backend only; the oracle path has
            no page table and raises rather than silently ignoring the flag.
            Default False.
        max_physical_pages (int | None): §7.1's allocator hard cap, in pages.
            `None` (default) provisions the dense limit `BH * owners_total`, at
            which `CAPPED` is unreachable. A finite value makes architecture §4.5's
            graceful degradation reachable and CHANGES THE FUNCTION COMPUTED when it
            binds -- which is reported in `last_paging_stats.capped_pages`, never
            silent.
        budget (BudgetConditioning | None): architecture §4.5 Tier 1 + usage-conditioning
            configuration. `None` (default) disables all conditioning machinery -- the
            "zero-budget-conditioning baseline is the unconstrained measurement" path
            (spec §4.5), and no conditioning parameters are allocated.
        temp_bias_init (float): target Tier-1 temperature `c` at an all-zero-logit
            actuator (`temp_W`/`temp_b` initialize to zero weight / this bias), only
            meaningful when `budget` configures at least one budgeted pair. Default 1.0.
            **Flagged model decision** (spec §4.5, D5 fix): unlike `gain_bias_init`
            (which defaults to `None`/raw-zero and is explicitly NOT neutral), this
            defaults to the actuator's neutral point -- `c = softplus(bias) +
            _TEMPERATURE_FLOOR` is never 1 at raw bias 0, so merely *configuring* a
            budgeted level would otherwise perturb an otherwise-baseline model's
            routing distribution before any training signal exists. `1.0` keeps a
            freshly budgeted level a no-op at init; pass a different target to start
            pre-biased toward a tighter/looser initial support.
        state_block (int | None): `BC`, the number of leafs one owner block owns --
            the CTA's resident state tile. **An EXPERT PIN over a planner-selected
            axis** (`THUNDER_RUN_SPEC.md` §4.3/§6 P6), no longer a contract the caller
            must supply: an explicit value shrinks the feasible set to itself and the
            planner never runs; `None` takes the LARGEST FEASIBLE `BC`.

            **WHAT `widths[-1]` WAS, AND WHY IT IS NOT THE DEFAULT ANY MORE.** The old
            default was the innermost level's width, defended as the unique `BC` whose
            owner rectangle is exactly one full innermost level (`j = D-1`, inner
            extent 1, so both alignment conditions hold identically for every
            topology). That defence is true and it is not the question: alignment is
            necessary, not sufficient. `widths[-1]` is not INSTANTIATED on a topology
            whose last level is wide -- `branches=[4, 256]` resolves it to `BC = 256`,
            which no kernel implements -- so the caller got a plan-time
            `NotImplementedError` out of the occupancy oracle instead of a
            construction-time error, on a value they never chose. It is also merely
            the smallest feasible `BC` on a topology whose last level is narrow, which
            costs the per-leaf amortization and the `N/BC`-way atomic reduction that
            make `BC` worth selecting at all.

            The largest feasible `BC` is the alignment-admissible set intersected with
            the instantiated ladder, maximized (`rola.planner.select.largest_feasible_bc`).
            Both are computed AT CONSTRUCTION, so a topology the ladder cannot serve
            and an inadmissible explicit pin are both construction-time errors and
            never plan-time surprises.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        head_v_dim: int = 32,
        levels_per_head: int = 1,
        states_per_level: int | list[int] = 16,
        decay: str = 'none',
        decay_target_leaf_rate: float = 2.0 ** -8,
        state_norm: str = 'global',
        router_bias: bool = False,
        gain_bias_init: float | None = None,
        gain: bool = True,
        gain_config: GainConfig | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        level_routing_configs: LevelRoutingConfig | list[LevelRoutingConfig] | None = None,
        progressive_memory: bool = True,
        budget: BudgetConditioning | None = None,
        temp_bias_init: float = 1.0,
        state_block: int | None = None,
        paging: bool = False,
        max_physical_pages: int | None = None,
        mma_precision: str = "bf16",
    ) -> RoLA:
        super().__init__()
        if decay not in ('none', 'leaf_mass'):
            raise ValueError(f"decay must be 'none' or 'leaf_mass', got {decay!r}")
        if level_routing_configs is None:
            level_routing_configs = LevelRoutingConfig()
        if state_norm not in ('raw', 'global'):
            raise ValueError(f"state_norm must be 'raw' or 'global', got {state_norm!r}")
        if mma_precision not in MMA_PRECISIONS:
            raise ValueError(
                f"mma_precision must be one of {MMA_PRECISIONS}, got {mma_precision!r}")
        if type(progressive_memory) is not bool:
            raise TypeError("progressive_memory must be bool")
        if budget is not None and type(budget) is not BudgetConditioning:
            raise TypeError(f"budget must be None or BudgetConditioning, got {budget!r}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        #: The readout MMA's NUMERIC FORMAT, forwarded into every plan this layer
        #: selects. `bf16` is the production default (`MMA_PRECISIONS` records the
        #: measurement and why the exactness gates pin `tf32x3` explicitly instead);
        #: it is a layer field rather than a plan-time default so that a gate
        #: comparing this layer against the fp64 oracle can ask for the exact arm.
        self.mma_precision = mma_precision
        self.head_v_dim = head_v_dim
        branches = resolve_b_spec(states_per_level, levels_per_head)
        self.level_routing_configs = resolve_level_routing_configs(level_routing_configs, levels_per_head)
        levels = tuple(
            LevelConfig(width, _resolve_level_routing(config))
            for width, config in zip(branches, self.level_routing_configs)
        )
        self.topology = Topology(levels=levels, d_v=head_v_dim)
        # WHICH (level, side) CELLS ARE FULL-SUPPORT BY CONSTRUCTION, decided ONCE,
        # here (§2.7's level-structure-awareness ruling, 2026-08-01). This is a
        # property of the configured activations -- read from the registry's
        # `exact_zeros` column by `StaticDensity.from_topology` -- so it cannot
        # change between steps, and it is what lets the planner substitute a closed
        # form for the dense side's tile statistics instead of measuring a rectangle
        # it already knows the shape of. The declaration is sound HERE, and only
        # here, because this layer's `produce_routing` applies exactly the
        # activations this topology names; a caller holding routing tensors of
        # unrelated provenance passes nothing and gets the fully measured pass.
        self.static_density = StaticDensity.from_topology(self.topology)
        if budget is not None:
            budget.validate_against(self.topology)
        self.budget = budget

        # `BC` -- contract, not a §6-selected axis. `owner_geometry` is the authority
        # on admissibility and it raises rather than re-aligning, so an inadmissible
        # `state_block` is a construction-time error and never a forward-time surprise.
        # §6 P6: the FEASIBLE SET and the largest-feasible default, both resolved at
        # CONSTRUCTION so an inadmissible topology or an inadmissible pin fails here and
        # not inside a forward. `feasible_bc` raises with the topology's own diagnosis
        # when the ladder cannot serve it at all.
        from rola.planner.select import feasible_bc, largest_feasible_bc

        if state_block is None:
            self.state_block = largest_feasible_bc(self.topology.widths)
            self.feasible_BC = feasible_bc(self.topology.widths)
        else:
            self.state_block = int(state_block)
            # An explicit pin SHRINKS THE FEASIBLE SET TO ITSELF (§4.3) -- the planner
            # then has nothing to choose and never runs. It is alignment-validated like
            # any other member rather than trusted, which is what makes it a pin over a
            # selected axis instead of a bypass of one.
            self.feasible_BC = feasible_bc(self.topology.widths, pinned=self.state_block)
        owner_geometry(self.topology.widths, self.state_block)

        # Execution-backend telemetry. Plain Python attributes (not buffers, not
        # parameters): they are diagnostic state, must not enter a state_dict, and
        # must not be cast by `_apply`. The dispatch gate asserts on these rather
        # than on wall-clock, because timing cannot distinguish "took the CUDA path"
        # from "took a fast oracle path" and this is exactly the distinction the
        # autograd trap turns on.
        self._execution_backend_counts = {'oracle': 0, 'cuda': 0}
        self.last_execution_backend: str | None = None
        self.last_selected_plan = None
        self.last_plan_axes = None
        #: `PagingResult` for the most recent paged forward, or `None`. Committed-state
        #: accounting (spec §7.1): peak pages, realized owners, resident vs dense
        #: bytes, and `capped_pages`. Same plain-attribute surface as the counters
        #: above -- diagnostic state, deliberately not a buffer.
        self.last_paging_stats = None

        self.decay_kind = decay
        self.decay_target_leaf_rate = decay_target_leaf_rate
        if decay == 'leaf_mass':
            self.decay_theta = nn.ParameterList([
                nn.Parameter(theta) for theta in self._init_decay_thetas(
                    num_heads, self.topology.widths, decay_target_leaf_rate)
            ])

        self.state_norm = state_norm
        self.router_bias = router_bias
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.layer_idx = layer_idx
        self.progressive_memory = progressive_memory
        if type(paging) is not bool:
            raise TypeError("paging must be bool")
        self.paging = paging
        self.max_physical_pages = max_physical_pages
        #: The §7.1 arena, built on first paged forward (it needs `BH`, which is a
        #: forward-time quantity). Plain attribute, like the other execution-backend
        #: diagnostics below: it is runtime state, must not enter a `state_dict`, and
        #: must not be cast by `_apply`.
        self._page_arena = None

        self.value_dim = num_heads * head_v_dim
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Real width for the packed router weight, and D + untied-read-slot packed layout,
        # both purely routing-plan facts -- unchanged in form from V1/V2 (`ResolvedRouting`
        # already had no Q/K dependence; see `routing/producer_v3.py:resolved_routing_from_topology`).
        real_width = max(self.topology.widths)
        D = self.topology.D
        untied_levels = tuple(
            index for index, level in enumerate(self.topology.levels)
            if isinstance(level.routing, UnionRouting) or not level.routing.tied
        )
        packed_slot_count = D + len(untied_levels)

        if use_short_conv:
            assert ShortConvolution is not None, (
                "use_short_conv=True needs the fused short convolution from "
                "flash-linear-attention, which is an optional companion of `rola`:\n"
                "    pip install 'rola[fla]'\n"
                "There is no vendored substitute: the convolution is not RoLA's "
                "code and forking it here would fork its maintenance. Pass "
                "use_short_conv=False to run without it.")
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation='silu')
            self.write_route_conv1d = ShortConvolution(hidden_size, conv_size, bias=conv_bias, activation=None)
            if untied_levels:
                self.read_route_conv1d = ShortConvolution(hidden_size, conv_size, bias=conv_bias, activation=None)

        self.route_W = nn.Parameter(torch.empty(num_heads, packed_slot_count, hidden_size, real_width))
        self._init_factor_router(self.route_W)
        if router_bias:
            self.route_bias_param = nn.Parameter(torch.zeros(num_heads, packed_slot_count, real_width))
        else:
            self.register_parameter('route_bias_param', None)

        # Side gains (spec §1.1, architecture §4.2; gain-surface ruling 2026-08-01).
        # `gain_layout` decides the columns from the algebra: the write gain is a real
        # dial (`write_gain=False` REMOVES it, rather than pinning it to 1, so no dead
        # parameter is optimized), and the read gain is a consequence of the
        # normalization mode -- present under `raw`, structurally absent under `global`.
        #: PUBLIC surface is the single `gain: bool`; `gain_config` is the internal
        #: two-boolean research surface behind it, for fine-grained combinations that
        #: are deliberately not public parameters.
        if gain_config is not None:
            self.gain_config = gain_config.validate(state_norm)
        else:
            self.gain_config = GainConfig.from_public(gain, state_norm).validate(state_norm)
        self.write_gain = self.gain_config.write_gain
        self.read_gain = self.gain_config.read_gain
        self.gain_sides = self.gain_config.layout
        gain_sides = len(self.gain_sides)
        self.gain_W = nn.Parameter(torch.empty(num_heads, gain_sides, hidden_size))
        if gain_sides:
            nn.init.kaiming_uniform_(
                self.gain_W.view(num_heads * gain_sides, hidden_size), a=5 ** 0.5)
        gain_bias_value = 0.0 if gain_bias_init is None else init_side_gain_bias(gain_bias_init)
        self.gain_bias = nn.Parameter(torch.full((num_heads, gain_sides), gain_bias_value))

        if budget is not None:
            pairs = budget.budgeted_pairs
            if pairs:
                self.temp_W = nn.Parameter(torch.zeros(num_heads, len(pairs), 2))
                # D5 fix: init the bias (not the weight) so an all-zero-conditioning-input
                # actuator starts at exactly `temp_bias_init` (default neutral c=1) instead
                # of the raw-zero-bias default, which is never neutral for this positive
                # parameterization (see `producer_v3.init_temp_bias`).
                self.temp_b = nn.Parameter(
                    torch.full((num_heads, len(pairs)), init_temp_bias(temp_bias_init)))
                # D5r fix: `temp_b` is materialized at the default (fp32) dtype, rounding the
                # fp64-precise `init_temp_bias` target before `_apply` ever runs. Record the
                # exact scalar target plus an at-construction snapshot so `_apply` can rebuild
                # the baseline exactly in whatever dtype a later `.double()` lands on, instead
                # of merely casting the already-rounded fp32 value forward.
                self._temp_bias_target = float(temp_bias_init)
                self._temp_b_init_snapshot = self.temp_b.data.clone()
            else:
                self.register_parameter('temp_W', None)
                self.register_parameter('temp_b', None)
            if budget.usage_bias_enabled:
                self.usage_bias_W = nn.Parameter(torch.zeros(num_heads, D, 4))
                self.usage_bias_b = nn.Parameter(torch.zeros(num_heads, D))
            else:
                self.register_parameter('usage_bias_W', None)
                self.register_parameter('usage_bias_b', None)
            self.budget_embed_W = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('temp_W', None)
            self.register_parameter('temp_b', None)
            self.register_parameter('usage_bias_W', None)
            self.register_parameter('usage_bias_b', None)
            self.register_parameter('budget_embed_W', None)

    @staticmethod
    def _init_factor_router(W: nn.Parameter):
        """Init the per-head per-level factor weights [H, slots, hidden, b] like an
        nn.Linear(hidden, b): the routing logit is h·W[head,slot] (== Linear with weight
        W[head,slot]^T), so the init must be a transposed Linear weight with kaiming
        fan_in == hidden. Unchanged from V1/V2 -- this has never depended on Q/K."""
        H, slots, hidden, b = W.shape
        wt = torch.empty(H * slots * b, hidden)
        nn.init.kaiming_uniform_(wt, a=5 ** 0.5)
        W.data.copy_(wt.view(H, slots, b, hidden).transpose(-1, -2))

    @staticmethod
    def _init_decay_thetas(
        num_heads: int, widths: tuple[int, ...], rho_init: float,
    ) -> tuple[torch.Tensor, ...]:
        """Symmetric product-rate initialization (architecture §4.4): every level gets an
        equal share of the target leaf rate, ``delta_l_init = rho_init ** (1/D)``, so that
        ``prod_l delta_l_init == rho_init`` exactly at init. Returns one ``[H, width_l]``
        logit tensor per level (``theta = logit(delta)``, since ``LeafMassDecay.dials``
        wants the sigmoid-parameterized ``delta`` directly, not a packed layout -- V3's
        per-level dial tensors make the V1/V2 packed-offset bookkeeping unnecessary)."""
        D = len(widths)
        delta_init = rho_init ** (1.0 / D)
        theta = math.log(delta_init) - math.log1p(-delta_init)
        return tuple(torch.full((num_heads, width), theta, dtype=torch.float32) for width in widths)

    def _apply(self, fn, recurse=True):
        """Apply module transforms while keeping compact decay parameters in fp32, and
        keeping ``temp_b``'s baseline exact under a dtype-widening transform (D5r).

        Mirrors the V1/V2 layer's ``theta`` hook, adapted for V3's per-level
        ``ParameterList`` (one dial tensor per level rather than one packed
        tensor): exclude it from ``Module._apply``'s default recursion, apply
        ``fn`` exactly once per level, then cast back to fp32 and restore
        registration -- so a ``.half()``/``.to(device)`` call moves the compact
        decay dials without leaving them in a reduced-precision dtype.

        ``temp_b`` gets a different treatment: it starts life fp32-rounded
        (``torch.full`` at the default dtype), so a naive ``fn(param.data)`` cast to
        fp64 would carry that rounding forward and never recover the exact
        ``init_temp_bias`` target. Instead, decompose the current value into
        ``init_snapshot + drift`` (``drift`` is exactly zero until training moves the
        parameter), cast the drift with ``fn`` like any other tensor, and rebuild the
        init term by *recomputing* ``init_temp_bias`` directly in the post-``fn``
        dtype rather than casting the stale fp32 constant. An untrained ``temp_b``
        therefore lands on the exact target in any dtype; a trained one keeps its
        drift with ordinary cast precision.
        """
        decay_theta = self._modules.get('decay_theta')
        temp_b = self._parameters.get('temp_b')
        if decay_theta is None and temp_b is None:
            return super()._apply(fn, recurse=recurse)
        if decay_theta is not None:
            del self._modules['decay_theta']
        if temp_b is not None:
            del self._parameters['temp_b']
        try:
            result = super()._apply(fn, recurse=recurse)
            with torch.no_grad():
                if decay_theta is not None:
                    for param in decay_theta:
                        param.data = fn(param.data).float()
                        if param.grad is not None:
                            param.grad = fn(param.grad).float()
                if temp_b is not None:
                    cast_ref = fn(temp_b.data)
                    exact_init = torch.full_like(cast_ref, init_temp_bias(self._temp_bias_target))
                    drift = fn(temp_b.data - self._temp_b_init_snapshot)
                    temp_b.data = exact_init + drift
                    if temp_b.grad is not None:
                        temp_b.grad = fn(temp_b.grad)
                    self._temp_b_init_snapshot = exact_init.clone()
            return result
        finally:
            if decay_theta is not None:
                self._modules['decay_theta'] = decay_theta
            if temp_b is not None:
                self._parameters['temp_b'] = temp_b

    @property
    def execution_backend_counts(self) -> dict[str, int]:
        """How many forwards took each backend, since construction.

        Read-only view. This is the gate surface for "the CUDA path was actually
        taken": a dispatch test asserts on the delta of these counters, which is a
        fact about which branch executed, whereas a timing comparison is an
        inference from a number that has other explanations.
        """
        return dict(self._execution_backend_counts)

    @property
    def has_read_gain(self) -> bool:
        """Whether this configuration projects a read gain. Read-only, by design."""
        return 'read' in self.gain_sides

    def select_execution_backend(self, *operation_inputs: torch.Tensor | None) -> str:
        """Choose ``'oracle'`` or ``'cuda'`` from the module's mode and the
        OPERATION's own inputs.

        Three structural conditions. The first is evaluated on ``self``; the other
        two are evaluated on the tensors the V3 operation consumes -- ``v``, the
        per-level read/write allocations, the side gains, and the decay dials when
        a decay config is in play -- rather than on the module's parameters or a
        configuration flag:

        1. **Training mode.** ``self.training`` alone selects ``'oracle'``,
           independent of ``torch.is_grad_enabled()``. Reentrant gradient
           checkpointing's first pass runs under ``torch.no_grad()`` with
           ``module.training`` still ``True`` (PyTorch's default when
           ``use_reentrant`` is unset), so a predicate keyed only on gradient
           reachability would select ``'cuda'`` for that first pass while backward
           recomputes the same step on the oracle -- two different
           implementations, not one graph. Gating on training mode closes this
           structurally: a module in training mode never takes the CUDA path,
           under any wrapper. Eval mode is where the CUDA path's inference speed
           is meant to matter.
        2. **Gradient reachability.** ``torch.is_grad_enabled()`` and any input
           requiring grad selects ``'oracle'``. The V3 CUDA consumer is
           forward-only; it has no native backward, so reaching it from a
           grad-requiring context would not raise, it would return a tensor with no
           ``grad_fn`` and silently sever the graph at this layer -- the
           ``x.grad = None`` failure class. The oracle is differentiable and is the
           only implemented training path. Calling it a fallback would be wrong in
           both directions: nothing failed to reach it, and nothing else exists to
           fall back from. The decay dials are included in ``operation_inputs``
           precisely because they are an operation input -- ``_consumer_forward``
           is called with ``decay=decay_config`` -- not because they are a module
           parameter; a dial-only-unfrozen forward has no other input requiring
           grad, and without the dials in this list the predicate could not see it.
        3. **Device.** The consumer is a CUDA kernel. On CPU tensors the oracle is
           the only implementation that exists for that device, which is a device
           dispatch and not a degradation -- there is no CPU lowering to decline.

        Everything else that could make a CUDA-context launch impossible -- an
        un-instantiated ``d_v`` or ``BC``, an unrealized warp tile -- is deliberately
        NOT a condition here. Those raise loudly downstream
        (``_consumer_forward``, the kernel's ``TORCH_CHECK``s). Folding them in would turn a loud unsupported
        configuration into a silent slow path, which is the pattern spec §7.4.1
        forbids.
        """
        if self.training:
            return 'oracle'
        # Conditions 2 and 3 are `rola.interface.select_backend`, shared verbatim with
        # the functional op, so the layer and `rola_func` cannot drift into two
        # slightly different answers to the same question. Condition 1 stays here:
        # `self.training` is a property of a module and has no meaning for a free
        # function.
        tensors = [t for t in operation_inputs if t is not None]
        return 'oracle' if select_backend(*tensors) == 'reference' else 'cuda'

    def _decay_config(self) -> LeafMassDecay | None:
        if self.decay_kind == 'none':
            return None
        return LeafMassDecay(dials=tuple(torch.sigmoid(theta) for theta in self.decay_theta))

    def _resolve_page_arena(self, plan, BH: int, device):
        """The layer's §7.1 arena, built once and reused across steps.

        Pooled across every `(batch, head)` pair of this layer -- V2's shape, and
        the reason the savings math works (§7.1): a per-pair arena would round each
        pair's tiny live fraction up to its own growth chunk and cap savings at the
        chunk-to-footprint ratio, while a pooled arena tracks the live fraction of
        the whole layer.

        Rebuilt (not resized) whenever `BH`, the device or `BC` changes.

        **`BT` LEFT THIS LIST WITH THE §4.5 RE-KEY**, and that is the point of the
        re-key rather than an incidental simplification. It was here because the old
        arena recorded `first_write_tile` in units of the plan's `BT` (findings
        `paging.md` D3), so live pages could not be reinterpreted under a different
        tile. Under PLANNED-EXACT-ASYNC allocation nothing records a tile at all: the
        footprint is the popcount of §4.1's atom condensation, admission order is
        ascending canonical leaf order, and neither mentions `BT`. Since `BT` is now
        selected PER LAUNCH (§4.3), an arena that had to be rebuilt on a `BT` flip would
        have been rebuilt constantly.

        `BC` remains, and remains for a different reason: it is not stored anywhere in
        the arena (every stored object is `BC`-invariant, gated), but it fixes the
        owner geometry the caller reasons about, so an arena built for one topology's
        `BC` should not silently serve another's.
        """
        from rola.ops.paging import PageArena

        cols = self.head_v_dim + (1 if self.state_norm == 'global' else 0)
        arena = self._page_arena
        if (arena is not None and arena.BH == BH and arena.device == device
                and arena.cols == cols and arena.BC == plan.BC):
            return arena
        arena = PageArena(
            widths=self.topology.widths, BC=plan.BC, cols=cols, BH=BH, device=device,
            max_physical_pages=self.max_physical_pages,
            allow_capping=self.max_physical_pages is not None)
        self._page_arena = arena
        return arena

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RoLAState | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        budget_r: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, None, RoLAState | None]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, (
                "Expected attention_mask as a [batch, seq_len] 0-1 padding matrix; "
                "[batch, seq_len, seq_len] masks are not supported."
            )
        x = hidden_states
        B, L, _ = x.shape
        H = self.num_heads
        cache_enabled = bool(use_cache)
        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get('cu_seqlens')
        if cu_seqlens is not None:
            raise NotImplementedError("cu_seqlens (varlen) is not supported by RoLA.")
        if budget_r is not None and self.budget is None:
            raise ValueError("budget_r was supplied but this layer has no BudgetConditioning configured")

        v = self.v_proj(x)

        if self.use_short_conv:
            carried_conv = None if last_state is None else last_state.conv_state
            conv_states = list(carried_conv) if carried_conv is not None else []
            while len(conv_states) < 3:
                conv_states.append(None)
            if not cache_enabled:
                conv_states = [state.clone() if state is not None else None for state in conv_states]
            conv_state_v, conv_state_wl, conv_state_rl = conv_states
            v, conv_state_v = self.v_conv1d(v, cache=conv_state_v,
                                             output_final_state=cache_enabled, cu_seqlens=cu_seqlens)
            wx, conv_state_wl = self.write_route_conv1d(
                x, cache=conv_state_wl, output_final_state=cache_enabled, cu_seqlens=cu_seqlens)
            has_untied = any(
                isinstance(level.routing, UnionRouting) or not level.routing.tied
                for level in self.topology.levels
            )
            if has_untied:
                rx, conv_state_rl = self.read_route_conv1d(
                    x, cache=conv_state_rl, output_final_state=cache_enabled, cu_seqlens=cu_seqlens)
            else:
                rx = wx
        else:
            wx = x
            rx = x

        v = rearrange(v, 'b l (h d) -> b l h d', d=self.head_v_dim)
        h_w = None if rx is wx else wx

        recurrent_state = last_state.recurrent_state if last_state is not None else None
        if recurrent_state is not None and torch.is_grad_enabled():
            raise RuntimeError("RoLA carried-state execution is inference-only")

        routing_out = produce_routing(
            rx, h_w, v, self.route_W, self.route_bias_param, self.gain_W, self.gain_bias,
            self.topology, self.state_norm,
            budget=self.budget, r=budget_r, budget_embed_W=self.budget_embed_W,
            temp_W=self.temp_W, temp_b=self.temp_b,
            usage_bias_W=self.usage_bias_W, usage_bias_b=self.usage_bias_b,
            write_gain=self.write_gain, read_gain=self.read_gain,
        )
        decay_config = self._decay_config()
        backend = self.select_execution_backend(
            v, routing_out.g_read, routing_out.g_write,
            *routing_out.read_levels, *routing_out.write_levels,
            # fix pass (wiring-review.md §2, BLOCKER): the dials are handed straight
            # to `_consumer_forward(..., decay=decay_config)` below, so they are an
            # OPERATION input and must be visible to the predicate -- a dial-only
            # unfrozen forward has no other grad-requiring operand.
            *(decay_config.dials if decay_config is not None else ()),
        )
        if backend == 'cuda':
            # THE plan build, per step. Coordinator ruling 2026-07-30: selection is
            # per-step, never cached per shape. The statistics it prices are exact
            # over the realized schedule, so a plan is only as valid as the routing
            # it was measured on, and a cached plan goes stale the moment training
            # moves the routing distribution -- a stale plan prices a `K`
            # distribution that no longer exists. What the selection itself costs is
            # scalar comparisons dispatching among precompiled template
            # instantiations, and it is reported as its own plane by
            # `benchmarks/ops/benchmark_rola_v3_layer.py` rather than assumed small
            # -- and that measurement is what motivated the fusion below.
            #
            # `plan_and_schedule` is that selection FUSED with the §4 schedule
            # build (planner increment, 2026-07-30). The two walk the same routing
            # tensors and were each recomputing the same prefix: the
            # `[B,T,H,w] -> [BH,T,w]` fold (a real copy), the exact-nonzero mask,
            # and the per-(level, digit) §4 tile bitmaps. Fusing them deletes the
            # duplicate work and NOTHING else -- the plan is the plan
            # `select_plan` returns on the same inputs and the schedule is the
            # schedule `build_schedule` builds from it, gated bit-identical by
            # `test_the_shared_view_changes_nothing_it_only_stops_recomputing`.
            # Selection stays PER-STEP: the shared view lives inside this one
            # forward, so there is nothing that can go stale across steps.
            plan, axes, schedule = plan_and_schedule(
                routing_out.read_levels, routing_out.write_levels,
                routing_out.g_read, routing_out.g_write, self.topology,
                self.state_block, normalization=self.state_norm,
                decay=decay_config is not None, mma_precision=self.mma_precision,
                # The build-time density declaration (see `__init__`): every side
                # it marks full-support skips the per-step live-leaf pass for the
                # closed form it provably equals.
                static_density=self.static_density,
                # The per-SM SMEM budget, device-queried once and RECORDED in the
                # plan (§10's ruling), so the plan carries the machine it reasoned
                # about rather than depending on the one it is replayed on.
                #
                # PROVENANCE ONLY as of item 6b: no selection rule reads it any more.
                # It backed the analytic occupancy mirror, which modelled the deleted
                # consumer's footprint and was measured INVERTING the bound it
                # claimed, so the mirror was deleted rather than re-fitted. The field
                # stays because a plan stating the machine it was built on is what
                # makes a replayed plan auditable; it is flagged in `rewrite.md`
                # §2.36 as the one plan field with no consuming rule.
                sm_shared_bytes=sm_shared_bytes())
            self.last_selected_plan, self.last_plan_axes = plan, axes
            if self.paging:
                arena = self._resolve_page_arena(plan, B * H, v.device)
                continue_paging = recurrent_state is not None
                if recurrent_state is None:
                    # A fresh forward starts from a ZERO state. `_consumer_forward`
                    # is call-scoped by default (fix pass BLOCKER B1,
                    # `workflows/20260730-wiring/paging-review.md`): its own
                    # `paging.plan(schedule, fresh=True)` resets the arena before
                    # planning, so nothing needs to be done here. (Before the fix
                    # pass this branch called `arena.reset()` itself, which is now
                    # redundant with the default and would have doubled the reset
                    # launch on every fresh forward.)
                    pass
                else:
                    # An explicit bridge, not an implicit one: `adopt_dense` provisions
                    # every owner, so it must be a visible act. What SHOULD cross a call
                    # boundary under paging is spec §7.3's question, which §7.1 says it
                    # does not settle -- so this layer keeps the existing dense cache
                    # contract and pays the bridge, rather than inventing a page-table
                    # continuation contract here. `arena.reset()` clears the page table
                    # before loading the dense state; `continue_paging=True` below is
                    # what then stops `_consumer_forward`'s own default reset from
                    # wiping the state `adopt_dense` just committed.
                    arena.reset()
                    arena.adopt_dense(recurrent_state.reshape(arena.BH, arena.N, arena.cols)
                                      .to(torch.float32).contiguous())
                out, _arena, _schedule = _consumer_forward(
                    v, routing_out.read_levels, routing_out.write_levels,
                    routing_out.g_read, routing_out.g_write, plan,
                    decay=decay_config, schedule=schedule, paging=arena,
                    continue_paging=continue_paging)
                self.last_paging_stats = arena.last_result
                final_state = arena.materialize().view(B, H, arena.N, arena.cols) \
                    if cache_enabled else None
            else:
                self.last_paging_stats = None
                out, final_state, _schedule = _consumer_forward(
                    v, routing_out.read_levels, routing_out.write_levels,
                    routing_out.g_read, routing_out.g_write, plan,
                    initial_state=recurrent_state, decay=decay_config, schedule=schedule)
        else:
            if self.paging:
                # [[layer-dispatch-autograd-gap]]'s shape of failure: a flag honoured on
                # one dispatch arm and silently ignored on the other. The oracle has no
                # page table, so a paged layer that falls back to it would silently run
                # unpaged -- exact here (paging is bitwise invisible), but it would also
                # silently discard the `max_physical_pages` cap, which is NOT invisible.
                raise NotImplementedError(
                    "paging=True is a CUDA-backend feature (spec §7.1); this forward "
                    "dispatched to the fp64 oracle, which has no page table. The cap "
                    "would be silently unenforced, so this raises rather than degrading.")
            out, final_state = naive_rola(
                v, routing_out.read_levels, routing_out.write_levels,
                routing_out.g_read, routing_out.g_write, self.topology, decay_config,
                self.state_norm, initial_state=recurrent_state,
                output_final_state=cache_enabled,
            )
        self._execution_backend_counts[backend] += 1
        self.last_execution_backend = backend

        if cache_enabled:
            # Cache contract across the two backends. The carried recurrent state has
            # the SAME SHAPE `[B, H, N, d_v + (norm == 'global')]` and the same
            # meaning on both paths; only its dtype differs (the oracle accumulates in
            # fp64, the consumer in fp32), and BOTH entry points accept either
            # (`naive._initial_cache` casts to fp64, `_consumer_forward` casts to
            # fp32). A continuation may therefore cross backends without
            # reinterpretation. The guard above only covers CONSUMPTION -- a
            # grad-context step with a prior `recurrent_state` raises -- so a
            # grad-context step never *consumes* a carried state; it is not
            # guarded against *producing* one (a fresh `use_cache=True` step under
            # grad, e.g. the oracle half of the cross-backend cache test below,
            # legitimately produces a `final_state`, since the oracle is fully
            # differentiable and there is no severance to prevent). Harmless today
            # -- both backends write the same layout -- but recorded precisely
            # rather than as a wider guard that does not exist (fix pass,
            # `wiring-review.md` §4).
            update_layer_cache(
                self, past_key_values, recurrent_state=final_state,
                conv_state=(conv_state_v, conv_state_wl, conv_state_rl) if self.use_short_conv else None,
                offset=L)

        o = self.o_proj(out.reshape(B, L, H * self.head_v_dim))
        return o, None, past_key_values

    # --- state accounting (zoology Hybrid.state_size / LM matched-state read these) ---
    def get_stats(self):
        d_v = self.head_v_dim + int(self.state_norm != 'raw')
        conv_state = 0
        if self.use_short_conv:
            has_untied = any(
                isinstance(level.routing, UnionRouting) or not level.routing.tied
                for level in self.topology.levels
            )
            conv_state = self.conv_size * (self.value_dim + 2 * self.hidden_size if has_untied
                                            else self.value_dim + self.hidden_size)
        return {
            'd_v': self.head_v_dim,
            'n_heads': self.num_heads,
            'num_chunks': self.topology.N,
            'state_floats': self.num_heads * self.topology.N * d_v + conv_state,
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        return self.get_stats()['state_floats']


__all__ = ['LevelRoutingConfig', 'RoLA', 'resolve_level_routing_configs']
