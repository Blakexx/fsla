import inspect
import pathlib
import sys
from unittest.mock import patch

import pytest
import torch

# `benchmarks/` on the path, so the bench PACKAGES import by their own names -- the
# repository's existing convention (`benchmarks/bench_regression.py` inserts the same
# directory and imports `bench.cells`). What still needs it is `bench.cells` and the
# bench drivers; the planner CALIBRATION package no longer does -- it moved to
# `rola/planner/calibration/` (post-merge, 2026-08-02) because an installed wheel has
# no `benchmarks/` sibling and every relocated name raised `ModuleNotFoundError` there.
# The runtime still imports none of it: `rola.planner.schedule` holds lazy call
# proxies, and the subprocess probe in `tests/unit/test_planner.py` asserts
# `import rola` leaves `sys.modules` free of the package.
_BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"
if _BENCHMARKS.is_dir() and str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

try:
    from torch.compiler import is_compiling
except ImportError:
    def is_compiling():
        return False

# The device's torch submodule (`torch.cuda`), resolved without depending on the
# fork's `fla.utils`: `rola` supports NVIDIA CUDA only, and the ratification gate
# is what decides which CUDA devices at that.
device_torch_lib = torch.cuda

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_ORIGINAL_EMPTY = torch.empty
_ORIGINAL_EMPTY_LIKE = torch.empty_like
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty


def _is_called_from_fla():
    """Check if the call is from the `rola` package."""
    frame = inspect.currentframe()
    try:
        # Skip the current frame and go up the call stack
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            if hasattr(frame, 'f_code') and hasattr(frame.f_code, 'co_filename'):
                filename = frame.f_code.co_filename
                # Skip conftest.py frames (where the guarded functions are defined)
                if 'conftest.py' in filename:
                    continue
                # Check if this frame is from a test file
                # Look for 'tests/' or 'test_' in the file path
                if 'tests/' in filename or 'test_' in filename:
                    return False

            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                # If call is from the `rola` package, apply guard
                if module.__name__.split('.')[0] == 'rola':
                    return True
    finally:
        del frame
    # Default to not guarding if we can't determine
    return False


def _poison(result):
    """Fill a scratch tensor with NaN. Skip requires_grad leaves: inductor's lazy init (e.g. pad_mm)
    allocates grad-leaf scratch via torch.empty, and an in-place fill_ on a leaf-that-requires-grad
    raises — so with torch.compile defaulted on, poisoning those would break compilation, not catch a
    real uninitialized-read bug. (Those tensors are inductor's, not fla buffers.)"""
    if result.requires_grad:
        return result
    if result.is_floating_point():
        result.fill_(float('nan'))
    elif result.is_complex():
        result.fill_(complex(float('nan'), float('nan')))
    return result


def _guarded_empty(*args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    dtype = kwargs.get('dtype') or torch.get_default_dtype()

    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY(*args, **kwargs)

    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY(*args, **kwargs)

    return _poison(_ORIGINAL_EMPTY(*args, **kwargs))


def _guarded_empty_like(input, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = input.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    return _poison(_ORIGINAL_EMPTY_LIKE(input, **kwargs))


def _guarded_new_empty(self, *args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = self.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    return _poison(_ORIGINAL_NEW_EMPTY(self, *args, **kwargs))


#: The tiers whose tests exercise `rola`'s tensor math directly (op-level: the two
#: semantics/geometry tiers; module-level: the layer tier). The unit tier is host-side
#: planner/schedule logic and is deliberately outside the guard. NOTE (coverage-closure
#: 2026-08-03): this previously gated on `tests/ops/` and `tests/modules/` — the FORK's
#: layout, neither of which exists here — so the NaN-poison guard had never run.
_POISON_TIERS = ('tests/tier1_semantics/', 'tests/tier2_geometry/', 'tests/layers/')


@pytest.fixture(scope="function", autouse=True)
def poison_torch_memory(request):
    # Only apply the guard to the tiers that reach rola's tensor math.
    path = str(request.node.fspath)
    if not any(tier in path for tier in _POISON_TIERS):
        yield
        return

    with patch('torch.empty', new=_guarded_empty), \
            patch('torch.empty_like', new=_guarded_empty_like), \
            patch('torch.Tensor.new_empty', new=_guarded_new_empty):
        yield
        if hasattr(device_torch_lib, 'synchronize'):
            device_torch_lib.synchronize()


# -----------------------------------------------------------------------------
# Building a model, in one place
# -----------------------------------------------------------------------------
#
# The public surface is the reduction theorem: a feature map, and linear attention.
# Assembling one is three lines -- producers, the gateway, the layer -- and every test
# that needs a model needs the same three, so they live here rather than being spelled
# out per file. A test that is ABOUT construction (validation, refusal, the expert pin)
# calls the real constructors directly; these are for the tests that need a model to
# have something to run.


def build_routes(*, hidden_size=64, num_heads=4, d_v=32, widths=(16,), routing=None,
                 normalization="global", gain=True, budget=None, bias=False,
                 expose_stats=False, gain_bias_init=None, gain_config=None):
    """One :class:`rola.RouteProducer` over one simplex producer covering every level."""
    from rola import RouteProducer, SimplexProducer, dense_routing

    producer = SimplexProducer(
        tuple(widths), dense_routing() if routing is None else routing,
        hidden_size=hidden_size, num_heads=num_heads, bias=bias, budget=budget,
        expose_stats=expose_stats)
    return RouteProducer(
        [producer], d_v=d_v, num_heads=num_heads, hidden_size=hidden_size,
        normalization=normalization, gain=gain, gain_bias_init=gain_bias_init,
        gain_config=gain_config)


def build_layer(*, hidden_size=64, num_heads=4, d_v=32, widths=(16,), routing=None,
                normalization="global", decay=None, decay_rate=2.0 ** -8, gain=True,
                budget=None, bias=False, expert=None, layer_idx=None, **kwargs):
    """A :class:`rola.RoLA` over :func:`build_routes`.

    ``decay`` takes ``None`` (no decay), ``'learned'`` or ``'constant'`` -- the two
    shipped decay SOURCES -- rather than a rate, because which source a model uses is
    the choice the slot exists to express.
    """
    from rola import ConstantDecay, LearnedDecay, RoLA

    routes = build_routes(hidden_size=hidden_size, num_heads=num_heads, d_v=d_v,
                          widths=widths, routing=routing, normalization=normalization,
                          gain=gain, budget=budget, bias=bias, **kwargs)
    sources = {"learned": LearnedDecay, "constant": ConstantDecay}
    source = None if decay is None else sources[decay](
        decay_rate, widths=routes.widths, num_heads=num_heads)
    return RoLA(hidden_size, num_heads, routes, decay=source, layer_idx=layer_idx,
                expert=expert)


# -----------------------------------------------------------------------------
# Pre-break producer adapter (API-contract migration, 2026-08-05)
# -----------------------------------------------------------------------------
#
# The low-level producer gates in tests/unit/test_producer.py and
# tests/unit/test_production_levels.py drive raw parameter tensors directly
# (route_W, gain_W, ... as bare leaves) rather than a module's parameters, which
# is what lets them isolate a backward-through-every-tensor gate. The break moved
# that orchestration inside `SimplexProducer`/`RouteProducer` (nn.Module methods
# over nn.Parameters), so there is no longer a free function taking bare tensors.
# These adapters reassemble the pre-break free-function surface
# (`produce_routing`, `resolved_routing_from_topology`, `_project_logits`,
# `embed_budget`) from the exact primitives the new methods call
# (`_packed_router_logits`, `production_routing_factor_levels`,
# `project_side_gains`, the budget arithmetic in `rola.routing.budget`, the mass
# fold `rola.routing.factors._fold`) -- so the tests keep exercising the real
# numerics, only the top-level wiring (which the old code also had as one free
# function) is reconstructed here rather than re-derived from a module.


def resolved_routing_from_topology(topology):
    """A `ResolvedRouting` over a `Topology`'s own levels -- exactly what
    `SimplexProducer.__init__` builds for itself
    (`self.routing = ResolvedRouting(levels=self.levels, branches=self.widths)`)."""
    from rola.routing.types import ResolvedRouting

    return ResolvedRouting(
        levels=tuple(level.routing for level in topology.levels), branches=topology.widths)


def _project_logits(x, route_W, routing, *, route_bias=None, h_w=None):
    """The packed router projection at `SimplexProducer._project`'s own precision
    rule: fp64 in, fp64 out; anything else projects in fp32."""
    from rola.routing.projection import _packed_router_logits

    projection_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return _packed_router_logits(
        x, route_W, routing, route_bias=route_bias, h_w=h_w, projection_dtype=projection_dtype)


def embed_budget(hidden, r, embed_W):
    """`r is None` is the identity (no conditioning stream to embed); otherwise
    the shipped `rola.routing.producer._embed_budget`."""
    if r is None:
        return hidden
    from rola.routing.producer import _embed_budget

    return _embed_budget(hidden, r, embed_W)


def produce_routing(
    x, x_w, v, route_W, route_bias, gain_W, gain_bias, topology, normalization,
    *, budget=None, r=None, budget_embed_W=None,
    temp_W=None, temp_b=None, usage_bias_W=None, usage_bias_b=None,
):
    """The pre-break `produce_routing(...) -> RoutingOutput`, reassembled.

    ``v`` is accepted (matching the old call sites) and unused: the old function
    never consumed it either, callers only needed it afterwards to run the
    oracle. The return carries the pre-break field names
    (`read_levels`/`write_levels`/`g_read`/`g_write`/`exact_support_counts`/
    `touched_bits`/`slack_history`) as a plain namespace.
    """
    from types import SimpleNamespace

    import torch.nn.functional as F

    from rola.routing.budget import (
        chunk_boundaries,
        chunk_target_count,
        exact_support_count,
        normalized_slack,
        touched_bitset_update,
        update_soft_usage,
    )
    from rola.routing.entmax.production import production_routing_factor_levels
    from rola.routing.factors import _fold
    from rola.routing.producer import _TEMPERATURE_FLOOR
    from rola.routing.side_gain import project_side_gains

    resolved = resolved_routing_from_topology(topology)
    B, L = x.shape[0], x.shape[1]
    H = route_W.shape[0]
    D = topology.D
    widths = topology.widths

    if budget_embed_W is not None and r is not None:
        x = embed_budget(x, r, budget_embed_W)
        if x_w is not None:
            x_w = embed_budget(x_w, r, budget_embed_W)

    conditioned = budget is not None and (
        (r is not None and budget.budgeted_pairs) or budget.usage_bias_enabled)

    if not conditioned:
        read_logits, write_logits = _project_logits(
            x, route_W, resolved, route_bias=route_bias, h_w=x_w)
        read_bh, write_bh = production_routing_factor_levels(read_logits, write_logits, resolved)
        slack_history: dict = {}
    else:
        device, dtype = x.device, x.dtype
        pairs = budget.budgeted_pairs
        pair_index = {pair: i for i, pair in enumerate(pairs)}
        read_logits_full, write_logits_full = _project_logits(
            x, route_W, resolved, route_bias=route_bias, h_w=x_w)
        actuators_active = (r is not None and bool(pair_index)) or budget.usage_bias_enabled

        def head_param(param_over_h):
            return param_over_h.unsqueeze(0).expand(B, *param_over_h.shape).reshape(
                B * H, *param_over_h.shape[1:])

        r_bh = None if r is None else r.to(device=device, dtype=dtype)[:, None].expand(
            B, H).reshape(B * H)
        running_slack = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
        running_count = {pair: torch.zeros(B * H, dtype=dtype, device=device) for pair in pairs}
        slack_lists: dict = {pair: [] for pair in pairs}
        running_usage = {level: torch.zeros(B * H, widths[level], dtype=dtype, device=device)
                         for level in range(D)}
        running_touched = {level: torch.zeros(B * H, widths[level], dtype=torch.bool, device=device)
                           for level in range(D)}
        tokens_seen = 0
        read_chunks: list = [[] for _ in range(D)]
        write_chunks: list = [[] for _ in range(D)]

        for start, end in chunk_boundaries(L, budget.chunk_size):
            read_logits = read_logits_full[:, start:end]
            write_logits = write_logits_full[:, start:end]
            if actuators_active:
                write_scale = torch.ones_like(write_logits)
                write_bias = torch.zeros_like(write_logits)
                read_scale = torch.ones_like(read_logits)
                if r_bh is not None:
                    for (level, side), idx in pair_index.items():
                        cond = torch.stack([r_bh, running_slack[(level, side)]], dim=-1)
                        temp_logit = (cond * head_param(temp_W[:, idx, :])).sum(-1)
                        if temp_b is not None:
                            temp_logit = temp_logit + head_param(
                                temp_b[:, idx:idx + 1]).squeeze(-1)
                        c = F.softplus(temp_logit) + _TEMPERATURE_FLOOR
                        buffer = write_scale if side == "write" else read_scale
                        buffer[:, :, resolved.level_logit_slice(level)] = c[:, None, None]
                if budget.usage_bias_enabled:
                    r_feature = r_bh if r_bh is not None else torch.zeros(
                        B * H, dtype=dtype, device=device)
                    for level in range(D):
                        usage_norm = running_usage[level] / max(tokens_seen, 1)
                        touched_f = running_touched[level].to(dtype)
                        write_slack = running_slack.get(
                            (level, "write"), torch.zeros(B * H, dtype=dtype, device=device))
                        w0, w1, w2, w3 = (head_param(usage_bias_W[:, level, i:i + 1])
                                          for i in range(4))
                        offset = (w0 * usage_norm + w1 * touched_f
                                  + w2 * r_feature[:, None] + w3 * write_slack[:, None]
                                  + head_param(usage_bias_b[:, level:level + 1]))
                        span = resolved.level_logit_slice(level)
                        write_bias[:, :, span] = write_bias[:, :, span] + offset[:, None, :]
                write_logits = write_logits * write_scale + write_bias
                read_logits = read_logits * read_scale

            read_bh_c, write_bh_c = production_routing_factor_levels(
                read_logits, write_logits, resolved)
            for level in range(D):
                read_chunks[level].append(read_bh_c[level])
                write_chunks[level].append(write_bh_c[level])
            tokens_seen += end - start
            for level in range(D):
                stored = write_bh_c[level] / write_bh_c[level].sum(dim=-1, keepdim=True)
                running_usage[level] = update_soft_usage(running_usage[level], stored)
                running_touched[level] = touched_bitset_update(running_touched[level], stored)
            for (level, side) in pairs:
                values = write_bh_c[level] if side == "write" else read_bh_c[level]
                if r_bh is not None:
                    running_count[(level, side)] = running_count[(level, side)] + exact_support_count(
                        values).detach().sum(dim=-1)
                    running_slack[(level, side)] = normalized_slack(
                        chunk_target_count(r_bh, widths[level], tokens_seen),
                        running_count[(level, side)])
                slack_lists[(level, side)].append(running_slack[(level, side)])

        read_bh = tuple(torch.cat(read_chunks[level], dim=1) for level in range(D))
        write_bh = tuple(torch.cat(write_chunks[level], dim=1) for level in range(D))
        slack_history = {pair: torch.stack(hist, dim=-1).reshape(B, H, -1)
                         for pair, hist in slack_lists.items()}

    def unfold(tensor):
        T, width = tensor.shape[1], tensor.shape[2]
        return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)

    read_unfold = tuple(unfold(t) for t in read_bh)
    write_unfold = tuple(unfold(t) for t in write_bh)
    read_levels, read_mass = _fold(read_unfold, B=B, H=H)
    write_levels, write_mass = _fold(write_unfold, B=B, H=H)

    projection_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    gains = project_side_gains(
        x, gain_W, gain_bias, normalization=normalization, projection_dtype=projection_dtype)
    g_write = gains["g_write"] * write_mass
    g_read = gains["g_read"]
    if normalization == "raw":
        g_read = g_read * read_mass

    exact_support_counts = {}
    for level in range(D):
        for side, levels in (("read", read_levels), ("write", write_levels)):
            exact_support_counts[(level, side)] = exact_support_count(levels[level])
    touched_bits = {
        level: touched_bitset_update(
            torch.zeros(B, H, widths[level], dtype=torch.bool, device=x.device),
            write_levels[level].permute(0, 2, 1, 3))
        for level in range(D)
    }

    return SimpleNamespace(
        read_levels=read_levels, write_levels=write_levels,
        g_read=g_read, g_write=g_write,
        exact_support_counts=exact_support_counts, touched_bits=touched_bits,
        slack_history=slack_history)
