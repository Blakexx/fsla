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
_POISON_TIERS = ('tests/tier1_semantics/', 'tests/tier2_geometry/', 'tests/layers/',
                 'tests/conformance/')


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
                 normalization="global", gain=True, bias=False,
                 expose_stats=False, gain_bias_init=None, gain_config=None):
    """One :class:`rola.RouteProducer` over one simplex producer covering every level."""
    from rola import RouteProducer, SimplexProducer, dense_routing

    producer = SimplexProducer(
        tuple(widths), dense_routing() if routing is None else routing,
        hidden_size=hidden_size, num_heads=num_heads, bias=bias,
        expose_stats=expose_stats)
    return RouteProducer(
        [producer], d_v=d_v, num_heads=num_heads, hidden_size=hidden_size,
        normalization=normalization, gain=gain, gain_bias_init=gain_bias_init,
        gain_config=gain_config)


def build_layer(*, hidden_size=64, num_heads=4, d_v=32, widths=(16,), routing=None,
                normalization="global", decay=None, decay_rate=2.0 ** -8, gain=True,
                bias=False, expert=None, layer_idx=None, **kwargs):
    """A :class:`rola.RoLA` over :func:`build_routes`.

    ``decay`` takes ``None`` (no decay), ``'learned'`` or ``'constant'`` -- the two
    shipped decay SOURCES -- rather than a rate, because which source a model uses is
    the choice the slot exists to express.
    """
    from rola import ConstantDecay, LearnedDecay, RoLA

    routes = build_routes(hidden_size=hidden_size, num_heads=num_heads, d_v=d_v,
                          widths=widths, routing=routing, normalization=normalization,
                          gain=gain, bias=bias, **kwargs)
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
# (`produce_routing`, `resolved_routing_from_topology`, `_project_logits`) from
# the exact primitives the new methods call (`_packed_router_logits`,
# `production_routing_factor_levels`, `project_side_gains`, the statistics
# primitives in `rola.routing.budget`, the mass fold `rola.routing.factors._fold`)
# -- so the tests keep exercising the real numerics, only the top-level wiring
# (which the old code also had as one free function) is reconstructed here
# rather than re-derived from a module.
#
# The §4.5 budget-conditioning arm this adapter used to also reassemble (a
# temperature actuator + a usage-bias head, both chunk-recurrent) is gone with
# the machinery it exercised (queue P29, 2026-08-06): the producer has no
# conditioning inputs left to drive.


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


def produce_routing(x, x_w, v, route_W, route_bias, gain_W, gain_bias, topology, normalization):
    """The pre-break `produce_routing(...) -> RoutingOutput`, reassembled.

    ``v`` is accepted (matching the old call sites) and unused: the old function
    never consumed it either, callers only needed it afterwards to run the
    oracle. The return carries the pre-break field names
    (`read_levels`/`write_levels`/`g_read`/`g_write`/`exact_support_counts`/
    `touched_bits`) as a plain namespace.
    """
    from types import SimpleNamespace

    from rola.routing.budget import exact_support_count, touched_bitset_update
    from rola.routing.entmax.production import production_routing_factor_levels
    from rola.routing.factors import _fold
    from rola.routing.side_gain import project_side_gains

    resolved = resolved_routing_from_topology(topology)
    B = x.shape[0]
    H = route_W.shape[0]
    D = topology.D
    widths = topology.widths

    read_logits, write_logits = _project_logits(
        x, route_W, resolved, route_bias=route_bias, h_w=x_w)
    read_bh, write_bh = production_routing_factor_levels(read_logits, write_logits, resolved)

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
        exact_support_counts=exact_support_counts, touched_bits=touched_bits)


# -----------------------------------------------------------------------------
# A routing bundle over HAND-BUILT tensors (P67 D2-b)
# -----------------------------------------------------------------------------


class RawRoutes:
    """The routing bundle `rola.ops.chunk` reads, over hand-built tensors.

    `RouteFactors` refuses direct construction on purpose -- a bundle whose
    statistics did not come from the derivation code could disagree with its own
    tensors -- and a gate that plants a specific routing pattern (a conformance
    regime cell, a sub-subnormal amplitude, a confined write support) is by
    definition not producer-derived. So those gates reach the kernel arm through
    this: EXACTLY the fields `rola.ops.chunk`'s `unsupported`/`arm`/`stats`/`run`
    read, and nothing else, so it cannot quietly stand in for a real bundle
    anywhere one is required -- `rola_op` refuses it, which is the point.

    It lives here rather than in one of its callers for the reason `build_routes`
    does: every gate that drives the kernel arm over planted tensors needs the
    same six fields, and a per-file copy is how they drift.
    """

    def __init__(self, topology, normalization, read, write, g_read, g_write):
        self.topology, self.normalization = topology, normalization
        self.read, self.write = read, write
        self.g_read, self.g_write = g_read, g_write

    @property
    def batch(self):
        return self.read[0].shape[0]

    @property
    def tokens(self):
        return self.read[0].shape[1]

    @property
    def heads(self):
        return self.read[0].shape[2]

    @property
    def device(self):
        return self.read[0].device
