"""Layer execution-backend dispatch gate (``ROLA_V3_SPEC.md`` §3-§6, §10 Stage 2).

`rola/layers/rola.py`'s forward has two execution backends -- the differentiable
Stage 1 oracle `naive_rola` and the plan-selected, **forward-only** V3 CUDA
consumer -- and `RoLA.select_execution_backend` chooses between them. This file is
the gate on that choice.

**Why this file exists and what it is guarding against.** A forward-only kernel
reachable from a grad-requiring context does not raise. It returns a tensor with no
`grad_fn`, the graph is severed at this layer, and everything upstream silently gets
`.grad = None` while training appears to run. That is not hypothetical: it is the
recorded `layer-dispatch-autograd-gap` class, and the fleet-smoke `x.grad = None`
failure was exactly an op-level-correct kernel reached through a LAYER's mode
dispatch that op-level tests could not see. Op-level tests here would be equally
blind -- `tests/ops/test_rola_v3_consumer.py` gates the kernel and
`test_the_plan_selected_path_is_oracle_exact_and_bit_identical` gates the selector,
and neither one goes through `RoLA.forward`.

Four properties, and they are the four the wiring increment claims:

1. **Grad context runs the oracle, and the graph survives it** -- every parameter
   gets a finite, NONZERO gradient, and the CUDA counter does not move. Finite is
   not enough: a severed graph produces `None`, but a partially severed one
   produces exact zeros, and `x.grad = None` is the failure this file is named for.
2. **No-grad context on CUDA runs the consumer** -- asserted on the layer's own
   backend COUNTER, never on wall-clock. Timing cannot distinguish "took the CUDA
   path" from "took a fast oracle path", and the counter is a fact about which
   branch executed.
3. **The two backends agree** on identical inputs at the consumer suite's own
   unmodified `RTOL`, across both `state_norm` arms, both decay arms, and the
   production sparse entmax router as well as the dense softmax one.
4. **Nothing degrades silently in either direction.** A CUDA-context configuration
   the consumer cannot launch RAISES; it does not quietly land on the oracle.

**MIGRATED (API-contract break, 2026-08-05).** `RoLA.__init__`'s flat kwargs
(`head_v_dim`, `levels_per_head`/`states_per_level`, `level_routing_configs`,
`state_norm`, `paging`, `mma_precision`) are gone; a layer is now built from a
`RouteProducer` (the feature-map slot) plus a decay SOURCE and a
`rola.expert.PlanOverrides` (`state_block`/`mma_precision`/`max_physical_pages`).
`BC` IS NOT A LAUNCH DECISION HERE ANY MORE. The kernel arm is the chunk consumer,
whose `BC` is a slot tile inside a BUILT arm and comes from the manifest rather than
from a per-launch derivation, so there is no launch geometry left for this file to
assert and no attribute carrying one. What it asserts instead is WHICH ARM
ran and why, which is the claim that survived: the module's conditions (training mode),
the operation's (gradient reachability, device), and the ENVELOPE
(`rola.interface.dispatch_for(...).no_kernel_arm` -- a configuration outside the built
matrix dispatches to the oracle and says which clause refused it).
"""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint
from conftest import build_layer, build_routes, produce_routing

import rola
from rola import RoLA, dense_routing, sparse_routing
from rola.expert import PlanOverrides
from rola.interface import dispatch_for

#: P26: `mma_precision` is a dead axis (bf16 is the only value `PlanOverrides`
#: accepts; `SUPPORTED_BT`/the `(BC, BT)` template pair are gone). Every backend
#: comparison in this file therefore runs bf16-vs-fp64-oracle, and the bound is the
#: bf16 family bound (`tests/tier1_semantics/tolerances.py:BF16_RTOL`),
#: quoted at its value rather than imported so that loosening it here would be a
#: visible edit to this file. Max absolute deviation divided by the reference's own
#: scale; whole-layer outputs sit behind `o_proj`, a linear map applied identically
#: to both backends, so the relative comparison basis carries through unchanged.
BF16_RTOL = 1e-2

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the V3 consumer is a CUDA kernel")

#: `branches=[16, 16]`, `d_v=64`, `H=8` -- a BUILT CHUNK ARM (`(C, BC, D, B) =
#: (32, 64, 2, 16)` in the manifest), at the same `N = 256` as the Stage 0 matched
#: cell it replaced. The cell moved with the unsupported-topology retirement: the
#: flagship `(8, 16)` is NON-UNIFORM and now dispatches to the oracle, so a dispatch
#: file fixtured on it could no longer assert the kernel arm at all. Its retirement is
#: asserted on its own, below, rather than lost.
#: Every `T` here is a multiple of the chunk `C = 32` for the same reason.
_BRANCHES = (16, 16)
_H, _DV = 8, 64
_HIDDEN = _H * _DV

#: Old `decay="none"`/`"leaf_mass"` -> the two shipped decay SOURCES `build_layer`
#: takes (`None`/`"learned"`); the exact rate is immaterial to these dispatch-only
#: gates, only whether a decay source is present at all.
_DECAY = {"none": None, "leaf_mass": "learned"}


def _routing_for(router: str):
    """Old `LevelRoutingConfig(read=router, write=router, tied=False, alpha=1.5)`
    with a symmetric side -- exactly the case `dense_routing()`/`sparse_routing()`
    (the tag vocabulary's own convenience constructors) name."""
    return dense_routing() if router == "dense" else sparse_routing(1.5)


def _layer(*, norm: str = "global", decay: str, router: str = "dense", device: str = "cuda",
           logit_gain: float = 1.0, head_v_dim: int = _DV, branches=None) -> RoLA:
    """The dispatch fixture. `norm` stays a parameter -- pinned to `"global"` by
    every caller now that `"raw"` is a dead axis validated at construction
    (`rola.routing.types.validate_normalization`) -- so a caller need not repeat the
    pin. `mma_precision` is not a parameter at all: `PlanOverrides`' default is
    `"bf16"` and it is the only value the dataclass accepts (P26, dead axis), so
    every backend comparison in this file runs bf16-vs-fp64-oracle at `BF16_RTOL`.
    """
    branches = _BRANCHES if branches is None else tuple(branches)
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=head_v_dim, widths=branches,
        routing=_routing_for(router), normalization=norm, decay=_DECAY[decay],
        layer_idx=0,
    ).to(device, torch.float32)
    if logit_gain != 1.0:
        # Sharpens an UNTRAINED exact-entmax router into a real sparse support. The
        # support is genuine (exact alpha=1.5, the production path -- never an
        # epsilon floor or a top-k stand-in); only its LEVEL is a synthetic stand-in
        # for training, and that is what makes this a usable sparse fixture rather
        # than a second dense one.
        with torch.no_grad():
            layer.routes.producers[0].route_W.mul_(logit_gain)
    return layer


def _decay_config(layer: RoLA):
    return None if layer.decay is None else layer.decay()


def _sparsity(layer: RoLA, x: torch.Tensor) -> float:
    """Realized exact-zero fraction of the router's write allocations, so a test
    that claims to exercise a sparse fixture can prove the fixture is sparse."""
    producer = layer.routes.producers[0]
    with torch.no_grad():
        out = produce_routing(
            x, None, torch.zeros(x.shape[0], x.shape[1], layer.num_heads, layer.head_v_dim,
                                 device=x.device, dtype=x.dtype),
            producer.route_W, producer.route_bias, layer.routes.side_gain.gain_W, layer.routes.side_gain.gain_bias,
            layer.topology, layer.normalization)
    zeros = sum(float((t == 0).float().mean()) * t.shape[-1] for t in out.write_levels)
    return zeros / sum(t.shape[-1] for t in out.write_levels)


def _support_fraction(layer: RoLA, x: torch.Tensor) -> float:
    """The realized WRITE-side support fraction on the same routing a forward used.

    Reconstructed rather than read off the layer: the routing bundle lives inside one
    forward, and a layer that retained a statistic off it would be retaining a per-step
    measurement past the step it describes.

    The row this serves is about the layer RE-MEASURING per step rather than caching
    per shape, so the exact statistic is not the point: the realized support fraction
    is what a router sharpening moves most directly, and it needs no launch geometry
    to state.
    """
    routes = layer.routes(x)
    live = sum(int((level != 0).sum()) for level in routes.write)
    total = sum(level.numel() for level in routes.write)
    return live / total


# ---------------------------------------------------------------------------
# 1. The grad context runs the oracle, and the graph survives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_a_grad_context_takes_the_oracle_and_every_parameter_gets_a_live_gradient(
        decay, router, gain):
    """Property 1. On CUDA, with grad enabled, the forward-only consumer MUST NOT
    run -- and the proof that it did not is the counter, plus a backward that
    actually reaches every parameter.

    **`out.grad_fn is not None` does NOT bind here, and asserting it would be a
    false gate.** Measured by mutation (forcing `select_execution_backend` to return
    `'cuda'` in a grad context, 2026-07-30): the layer output still carries a
    `grad_fn`, because `o_proj` is a `nn.Linear` applied AFTER the severance and it
    re-attaches a graph downstream of the break. What the same mutation produces is
    `x.grad = None`, `route_W.grad = None`, `gain_W.grad = None`,
    `v_proj.weight.grad = None`, and `o_proj.weight.grad = 2.9e-4` -- exactly one
    surviving parameter, on the far side of the cut. So the assertions that bind are
    the backend counter and the per-parameter gradient, and those are the ones made.
    """
    torch.manual_seed(3)
    layer = _layer(decay=decay, router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda", requires_grad=True)

    out, _ = layer(x)
    out.square().mean().backward()

    assert layer.last_execution_backend == "oracle"
    assert layer.execution_backend_counts == {"oracle": 1, "cuda": 0, "decode": 0}, (
        "a grad-requiring forward reached the forward-only CUDA consumer")
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().max()) > 0

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} received a non-finite gradient"
        assert float(param.grad.abs().max()) > 0, f"{name} received an all-zero gradient"


@_CUDA
def test_grad_enabled_but_nothing_requiring_grad_is_not_a_grad_context():
    """The predicate is gradient REACHABILITY, not the global grad flag.

    With every parameter frozen and an input that does not require grad, there is no
    graph to sever, so the CUDA path is correct and taking the oracle instead would
    be a silent slowdown with no compensating property. This pins the predicate to
    the operation's inputs rather than to `torch.is_grad_enabled()` alone.
    """
    torch.manual_seed(5)
    layer = _layer(norm="global", decay="none")
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    for param in layer.parameters():
        param.requires_grad_(False)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    assert torch.is_grad_enabled()
    out, _ = layer(x)
    assert layer.last_execution_backend == "cuda"
    assert out.grad_fn is None


# ---------------------------------------------------------------------------
# 2. The no-grad context runs the consumer, on a SELECTED plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_no_grad_cuda_context_takes_the_kernel_arm(decay):
    """Property 2, asserted on the counter -- not on timing.

    The launch geometry is NOT asserted here any more, and there is nothing left to
    assert it about: the kernel arm is the chunk consumer and its `BC` is a slot tile
    inside a built arm, fixed by the manifest rather than derived per launch.
    """
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    torch.manual_seed(7)
    layer = _layer(decay=decay)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
        #: inside the no-grad context on purpose: `dispatch_for` answers the gradient
        #: question FIRST and short-circuits, so an envelope read under grad would say
        #: `no_kernel_arm is None` about every topology and assert nothing.
        envelope = dispatch_for(layer.routes(x), x)
    assert layer.execution_backend_counts == {"oracle": 0, "cuda": 1, "decode": 0}
    assert envelope.no_kernel_arm is None, (
        "fixture rot: this topology must be a BUILT chunk arm for the row above to "
        "mean anything")


@_CUDA
def test_the_selected_plan_is_re_measured_every_step_not_cached():
    """The coordinator's per-step ruling, gated rather than commented.

    Moving the routing distribution (here by sharpening the router between two no-grad
    forwards on the same shape) must move the measured statistics the plan was priced
    from. A per-shape cache would return the first step's statistics for the second
    step's routing, which is precisely the staleness the ruling rejects.

    **RE-POINTED THREE TIMES, AND EACH TIME THE STATISTIC MOVED, NOT THE CLAIM.**
    `last_plan_axes` left the layer (the API contract), then `k_hat` was replaced
    because at THIS cell it could not move at all (`QPO = 1`, so every candidate
    pair quantizes to one k-step whether its support is 94 % or 19 %), and at P67
    D2-b the planner summary it was read from was deleted (R-6). What the row has
    always asserted is that a layer re-measures its routing per STEP rather than
    caching per shape; the realized write-side support fraction carries a 16x
    router sharpening directly and depends on no launch geometry to be stated.
    """
    torch.manual_seed(11)
    layer = _layer(decay="none", router="sparse")
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 128, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
        dense = _support_fraction(layer, x)
        layer.routes.producers[0].route_W.mul_(16.0)
        layer(x)
        sharp = _support_fraction(layer, x)
    assert layer.execution_backend_counts == {"oracle": 0, "cuda": 2, "decode": 0}
    assert sharp < 0.5 < dense, (
        "the sharpening did not move the routing it is supposed to move: the realized "
        f"write support went {dense:.3f} -> {sharp:.3f}")


# ---------------------------------------------------------------------------
# 3. The two backends agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_the_two_backends_agree_on_the_same_inputs(decay, router, gain):
    """Property 3, on ONE layer instance and ONE input, so the only difference
    between the two runs is which backend executed.

    The grad-context path is forced by `requires_grad`, the no-grad path by
    `torch.no_grad()`; both consume the identical parameters and the identical `x`.
    `mma_precision` is a dead axis (bf16 is the only value `PlanOverrides` accepts,
    P26), so this is also THE layer's default-arm numerical gate -- there is no
    second, tighter "structural" arm left to compare against, and the comparison
    runs at `BF16_RTOL`, the bf16 family bound.
    """
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    torch.manual_seed(13)
    layer = _layer(decay=decay, router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    if router == "sparse":
        assert _sparsity(layer, x) > 0.2, "the sparse fixture is vacuously dense"

    x_grad = x.clone().requires_grad_(True)
    oracle_out, _ = layer(x_grad)
    assert layer.last_execution_backend == "oracle"
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    with torch.no_grad():
        cuda_out, _ = layer(x)
    assert layer.last_execution_backend == "cuda"

    reference = oracle_out.detach()
    scale = float(reference.abs().max())
    deviation = float((cuda_out - reference).abs().max()) / scale
    assert deviation < BF16_RTOL, (
        f"backends disagree at {deviation:.3e} (BF16_RTOL={BF16_RTOL:.1e}); "
        f"decay={decay} router={router}")


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_the_two_backends_agree_on_the_carried_state_they_hand_back(decay):
    """The cache surface, which is the one place the two backends' outputs are not
    just compared but INTERCHANGED.

    Both write `[B, H, N, d_v + 1]` with the same meaning (the mass column is
    unconditional now that `"raw"` is dead); only the accumulation dtype differs
    (fp64 oracle, fp32 consumer) and both entry points accept either. A
    continuation may therefore cross backends, which is what this asserts: same
    shape, and the same values at `BF16_RTOL`.

    Fix pass item 3f (wiring-review.md §9 NOTE): this gate was `decay='none'` only;
    it is now parametrized over both decay arms, closing the narrower-than-the-claim
    gap the review flagged (the `leaf_mass` arm's carried state was never exercised
    here, only asserted empirically by the reviewer).
    """
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    torch.manual_seed(17)
    layer = _layer(decay=decay)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode;
    # the oracle leg below still takes 'oracle' because `x` does not require grad but
    # the (unfrozen) projection parameters do, so `v` requires grad under enable_grad.
    import rola
    x = torch.randn(1, 64, _HIDDEN, device="cuda")

    states = {}
    for tag, context in (("oracle", torch.enable_grad()), ("cuda", torch.no_grad())):
        carried = rola.state()
        with context:
            layer(x, state=carried)
        assert layer.last_execution_backend == tag
        states[tag] = carried.materialize()

    assert states["oracle"].shape == states["cuda"].shape
    scale = float(states["oracle"].abs().max())
    deviation = float((states["cuda"].double() - states["oracle"].double()).abs().max()) / scale
    assert deviation < BF16_RTOL, (
        f"carried states disagree at {deviation:.3e} (BF16_RTOL={BF16_RTOL:.1e})")


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_carried_state_actually_crosses_backends_on_a_real_continuation(decay):
    """Fix pass item 3f (wiring-review.md §9 NOTE): the prior gate compared two
    INDEPENDENTLY produced states; it never fed one backend's state into the other
    backend's entry point, which is the "entry points accept either dtype" half of
    the claim. This does: an oracle-produced (fp64) carried state is handed to a
    SECOND forward that takes the CUDA backend, through the ordinary cache
    machinery, and the continuation must run and produce a finite output.

    The reverse direction -- feeding a CUDA-produced (fp32) state into a
    grad-context oracle step -- is not reachable through the layer at all: the
    layer's own guard (`forward`, "carried-state execution is inference-only")
    raises on ANY grad-context step that consumes a prior `recurrent_state`, and the
    oracle backend is only ever selected in a grad context. That half of the claim
    is exercised directly against the ops entry points below, matching how the
    reviewer checked it.
    """
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    torch.manual_seed(37)
    layer = _layer(decay=decay)
    layer.eval()
    import rola
    x0 = torch.randn(1, 48, _HIDDEN, device="cuda")
    x1 = torch.randn(1, 32, _HIDDEN, device="cuda")

    # Step 1: produce a carried state on the oracle (grad context).
    carried = rola.state()
    with torch.enable_grad():
        layer(x0, state=carried)
    assert layer.last_execution_backend == "oracle"
    oracle_state = carried.materialize()
    assert oracle_state.dtype == torch.float64
    assert not carried.paged, "the reference arm's result is dense-backed"

    # Step 2: continue on the CUDA backend, consuming that oracle-produced state --
    # which re-admits the dense contents into pages (docs/internals/state.md §5).
    with torch.no_grad():
        out, _state = layer(x1, state=carried)
    assert layer.last_execution_backend == "cuda"
    assert torch.isfinite(out).all()
    continued_state = carried.materialize()
    assert continued_state.dtype == torch.float32
    assert continued_state.shape == oracle_state.shape
    assert torch.isfinite(continued_state).all()


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_cuda_produced_state_feeds_a_grad_context_oracle_step_at_the_ops_level(decay):
    """The other half of item 3f's cross-feed: a CUDA-produced (fp32) carried state
    consumed by a grad-context oracle step. The LAYER refuses this categorically
    (consuming any carried state under grad raises, regardless of which backend
    produced it), so this calls `naive_rola` directly -- exactly the surface the
    reviewer exercised by hand in wiring-review.md §9 -- to gate the claim that both
    entry points accept either dtype, independent of the layer's stricter policy.
    """
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    from rola.ops import chunk as _chunk
    from rola.ops.naive import naive_rola
    from tests.conftest import RawRoutes

    torch.manual_seed(41)
    layer = _layer(decay=decay)
    layer.eval()
    producer = layer.routes.producers[0]
    #: `T` is a multiple of the operator's chunk (P67 D2-b: it was 48, which the
    #: chunk arm refuses); the topology is the file's, a built arm.
    T = 2 * _chunk.CHUNK_TOKENS
    x = torch.randn(1, T, _HIDDEN, device="cuda")

    with torch.no_grad():
        v = layer.v_proj(x).view(1, T, _H, _DV)
        ro = produce_routing(
            x, None, v, producer.route_W, producer.route_bias, layer.routes.side_gain.gain_W,
            layer.routes.side_gain.gain_bias, layer.topology, "global")
        routes = RawRoutes(layer.topology, "global", ro.read_levels, ro.write_levels,
                           ro.g_read, ro.g_write)
        refusal = _chunk.unsupported(routes, None)
        assert refusal is None, f"the fixture left the built chunk matrix: {refusal}"
        _out, plane = _chunk.run(routes, v, want_state=True)
        #: the kernel's atom-major plane IS the oracle's `[B, H, N, cols]` layout
        #: byte for byte (`rola.ops.chunk.state_plane`), so this is a reshape.
        cuda_state = plane.reshape(1, _H, layer.topology.N, _DV + 1)
    assert cuda_state.dtype == torch.float32

    # Consume that fp32 CUDA-produced state as the PRIOR state of a grad-context
    # oracle step -- the direction the layer forbids, exercised at the ops level.
    v2 = layer.v_proj(x).view(1, T, _H, _DV)
    with torch.enable_grad():
        ro2 = produce_routing(
            x, None, v2, producer.route_W, producer.route_bias, layer.routes.side_gain.gain_W,
            layer.routes.side_gain.gain_bias, layer.topology, "global")
        out2, oracle_state2 = naive_rola(
            v2, ro2.read_levels, ro2.write_levels, ro2.g_read, ro2.g_write,
            layer.topology, _decay_config(layer), "global",
            initial_state=cuda_state, output_final_state=True)
    assert torch.isfinite(out2).all()
    assert oracle_state2.dtype == torch.float64
    assert oracle_state2.shape == cuda_state.shape


# ---------------------------------------------------------------------------
# 4. No silent degradation in either direction
# ---------------------------------------------------------------------------


@_CUDA
def test_a_d_v_outside_the_built_matrix_dispatches_to_the_oracle_and_says_so():
    """Property 4, AS THE UNSUPPORTED-TOPOLOGY RETIREMENT LEFT IT (2026-08-14).

    `d_v = 48` is outside the chunk consumer's built matrix, which carries `d_v = 64`
    only. It used to reach the kernel and be refused there, on the reasoning that
    folding launchability into the dispatch would turn a loud unsupported
    configuration into a silent oracle-speed path. The retirement answers that
    objection differently rather than ignoring it: the path is not SILENT. The arm is
    the oracle, `no_kernel_arm` names the clause that refused, and the caller can ask
    before the call. It is a labelled research surface, the same treatment an
    unratified activation gets -- and it remains a real capability narrowing, which is
    why the label exists at all.

    The override is still loud, which is the other half: a caller who asks for the
    kernel BY NAME on this configuration gets a refusal, not the oracle.
    """
    torch.manual_seed(19)
    layer = _layer(decay="none", head_v_dim=48)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 64, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
    assert layer.execution_backend_counts == {"oracle": 1, "cuda": 0, "decode": 0}
    with torch.no_grad():
        routes = layer.routes(x)
        envelope = dispatch_for(routes, x)
    assert envelope.backend == "reference"
    assert "d_v=48" in envelope.no_kernel_arm
    v = torch.randn(1, 64, _H, 48, device="cuda")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="d_v=48"):
        rola.rola_op(routes, v, _arm="kernel")


# ---------------------------------------------------------------------------
# 5. Fix pass item 1 (wiring-review.md §2, BLOCKER) -- the decay dials are a
#    grad-requiring operation input to the predicate
# ---------------------------------------------------------------------------


def _freeze_all_but(layer: RoLA, *keep_prefixes: str) -> None:
    for name, param in layer.named_parameters():
        if not any(name.startswith(prefix) for prefix in keep_prefixes):
            param.requires_grad_(False)


@_CUDA
def test_the_decay_dials_are_a_grad_requiring_operation_input():
    """Property, per the fix pass: a context where ONLY the decay source's `theta`
    requires grad (`v`, `g_read`, `g_write` and the read/write allocations do not,
    because `v_proj`/the producer's `route_W`/the gateway's `side_gain.gain_W`/`.gain_bias` are
    frozen and `x` does not require grad) IS a grad-requiring operation input,
    because the dials are handed straight to the operation as `decay=`.

    `o_proj` is left trainable (matching the reviewer's fixture, wiring-review.md
    §2): its weight requiring grad is what lets `.backward()` complete regardless of
    whether the dispatch is correct, exactly as measured -- `o_proj` sits downstream
    of the consumer and re-attaches a graph there whether or not the upstream
    graph is severed. So the assertion that binds is `decay.theta.grad`, not
    whether backward raised.

    Old `layer.decay_theta` (the layer's own `ParameterList`) -> `layer.decay.theta`
    (`rola.LearnedDecay`'s own): the dials live on the decay SOURCE now, not on the
    layer that consumes it.
    """
    torch.manual_seed(29)
    layer = _layer(norm="global", decay="leaf_mass")
    layer.eval()
    _freeze_all_but(layer, "decay.theta", "o_proj")
    x = torch.randn(1, 64, _HIDDEN, device="cuda")  # does not require grad
    assert torch.is_grad_enabled()

    out, _ = layer(x)
    out.square().mean().backward()

    assert layer.last_execution_backend == "oracle"
    assert layer.execution_backend_counts == {"oracle": 1, "cuda": 0, "decode": 0}, (
        "a context where only the decay dials require grad reached the forward-only "
        "CUDA consumer")
    for theta in layer.decay.theta:
        assert theta.grad is not None, "the decay dials received no gradient"
        assert torch.isfinite(theta.grad).all(), "the decay dials received a non-finite gradient"
        assert float(theta.grad.abs().max()) > 0, "the decay dials received an all-zero gradient"


@_CUDA
def test_forcing_cuda_in_the_decay_only_grad_context_severs_the_decay_gradient(monkeypatch):
    """Mutation test mirroring the measurement in wiring-review.md §2: this is what
    WOULD happen on the exact fixture above if the dials were not in the predicate.
    Forcing the dispatch to `'cuda'` severs the decay dials' graph (`grad is None`)
    while `o_proj.weight` survives -- it is applied AFTER the severance and
    re-attaches a graph downstream of the cut, so training would *appear* to work.
    This exists to prove the gate can actually detect the bug class the test above
    closes; the unmutated dispatch (proven by the test above) is what avoids it in
    production.

    **MIGRATED, with an extra layer of forcing.** `rola_op` (the op the layer's
    `else` branch calls for both remaining arms) now RE-DERIVES its own backend
    from operand gradient-reachability (`rola.interface._backend_for`) whenever the
    layer passes `_arm=None` -- i.e. whenever the layer itself did not choose
    `'oracle'`. That is defense in depth the pre-break code did not have: mutating
    only `RoLA.select_execution_backend` (as the pre-break mutation did) no longer
    reaches the consumer here, because `rola_op`'s own check independently sees the
    decay dials requiring grad and still routes to the oracle -- measured: the old
    mutation alone left `theta.grad` NONZERO, i.e. the graph did NOT sever. Forcing
    the CUDA arm all the way through now requires mutating `rola.interface.
    select_backend` as well, which is what this does.
    """
    pytest.skip("premise retired: the kernel arm no longer runs decay at all (decay routes to the oracle); the severed-decay-gradient scenario cannot arise")
    torch.manual_seed(29)
    layer = _layer(norm="global", decay="leaf_mass")
    layer.eval()
    _freeze_all_but(layer, "decay.theta", "o_proj")
    x = torch.randn(1, 64, _HIDDEN, device="cuda")

    import rola.interface as interface_module

    monkeypatch.setattr(RoLA, "select_execution_backend", lambda *args, **kwargs: "cuda")
    monkeypatch.setattr(interface_module, "select_backend", lambda *args, **kwargs: "cuda")
    out, _ = layer(x)
    out.square().mean().backward()

    assert layer.last_execution_backend == "cuda"
    for theta in layer.decay.theta:
        assert theta.grad is None, "the mutation test's premise did not hold: the decay dials got a grad"
    assert layer.o_proj.weight.grad is not None, (
        "the mutation test's premise did not hold: o_proj did not survive the severance")


# ---------------------------------------------------------------------------
# 6. Fix pass item 2 (wiring-review.md §3, REQUIRED-BEFORE-MERGE) -- training mode
#    forces the oracle, even under a no_grad reentrant-checkpoint first pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_reentrant", [True, False])
@_CUDA
def test_checkpointing_in_training_mode_never_reaches_the_cuda_consumer(use_reentrant):
    """Reentrant checkpointing's first pass runs the wrapped module under
    `torch.no_grad()` with `module.training` still `True`; before the fix pass this
    selected `'cuda'` for that pass while backward recomputed the identical step on
    the oracle (measured, wiring-review.md §3: after fwd `{'oracle': 0, 'cuda': 1}`,
    after bwd `{'oracle': 1, 'cuda': 1}`) -- two different implementations agreeing
    only to `RTOL`, not one graph. `self.training` now gates the CUDA path
    independently of grad-enablement, so BOTH the forward pass and the recompute
    take the oracle, under both checkpoint modes, and gradients reach every
    parameter.
    """
    torch.manual_seed(31)
    layer = _layer(norm="global", decay="none")
    assert layer.training
    x = torch.randn(1, 96, _HIDDEN, device="cuda", requires_grad=True)

    def run(inp):
        out, _ = layer(inp)
        return out

    out = torch_checkpoint.checkpoint(run, x, use_reentrant=use_reentrant)
    out.square().mean().backward()

    assert layer.execution_backend_counts["cuda"] == 0, (
        "a training-mode forward under checkpointing reached the CUDA consumer")
    assert layer.execution_backend_counts["oracle"] >= 1
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().max()) > 0
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} received a non-finite gradient"
        assert float(param.grad.abs().max()) > 0, f"{name} received an all-zero gradient"


def test_cpu_tensors_take_the_oracle_because_no_cpu_lowering_exists():
    """Device dispatch, not degradation: there is no CPU lowering of the consumer to
    decline. Runs without CUDA so it is part of the CPU gate too."""
    torch.manual_seed(23)
    layer = _layer(norm="global", decay="leaf_mass", device="cpu")
    x = torch.randn(1, 32, _HIDDEN)
    with torch.no_grad():
        layer(x)
    assert layer.execution_backend_counts == {"oracle": 1, "cuda": 0, "decode": 0}


#: RETIRED at P67 D2 (ruling R-2): the two rows whose subject was the layer's
#: CONSTRUCTION-time `BC` ladder -- the largest-feasible default and the
#: inadmissible-pin refusal. `RoLA.feasible_BC`/`state_block` and the `SUPPORTED_BC`
#: ladder they read are the tiled consumer's, deleted in D2-c. The chunk arm's `BC`
#: belongs to the built arm the manifest carries for a topology and is not
#: host-selectable, so there is no ladder to take a largest member of, no feasible
#: SET for a pin to shrink, and no admissibility question a constructor could answer.
#: The claim that survives is that a pin the kernel arm cannot honor is refused BY
#: NAME rather than silently ignored, and it is held one row below.


# ---------------------------------------------------------------------------
# H2 (coverage-closure): the dispatch on a SECOND built geometry, deep rather than
# shallow. Every cell above runs one two-level arm, so a dispatch that happened to
# be right only at `D = 2` would pass the whole file
# (the capacity-fair comparison rule: small-regime-only gates invert).
# ---------------------------------------------------------------------------

_DEEP_BRANCHES = (8, 8, 8, 8)  # the manifest's D=4, B=8 arm; N = 4096


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_the_no_grad_dispatch_holds_on_a_deeper_built_arm(decay):
    """`test_a_no_grad_cuda_context_takes_the_kernel_arm`'s property, at `D = 4`."""
    if decay == "leaf_mass":
        pytest.skip("decay routes to the oracle now (retired from the operator); covered by the decay-routing layer test")
    torch.manual_seed(7)
    layer = _layer(decay=decay, branches=_DEEP_BRANCHES)
    layer.eval()
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
        envelope = dispatch_for(layer.routes(x), x)
    assert envelope.no_kernel_arm is None, (
        "fixture rot: this topology must be a BUILT chunk arm")
    assert layer.execution_backend_counts == {"oracle": 0, "cuda": 1, "decode": 0}


@pytest.mark.cuda
@_CUDA
def test_an_uninstantiated_but_aligned_state_block_is_refused_host_side():
    """H12 (coverage-closure): `state_block = 32`, a pin the kernel arm cannot honor.

    Construction accepts any int, because the layer holds no `BC` ladder to check one
    against. The kernel arm is
    the chunk consumer, whose `BC` is a slot tile inside a BUILT arm and is fixed by
    the manifest rather than chosen per launch, so the pin is over an axis this arm
    does not have. It must be REFUSED by a HOST-side check that names the contract --
    never accepted and ignored, and never a device-side assert, which would poison the
    context for every subsequent test in the process (the fleet-freeze failure mode).
    The context-health probe after the raise is the second half of the claim.

    What is pinned here is that behaviour -- host-side, specific, and
    context-preserving. A knob that silently does nothing is how a caller comes to
    believe a launch was shaped a way it never was, which is the failure this closes.
    """
    layer = _layer(norm="global", decay="none")
    routes_32 = build_routes(hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_BRANCHES)
    layer_32 = RoLA(
        _HIDDEN, _H, routes_32, expert=PlanOverrides(state_block=32), layer_idx=0,
    ).to("cuda", torch.float32).eval()
    x = torch.randn(1, 32, _HIDDEN, device="cuda")
    with pytest.raises(NotImplementedError, match="state_block"), torch.no_grad():
        layer_32(x)
    # The refusal must be host-side: the CUDA context is still usable.
    assert float(torch.ones(4, device="cuda").sum()) == 4.0
    with torch.no_grad():
        layer.eval()
        layer(x)
    assert layer.execution_backend_counts["cuda"] == 1


# ---------------------------------------------------------------------------
# H8 (coverage-closure): bf16 x paging. Every paging gate pins tf32x3 (so the
# derived fp32 envelope stays the one that was derived), and every bf16 gate runs
# unpaged. `mma_precision` is a dead axis now (bf16 is the only value
# `PlanOverrides` accepts, P26; it never even reaches `Plan` any more), so every
# gate in this file is already bf16 and this row's remaining, still-live claim is
# paging invisibility itself. The state is atomic-free and deterministic, so
# paged-vs-unpaged state stays torch.equal; y is held to the oracle at BF16_RTOL,
# the bf16 family bound.
# ---------------------------------------------------------------------------


@_CUDA
def test_the_bf16_arm_is_paging_invisible_at_its_own_tolerance():
    """REPOINTED (P67 S3b): the backing is the STATE's, so the two arms are two states.

    Paging stopped being a layer configuration -- it is chosen when a state binds, and
    `PlanOverrides(paging=False)` is the surviving pin -- but the claim is the one it
    always was: the bf16 arm computes the same carried state through either backing,
    `torch.equal`, because the state path has no atomic and no precision excuse.
    """
    import rola

    torch.manual_seed(23)
    #: UNIFORM widths at `d_v = 64`: the paged backing lives on the CHUNK arm, and the
    #: legacy tiled consumer (which `_BRANCHES = (8, 16)` falls to) takes raw tensors
    #: and hands back dense contents whatever the state prefers. A cell that wants two
    #: backings has to run on the arm that has two.
    kw = dict(hidden_size=_HIDDEN, num_heads=_H, d_v=64, widths=(16, 16),
              routing=dense_routing(), normalization="global", decay=None, layer_idx=0)
    layer = build_layer(**kw).to("cuda", torch.float32)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")

    x_grad = x.clone().requires_grad_(True)
    oracle_out, _ = layer(x_grad)
    assert layer.last_execution_backend == "oracle"

    layer.eval()
    dense_pin = build_layer(expert=PlanOverrides(paging=False), **kw).to(
        "cuda", torch.float32).eval()
    dense_pin.load_state_dict(layer.state_dict())
    with torch.no_grad():
        state_u, state_p = rola.state(), rola.state()
        y_u, state_u = dense_pin(x, state=state_u)
        y_p, state_p = layer(x, state=state_p)

    assert layer.last_execution_backend == "cuda"
    assert not state_u.paged and state_p.paged, (
        "both arms took the same backing; the cell is vacuous")
    assert torch.equal(state_u.materialize(), state_p.materialize()), (
        "bf16 x paging moved the carried state; the state path has no atomic and "
        "no precision excuse")
    ref = oracle_out.detach()
    scale = float(ref.abs().max())
    assert float((y_p - ref).abs().max()) / scale < BF16_RTOL, (
        "the paged bf16 output left the bf16 arm's own tier bound vs the oracle")
    assert float((y_u - ref).abs().max()) / scale < BF16_RTOL
