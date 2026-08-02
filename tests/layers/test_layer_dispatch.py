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
"""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

from rola.layers.rola import LevelRoutingConfig, RoLA

#: The consumer suite's own fp32-against-fp64 tolerance
#: (`tests/ops/test_rola_v3_consumer.py:RTOL`), quoted at its value rather than
#: imported so that loosening it here would be a visible edit to this file. It is
#: applied the same way that suite applies it: max absolute deviation divided by
#: the reference's own scale. Whole-layer outputs sit behind `o_proj`, a linear map
#: applied identically to both backends, so the relative comparison basis carries
#: through unchanged.
RTOL = 2e-5

#: The bf16 MMA arm's own tolerance (`tests/ops/test_rola_v3_consumer.py:BF16_RTOL`),
#: quoted at its value on the same terms as `RTOL` above. It is a FORMAT tolerance --
#: bf16 A/B operands carry 8 mantissa bits and the readout contracts over `BC` of them
#: -- and it is the tolerance that suite's 43 bf16 gates already run at, not a new one
#: invented for the layer. It applies ONLY where `mma_precision="bf16"` is asked for.
BF16_RTOL = 1e-2

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the V3 consumer is a CUDA kernel")

#: `branches=[8, 16]`, `d_v=64`, `H=8` -- the Stage 0 matched cell, and the
#: `(d_v, BC) = (64, 16)` instantiation the kernel actually compiles. `BC` is left
#: at its default, which resolves to the innermost width 16.
_BRANCHES = [8, 16]
_H, _DV = 8, 64
_HIDDEN = _H * _DV


def _layer(*, norm: str, decay: str, router: str = "dense", device: str = "cuda",
           logit_gain: float = 1.0, head_v_dim: int = _DV, branches=None,
           mma_precision: str = "tf32x3") -> RoLA:
    """The dispatch fixture, pinned to the EXACT MMA arm by default.

    The layer's own default is `bf16` (the production numeric format, `harvest.md`
    §30). Every gate in this file compares one backend against another or against
    the fp64 oracle at the consumer suite's `RTOL`, and bf16's ~1e-2 per-product
    error would swamp exactly the structural disagreement those gates exist to
    detect -- the split `MMA_PRECISIONS` and `v3_readout_mma.cuh` both document. So
    the STRUCTURAL gates pin `tf32x3` here and the default itself is asserted
    separately, rather than the tolerance being widened.
    """
    branches = _BRANCHES if branches is None else branches
    layer = RoLA(
        hidden_size=_HIDDEN, num_heads=_H, head_v_dim=head_v_dim,
        levels_per_head=len(branches), states_per_level=branches,
        decay=decay, state_norm=norm, mma_precision=mma_precision,
        level_routing_configs=LevelRoutingConfig(read=router, write=router, tied=False, alpha=1.5),
        layer_idx=0,
    ).to(device, torch.float32)
    if logit_gain != 1.0:
        # Sharpens an UNTRAINED exact-entmax router into a real sparse support. The
        # support is genuine (exact alpha=1.5, the production path -- never an
        # epsilon floor or a top-k stand-in); only its LEVEL is a synthetic stand-in
        # for training, and that is what makes this a usable sparse fixture rather
        # than a second dense one.
        with torch.no_grad():
            layer.route_W.mul_(logit_gain)
    return layer


def _sparsity(layer: RoLA, x: torch.Tensor) -> float:
    """Realized exact-zero fraction of the router's write allocations, so a test
    that claims to exercise a sparse fixture can prove the fixture is sparse."""
    from rola.routing.producer import produce_routing
    with torch.no_grad():
        out = produce_routing(
            x, None, torch.zeros(x.shape[0], x.shape[1], layer.num_heads, layer.head_v_dim,
                                 device=x.device, dtype=x.dtype),
            layer.route_W, layer.route_bias_param, layer.gain_W, layer.gain_bias,
            layer.topology, layer.state_norm, budget=layer.budget, r=None,
            budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W, temp_b=layer.temp_b,
            usage_bias_W=layer.usage_bias_W, usage_bias_b=layer.usage_bias_b)
    zeros = sum(float((t == 0).float().mean()) * t.shape[-1] for t in out.write_levels)
    return zeros / sum(t.shape[-1] for t in out.write_levels)


# ---------------------------------------------------------------------------
# 1. The grad context runs the oracle, and the graph survives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_a_grad_context_takes_the_oracle_and_every_parameter_gets_a_live_gradient(
        norm, decay, router, gain):
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
    layer = _layer(norm=norm, decay=decay, router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda", requires_grad=True)

    out, _, _ = layer(x)
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
    out, _, _ = layer(x)
    assert layer.last_execution_backend == "cuda"
    assert out.grad_fn is None


# ---------------------------------------------------------------------------
# 2. The no-grad context runs the consumer, on a SELECTED plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_no_grad_cuda_context_takes_the_plan_selected_consumer(norm, decay):
    """Property 2, asserted on the counter and on the stamped plan -- not on timing.

    The plan the layer ran is recorded, so this also pins that the layer went
    through `select_plan` (the axes are a QUERY of the realized routing) rather
    than through a hand-written plan constant: `last_plan_axes` carries the measured
    `TileUnionStats` the decision was priced from.
    """
    torch.manual_seed(7)
    layer = _layer(norm=norm, decay=decay)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
    assert layer.execution_backend_counts == {"oracle": 0, "cuda": 1, "decode": 0}
    plan, axes = layer.last_selected_plan, layer.last_plan_axes
    assert plan is not None and axes is not None
    assert plan.BC == layer.state_block == _BRANCHES[-1]
    assert plan.decay is (decay == "leaf_mass")
    assert plan.normalization == norm
    assert axes.stats.BT == plan.BT and axes.stats.BC == plan.BC


@_CUDA
def test_the_selected_plan_is_re_measured_every_step_not_cached():
    """The coordinator's per-step ruling, gated rather than commented.

    Moving the routing distribution (here by sharpening the router between two
    no-grad forwards on the same shape) must move the measured statistics the plan
    was priced from. A per-shape cache would return the first step's stats for the
    second step's routing, which is precisely the staleness the ruling rejects.
    """
    torch.manual_seed(11)
    layer = _layer(norm="raw", decay="none", router="sparse")
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 128, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
        dense_stats = layer.last_plan_axes.stats
        layer.route_W.mul_(16.0)
        layer(x)
        sharp_stats = layer.last_plan_axes.stats
    assert layer.execution_backend_counts == {"oracle": 0, "cuda": 2, "decode": 0}
    assert sharp_stats.mean_K < dense_stats.mean_K, (
        f"the per-step selection did not re-measure: K went {dense_stats.mean_K} -> "
        f"{sharp_stats.mean_K} after a 16x router sharpening")


# ---------------------------------------------------------------------------
# 3. The two backends agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_the_two_backends_agree_on_the_same_inputs(norm, decay, router, gain):
    """Property 3, on ONE layer instance and ONE input, so the only difference
    between the two runs is which backend executed.

    The grad-context path is forced by `requires_grad`, the no-grad path by
    `torch.no_grad()`; both consume the identical parameters and the identical `x`.
    Comparison basis and tolerance are the consumer suite's, unmodified.
    """
    torch.manual_seed(13)
    layer = _layer(norm=norm, decay=decay, router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    if router == "sparse":
        assert _sparsity(layer, x) > 0.2, "the sparse fixture is vacuously dense"

    x_grad = x.clone().requires_grad_(True)
    oracle_out, _, _ = layer(x_grad)
    assert layer.last_execution_backend == "oracle"
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    with torch.no_grad():
        cuda_out, _, _ = layer(x)
    assert layer.last_execution_backend == "cuda"

    reference = oracle_out.detach()
    scale = float(reference.abs().max())
    deviation = float((cuda_out - reference).abs().max()) / scale
    assert deviation < RTOL, (
        f"backends disagree at {deviation:.3e} (RTOL={RTOL:.1e}); norm={norm} decay={decay} "
        f"router={router}")


@_CUDA
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_the_shipped_bf16_default_layer_agrees_with_the_oracle(norm, decay):
    """The DEFAULT arm, numerically, through `RoLA.forward` -- not a plan field.

    Every other numerical gate in this file and in `test_rola_v3_layer_paging.py`
    pins `tf32x3`, because their comparison bases (the consumer suite's `RTOL`, and
    paging's derived fp32 reassociation envelope) are bases bf16 rounding would
    swamp. That pinning is correct and stays, but on its own it would leave the arm
    that actually SHIPS with no layer-level number checked anywhere -- the default
    gate next to this one asserts `last_selected_plan.mma_precision`, which a
    bf16-only regression in the readout would sail straight past.

    So this is the same property as `test_the_two_backends_agree_on_the_same_inputs`
    -- one layer instance, one input, the oracle backend against the CUDA backend --
    run on `mma_precision="bf16"` at that arm's OWN declared `BF16_RTOL`. Both decay
    arms run for coverage, but only `decay=None` DISCRIMINATES here and the
    difference is measured rather than assumed: at this cell (`branches=[8,16]`,
    `BC=16`, `L=96`, `realized_leaf` both arms) the deviation is 2.4e-7 on tf32x3 and
    4.5e-4 on bf16 with `decay=None`, while under LeafMassDecay both formats land on
    the same 6.4e-7 -- the element setting the max there is fixed by the
    token-sequential intra walk, which the MMA format does not touch. So the
    falsifiability assertion below is made on the leg that carries it, and the decay
    leg is a coverage gate; the axis is separately proved live under decay at the
    CONSUMER level (`test_rola_v3_consumer.py::test_the_compiled_out_gather_is_exact_
    under_decay` and `..._mma_precision_is_a_live_axis_on_both_decay_arms`).
    """
    torch.manual_seed(29)
    layer = _layer(norm=norm, decay=decay, mma_precision="bf16")
    assert layer.mma_precision == "bf16"
    x = torch.randn(1, 96, _HIDDEN, device="cuda")

    x_grad = x.clone().requires_grad_(True)
    oracle_out, _, _ = layer(x_grad)
    assert layer.last_execution_backend == "oracle"
    layer.eval()
    with torch.no_grad():
        cuda_out, _, _ = layer(x)
    assert layer.last_execution_backend == "cuda"
    assert layer.last_selected_plan.mma_precision == "bf16", (
        "the layer did not forward its own bf16 default into the plan, so this gate "
        "would be silently re-measuring the tf32x3 arm")

    reference = oracle_out.detach()
    deviation = float((cuda_out - reference).abs().max()) / float(reference.abs().max())
    assert deviation < BF16_RTOL, (
        f"the shipped bf16 layer disagrees with the oracle at {deviation:.3e} "
        f"(BF16_RTOL={BF16_RTOL:.1e}); norm={norm} decay={decay}")
    if decay == "none":
        # The arm is REAL, not tf32x3 under another name: on this leg bf16's format
        # error is three orders above the exact arm's, so a deviation at `RTOL` would
        # mean the pin leaked and this gate was re-measuring tf32x3.
        assert deviation > RTOL, (
            f"deviation {deviation:.3e} is at the tf32x3 arm's tolerance -- this gate "
            "is not exercising bf16 and would pass with the format flipped back")


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_the_two_backends_agree_on_the_carried_state_they_hand_back(norm, decay):
    """The cache surface, which is the one place the two backends' outputs are not
    just compared but INTERCHANGED.

    Both write `[B, H, N, d_v + (norm == 'global')]` with the same meaning; only the
    accumulation dtype differs (fp64 oracle, fp32 consumer) and both entry points
    accept either. A continuation may therefore cross backends, which is what this
    asserts: same shape, and the same values at `RTOL`.

    Fix pass item 3f (wiring-review.md §9 NOTE): this gate was `decay='none'` only;
    it is now parametrized over both decay arms, closing the narrower-than-the-claim
    gap the review flagged (the `leaf_mass` arm's carried state was never exercised
    here, only asserted empirically by the reviewer).
    """
    torch.manual_seed(17)
    layer = _layer(norm=norm, decay=decay)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode;
    # the oracle leg below still takes 'oracle' because `x` does not require grad but
    # the (unfrozen) projection parameters do, so `v` requires grad under enable_grad.
    from rola.state import RoLAState
    x = torch.randn(1, 64, _HIDDEN, device="cuda")

    states = {}
    for tag, context in (("oracle", torch.enable_grad()), ("cuda", torch.no_grad())):
        cache = RoLAState()
        with context:
            layer(x, past_key_values=cache, use_cache=True)
        assert layer.last_execution_backend == tag
        states[tag] = cache[layer.layer_idx].recurrent_state

    assert states["oracle"].shape == states["cuda"].shape
    scale = float(states["oracle"].abs().max())
    deviation = float((states["cuda"].double() - states["oracle"].double()).abs().max()) / scale
    assert deviation < RTOL, f"carried states disagree at {deviation:.3e} (RTOL={RTOL:.1e})"


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_carried_state_actually_crosses_backends_on_a_real_continuation(norm, decay):
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
    torch.manual_seed(37)
    layer = _layer(norm=norm, decay=decay)
    layer.eval()
    from rola.state import RoLAState
    x0 = torch.randn(1, 48, _HIDDEN, device="cuda")
    x1 = torch.randn(1, 32, _HIDDEN, device="cuda")

    # Step 1: produce a carried state on the oracle (grad context).
    cache = RoLAState()
    with torch.enable_grad():
        layer(x0, past_key_values=cache, use_cache=True)
    assert layer.last_execution_backend == "oracle"
    oracle_state = cache[layer.layer_idx].recurrent_state
    assert oracle_state.dtype == torch.float64

    # Step 2: continue on the CUDA backend, consuming that oracle-produced state.
    with torch.no_grad():
        out, _, _ = layer(x1, past_key_values=cache, use_cache=True)
    assert layer.last_execution_backend == "cuda"
    assert torch.isfinite(out).all()
    continued_state = cache[layer.layer_idx].recurrent_state
    assert continued_state.dtype == torch.float32
    assert continued_state.shape == oracle_state.shape
    assert torch.isfinite(continued_state).all()


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@_CUDA
def test_a_cuda_produced_state_feeds_a_grad_context_oracle_step_at_the_ops_level(norm, decay):
    """The other half of item 3f's cross-feed: a CUDA-produced (fp32) carried state
    consumed by a grad-context oracle step. The LAYER refuses this categorically
    (consuming any carried state under grad raises, regardless of which backend
    produced it), so this calls `naive_rola` directly -- exactly the surface the
    reviewer exercised by hand in wiring-review.md §9 -- to gate the claim that both
    entry points accept either dtype, independent of the layer's stricter policy.
    """
    from rola.ops.consumer import _consumer_forward
    from rola.ops.naive import naive_rola
    from rola.planner.schedule import select_plan
    from rola.routing.producer import produce_routing

    torch.manual_seed(41)
    layer = _layer(norm=norm, decay=decay)
    layer.eval()
    x = torch.randn(1, 48, _HIDDEN, device="cuda")

    with torch.no_grad():
        v = layer.v_proj(x).view(1, 48, _H, _DV)
        ro = produce_routing(
            x, None, v, layer.route_W, layer.route_bias_param, layer.gain_W,
            layer.gain_bias, layer.topology, norm, budget=layer.budget, r=None,
            budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W, temp_b=layer.temp_b,
            usage_bias_W=layer.usage_bias_W, usage_bias_b=layer.usage_bias_b)
        decay_config = layer._decay_config()
        plan, _axes = select_plan(
            ro.read_levels, ro.write_levels, layer.topology, layer.state_block,
            normalization=norm, decay=decay_config is not None)
        _out, cuda_state, _sched = _consumer_forward(
            v, ro.read_levels, ro.write_levels, ro.g_read, ro.g_write, plan,
            initial_state=None, decay=decay_config)
    assert cuda_state.dtype == torch.float32

    # Consume that fp32 CUDA-produced state as the PRIOR state of a grad-context
    # oracle step -- the direction the layer forbids, exercised at the ops level.
    v2 = layer.v_proj(x).view(1, 48, _H, _DV)
    with torch.enable_grad():
        ro2 = produce_routing(
            x, None, v2, layer.route_W, layer.route_bias_param, layer.gain_W,
            layer.gain_bias, layer.topology, norm, budget=layer.budget, r=None,
            budget_embed_W=layer.budget_embed_W, temp_W=layer.temp_W, temp_b=layer.temp_b,
            usage_bias_W=layer.usage_bias_W, usage_bias_b=layer.usage_bias_b)
        out2, oracle_state2 = naive_rola(
            v2, ro2.read_levels, ro2.write_levels, ro2.g_read, ro2.g_write,
            layer.topology, layer._decay_config(), norm,
            initial_state=cuda_state, output_final_state=True)
    assert torch.isfinite(out2).all()
    assert oracle_state2.dtype == torch.float64
    assert oracle_state2.shape == cuda_state.shape


# ---------------------------------------------------------------------------
# 4. No silent degradation in either direction
# ---------------------------------------------------------------------------


@_CUDA
def test_a_cuda_context_the_consumer_cannot_launch_raises_rather_than_using_the_oracle():
    """Property 4. `d_v = 48` is outside the kernel's instantiated `{32, 64}` set.

    The dispatch deliberately does NOT test for this. Folding launchability into
    `select_execution_backend` would convert a loud unsupported configuration into a
    silent oracle-speed path -- the pattern spec §7.4.1 forbids -- so the
    configuration must reach the kernel and be rejected there.
    """
    torch.manual_seed(19)
    layer = _layer(norm="raw", decay="none", head_v_dim=48)
    layer.eval()  # fix pass item 2: the CUDA path additionally requires eval mode
    x = torch.randn(1, 64, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(RuntimeError, match="d_v"):
        layer(x)
    assert layer.execution_backend_counts["oracle"] == 0, (
        "an un-instantiated d_v silently fell back to the oracle instead of raising")


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
    """Property, per the fix pass: a context where ONLY `decay_theta` requires grad
    (`v`, `g_read`, `g_write` and the read/write allocations do not, because
    `v_proj`/`route_W`/`gain_W`/`gain_bias` are frozen and `x` does not require
    grad) IS a grad-requiring operation input, because the dials are handed straight
    to `_consumer_forward(..., decay=decay_config)`.

    `o_proj` is left trainable (matching the reviewer's fixture, wiring-review.md
    §2): its weight requiring grad is what lets `.backward()` complete regardless of
    whether the dispatch is correct, exactly as measured -- `o_proj` sits downstream
    of the consumer and re-attaches a graph there whether or not the upstream
    graph is severed. So the assertion that binds is `decay_theta.grad`, not
    whether backward raised.
    """
    torch.manual_seed(29)
    layer = _layer(norm="global", decay="leaf_mass")
    layer.eval()
    _freeze_all_but(layer, "decay_theta", "o_proj")
    x = torch.randn(1, 64, _HIDDEN, device="cuda")  # does not require grad
    assert torch.is_grad_enabled()

    out, _, _ = layer(x)
    out.square().mean().backward()

    assert layer.last_execution_backend == "oracle"
    assert layer.execution_backend_counts == {"oracle": 1, "cuda": 0, "decode": 0}, (
        "a context where only decay_theta requires grad reached the forward-only "
        "CUDA consumer")
    for theta in layer.decay_theta:
        assert theta.grad is not None, "decay_theta received no gradient"
        assert torch.isfinite(theta.grad).all(), "decay_theta received a non-finite gradient"
        assert float(theta.grad.abs().max()) > 0, "decay_theta received an all-zero gradient"


@_CUDA
def test_forcing_cuda_in_the_decay_only_grad_context_severs_the_decay_gradient(monkeypatch):
    """Mutation test mirroring the measurement in wiring-review.md §2: this is what
    WOULD happen on the exact fixture above if the dials were not in the predicate.
    Forcing `select_execution_backend` to return `'cuda'` severs `decay_theta`'s
    graph (`grad is None`) while `o_proj.weight` survives -- it is applied AFTER the
    severance and re-attaches a graph downstream of the cut, so training would
    *appear* to work. This exists to prove the gate can actually detect the bug
    class the test above closes; the unmutated dispatch (proven by the test above)
    is what avoids it in production.
    """
    torch.manual_seed(29)
    layer = _layer(norm="global", decay="leaf_mass")
    layer.eval()
    _freeze_all_but(layer, "decay_theta", "o_proj")
    x = torch.randn(1, 64, _HIDDEN, device="cuda")

    monkeypatch.setattr(RoLA, "select_execution_backend", lambda *args, **kwargs: "cuda")
    out, _, _ = layer(x)
    out.square().mean().backward()

    assert layer.last_execution_backend == "cuda"
    for theta in layer.decay_theta:
        assert theta.grad is None, "the mutation test's premise did not hold: decay_theta got a grad"
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
        out, _, _ = layer(inp)
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


def test_an_inadmissible_state_block_is_rejected_at_construction():
    """`BC` is contract, and `owner_geometry` is its authority. 24 divides
    `N = 128`? It does not -- and even a divisor that does not name a `BC`-aligned
    rectangle must be refused at construction rather than at the first CUDA forward,
    so the error names the configuration and not a launch."""
    with pytest.raises(ValueError):
        RoLA(hidden_size=_HIDDEN, num_heads=_H, head_v_dim=_DV, levels_per_head=2,
             states_per_level=_BRANCHES, state_block=24)
    with pytest.raises(ValueError):
        # 32 divides N=128 but implies c = 32/16 = 2 digits of level 0 (width 8) --
        # aligned -- while 12 divides nothing: the second case is the alignment one.
        RoLA(hidden_size=_HIDDEN, num_heads=_H, head_v_dim=_DV, levels_per_head=2,
             states_per_level=_BRANCHES, state_block=12)


def test_the_default_state_block_is_the_innermost_width_on_every_topology():
    """The documented structural default, over several mixed radices: it is always
    admissible because the owner rectangle is exactly one full innermost level."""
    for branches in ([8, 16], [4, 4, 8], [16], [2, 3, 16]):
        layer = RoLA(hidden_size=32, num_heads=2, head_v_dim=8,
                     levels_per_head=len(branches), states_per_level=branches)
        assert layer.state_block == branches[-1]
        assert layer.last_execution_backend is None


def test_the_layer_selects_the_production_bf16_readout_by_default():
    """The LAYER's default is the production format, and it reaches the plan.

    `_layer` pins `tf32x3` so the structural gates in this file measure structure
    rather than bf16 rounding, which means nothing else here would notice if the
    layer stopped forwarding its own default. This asserts both halves: the
    constructor default is `bf16`, and the plan the forward selects carries it.
    """
    layer = _layer(norm="global", decay="none", mma_precision="bf16")
    assert layer.mma_precision == "bf16"
    layer.eval()
    with torch.no_grad():
        layer(torch.randn(1, 96, _HIDDEN, device="cuda"))
    assert layer.last_selected_plan.mma_precision == "bf16"
    assert RoLA(hidden_size=_HIDDEN, num_heads=_H, head_v_dim=_DV,
                levels_per_head=len(_BRANCHES), states_per_level=_BRANCHES,
                layer_idx=0).mma_precision == "bf16"
