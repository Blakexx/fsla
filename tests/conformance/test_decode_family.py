"""CONFORMANCE — the decode family (ratified design, 2026-08-05).

Decode joins the family scheme as its own fixture family: the decomposed
oracle already covers it, because the textbook RECURRENT form IS decode — the
same glue, stepped. Four decode-specific axes ride on top of the shared ones,
one test section each:

1. **PREFILL -> DECODE HANDOFF** — chunked-path state consumed by the
   recurrent path, an implementation seam, exercised here across the regime
   arms (both norms x both decay arms, sparse routing, a RAGGED prefill) on
   top of tier 1's G7 single-arm seam gate.
2. **HORIZON** — multi-step conformance threading state through k steps
   against the stepwise fp64 oracle WITH CHECKPOINTS ALONG THE HORIZON. This
   closes the per-call blindness to cross-call compounding named in the
   bf16-state kill: a per-step gate cannot see an error that accumulates, a
   checkpointed horizon can.
3. **FROZEN-PLAN VALIDITY UNDER DRIFT** — the carrier's declarations are
   frozen at the boundary while the realized routing drifts; declarations are
   performance-only, so the answers must not move, and the carrier's
   certificates (arm mismatches) must refuse rather than degrade.
4. **DISPATCH-BOUNDARY CROSSINGS, ONE TEST, FWD + BWD** — the layer-dispatch
   autograd-gap lesson: op-level suites missed the layer's mode dispatch, so
   the walk that crosses oracle -> tiled -> decode -> oracle lives in ONE test
   with the gradient asserted alive on every grad-requiring leg.

The cold-read corner is NATIVE here (a decode step's read support routinely
lands on leaves the prefill never wrote) and section 1's sparse fixtures
realize it, asserted, rather than staging it.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.decode import decode_forward, freeze_decode_config
from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan
from rola.routing.types import LeafMassDecay
from tests.conformance import generators as g
from tests.tier1_semantics.test_decode_vs_oracle import DECODE_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

_WIDTHS, _DV, _B, _H = (8, 16), 64, 2, 2


def _draw(gen, device, T, p_read, p_write, *, widths=_WIDTHS, H=_H, B=_B, dv=_DV,
          norm="global"):
    read = tuple(g._simplex((B, T, H, w), p_read, gen, device) for w in widths)
    write = tuple(g._simplex((B, T, H, w), p_write, gen, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    g_read = (None if norm == "global" else
              torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5)
    v = torch.randn(B, T, H, dv, device=device, dtype=torch.float64, generator=gen)
    return dict(read=read, write=write, g_read=g_read, g_write=g_write, v=v)


def _f32(d):
    return {k: (tuple(x.float() for x in v) if isinstance(v, tuple)
                else (None if v is None else v.float()))
            for k, v in d.items()}


# ---------------------------------------------------------------------------
# 1. The prefill -> decode handoff, across the regime arms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay_on", [False, True])
def test_the_handoff_holds_across_the_arms_from_a_ragged_prefill(norm, decay_on):
    """Tiled prefill over T = 33 (a ragged tail — the seam consumes a state the
    chunked path produced through a partial tile), then four decode steps, each
    graded against the chained fp64 oracle. Sparse routing, so decode's read
    support lands on prefill-cold leaves — asserted, since that is the corner
    this family owns natively."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(9000 + int(decay_on))
    T, steps = 33, 4
    decay = (None if not decay_on else
             LeafMassDecay(dials=g._decay_dials(_WIDTHS, _H, gen, device)))
    pre = _draw(gen, device, T, 0.5, 0.4, norm=norm)
    plan = Plan(topology=g._topology(_WIDTHS, _DV), normalization=norm, BT=16, BC=16,
                warp_tile=(1, _DV // 32), decay=decay_on, mma_precision="tf32x3")
    f = _f32(pre)
    _y_pre, state, _ = _consumer_forward(
        f["v"], f["read"], f["write"], f["g_read"], f["g_write"], plan, decay=decay)

    _y_ref, ref_state = naive_rola(
        pre["v"], pre["read"], pre["write"], pre["g_read"], pre["g_write"],
        plan.topology, decay, norm, output_final_state=True)

    config = freeze_decode_config(plan.topology, norm, decay=decay_on, BH=_B * _H,
                                  device=device, mma_precision="tf32x3")
    state = state.reshape(_B, _H, config.N, config.cols).clone()
    workspace, cold_seen = None, False
    for s in range(steps):
        step = _draw(gen, device, 1, 0.5, 0.4, norm=norm)
        fs = _f32(step)
        y, state, workspace = decode_forward(
            fs["v"], fs["read"], fs["write"], fs["g_read"], fs["g_write"],
            config, state, decay=decay, workspace=workspace)
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_read"], step["g_write"],
            plan.topology, decay, norm, initial_state=ref_state, output_final_state=True)
        assert g.relative(y, y_ref) < DECODE_RTOL, f"step {s}: y off the oracle"
        assert g.relative(state, ref_state) < DECODE_RTOL, f"step {s}: state off the oracle"

        R = g._leaf_product(step["read"], _WIDTHS) != 0
        written = (g._leaf_product(pre["write"], _WIDTHS) != 0).any(dim=1, keepdim=True)
        cold_seen = cold_seen or bool((R & ~written).any())
    assert cold_seen, (
        "no decode step read a prefill-cold leaf; the fixture stopped realizing "
        "the corner this family owns natively")


# ---------------------------------------------------------------------------
# 2. The horizon, with checkpoints along it
# ---------------------------------------------------------------------------

_CHECKPOINTS = (1, 2, 4, 8, 16, 32)


def test_the_horizon_is_conformant_at_every_checkpoint_not_just_the_end():
    """32 steps, decay on, global arm, from a carried state — the compounding
    regime — with the kernel-vs-oracle distance ASSERTED AT EVERY CHECKPOINT.
    An endpoint-only gate (tier 1's G6) proves the destination; the checkpoints
    prove the trajectory, which is what a slowly compounding defect (the
    bf16-state class) hides from."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(9100)
    decay = LeafMassDecay(dials=g._decay_dials(_WIDTHS, _H, gen, device))
    topology = g._topology(_WIDTHS, _DV)
    N, cols = topology.N, _DV + 1
    m0 = 0.1 * torch.randn(_B, _H, N, cols, device=device, dtype=torch.float64,
                           generator=gen)
    m0[..., _DV] = m0[..., _DV].abs()

    config = freeze_decode_config(topology, "global", decay=True, BH=_B * _H,
                                  device=device, mma_precision="tf32x3")
    state, ref_state = m0.float().clone(), m0
    workspace, errors = None, {}
    for s in range(1, max(_CHECKPOINTS) + 1):
        step = _draw(gen, device, 1, 0.5, 0.4, norm="global")
        fs = _f32(step)
        y, state, workspace = decode_forward(
            fs["v"], fs["read"], fs["write"], fs["g_read"], fs["g_write"],
            config, state, decay=decay, workspace=workspace)
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_read"], step["g_write"],
            topology, decay, "global", initial_state=ref_state, output_final_state=True)
        if s in _CHECKPOINTS:
            err_y, err_s = g.relative(y, y_ref), g.relative(state, ref_state)
            errors[s] = (err_y, err_s)
            assert err_y < DECODE_RTOL and err_s < DECODE_RTOL, (
                f"checkpoint {s}: (y, state) = ({err_y:.3e}, {err_s:.3e}) exceeds "
                f"{DECODE_RTOL:.0e}; the profile so far is {errors} — a growing "
                "profile inside the band is compounding to watch, one outside it "
                "is the defect this horizon exists to catch")
    # NON-VACUITY: the horizon must actually accumulate fp32-vs-fp64 distance —
    # a zero at the far end would mean the comparison is comparing nothing.
    assert errors[max(_CHECKPOINTS)][1] > 0.0
    assert not torch.equal(state.double(), m0), "32 steps left the state untouched"


# ---------------------------------------------------------------------------
# 3. Frozen declarations under drifting realized routing
# ---------------------------------------------------------------------------

def test_frozen_density_declarations_are_performance_only_under_drift():
    """Freeze THREE carriers — dense-declared, sparse-declared, undeclared —
    then run the SAME drifted stream (declared-dense arm sees realized-sparse
    routing and vice versa) through each. The declarations size the launch
    (``n_split``) and nothing else, so the states must be BYTE-IDENTICAL
    across carriers (decode state is n_split-invariant by design) and ``y``
    inside the oracle band. A declaration that could move a number would make
    the prefill's density snapshot a semantic input, which is exactly what
    §6b's frozen boundary forbids."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(9200)
    topology = g._topology(_WIDTHS, _DV)
    D = len(_WIDTHS)
    configs = {
        "dense-declared": freeze_decode_config(
            topology, "global", decay=False, BH=_B * _H, device=device,
            static_density=((True, True),) * D, mma_precision="tf32x3"),
        "sparse-declared": freeze_decode_config(
            topology, "global", decay=False, BH=_B * _H, device=device,
            static_density=((False, False),) * D, mma_precision="tf32x3"),
        "undeclared": freeze_decode_config(
            topology, "global", decay=False, BH=_B * _H, device=device,
            mma_precision="tf32x3"),
    }
    N, cols = topology.N, _DV + 1
    m0 = 0.1 * torch.randn(_B, _H, N, cols, device=device, dtype=torch.float64,
                           generator=gen)
    m0[..., _DV] = m0[..., _DV].abs()
    #: The drift: half the steps realized-sparse, half realized-dense, one stream.
    steps = [_draw(gen, device, 1, p, p, norm="global")
             for p in (0.2, 0.2, 1.0, 1.0, 0.3, 1.0)]

    outcomes = {}
    for name, config in configs.items():
        state, workspace, ys = m0.float().clone(), None, []
        for step in steps:
            fs = _f32(step)
            y, state, workspace = decode_forward(
                fs["v"], fs["read"], fs["write"], fs["g_read"], fs["g_write"],
                config, state, workspace=workspace)
            ys.append(y)
        outcomes[name] = (ys, state)

    ref_state, ys_ref = m0, []
    for step in steps:
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_read"], step["g_write"],
            topology, None, "global", initial_state=ref_state, output_final_state=True)
        ys_ref.append(y_ref)

    base = outcomes["undeclared"]
    for name, (ys, state) in outcomes.items():
        assert torch.equal(state, base[1]), (
            f"the {name} carrier's STATE differs from the undeclared one: a "
            "frozen density declaration moved stored numbers")
        for s, (y, y_ref) in enumerate(zip(ys, ys_ref)):
            assert g.relative(y, y_ref) < DECODE_RTOL, (
                f"{name}, step {s}: y left the oracle band under drift")

    # The certificates half: an arm mismatch REFUSES rather than degrades.
    step = _f32(steps[0])
    with pytest.raises(ValueError, match="has_decay"):
        decode_forward(step["v"], step["read"], step["write"], step["g_read"],
                       step["g_write"], configs["undeclared"],
                       base[1].clone(),
                       decay=LeafMassDecay(dials=g._decay_dials(_WIDTHS, _H, gen, device)))


# ---------------------------------------------------------------------------
# 4. Dispatch-boundary crossings, one test, forward AND backward
# ---------------------------------------------------------------------------

def test_one_walk_crosses_every_dispatch_boundary_with_the_graph_alive():
    """oracle (train, bwd) -> tiled prefill (eval) -> decode step (eval) ->
    oracle again (train, bwd), in ONE test — the layer-dispatch autograd-gap
    lesson says the seams between modes are where op-level suites go blind.
    Every grad-requiring leg asserts ``x.grad is not None`` (the fab325ae
    failure class); every eval leg asserts the counter of the arm it claims;
    and the decode leg's output is cross-checked against one tiled prefill of
    the whole sequence, which is the layer-level statement of the seam."""
    from conftest import build_layer

    from rola.state import RoLAState

    device = "cuda"
    torch.manual_seed(9300)
    layer = build_layer(hidden_size=128, num_heads=2, d_v=64, widths=(8, 16),
                        normalization="global", decay="learned",
                        layer_idx=0).to(device, torch.float32)
    T = 8

    # LEG 1 — training: the oracle arm, forward AND backward.
    layer.train()
    x1 = torch.randn(1, T, 128, device=device, requires_grad=True)
    before = dict(layer.execution_backend_counts)
    out1, _, _ = layer(x1)
    out1.sum().backward()
    after = dict(layer.execution_backend_counts)
    assert after["oracle"] == before["oracle"] + 1
    assert x1.grad is not None, "the training leg severed the graph (fab325ae class)"
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g_ is not None for g_ in grads), "no parameter received a gradient"

    # LEG 2 — eval prefill: the tiled CUDA arm, carried state out.
    layer.eval()
    x_seq = torch.randn(1, T + 1, 128, device=device)
    cache = RoLAState()
    before = dict(layer.execution_backend_counts)
    with torch.no_grad():
        layer(x_seq[:, :T], past_key_values=cache, use_cache=True)
    after = dict(layer.execution_backend_counts)
    assert after["cuda"] == before["cuda"] + 1, "the prefill leg did not take the tiled arm"

    # LEG 3 — eval decode: the decode arm, consuming the tiled state.
    before = after
    with torch.no_grad():
        y_dec, _, _ = layer(x_seq[:, T:T + 1], past_key_values=cache, use_cache=True)
    after = dict(layer.execution_backend_counts)
    assert after["decode"] == before["decode"] + 1, "the step did not take the decode arm"

    # The seam, stated at layer level: one tiled prefill of ALL T + 1 tokens
    # must agree with prefill-T-then-decode-1 on the last token. The layer's
    # default readout is the bf16 production arm, so the band is the bf16 one.
    from tests.tier1_semantics.test_consumer_vs_oracle import BF16_RTOL

    with torch.no_grad():
        y_full, _, _ = layer(x_seq)
    rel = g.relative(y_dec[:, 0], y_full[:, T])
    assert rel < BF16_RTOL, (
        f"the layer-level prefill/decode seam moved the last token by {rel:.3e}")

    # LEG 4 — back to training on a fresh sequence: the graph must be alive
    # again (a carried state under grad correctly RAISES and is gated in
    # tests/layers/test_decode_dispatch.py, so this leg starts clean).
    layer.train()
    layer.zero_grad(set_to_none=True)
    x4 = torch.randn(1, T, 128, device=device, requires_grad=True)
    out4, _, _ = layer(x4)
    out4.sum().backward()
    assert x4.grad is not None, "returning to training after decode severed the graph"
