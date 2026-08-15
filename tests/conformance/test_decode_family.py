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

from rola.ops import chunk as _chunk
from rola.ops.decode import _decode_step, freeze_decode_config
from rola.ops.naive import naive_rola
from rola.routing.types import LeafMassDecay
from tests.conformance import generators as g
from tests.conftest import RawRoutes
from tests.tier1_semantics.test_decode_vs_oracle import DECODE_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

#: P67 D2-b: the prefill leg runs on the CHUNK arm, whose built manifest is
#: uniform-width, so the family's topology is a built `(D, B)` and its prefill
#: length is a multiple of `rola.ops.chunk.CHUNK_TOKENS`. DECODE's own envelope
#: is unchanged -- it is BC-blind and topology-general -- so that is a
#: constraint the PREFILL puts on the fixture, not one decode acquired.
_WIDTHS, _DV, _B, _H = (16, 16), 64, 2, 2


def _draw(gen, device, T, p_read, p_write, *, widths=_WIDTHS, H=_H, B=_B, dv=_DV,
          norm="global", write_live=None):
    """`write_live` confines level 0's WRITE support to its first `write_live`
    digits, which makes every leaf under a higher digit provably never written.
    The prefill uses it so the cold-read corner this family owns natively is
    STRUCTURAL rather than a property the draw happened to have: at a built
    uniform topology a Bernoulli write side reaches every leaf within a few
    tokens, and the corner would be asserted and never realized."""
    read = tuple(g._simplex((B, T, H, w), p_read, gen, device) for w in widths)
    if write_live is None:
        write = tuple(g._simplex((B, T, H, w), p_write, gen, device) for w in widths)
    else:
        mask = torch.zeros(1, 1, 1, widths[0])
        mask[..., :write_live] = 1.0
        write = (g._masked_simplex((B, T, H, widths[0]), mask, gen, device),) + tuple(
            g._simplex((B, T, H, w), p_write, gen, device) for w in widths[1:])
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

@pytest.mark.parametrize("decay_on", [False, True])
def test_the_handoff_holds_across_the_arms_from_a_chunked_prefill(decay_on):
    """Chunk-arm prefill, then four decode steps, each graded against the
    chained fp64 oracle. Sparse routing, so decode's read support lands on
    prefill-cold leaves -- asserted, since that is the corner this family owns
    natively.

    P67 D2-b: the prefill leg was the TILED consumer over a RAGGED `T = 33`.
    The chunk arm has no ragged tail -- `T` must be a multiple of
    `rola.ops.chunk.CHUNK_TOKENS` or the dispatch routes the call to the fp64
    oracle -- so the handoff-through-a-partial-tile half of this claim has no
    kernel left to be made against and does not survive. What does survive, and
    is what the seam is about, is that the state the PRODUCTION prefill arm
    writes is the state the recurrent arm continues. The ragged path is still
    walked end to end at layer level further down this file, where `T + 1` is
    deliberately not a chunk multiple and therefore runs on the oracle.
    """
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(9000 + int(decay_on))
    T, steps = 2 * _chunk.CHUNK_TOKENS, 4
    topology = g._topology(_WIDTHS, _DV)
    decay = (None if not decay_on else
             LeafMassDecay(dials=g._decay_dials(_WIDTHS, _H, gen, device)))
    pre = _draw(gen, device, T, 0.5, 0.4, norm="global",
                write_live=_WIDTHS[0] // 2)
    f = _f32(pre)
    bundle = RawRoutes(topology, "global", f["read"], f["write"], f["g_read"],
                       f["g_write"])
    if decay_on:
        # The decay retirement: decay is not a feature of the operator. The
        # kernel arm names the refusal and the dispatch routes such a call to
        # the oracle; the refusal IS this cell's contract until the reopen
        # conditions, and the oracle below keeps the handoff semantics alive.
        reason = _chunk.unsupported(bundle, decay)
        assert reason is not None and "decay" in reason, reason
        naive_rola(pre["v"], pre["read"], pre["write"], pre["g_read"],
                   pre["g_write"], topology, decay, "global",
                   output_final_state=True)
        return
    assert _chunk.unsupported(bundle, decay) is None, "the prefill fixture left the envelope"
    _y_pre, state = _chunk.run(bundle, f["v"], want_state=True)

    _y_ref, ref_state = naive_rola(
        pre["v"], pre["read"], pre["write"], pre["g_read"], pre["g_write"],
        topology, decay, "global", output_final_state=True)

    config = freeze_decode_config(topology, "global", decay=decay_on, BH=_B * _H,
                                  device=device, mma_precision="bf16")
    state = state.reshape(_B, _H, config.N, config.cols).clone()
    workspace, cold_seen = None, False
    for s in range(steps):
        step = _draw(gen, device, 1, 0.5, 0.4, norm="global")
        fs = _f32(step)
        y, state, workspace = _decode_step(
            fs["v"], fs["read"], fs["write"], fs["g_read"], fs["g_write"],
            config, state, decay=decay, workspace=workspace)
        y_ref, ref_state = naive_rola(
            step["v"], step["read"], step["write"], step["g_read"], step["g_write"],
            topology, decay, "global", initial_state=ref_state, output_final_state=True)
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
                                  device=device, mma_precision="bf16")
    state, ref_state = m0.float().clone(), m0
    workspace, errors = None, {}
    for s in range(1, max(_CHECKPOINTS) + 1):
        step = _draw(gen, device, 1, 0.5, 0.4, norm="global")
        fs = _f32(step)
        y, state, workspace = _decode_step(
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
# 3. The launch split under drifting realized routing
# ---------------------------------------------------------------------------

def test_the_decode_split_is_performance_only_under_drift():
    """Freeze THREE carriers at three ``n_split`` values, then run the SAME
    drifted stream (sparse steps and dense steps in one sequence) through each.
    ``n_split`` sizes the LAUNCH -- how many CTAs divide the row space -- and
    nothing else, so the states must be BYTE-IDENTICAL across carriers and ``y``
    inside the oracle band. A split that could move a stored number would make
    how many CTAs ran a semantic input, and the whole reason the split can be
    frozen once at the boundary is that it cannot."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(9200)
    topology = g._topology(_WIDTHS, _DV)
    configs = {
        f"n_split={n}": freeze_decode_config(
            topology, "global", decay=False, BH=_B * _H, device=device,
            n_split=n, mma_precision="bf16")
        for n in (1, 3, 8)
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
            y, state, workspace = _decode_step(
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

    base = outcomes["n_split=1"]
    for name, (ys, state) in outcomes.items():
        assert torch.equal(state, base[1]), (
            f"the {name} carrier's STATE differs from the single-CTA one: the "
            "launch split moved stored numbers")
        for s, (y, y_ref) in enumerate(zip(ys, ys_ref)):
            assert g.relative(y, y_ref) < DECODE_RTOL, (
                f"{name}, step {s}: y left the oracle band under drift")

    # The certificates half: an arm mismatch REFUSES rather than degrades.
    step = _f32(steps[0])
    with pytest.raises(ValueError, match="has_decay"):
        _decode_step(step["v"], step["read"], step["write"], step["g_read"],
                       step["g_write"], configs["n_split=1"],
                       base[1].clone(),
                       decay=LeafMassDecay(dials=g._decay_dials(_WIDTHS, _H, gen, device)))


# ---------------------------------------------------------------------------
# 4. Dispatch-boundary crossings, one test, forward AND backward
# ---------------------------------------------------------------------------

def test_one_walk_crosses_every_dispatch_boundary_with_the_graph_alive():
    """oracle (train, bwd) -> kernel prefill (eval) -> decode step (eval) ->
    oracle again (train, bwd), in ONE test — the layer-dispatch autograd-gap
    lesson says the seams between modes are where op-level suites go blind.
    Every grad-requiring leg asserts ``x.grad is not None`` (the fab325ae
    failure class); every eval leg asserts the counter of the arm it claims;
    and the decode leg's output is cross-checked against one prefill of the whole
    sequence, which is the layer-level statement of the seam.

    The topology and ``T`` are the BUILT chunk arm's: uniform widths and a multiple
    of the chunk. A cell outside the built matrix dispatches to the oracle (the
    unsupported-topology retirement), which would leave this walk no boundary to
    cross — the same reason decay is not configured on this layer."""
    from conftest import build_layer

    import rola

    device = "cuda"
    torch.manual_seed(9300)
    #: decay dropped from THIS layer (the decay retirement): a decay-configured layer
    #: routes every arm to the oracle, which would leave this walk no
    #: boundary to cross -- the decay routing has its own test below.
    layer = build_layer(hidden_size=128, num_heads=2, d_v=64, widths=(16, 16),
                        normalization="global", decay=None,
                        layer_idx=0).to(device, torch.float32)
    T = 32

    # LEG 1 — training: the oracle arm, forward AND backward.
    layer.train()
    x1 = torch.randn(1, T, 128, device=device, requires_grad=True)
    before = dict(layer.execution_backend_counts)
    out1, _ = layer(x1)
    out1.sum().backward()
    after = dict(layer.execution_backend_counts)
    assert after["oracle"] == before["oracle"] + 1
    assert x1.grad is not None, "the training leg severed the graph (fab325ae class)"
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g_ is not None for g_ in grads), "no parameter received a gradient"

    # LEG 2 — eval prefill: the chunk CUDA arm, carried state out.
    layer.eval()
    x_seq = torch.randn(1, T + 1, 128, device=device)
    carried = rola.state()
    before = dict(layer.execution_backend_counts)
    with torch.no_grad():
        layer(x_seq[:, :T], state=carried)
    after = dict(layer.execution_backend_counts)
    assert after["cuda"] == before["cuda"] + 1, "the prefill leg did not take the kernel arm"

    # LEG 3 — eval decode: the decode arm, consuming the kernel arm's state.
    before = after
    with torch.no_grad():
        y_dec, _ = layer(x_seq[:, T:T + 1], state=carried)
    after = dict(layer.execution_backend_counts)
    assert after["decode"] == before["decode"] + 1, "the step did not take the decode arm"

    # The seam, stated at layer level: one prefill of ALL T + 1 tokens must agree
    # with prefill-T-then-decode-1 on the last token. `T + 1` is not a multiple of
    # the chunk, so that whole-sequence leg runs on the oracle — which makes it an
    # INDEPENDENT reference for the seam rather than the same arm twice. The band is
    # the bf16 family's, since the two legs under test are the bf16 production arms.
    from tests.tier1_semantics.tolerances import BF16_RTOL

    with torch.no_grad():
        y_full, _ = layer(x_seq)
    rel = g.relative(y_dec[:, 0], y_full[:, T])
    assert rel < BF16_RTOL, (
        f"the layer-level prefill/decode seam moved the last token by {rel:.3e}")

    # LEG 4 — back to training on a fresh sequence: the graph must be alive
    # again (a carried state under grad correctly RAISES and is gated in
    # tests/layers/test_decode_dispatch.py, so this leg starts clean).
    layer.train()
    layer.zero_grad(set_to_none=True)
    x4 = torch.randn(1, T, 128, device=device, requires_grad=True)
    out4, _ = layer(x4)
    out4.sum().backward()
    assert x4.grad is not None, "returning to training after decode severed the graph"


def test_a_decay_configured_layer_routes_every_arm_to_the_oracle():
    """The decay retirement: decay is not implemented in the kernel arm.  A layer
    CONFIGURED with decay dispatches to the oracle at eval exactly as an
    unratified activation would -- counted, not silently degraded -- and the
    direct consumer entries keep the loud refusal."""
    from conftest import build_layer

    device = "cuda"
    torch.manual_seed(9301)
    layer = build_layer(hidden_size=128, num_heads=2, d_v=64, widths=(8, 16),
                        normalization="global", decay="learned",
                        layer_idx=0).to(device, torch.float32).eval()
    x = torch.randn(2, 8, 128, device=device)
    before = dict(layer.execution_backend_counts)
    with torch.no_grad():
        layer(x)
    after = dict(layer.execution_backend_counts)
    assert after["oracle"] == before["oracle"] + 1
    assert after["cuda"] == before["cuda"]
