"""CONFORMANCE — the producer-source families: realized routing at swept gains.

Some correlations only real routing produces (token coherence inherited from the
hidden stream, read-write coincidence from the shared projection), so these
families drive the REAL gateway — ``RouteProducer`` over ``sparse_routing`` —
rather than a sampler, and the gain on the hidden stream is the knob that sweeps
the realized density, exactly as it does in training.

Both arms of ``rola_op`` run on each realized bundle and must agree at the
tier-1 exactness tolerance. The stale-plan family is the P16 seam used as
designed: a plan derived from one end of the sweep executed against the other
end's bundle through ``rola.expert`` — the adversarial handle a
derive-then-immediately-use pipeline would not have.

NON-VACUITY here is sweep-shaped: ``assert_swept`` proves the gain knob actually
moved the realized density (>= 4x span, monotone), and the shared regime
checkers run per arm with the family's cell required to be demonstrated by the
sweep as a whole (an entmax sweep is dense at its cold end BY DESIGN, so
requiring every arm to sit in the sparse cell would demand a vacuous sweep).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.conformance import generators as g
from tests.conformance.families import FAMILY_BY_NAME
from tests.tier1_semantics.test_consumer_vs_oracle import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]


def _producer_and_stream(family):
    """The gateway, and a temporally COHERENT hidden stream (AR(1), phi = 0.95):
    coherence is a property realized routing inherits from its input, so the
    input has to carry it."""
    from rola import sparse_routing
    from tests.conftest import build_routes

    c = family.config
    torch.manual_seed(c["seed"])
    producer = build_routes(
        hidden_size=64, num_heads=c["H"], d_v=c["d_v"], widths=c["widths"],
        routing=sparse_routing(1.5), normalization=c["norm"]).cuda()
    B, T = c["B"], c["T"]
    gen = torch.Generator(device="cuda").manual_seed(c["seed"] + 1)
    eps = torch.randn(B, T, 64, device="cuda", generator=gen)
    x = torch.empty_like(eps)
    x[:, 0] = eps[:, 0]
    for t in range(1, T):
        x[:, t] = 0.95 * x[:, t - 1] + (1 - 0.95**2) ** 0.5 * eps[:, t]
    v = torch.randn(B, T, c["H"], c["d_v"], device="cuda", generator=gen)
    return producer, x, v


def _density(routes) -> float:
    """The realized exact-nonzero fraction of the write side."""
    total = live = 0
    for level in routes.write:
        live += int((level != 0).sum())
        total += level.numel()
    return live / total


def _view(family, routes):
    c = family.config
    return SimpleNamespace(
        read_levels=routes.read, write_levels=routes.write, widths=tuple(c["widths"]),
        T=c["T"], BT=g.TOKEN_WORD, initial_state=None)


def _demonstrated_by_some_arm(family, views):
    """Each (axis, value) of the family's regime cell must be demonstrated by AT
    LEAST ONE arm of the sweep (`density = swept` is proven by `assert_swept`
    over all arms instead)."""
    for axis, value in family.regime.items():
        if (axis, value) == ("density", "swept"):
            continue
        errors = []
        for view in views:
            try:
                g.CHECKERS[(axis, value)](view)
                break
            except AssertionError as error:
                errors.append(str(error))
        else:
            raise AssertionError(
                f"no arm of the sweep demonstrates ({axis}, {value}): {errors}")


#: The kernel arm is the chunk consumer, whose envelope is uniform
#: D>=2, d_v=64, global, no state output yet (pending the paged state I/O work).  INSIDE the envelope the
#: arms must AGREE on y; OUTSIDE it the kernel arm must REFUSE LOUDLY with
#: the card that adds the capability -- both are the documented contract, and
#: this test asserts whichever one the cell falls in.  Final-state agreement
#: rejoins the in-envelope family when the paged state I/O lands.
@pytest.mark.parametrize("name", ["producer/swept-gain", "producer/swept-gain-d64",
                                  "producer/swept-gain-raw-d2"])
def test_the_swept_gain_family_agrees_across_both_arms(name):
    from rola import rola_op
    from rola.expert import PlanOverrides

    family = FAMILY_BY_NAME[name]
    producer, x, v = _producer_and_stream(family)
    densities, views = [], []
    y_k = None
    for gain in family.params["gains"]:
        with torch.no_grad():
            routes = producer(gain * x)
            try:
                y_k, _ = rola_op(routes, v, expert=PlanOverrides())
            except NotImplementedError as e:
                # the refusal is part of the contract: it names the boundary,
                # and the reference arm still serves the cell.
                assert "chunk consumer" in str(e)
                y_k = None
        y_r, _ = rola_op(routes, v, _arm="reference")
        if y_k is not None:
            rel_y = g.relative(y_k, y_r)
            assert rel_y < BF16_RTOL, f"gain {gain}: y differs across arms by {rel_y:.3e}"
        densities.append(_density(routes))
        views.append(_view(family, routes))
    if name == "producer/swept-gain-d64":
        assert y_k is not None, "the in-envelope family must actually run the kernel arm"

    # NON-VACUITY: the knob moved the distribution, and the cell is demonstrated.
    g.assert_swept(densities)  # ascending gain -> monotone descending density
    _demonstrated_by_some_arm(family, views)


def test_the_stale_plan_family_executes_a_plan_priced_on_a_dead_distribution():
    from rola import rola_op
    from rola.expert import PlanOverrides, derive_plan, execute

    family = FAMILY_BY_NAME["producer/stale-plan"]
    producer, x, v = _producer_and_stream(family)
    with torch.no_grad():
        routes_plan = producer(family.params["plan_gain"] * x)
        routes_data = producer(family.params["data_gain"] * x)

    # NON-VACUITY: the plan's bundle and the executed bundle realize genuinely
    # different distributions — otherwise "stale" is a label, not a fact.
    d_plan, d_data = _density(routes_plan), _density(routes_data)
    assert d_plan / max(d_data, 1e-9) >= 4.0, (
        f"the plan bundle ({d_plan:.2f}) and data bundle ({d_data:.2f}) are not "
        "4x apart in realized density; the plan is not meaningfully stale")

    plan = derive_plan(routes_plan, expert=PlanOverrides(), device=v.device)
    with torch.no_grad():
        y, state, _schedule = execute(plan, routes_data, v)
    import rola

    y_ref, state_ref = rola_op(routes_data, v, _arm="reference", state=rola.state())
    state_ref = state_ref.materialize()
    rel_y, rel_s = g.relative(y, y_ref), g.relative(state, state_ref)
    assert rel_y < BF16_RTOL, (
        f"a stale plan moved y by {rel_y:.3e}: the schedule is assembled from the "
        "REALIZED masks inside _execute, so a plan may only be slow, never wrong")
    assert rel_s < BF16_RTOL, f"a stale plan moved the state by {rel_s:.3e}"
