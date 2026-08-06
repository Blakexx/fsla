"""CONFORMANCE — execute every declared family cell against the oracle.

One test body for the whole tier: build the family's fixture, PROVE it sits in
its claimed regime cell (`assert_nonvacuous` — a family that cannot demonstrate
its regime FAILS HERE, before any kernel comparison, which is what "a vacuous
fixture is treated as absent" means operationally), then compare the kernel to
the decomposed fp64 oracle at the tier-1 tolerance for the family's declared
numeric arm. The tolerances are IMPORTED from the tier-1 gate, never restated:
no family widens a number.

The producer-source families run in ``test_producer_families.py`` (they drive
the real gateway and both arms of ``rola_op``); the K = 0 corner's launch-side
witness is here, next to the run it inspects.
"""
from __future__ import annotations

import pytest
import torch

from tests.conformance import generators as g
from tests.conformance.families import FAMILIES, FAMILY_BY_NAME
from tests.tier1_semantics.test_consumer_vs_oracle import BF16_RTOL, RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]

_KERNEL_FAMILIES = [f for f in FAMILIES if f.source != "producer"]


def _tolerance(family):
    return RTOL if family.config["mma"] == "tf32x3" else BF16_RTOL


@pytest.mark.parametrize("family", _KERNEL_FAMILIES, ids=lambda f: f.name)
def test_the_family_is_nonvacuous_and_matches_the_oracle(family):
    case = g.build_case(family)
    # IN-TEST NON-VACUITY, first. A fixture that cannot demonstrate its claimed
    # regime is absent from the census whatever the kernel comparison would say.
    g.assert_nonvacuous(family, case)

    y_ref, state_ref = case.oracle()
    y, state = case.kernel()
    tol = _tolerance(family)
    rel_y, rel_s = g.relative(y, y_ref), g.relative(state, state_ref)
    assert rel_y < tol, (
        f"{family.name}: y is {rel_y:.3e} from the oracle (tol {tol:.0e}) in the "
        f"regime cell {dict(family.regime)}")
    assert rel_s < tol, (
        f"{family.name}: state is {rel_s:.3e} from the oracle (tol {tol:.0e}) in "
        f"the regime cell {dict(family.regime)}")


def test_the_k0_corner_actually_reaches_an_empty_summary():
    """The launch-side witness for ``corner/k0-fold``: the family's claim is that
    some summarized segment compacts to K = 0, and ``ws_count`` is where the
    kernel says so. ``-1`` is unreachable, so it distinguishes "pass 1 wrote
    K = 0" from "pass 1 never touched the slot"."""
    from rola.routing.sequence_parallel import build_sequence_parallel_tensors, owner_slot_counts

    family = FAMILY_BY_NAME["corner/k0-fold"]
    case = g.build_case(family)
    sp = case.forced(case.P)
    tensors = build_sequence_parallel_tensors(sp, case.schedule)
    tensors.ws_count.fill_(-1)
    y, state = case.kernel(sequence_parallel_tensors=tensors)

    offsets = sp.slot_offsets()
    stash = {o for o, n in zip(offsets, owner_slot_counts(sp.P)) if n}
    summary = [k for s, k in enumerate(tensors.ws_count.tolist()) if s not in stash]
    assert 0 in summary, (
        "no summary slot compacted to K = 0; the corner's write mask no longer "
        "confines segment 0 to one owner and the family is vacuous")
    assert any(k > 0 for k in summary), (
        "every summary is empty; the fold is never exercised and the K = 0 rows "
        "prove nothing about it")

    # The same launch is still the oracle's function — the corner is a regime,
    # not a dispensation.
    y_ref, state_ref = case.oracle()
    assert g.relative(y, y_ref) < RTOL
    assert g.relative(state, state_ref) < RTOL
