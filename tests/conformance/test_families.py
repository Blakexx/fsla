"""CONFORMANCE — execute every declared family cell against the oracle.

One test body for the whole tier: build the family's fixture, PROVE it sits in
its claimed regime cell (`assert_nonvacuous` — a family that cannot demonstrate
its regime FAILS HERE, before any kernel comparison, which is what "a vacuous
fixture is treated as absent" means operationally), then compare the kernel to
the decomposed fp64 oracle at the tier-1 tolerance for the family's declared
numeric arm. The tolerances are IMPORTED from the tier-1 gate, never restated:
no family widens a number.

The producer-source families run in ``test_producer_families.py`` (they drive
the real gateway and both arms of ``rola_op``).
"""
from __future__ import annotations

import pytest
import torch

from tests.conformance import generators as g
from tests.conformance.families import FAMILIES
from tests.tier1_semantics.test_consumer_vs_oracle import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]

_KERNEL_FAMILIES = [f for f in FAMILIES if f.source != "producer"]


def _tolerance(family):
    # P26: `mma_precision` is a dead axis (bf16 is the only value `PlanOverrides`
    # accepts); `RTOL`, the tf32x3-exact arm's tolerance, is gone with it
    # (`tests/tier1_semantics/test_consumer_vs_oracle.py` no longer defines it).
    # Every family now runs bf16-vs-fp64-oracle at the bf16 family bound.
    del family
    return BF16_RTOL


@pytest.mark.parametrize("family", _KERNEL_FAMILIES, ids=lambda f: f.name)
def test_the_family_is_nonvacuous_and_matches_the_oracle(family):
    case = g.build_case(family)
    # IN-TEST NON-VACUITY, first. A fixture that cannot demonstrate its claimed
    # regime is absent from the census whatever the kernel comparison would say.
    g.assert_nonvacuous(family, case)

    y_ref, state_ref = case.oracle()
    # P66/P69 D2a: decay is not a feature of the operator -- its arms are no
    # longer compiled, and the dispatch's generated -1 lookup REFUSES a decay
    # request loudly.  For decay cells the refusal IS the contract; the
    # oracle side above keeps the semantics alive for the reopen conditions.
    if case.decay is not None:
        with pytest.raises(Exception, match="instantiation|arm|decay|unsupported|refus"):
            case.kernel()
        return
    y, state = case.kernel()
    tol = _tolerance(family)
    rel_y, rel_s = g.relative(y, y_ref), g.relative(state, state_ref)
    assert rel_y < tol, (
        f"{family.name}: y is {rel_y:.3e} from the oracle (tol {tol:.0e}) in the "
        f"regime cell {dict(family.regime)}")
    assert rel_s < tol, (
        f"{family.name}: state is {rel_s:.3e} from the oracle (tol {tol:.0e}) in "
        f"the regime cell {dict(family.regime)}")


