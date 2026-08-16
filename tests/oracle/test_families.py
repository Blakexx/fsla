"""CONFORMANCE — execute every declared family cell against the oracle.

One test body for the whole tier: build the family's fixture, PROVE it sits in
its claimed regime cell (`assert_nonvacuous` — a family that cannot demonstrate
its regime FAILS HERE, before any kernel comparison, which is what "a vacuous
fixture is treated as absent" means operationally), then compare the kernel to
the decomposed fp64 oracle at the tier-1 tolerance for the family's declared
numeric arm. The tolerances are IMPORTED from the tier-1 gate, never restated:
no family widens a number.

A family OUTSIDE the built chunk matrix has no kernel comparison to make, and
its contract is the REFUSAL: `rola.ops.chunk` names the refusing clause, and
`rola_op` raises it (`docs/api.md` §1). That is asserted here BY NAME rather than
by catching any exception, so a family that started being refused for a different
reason than it declares fails instead of passing quietly. The FACADE half -- the
refusal reaching a caller -- is gated where a real `RouteFactors` exists to pass
(`tests/integration/test_layer_dispatch.py`, `tests/unit/test_interface.py`); a
hand-built fixture cannot reach `rola_op`, which refuses any bundle the producer
did not derive. The family's regime cell is proven either way -- non-vacuity is a
fact about the FIXTURE, and it is what the census join is built on.

The oracle is CALLED here, never dispatched to. The producer-source families run
in ``test_producer_families.py`` (they drive the real producer and the kernel arm
against the oracle).
"""
from __future__ import annotations

import pytest
import torch

from tests.oracle import generators as g
from tests.oracle.families import FAMILIES
from tests.oracle.instantiation import REFUSAL_CLAUSES
from tests.oracle.tolerances import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]

_KERNEL_FAMILIES = [f for f in FAMILIES if f.source != "producer"]


def _tolerance(family):
    del family
    return BF16_RTOL


@pytest.mark.parametrize("family", _KERNEL_FAMILIES, ids=lambda f: f.name)
def test_the_family_is_nonvacuous_and_matches_the_oracle(family):
    case = g.build_case(family)
    # IN-TEST NON-VACUITY, first. A fixture that cannot demonstrate its claimed
    # regime is absent from the census whatever the kernel comparison would say.
    g.assert_nonvacuous(family, case)

    y_ref, state_ref = case.oracle()
    # OUT OF ENVELOPE: the refusal IS the contract. Two halves, both required --
    # the reason the family DECLARES is the reason production gives (a family
    # refused for a different clause than it says is a stale declaration), and
    # the oracle side above keeps the semantics alive whatever the envelope does.
    if not family.runs_on_kernel:
        assert case.refusal is not None, (
            f"{family.name} declares a refusal the kernel arm does not make")
        token = REFUSAL_CLAUSES[family.refusal]
        assert token in case.refusal, (
            f"{family.name} declares the {family.refusal!r} clause but production "
            f"refuses with {case.refusal!r}")
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

