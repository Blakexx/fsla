# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K39 -- a hypothesis-drawn `cells` strategy over the carry oracle's cell space.

A SUPPLEMENT to `test_carry_vs_oracle.py`'s fixed named cells
(`rola-oracle-tests`: "never invent a new one ad hoc" -- the fixed cells stay the
default battery), not a replacement: this file exists so an oracle red yields a
minimal failing (widths, k_tok, seed, cohort, d_v) automatically, via shrinking,
rather than only at whichever fixed point someone happened to name. Opt-in
(`hypothesis_cells` marker, deselected by default alongside `bench` --
`pyproject.toml`'s `addopts`) so its search runs on request, never inside the
default battery's runtime budget.

The strategy draws only over axes the ARM LIST actually declares (`widths`,
`d_v`) or that are properties of the DRAW and never told to the kernel
(`k_tok`, `seed`, `cohort` -- `tests/oracle/fixtures.py`'s own module docstring).
Drawing an undeclared topology would only ever exercise `require_arm`'s own
refusal, which is `test_arms_are_the_declared_solver_rows`'s claim, not this
one's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.oracle.fixtures import assert_fresh_binary, relative
from tests.oracle.test_carry_vs_oracle import CARVE, FLAG, SMALL, alt_cell, ref, run
from tests.oracle.tolerances import BF16_RTOL

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gen_shards  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required"),
    pytest.mark.hypothesis_cells,
]

#: THE DECLARED d_v SET, READ OFF THE ARM LIST -- never hardcoded, so a future arm
#: at a second `d_v` is picked up here without editing this file (the same
#: "read from the census, don't restate it" discipline `fixtures.built_arms`
#: applies to topologies).
_DECLARED_D_V = sorted({row[4] for row in gen_shards.CARRY_ARMS})

#: (arm, canonical T) -- the two SHIPPED flat topologies `test_carry_vs_oracle.py`
#: already exercises at M3. Widths are drawn from THIS closed set, not from
#: arbitrary integers: an undeclared width only reaches `require_arm`'s refusal,
#: which is a different test's claim.
_TOPOLOGIES = ((SMALL, 256), (FLAG, 1024))


@st.composite
def carry_cells(draw):
    """One (arm, T, k_tok, seed, cohort, d_v) draw, realized as an `alt_cell`."""
    arm, T = draw(st.sampled_from(_TOPOLOGIES))
    d_v = draw(st.sampled_from(_DECLARED_D_V))
    k_tok = draw(st.sampled_from([2, 4, 8, 16, None]))
    seed = draw(st.integers(min_value=0, max_value=2**16))
    #: `cohort` must divide T exactly (`clustered`'s `T // cohort` reshape) and is
    #: only meaningful once `k_tok` narrows the support at all.
    cohort = draw(st.none() if k_tok is None else st.sampled_from([None, 32, 64]))
    return arm, T, k_tok, seed, cohort, d_v


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(cell_args=carry_cells())
def test_carry_vs_oracle_hypothesis(cell_args):
    """M3's two-sided ALT gate, over a hypothesis-drawn cell instead of a fixed one.

    Same comparison `test_m3_two_sided_alt_*` makes (`alt_cell` + `run` + `ref`,
    the repository's one bf16 band) -- only the cell is drawn, not the assertion.
    """
    arm, T, k_tok, seed, cohort, d_v = cell_args
    assert_fresh_binary()
    cell = alt_cell(arm[0], T=T, k_tok=k_tok, seed=seed, cohort=cohort, d_v=d_v)
    num, den, _ = run(cell, arm, CARVE)
    n_ref, d_ref, _ = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
