# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE SEAM: statistics bundle -> selector -> choice -> the dispatch table.

The card's BINDING CONSTRAINT is that selection is REPLACEABLE: the cost model that
decides today is one implementation behind an interface, labelled interim because its
constants are knowingly unaudited, and swapping it must be an assignment rather than
surgery. That is a property of the SEAM, not of the model, and a seam nobody exercises
is a seam that quietly stops being one -- the first replacement is exactly when its
absence would bite.

So each row here holds one piece of the seam, and each is FAIL-DEMONSTRABLE in the way
this tree requires: perturb the thing the row is about, observe the row fail, restore.
Where a row asserts that something is REFUSED, the refusal itself is the observation.

Everything here runs on the CPU with an INJECTED occupancy oracle. Selection is a pure
function of its recorded inputs, which is what makes it testable without a device; a
row that needed one would be testing the driver.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from rola.planner.schedule import DEFAULT_SM_COUNT, Plan, StaticDensity
from rola.planner.select import (
    DEFAULT_SELECTOR,
    GROUP_TOKENS,
    MMA_K_QUANTUM,
    SELECTOR_INTERFACE_VERSION,
    ConfigChoice,
    ConfigEvidence,
    CostModelSelector,
    FixedChoiceSelector,
    HardwareFacts,
    StaticPlanInputs,
    StatisticsBundle,
    config_for_step,
    select_config,
    select_with,
)
from rola.routing.summaries import RoutingSummaries
from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology

# ---------------------------------------------------------------------------
# Fixtures: one plannable cell, no device
# ---------------------------------------------------------------------------


def _topology(widths, d_v=64):
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=d_v)


def _summaries(k_hat=1.0, candidate_frac=0.5, BC=64, D=2):
    QPO = BC // MMA_K_QUANTUM
    hist = [0.0] * (QPO + 1)
    lo = min(int(k_hat), QPO)
    hist[lo] = 1.0 - (k_hat - lo)
    if lo + 1 <= QPO:
        hist[lo + 1] = k_hat - lo
    return RoutingSummaries(
        occupancy=torch.full((2, D), 0.5, dtype=torch.float64),
        quanta_hist=torch.tensor(hist, dtype=torch.float64),
        active_owner_frac=1.0,
        candidate_frac=candidate_frac,
        written_leaf_bitmap=torch.zeros((1, 1), dtype=torch.int32))


def _static(*, decay=False, widths=(64, 64), BC=64, BH=8):
    return StaticPlanInputs(
        topology=_topology(widths), normalization="global", decay=decay,
        mma_precision="bf16",
        density=StaticDensity(read=(False,) * len(widths), write=(False,) * len(widths)),
        feasible_BC=(BC,), BH=BH)


def _hw(blocks=3):
    return HardwareFacts(sm_count=DEFAULT_SM_COUNT,
                         occupancy=lambda *a, **k: blocks)


def _bundle(**kw):
    return StatisticsBundle(static=kw.pop("static", _static()),
                            hw=kw.pop("hw", _hw()),
                            summaries=kw.pop("summaries", _summaries()), **kw)


# ---------------------------------------------------------------------------
# 1. THE BUNDLE: typed, versioned, and it exposes what a selector read
# ---------------------------------------------------------------------------


def test_a_bundle_from_a_different_interface_version_is_refused():
    """The version guard, and it is the reason the field exists.

    A bundle is the CONTRACT between a producer of statistics and a selector. If the
    fields move, a selector reading the new names off an old bundle does not fail --
    it returns a different plan, silently. The refusal is what converts that into an
    error.

    FAIL-DEMONSTRABLE: constructing at the live version must SUCCEED (so the guard is
    not refusing everything), and at any other version must RAISE.
    """
    StatisticsBundle(static=_static(), hw=_hw(), summaries=_summaries(),
                     version=SELECTOR_INTERFACE_VERSION)
    with pytest.raises(ValueError, match="the fields a selector reads have moved"):
        StatisticsBundle(static=_static(), hw=_hw(), summaries=_summaries(),
                         version=SELECTOR_INTERFACE_VERSION + 1)


def test_the_bundle_exposes_the_named_statistics_a_selector_may_read():
    """STAT-EXPOSE: what was read, as scalars, recordable beside what was chosen.

    The key set is asserted EXACTLY rather than by sampling: `expose()` exists so a
    ledger row can be written from it, and a key that silently disappears takes a
    column of that ledger with it.
    """
    bundle = _bundle()
    got = bundle.expose()
    assert set(got) == {
        "interface_version", "D", "N", "BH", "d_v", "decay", "normalization",
        "mma_precision", "feasible_BC", "sm_count", "atomic_coeff", "per_bt_summaries",
        f"candidate_frac@G{GROUP_TOKENS}", f"k_hat@G{GROUP_TOKENS}",
    }
    assert got["interface_version"] == SELECTOR_INTERFACE_VERSION
    assert got["D"] == 2 and got["N"] == 64 * 64
    assert got["per_bt_summaries"] is False


def test_expose_reports_the_per_bt_statistic_when_one_was_supplied():
    """`candidate_frac` and `k_hat` are statistics OF A TILING and are not BT-invariant.

    FAIL-DEMONSTRABLE by construction: the two granularities carry DIFFERENT numbers,
    so a bundle that ignored `summaries_by_bt` would report the wrong one's value at
    `GROUP_TOKENS` and this row would fail.
    """
    by_bt = {16: _summaries(k_hat=1.0, candidate_frac=0.19),
             32: _summaries(k_hat=3.0, candidate_frac=0.36)}
    bundle = _bundle(summaries=by_bt[16], summaries_by_bt=by_bt)
    got = bundle.expose()
    assert got["per_bt_summaries"] is True
    assert got[f"candidate_frac@G{GROUP_TOKENS}"] == pytest.approx(0.19)
    assert bundle.at(16) is by_bt[16] and bundle.at(32) is by_bt[32]

    #: And with no per-BT mapping the SAME object answers at every granularity --
    #: the single-granularity arm, stated rather than assumed.
    single = _bundle()
    assert single.at(16) is single.at(32) is single.summaries


# ---------------------------------------------------------------------------
# 2. THE SELECTOR PROTOCOL: who decided, and was it audited
# ---------------------------------------------------------------------------


def test_the_default_selector_is_the_cost_model_and_it_declares_itself_interim():
    """The label the ruling is FOR.

    `INTERIM` is not decoration: it is the one machine-readable statement that this
    plan's numbers came from constants the calibration inventory records as not yet
    personally audited. A selector that stopped declaring it would make an audited and
    an unaudited plan indistinguishable.
    """
    assert isinstance(DEFAULT_SELECTOR, CostModelSelector)
    assert (DEFAULT_SELECTOR.name, DEFAULT_SELECTOR.status) == ("cost-model", "INTERIM")
    #: Every selector shipped answers both questions -- the protocol's whole surface.
    fixed = FixedChoiceSelector(choice=ConfigChoice(BC=64, feeder_flags=None))
    assert (fixed.name, fixed.status) == ("fixed", "DEBUG")


def test_select_with_runs_the_selector_it_is_given_and_returns_its_choice():
    """The swappable call itself.

    FAIL-DEMONSTRABLE: the fixed selector is pinned to a choice the cost model does
    NOT make on this cell (asserted here, not assumed), so a `select_with` that
    ignored its argument and ran the default would return the other `BC` and fail.
    """
    bundle = _bundle()
    cost_choice, cost_evidence = select_with(CostModelSelector(), bundle)
    assert isinstance(cost_evidence, ConfigEvidence)

    other_BC = 16 if cost_choice.BC == 64 else 64
    pinned = ConfigChoice(BC=other_BC, feeder_flags=cost_choice.feeder_flags)
    choice, evidence = select_with(FixedChoiceSelector(choice=pinned), bundle)
    assert choice is pinned
    assert choice.BC == other_BC != cost_choice.BC
    #: NO EVIDENCE, deliberately: it weighed nothing, and an empty `ConfigEvidence`
    #: would be a claim that a decision was reasoned.
    assert evidence is None


def test_a_fixed_selector_reads_nothing_from_the_bundle():
    """`FixedChoiceSelector` must be usable where a bundle is degenerate.

    FAIL-DEMONSTRABLE: the bundle's statistics are replaced with an object that RAISES
    on any attribute access, so any read at all fails the row.
    """
    class Explodes:
        def __getattr__(self, name):
            raise AssertionError(f"the fixed selector read {name!r} from the summaries")

    pinned = ConfigChoice(BC=64, feeder_flags=None)
    bundle = StatisticsBundle(static=_static(), hw=_hw(), summaries=Explodes())
    assert select_with(FixedChoiceSelector(choice=pinned), bundle)[0] is pinned


# ---------------------------------------------------------------------------
# 3. THE FROZEN ENTRY POINTS TAKE A SELECTOR, KEYWORD-ONLY
# ---------------------------------------------------------------------------


def test_select_config_keeps_its_frozen_positional_signature():
    """§4.3 freezes `select_config(static, hw, summaries)`. The seam may not move it.

    FAIL-DEMONSTRABLE: the call is made POSITIONALLY with exactly three arguments, so
    an added positional parameter breaks this row.
    """
    choice = select_config(_static(), _hw(), _summaries())
    assert isinstance(choice, ConfigChoice)
    assert choice.BC == 64
    with pytest.raises(TypeError):
        select_config(_static(), _hw(), _summaries(), FixedChoiceSelector(choice=choice))


def test_a_selector_passed_to_the_frozen_entry_points_is_the_one_that_decides():
    """The swap, through the two entry points a caller actually uses.

    FAIL-DEMONSTRABLE the same way as `select_with`: the pin is the choice the default
    does NOT make, so an entry point that ignored `selector=` returns the other one.
    """
    static, hw, summaries = _static(), _hw(), _summaries()
    default_choice = select_config(static, hw, summaries)
    other_BC = 16 if default_choice.BC == 64 else 64
    pinned = ConfigChoice(BC=other_BC, feeder_flags=default_choice.feeder_flags)
    selector = FixedChoiceSelector(choice=pinned)

    assert select_config(static, hw, summaries, selector=selector) is pinned
    choice, evidence = config_for_step(static, hw, T=128, summaries=summaries,
                                       selector=selector)
    assert choice is pinned and evidence is None
    #: And WITHOUT the keyword the default still decides -- the swap is opt-in.
    assert config_for_step(static, hw, T=128, summaries=summaries)[0].BC == default_choice.BC


def test_cost_model_kwargs_beside_a_selector_are_refused_rather_than_dropped():
    """`summaries_by_bt` belongs to the cost model's signature, not to the protocol.

    Silently dropping it would run the named selector on statistics the caller thought
    it had supplied -- the failure mode is a plan that looks reasoned and is not.
    """
    static, hw, summaries = _static(), _hw(), _summaries()
    selector = FixedChoiceSelector(choice=ConfigChoice(BC=64, feeder_flags=None))
    with pytest.raises(TypeError, match="those arguments are the cost model's"):
        config_for_step(static, hw, T=128, summaries=summaries, selector=selector,
                        summaries_by_bt={16: summaries, 32: summaries})
    #: The same kwarg WITHOUT a selector is accepted -- so the refusal is about the
    #: combination and not about the argument.
    config_for_step(static, hw, T=128, summaries=summaries,
                    summaries_by_bt={16: summaries, 32: summaries})


def test_the_decode_bypass_still_precedes_the_selector():
    """§6b is ARCHITECTURAL: at `T = 1` no selector runs, named or not.

    FAIL-DEMONSTRABLE: the selector passed in would raise if it were called.
    """
    class Explodes:
        name, status = "explodes", "DEBUG"

        def __call__(self, bundle):
            raise AssertionError("a selector ran on the decode path")

    choice, evidence = config_for_step(_static(), _hw(), T=1, selector=Explodes())
    assert evidence is None and isinstance(choice, ConfigChoice)


# ---------------------------------------------------------------------------
# 4. THE CHOICE REACHES THE TABLE, AND THE TABLE REFUSES WHAT IT HAS NO ARM FOR
# ---------------------------------------------------------------------------


def test_a_choice_the_table_has_no_arm_for_is_refused_by_the_table():
    """The last stage of the seam: the table CONSUMES a choice and never repairs one.

    A fixed selector is the easiest way to produce an off-matrix choice, which is
    exactly why the refusal has to live in the table rather than in the selector: a
    replacement selector is not obliged to be careful, and the dispatch is.

    Runs on the ROW SET, not on a device: `DispatchTable` is data.
    """
    from rola.planner.dispatch_table import DispatchRow, DispatchTable

    rows = [DispatchRow(BC=64, DV=64, D=2, decay=False,
                        smem_bytes=27096, shard="s", src_key="k")]
    table = DispatchTable(rows)
    assert table.resolve(64, 64, 2, False) is rows[0]
    with pytest.raises(KeyError, match="no compiled arm"):
        table.resolve(32, 64, 2, False)


# ---------------------------------------------------------------------------
# 5. THE PLAN RECORDS WHO DECIDED
# ---------------------------------------------------------------------------


def test_a_plan_carries_the_selector_name_and_status_advisorily():
    """F3's fix: the fields the `Selector` docstring promises actually exist.

    ADVISORY on the same contract `occupancy_blocks_per_sm` has -- recorded beside the
    decision, read by no rule. Both halves are asserted: the fields carry what was put
    in them, and a plan built WITHOUT a selector says so with an empty string rather
    than with a plausible default.
    """
    topology = _topology((64, 64))
    plan = Plan(topology=topology, normalization="global", BC=64,
                warp_tile=(1, 2), selector_name=DEFAULT_SELECTOR.name,
                selector_status=DEFAULT_SELECTOR.status)
    assert (plan.selector_name, plan.selector_status) == ("cost-model", "INTERIM")

    hand_stamped = Plan(topology=topology, normalization="global", BC=64,
                        warp_tile=(1, 2))
    assert (hand_stamped.selector_name, hand_stamped.selector_status) == ("", "")

    #: NO RULE READS THEM. Two plans differing ONLY in provenance must be identical
    #: everywhere a rule looks -- which is what "advisory" has to mean to be worth
    #: writing down.
    other = dataclasses.replace(plan, selector_name="whatever", selector_status="AUDITED")
    ignored = {"selector_name", "selector_status"}
    assert {f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)
            if f.name not in ignored} == {
        f.name: getattr(other, f.name) for f in dataclasses.fields(other)
        if f.name not in ignored}


def test_the_op_stamps_the_selector_it_actually_ran():
    """The stamp is taken FROM the selector object, not written out beside it.

    `rola/interface.py`'s `_derive_plan` -- the private seam, which is where selection
    now lives, because choosing a cost model is not describing a model -- resolves the
    selector (the expert override or `DEFAULT_SELECTOR`), runs it through `select_with`,
    and stamps `selector.name` / `selector.status`. Asserting the SOURCE rather than the
    string is what keeps a swapped selector's plan honest: this row passes only while
    the two come from one object.
    """
    import inspect

    import rola.interface as layer_mod

    src = inspect.getsource(layer_mod)
    assert "selector = overrides.selector or DEFAULT_SELECTOR" in src
    assert "select_with(" in src
    assert "selector_name=selector.name" in src
    assert "selector_status=selector.status" in src

