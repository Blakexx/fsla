"""The measurement RECORD: the ledger's schema, the paired test, and the debounce.

Every check here plants the violation its instrument exists to catch and requires
the instrument to fire. A gate that cannot be shown failing is treated as absent,
so "it passed" is not evidence about any of these three until the planted case is
also shown rejected.

No device: the ledger is a file format, the test is arithmetic, and the debounce is
a run-length rule.
"""
from __future__ import annotations

import json

import pytest

from bench.ledger import (
    REQUIRED,
    SCHEMA,
    append,
    baseline_series,
    fingerprint,
    load,
    make_row,
)
from bench.regression import classify
from bench.stats import ALPHA, MIN_ROUNDS, classify_flags, paired_verdict

ENV = {"gpu": "NVIDIA GeForce RTX 3080 Ti", "sm": "sm_86", "driver_cuda": "12.4",
       "torch": "2.6.0+cu124", "ptxas": "cuda_12.4.r12.4"}


def _row(**over):
    base = dict(probe="consumer_forward", probe_version=1, arm="cell|global|none",
                env=ENV, source="cuda_events", metric="duration", unit="ms",
                values=[1.0, 1.1, 0.9], tier="landing", run_id="run-1",
                csrc_sha256="a" * 64, sha="b" * 40, dirty=False)
    base.update(over)
    return make_row(**base)


# ---------------------------------------------------------------------------
# THE SCHEMA
# ---------------------------------------------------------------------------

def test_a_row_carries_every_field_a_reader_would_otherwise_guess():
    row = _row()
    assert [k for k in REQUIRED if k not in row] == []
    assert row["schema"] == SCHEMA
    #: RAW SAMPLES, in run order, and no derived scalar beside them to disagree.
    assert row["values"] == [1.0, 1.1, 0.9]
    assert "median_ms" not in row and "iqr_ms" not in row


def test_a_row_missing_its_provenance_is_refused():
    """PLANTED VIOLATION: the field that ties a number to a kernel is dropped."""
    with pytest.raises(ValueError, match="missing required field"):
        _row(csrc_sha256=None)
    with pytest.raises(ValueError, match="missing required field"):
        _row(tier=None)
    with pytest.raises(ValueError, match="records nothing"):
        _row(values=[])


def test_the_fingerprint_separates_what_must_never_be_compared():
    """PLANTED VIOLATION, four ways: a different box, a different arch, a different
    assembler, and a redefinition of the probe must each break the series."""
    base = fingerprint(probe="p", probe_version=1, arm="a", env=ENV)
    assert base == fingerprint(probe="p", probe_version=1, arm="a", env=dict(ENV))
    for field, value in (("sm", "sm_90"), ("gpu", "A100"), ("ptxas", "cuda_12.6")):
        assert base != fingerprint(probe="p", probe_version=1, arm="a",
                                   env=dict(ENV, **{field: value}))
    assert base != fingerprint(probe="p", probe_version=2, arm="a", env=ENV)
    assert base != fingerprint(probe="p", probe_version=1, arm="b", env=ENV)


def test_a_legacy_import_can_never_enter_a_disciplined_series():
    """`probe_version: 0` is inside the fingerprint, which is what makes the banked
    markdown numbers un-comparable BY CONSTRUCTION rather than by remembering."""
    live = fingerprint(probe="p", probe_version=1, arm="a", env=ENV)
    legacy = fingerprint(probe="p", probe_version=0, arm="a", env=ENV)
    assert live != legacy


def test_the_ledger_round_trips_and_dirty_rows_stay_out_of_baselines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    clean = [_row(run_id=f"r{i}") for i in range(3)]
    dirty = _row(run_id="r-dirty", dirty=True)
    append(clean + [dirty], path=path)
    rows = load(path=path)
    assert len(rows) == 4
    fp = clean[0]["fingerprint"]
    kept = baseline_series(rows, fp, source="cuda_events")
    assert [r["run_id"] for r in kept] == ["r0", "r1", "r2"]


def test_a_row_from_an_unknown_schema_is_refused_not_guessed_at(tmp_path):
    """PLANTED VIOLATION: a future writer's row must stop this reader, not be
    half-interpreted by it."""
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(dict(_row(), schema=SCHEMA + 1)) + "\n")
    with pytest.raises(ValueError, match="this reader knows schema"):
        load(path=path)


# ---------------------------------------------------------------------------
# THE PAIRED TEST (I-4)
# ---------------------------------------------------------------------------

def test_the_paired_test_detects_a_planted_shift_and_clears_an_unshifted_pair():
    """NON-VACUITY: the same n, one arm shifted and one not."""
    shifted = paired_verdict([0.05] * MIN_ROUNDS)
    assert shifted["verdict"] == "b_slower"
    assert shifted["p"] <= ALPHA and shifted["exact"]

    #: The planted null: differences that alternate sign with equal magnitude carry
    #: no evidence at all, and the test must say so rather than pick a direction.
    assert paired_verdict([0.05, -0.05] * (MIN_ROUNDS // 2))["verdict"] == "same"


def test_the_paired_test_sees_direction():
    assert paired_verdict([-0.05] * MIN_ROUNDS)["verdict"] == "b_faster"


def test_the_round_floor_is_derived_from_alpha_and_is_binding():
    """The floor is not a chosen number: one round fewer makes EVERY outcome
    non-significant, which is the definition of an undefined test rather than a
    strict one. Both sides are asserted, so the floor cannot drift off its alpha.
    """
    from bench.stats import _exact_two_sided_p, _signed_rank_statistic

    assert paired_verdict([0.05] * MIN_ROUNDS)["p"] <= ALPHA
    #: One round short, computed directly: the STRONGEST possible evidence (every
    #: difference the same way) still cannot reach alpha, which is why the harness
    #: refuses to report a verdict there at all.
    short = [0.05] * (MIN_ROUNDS - 1)
    w, ranks = _signed_rank_statistic(short)
    assert _exact_two_sided_p(w, ranks) > ALPHA
    assert paired_verdict(short)["verdict"] == "insufficient_data"


def test_a_shift_smaller_than_the_noise_is_not_called():
    """The instrument must not manufacture precision: a shift buried in scatter of
    its own size is `same`, not a quiet pass."""
    diffs = [0.01, -0.4, 0.35, -0.30, 0.28, -0.25, 0.22, -0.18, 0.30]
    assert paired_verdict(diffs)["verdict"] == "same"


def test_too_few_rounds_is_reported_as_undecided_never_as_no_regression():
    got = paired_verdict([0.05, 0.05])
    assert got["verdict"] == "insufficient_data"
    assert got["p"] is None


def test_the_exact_and_approximate_tails_agree_where_they_meet():
    """The EXACT_MAX_N boundary is structural, not a knob: the two computations must
    describe the same distribution at the size where the switch happens."""
    from bench.stats import _normal_two_sided_p, _signed_rank_statistic

    diffs = [0.3, -0.1, 0.25, 0.4, -0.05, 0.2, 0.35, 0.15, -0.2, 0.45,
             0.05, 0.5, -0.15, 0.28, 0.33]
    w, ranks = _signed_rank_statistic(diffs)
    exact = paired_verdict(diffs)["p"]
    assert exact == pytest.approx(_normal_two_sided_p(w, ranks), abs=0.02)


# ---------------------------------------------------------------------------
# THE DEBOUNCE (I-5)
# ---------------------------------------------------------------------------

def test_one_unlucky_run_is_not_a_regression_and_two_are():
    """PLANTED VIOLATION on both sides of the rule: the single thermal outlier the
    debounce exists to absorb, and the persistent pair it must still catch."""
    assert classify_flags([False, False, False, True]) == "no_regression"
    assert classify_flags([False, False, True, True]) == "regression"


def test_a_healed_run_is_suspicious_not_silent():
    assert classify_flags([True, True, True, False, False]) == "suspicious"


def test_a_short_history_is_undecided():
    assert classify_flags([True, True]) == "insufficient_data"


# ---------------------------------------------------------------------------
# THE THREE GATES TOGETHER
# ---------------------------------------------------------------------------

def _history(values, n=5):
    return [_row(run_id=f"h{i}", values=values) for i in range(n)]


def test_all_three_gates_must_fire_before_anything_is_called_a_regression():
    history = _history([1.00, 1.01, 0.99, 1.00])
    #: A candidate far above the derived limit, persistent, and significant.
    slow = _row(run_id="head", values=[1.60, 1.61, 1.59, 1.60])
    got = classify(history + [slow], slow, paired_diffs=[0.6] * MIN_ROUNDS)
    assert got["verdict"] == "regression"
    assert got["median_ms"] > got["limit_ms"]


def test_an_effect_without_significance_is_flagged_but_not_confirmed():
    """The planted case the significance gate exists for: the median moved, the
    per-round differences do not agree that it did."""
    history = _history([1.00, 1.01, 0.99, 1.00])
    slow = _row(run_id="head", values=[1.60, 1.61, 1.59, 1.60])
    got = classify(history + [slow], slow, paired_diffs=[0.6, -0.6, 0.6, -0.6])
    assert got["verdict"] == "flagged_not_confirmed"


def test_an_unchanged_kernel_does_not_fail_its_own_gate():
    history = _history([1.00, 1.01, 0.99, 1.00])
    same = _row(run_id="head", values=[1.00, 1.00, 1.01, 0.99])
    assert classify(history, same, paired_diffs=[0.0] * 6)["verdict"] == "no_regression"


def test_a_series_too_short_to_judge_says_so():
    short = _history([1.0, 1.0], n=2)
    got = classify(short, short[-1], paired_diffs=[0.5] * 5)
    assert got["verdict"] == "insufficient_data"
