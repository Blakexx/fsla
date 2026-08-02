"""The per-commit latency gate, as a ``pytest-benchmark`` test.

``pytest-benchmark`` supplies the storage and comparison plumbing (§6.9): it runs
each arm, keeps the distribution, and can save/compare runs across commits with
``--benchmark-autosave`` / ``--benchmark-compare``. What it does NOT supply, and
what this file therefore keeps, is the JUDGEMENT: the pass/fail line comes from
:func:`bench.regression.threshold_ms`, derived from the arm's own committed IQR,
because the library's own comparison flags are percentages of a previous run and a
percentage is a number about nobody's hardware.

Run it explicitly -- it is marked ``bench`` and deselected everywhere else:

    flock /tmp/rola_gpu.lock pytest benchmarks/test_latency_regression.py -m bench

This is a GATE, never a recorder. Re-recording is
``benchmarks/bench_regression.py --write``, a separate deliberate act, because a
test that can rewrite its own baseline cannot fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_regression import ARMS, arm_id, measure  # noqa: E402
from bench.regression import load_baseline, summarize, threshold_ms  # noqa: E402

pytestmark = [
    pytest.mark.bench,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]


@pytest.mark.parametrize("spec", ARMS, ids=[arm_id(*s) for s in ARMS])
def test_an_arm_has_not_regressed_past_its_own_measured_spread(benchmark, spec):
    key = arm_id(*spec)
    baseline = load_baseline()
    exp = baseline["arms"].get(key)
    assert exp is not None, (
        f"{key} has no committed baseline. A new arm is a MEASUREMENT: record the "
        f"whole baseline with `python benchmarks/bench_regression.py --write` under "
        f"the lock, and commit it with its provenance.")

    #: pytest-benchmark drives the repetition and keeps the distribution (that is the
    #: plumbing being adopted), and its own reported columns time the WHOLE arm sweep
    #: -- warm-ups included -- which is not the quantity being gated. The gated number
    #: is the returned per-launch summary, taken with the harness's own timer so it is
    #: the same quantity as the baseline. Two different numbers on purpose: the
    #: library's is for its cross-commit history, ours is for the pass/fail line.
    times = benchmark(measure, *spec, reps=11)
    got = summarize(times)

    limit = threshold_ms(exp["median_ms"], exp["iqr_ms"])
    assert got["median_ms"] <= limit, (
        f"{key} regressed: {got['median_ms']:.4f} ms against a limit of {limit:.4f} ms "
        f"(baseline median {exp['median_ms']:.4f}, IQR {exp['iqr_ms']:.4f}, "
        f"{baseline['alarm_sigma']}-sigma one-sided).\n"
        f"  The limit is DERIVED from this arm's own measured spread. If the arm is "
        f"legitimately slower, re-record the baseline as a measurement and say in the "
        f"commit why it moved -- do not widen the sigma.\n"
        f"  Baseline provenance: {baseline['provenance']}")


def test_the_derived_limit_can_actually_be_exceeded():
    """NON-VACUITY of the gate above. A threshold that nothing can exceed is not a
    threshold, and the derivation is arithmetic, so this needs no device.

    Two properties, both required: the limit is STRICTLY ABOVE the baseline median
    (or an unchanged kernel would fail its own gate), and it is FINITE and close
    enough that a real slowdown lands outside it (or the gate is decoration).
    """
    from bench.regression import ALARM_SIGMA, IQR_PER_SIGMA

    median, spread = 1.0, 0.10
    limit = threshold_ms(median, spread)
    assert limit == pytest.approx(median + ALARM_SIGMA * spread / IQR_PER_SIGMA)
    assert median < limit < median + ALARM_SIGMA * spread
    assert limit < median * 1.25, "a 10 % IQR must not buy a 25 % regression"
    #: The comparison the gate makes, on both sides of its own line.
    assert not (limit * 0.99 > limit)
    assert limit * 1.01 > limit


def test_a_zero_spread_arm_still_has_a_usable_limit():
    """The degenerate end: a perfectly reproducible arm gets a ZERO-width band, so
    any slowdown at all fires. That is the correct behaviour and it is pinned here,
    because the tempting fix -- adding a floor percentage -- would reintroduce
    exactly the invented number the derivation exists to avoid."""
    assert threshold_ms(1.0, 0.0) == 1.0
