"""The per-commit latency gate.

The pass/fail line comes from :func:`bench.regression.threshold_ms`, derived from
the arm's own committed IQR, because every off-the-shelf comparison flag is a
percentage of a previous run and a percentage is a number about nobody's hardware.

**THE GATED NUMBER IS TAKEN DIRECTLY, NOT FROM INSIDE A BENCHMARK RUNNER'S ROUND
LOOP -- MEASURED, THIRD CORRECTION TO THIS FILE'S SPREAD.** It ran under
``pytest-benchmark``, which re-enters ``measure`` dozens of times on its own
calibration schedule and reports the LAST call's samples. Under the host timer that
was invisible: the baseline's IQR was 0.03-0.06 ms and absorbed it. Under device
events the committed spread is 0.001 ms and it stopped being absorbed -- three of
four arms read up to +51 % (``global|warm`` 0.1362 ms against a 0.0924 ms limit),
while the SAME arms measured alone, or in the same session under
``--benchmark-disable``, read their baseline exactly (8/8 pass). The runner's
rounds, not the kernel, were the difference; the arm's median is now measured once,
the way the recorder measures it.

The storage the library was adopted for is `perf/ledger.jsonl` now, which keeps the
raw samples, the instrument and the provenance the library never had
(`docs/measurement.md`).

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

from bench.instruments import settle_clocks  # noqa: E402
from bench.pairing import CANONICAL_INSTRUMENT  # noqa: E402
from bench.regression import check_instrument, load_baseline, summarize, threshold_ms  # noqa: E402

pytestmark = [
    pytest.mark.bench,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]


@pytest.fixture(scope="session", autouse=True)
def _clocks_settled():
    """Item 0b: match the RECORDER's clock state before the gate times anything.

    The recorder's 35 * 5 * 7 reps ramp the box as a side effect; this gate's 11
    reps per arm do not, which is exactly the gap `extraction.md`'s gate-defect note
    traces the two wave-quantized `topo-*` cells' flakiness to (§`bench.instruments`
    docstring for the full account). Session-scoped and autouse: it runs once,
    before the first parametrized arm, on the same device every arm in this process
    will be timed on -- not per-arm, because the clock is a property of the box, not
    of the arm.

    **A gate that times arms against an unsettled clock is not gating anything**
    (the whole point of item 0b): if `settle_clocks()` reports `settled: False`
    (no `nvidia-smi`, a query failure, or a genuine `timeout_s` timeout without
    plateauing), this fails the session outright rather than proceeding to compare
    every arm's timing against a baseline recorded under a different clock state.
    """
    if torch.cuda.is_available():
        info = settle_clocks()
        print(f"\nclock-settle: {info}")
        if not info["settled"]:
            pytest.fail(f"clock did not settle before the gate: {info}")
    yield


@pytest.mark.parametrize("spec", ARMS, ids=[arm_id(*s) for s in ARMS])
def test_an_arm_has_not_regressed_past_its_own_measured_spread(spec):
    key = arm_id(*spec)
    baseline = load_baseline()
    exp = baseline["arms"].get(key)
    assert exp is not None, (
        f"{key} has no committed baseline. A new arm is a MEASUREMENT: record the "
        f"whole baseline with `python benchmarks/bench_regression.py --write` under "
        f"the lock, and commit it with its provenance.")

    #: ONE call, the recorder's own entry point at the recorder's rep count, so the
    #: gate and the baseline are the same quantity by construction rather than by
    #: two schedules happening to agree (see the module docstring's third correction).
    got = summarize(measure(*spec, reps=11))
    #: The two sides must be the same quantity before they can be the same
    #: comparison. `measure` times with the committed instrument; the baseline says
    #: which one took it, and a disagreement is an error, never a widened band.
    check_instrument(exp, got["source"], key)

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


def test_a_baseline_from_a_different_stopwatch_is_refused():
    """NON-VACUITY of the one-stopwatch rule (I-2). Plant the violation the rule
    exists to catch -- a baseline recorded with the host timer, gated against a
    run taken with device events -- and require the refusal.

    The planted arm carries the SAME median and IQR as a passing one, so nothing but
    the instrument name distinguishes it: if the check were decorative the assertion
    below would fail by the comparison quietly succeeding.
    """
    passing = {"median_ms": 1.0, "iqr_ms": 0.1, "reps": 11,
               "source": CANONICAL_INSTRUMENT}
    check_instrument(passing, CANONICAL_INSTRUMENT, "arm")     # the honest case

    planted = dict(passing, source="perf_counter")
    with pytest.raises(ValueError, match="different quantities"):
        check_instrument(planted, CANONICAL_INSTRUMENT, "arm")

    #: An ncu duration is a third instrument and is refused by the same rule -- the
    #: measured gap against event timing on this box is 19 %.
    with pytest.raises(ValueError, match="different quantities"):
        check_instrument(dict(passing, source="ncu"), CANONICAL_INSTRUMENT, "arm")

    #: A baseline that never said which stopwatch took it is unknown, not compatible.
    with pytest.raises(ValueError, match="does not say which stopwatch"):
        check_instrument({"median_ms": 1.0, "iqr_ms": 0.1}, CANONICAL_INSTRUMENT,
                         "arm")


def test_every_committed_baseline_arm_names_its_stopwatch():
    """The committed file, not a synthetic dict: a rule the tree does not satisfy
    is a rule nothing enforces."""
    for key, arm in load_baseline()["arms"].items():
        check_instrument(arm, arm["source"], key)


def test_a_zero_spread_arm_still_has_a_usable_limit():
    """The degenerate end: a perfectly reproducible arm gets a ZERO-width band, so
    any slowdown at all fires. That is the correct behaviour and it is pinned here,
    because the tempting fix -- adding a floor percentage -- would reintroduce
    exactly the invented number the derivation exists to avoid."""
    assert threshold_ms(1.0, 0.0) == 1.0
