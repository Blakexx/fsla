"""The per-(subject, cell) latency gate: measured HEAD against the committed baseline.

Marked `bench`, which is opt-in (`pyproject.toml` `addopts = -m 'not bench'`): a latency
measurement is not a correctness gate and must never be able to fail a merge from a noisy
machine. Run it deliberately, on a quiet box, wrapped:

    flock /tmp/rola_gpu.lock pytest tests/unit/test_chunk_latency_regression.py -m bench

**THIS FILE NEVER WRITES A BASELINE.** Recording is
`benchmarks/bench_chunk.py baseline --write`, in one quiet session, on a clean tree. A
test that could re-record the thing it compares against is a test that cannot fail twice.

The two sides are the same quantity by construction: the gate and the recorder both go
through `bench.pairing.measure_arm` on a fixture from the same registry. The retired
harness's gate read its number from inside a benchmark runner's round loop instead, and
that runner's schedule -- not the kernel -- set the value it compared.
"""
from __future__ import annotations

import pytest
import torch

from bench import chunk_baseline
from bench.chunk_cells import CELLS
from bench.discipline import disciplined
from bench.driver import TIER_ROUNDS, build_arm
from bench.pairing import REPS_PER_ROUND, measure_arm
from bench.subjects import applicable

pytestmark = [pytest.mark.bench, pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]

_ARMS = [(subject, name) for name, cell in CELLS.items() for subject in applicable(cell)]


@pytest.fixture(scope="module")
def measured():
    """One disciplined session for the whole module: the lock is proven once, the clock
    settles once, and every arm in the file is timed under that one settled state."""
    baseline = chunk_baseline.load()
    results, cache = {}, {}
    with disciplined(tier="landing", cmd="pytest test_chunk_latency_regression.py"):
        for subject, cell in _ARMS:
            arm = build_arm(subject, cell, cache)
            results[chunk_baseline.key(subject, cell)] = measure_arm(
                arm, rounds=TIER_ROUNDS["landing"], reps_per_round=REPS_PER_ROUND)
    return baseline, results


@pytest.mark.parametrize("subject,cell", _ARMS, ids=[f"{s}|{c}" for s, c in _ARMS])
def test_the_cell_stays_inside_the_limit_derived_from_its_own_recorded_spread(
        measured, subject, cell):
    baseline, results = measured
    verdict = chunk_baseline.compare(baseline, subject, cell,
                                     results[chunk_baseline.key(subject, cell)])
    assert verdict["verdict"] == "within_limit", (
        f"{verdict['key']} measured {verdict['median_ms']:.4f} ms against a limit of "
        f"{verdict['limit_ms']:.4f} ms (baseline median {verdict['baseline_median_ms']:.4f} "
        f"ms, IQR {verdict['baseline_iqr_ms']:.4f}). EFFECT SIZE ONLY: confirm with an "
        f"interleaved A/B before calling it a regression -- "
        f"`bench_chunk.py run --ab {subject}@{cell}:... --tier landing`.")
