"""The CI regression layer: standard plumbing around a custom methodology (§6.9).

**WHAT IS STANDARDIZED AND WHAT IS NOT.** The methodology stays this repository's
own -- no library provides interleaved pairing, per-pair ratio IQRs, counter
corroboration, or a committed statistics manifest (:mod:`bench.pairing` is where it
lives). What a library DOES provide, and what was being hand-rolled, is storage and
comparison: run this commit's latency per ``(cell, arm)``, keep it, and compare it
to a recorded baseline. That is ``pytest-benchmark``'s job and it is now doing it.

**THE THRESHOLD IS DERIVED FROM THE CELL'S OWN MEASURED IQR, NEVER INVENTED.** A
flat "fail at +5 %" is a number about nobody's hardware: on a cell whose IQR is 12 %
it fires constantly, and on a cell whose IQR is 0.4 % it lets a real 4 % regression
through. The derivation:

    for a roughly normal sample, IQR = 1.349 * sigma          (Q3 - Q1 = 1.349 sigma)
    a one-sided 3-sigma alarm is therefore  median + 3 * IQR / 1.349
                                          = median + 2.224 * IQR

so ``ALARM_SIGMA = 3.0`` is the only free choice, it is stated as a false-alarm
rate (~0.1 % one-sided per cell per run) rather than as a percentage of anything,
and the per-cell number falls out of that cell's own spread. A cell too noisy to
discriminate the effect it is asked about says so by having a wide threshold --
which is the honest outcome, and the same rule §6.9 states for reporting.

**PROVENANCE IS NORMALIZED (O-11).** GPU model, driver, CUDA runtime, torch,
assembler, and the ratification digest the binary was gated against. **No hostname,
no username, no path, no account identifier** -- these files are published, and a
fleet hostname is not a fact about the kernel.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

BASELINE = Path(__file__).resolve().parents[1] / "cells" / "latency_baseline.json"

#: The one free choice. A one-sided 3-sigma alarm: ~0.1 % false alarm per cell per
#: run. Everything else about the threshold is that cell's measured spread.
ALARM_SIGMA = 3.0

#: Q3 - Q1 of a normal sample, in sigma. The constant that turns a measured IQR into
#: a sigma, which is what makes the threshold derived rather than chosen.
IQR_PER_SIGMA = 1.349


def threshold_ms(median_ms: float, iqr_ms: float) -> float:
    """The latency above which this cell is declared REGRESSED. Derived, see above."""
    return median_ms + ALARM_SIGMA * iqr_ms / IQR_PER_SIGMA


def provenance() -> dict:
    """Everything a reader needs to know WHERE a number came from, and nothing that
    identifies WHOSE machine it was."""
    from rola._build_config import BUILD_CONFIG

    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "sm": f"sm_{props.major}{props.minor}",
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "ptxas": BUILD_CONFIG["ptxas"],
        "manifest_sha256": BUILD_CONFIG["manifest_sha256"],
        "rola": BUILD_CONFIG["version"],
    }


def iqr(xs) -> float:
    """The same order statistic :mod:`bench.pairing` reports, so a baseline's spread
    and a run's spread are the same quantity."""
    xs = sorted(xs)
    n = len(xs)
    return xs[int(0.75 * n)] - xs[int(0.25 * n)]


def summarize(times_ms) -> dict:
    return {"median_ms": round(statistics.median(times_ms), 6),
            "iqr_ms": round(iqr(times_ms), 6),
            "reps": len(times_ms)}


def load_baseline() -> dict:
    if not BASELINE.exists():
        raise SystemExit(
            f"no committed latency baseline at {BASELINE}; record one with\n"
            f"    flock /tmp/rola_gpu.lock python benchmarks/bench_regression.py --write")
    return json.loads(BASELINE.read_text())


def write_baseline(arms: dict) -> None:
    """A baseline is a MEASUREMENT, recorded whole with its provenance.

    Never a partial update: half a baseline taken on one machine and half on another
    would compare two runs that were never comparable, and nothing in the file would
    say so.
    """
    BASELINE.write_text(json.dumps(
        {"provenance": provenance(), "alarm_sigma": ALARM_SIGMA, "arms": arms},
        indent=1, sort_keys=True) + "\n")
