"""The committed per-cell latency baseline of the CHUNK arm, and the gate that reads it.

The threshold's spread is `max(the recorded cross-session IQR, the instrument's measured
resolution)`. Both halves are measured; neither is chosen. See
:func:`instrument_resolution`.

`benchmarks/cells/chunk_latency_baseline.json`. A sibling of the tiled arm's PARKED
`latency_baseline.json` and never a replacement for it: that file's arms name a kernel
that no longer exists, and no chunk number may be compared against one.

**A BASELINE IS A MEASUREMENT, AND IT IS RECORDED ACROSS SESSIONS.** It is recorded
whole, on a clean tree, by :func:`write` -- never by a test, never partially. Each row's
median and IQR are taken over `sessions` SESSION MEDIANS, one per child process, because
the gate's comparison crosses two processes and a within-session block IQR does not bound
that gap: measured here, a single-session baseline put 17 of 44 arms over their limit on
the very next run of the same binary. The within-session spread is kept beside it as
`within_session_iqr_ms`, as the record of the contrast. The gate side (:func:`compare`)
READS ONLY.

The threshold is `bench.regression.threshold_ms`, derived from the recorded cell's own
IQR; the instrument check is `bench.regression.check_instrument`. Neither is re-stated
here, because a second copy of a rule is a rule that can disagree with itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.regression import ALARM_SIGMA, check_instrument, provenance, threshold_ms

BASELINE = Path(__file__).resolve().parents[1] / "cells" / "chunk_latency_baseline.json"

#: The identity a baseline row is keyed by. `subject|cell`, the same two names the ledger
#: row's `arm` carries, so a baseline row and a ledger row are the same question asked
#: once each.
def key(subject: str, cell: str) -> str:
    return f"{subject}|{cell}"


def load(path: Path = BASELINE) -> dict:
    if not path.exists():
        raise SystemExit(
            f"no committed chunk baseline at {path}; record one in a quiet locked "
            f"session with\n"
            f"    flock /tmp/rola_gpu.lock python benchmarks/bench_chunk.py baseline --write")
    return json.loads(path.read_text())


#: A baseline row must be built from at least this many SESSIONS. The gate compares two
#: processes, so the spread it is entitled to be judged against is the spread across
#: processes. MEASURED: a first single-session baseline put 17 of 44 arms over their limit
#: on the very next run of the same binary, because a within-session block IQR is often
#: 0.0000 ms and bounds nothing the comparison actually crosses.
MIN_SESSIONS = 2


def refuse_if_dirty(dirty: bool) -> None:
    """The dirty-tree refusal, callable BEFORE the sessions run as well as at the write.

    It fired once at the write, after seven sessions had already been measured -- a
    correct refusal delivered at the most expensive moment there is. The recorder captures
    the tree's identity before its first session and checks it there too; the check at the
    write stays, because the file must not be writable by any path that skipped it.
    """
    if dirty:
        raise SystemExit(
            "REFUSING to record a baseline from a DIRTY tree: the number would describe "
            "edits that are not in any commit, and every later gate would compare against "
            "a tree nobody can check out. Commit first, then record.")


def write(arms: dict, *, dirty: bool, sha: str, build_stamp: dict, sessions: int = 1,
          path: Path = BASELINE) -> None:
    """Record the whole baseline with its provenance. A DIRTY TREE IS REFUSED.

    The reason is the ledger's: a number taken over uncommitted edits describes a tree
    that no longer exists, so it can never be the thing later numbers are compared
    against.
    """
    if sessions < MIN_SESSIONS:
        raise SystemExit(
            f"REFUSING to write a baseline from {sessions} session(s): the gate compares "
            f"across PROCESSES, so the spread it is judged against has to be measured "
            f"across processes. A within-session block IQR is the spread of eleven reps on "
            f"one warm fixture, and it bounds nothing the comparison crosses -- measured, "
            f"17 of 44 arms over their limit on an unchanged binary. "
            f"Record with --sessions >= {MIN_SESSIONS}.")
    refuse_if_dirty(dirty)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance(), "alarm_sigma": ALARM_SIGMA, "sha": sha,
               "build_stamp": build_stamp, "sessions": sessions, "arms": arms}
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def instrument_resolution(values) -> float:
    """The smallest gap this stopwatch RESOLVES, measured from the run's own samples.

    A floor on the spread, and a derived one. The device-event timer quantizes: seven
    sessions of a small kernel can land on the same value every time, and the recorded IQR
    is then exactly 0.0 -- which makes `threshold_ms` a zero-width limit that the next tick
    of the same timer exceeds. MEASURED consequence: two arms over their limit on an
    unchanged binary, both of them rows whose cross-session IQR was 0.0000.

    A spread of zero is a statement about the instrument, not about the kernel, so the
    floor is the instrument's own tick -- the smallest nonzero gap between the distinct
    values the run actually produced. It is measured every run rather than stored, because
    it is a property of the stopwatch and the box and a stored copy could disagree with
    the samples beside it.

    **PER ARM, not pooled across the roster.** Pooling was tried first and is INERT: the
    minimum gap over every arm's samples together is whatever the finest-grained arm on the
    board happened to resolve (0.000032 ms measured here), which floors nothing for the
    coarse sub-10-microsecond arms that need it. The gap this timer resolves depends on the
    magnitude being timed, so it is measured at the magnitude in question.
    """
    distinct = sorted({round(float(v), 9) for v in values})
    gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
    if not gaps:
        raise ValueError("every sample in this run is identical; the instrument's "
                         "resolution cannot be measured from it, and neither can anything "
                         "else")
    return min(gaps)


def compare(baseline: dict, subject: str, cell: str, measured: dict,
            *, resolution_ms: float) -> dict:
    """One cell's measured median against its committed limit. READ ONLY.

    `measured` is a `bench.pairing.measure_arm` result. The verdict is `over_limit` or
    `within_limit` -- never "pass": this is an effect-size gate alone, and the flagging
    rule needs the significance and debounce gates behind it before anything is called a
    regression (`docs/measurement.md`).
    """
    k = key(subject, cell)
    arms = baseline["arms"]
    if k not in arms:
        raise SystemExit(
            f"{k} has no committed baseline row. A cell that is measured but not "
            f"recorded is a cell whose gate silently checks nothing; re-record the "
            f"baseline whole rather than adding one row to it.")
    row = arms[k]
    check_instrument(row, measured["source"], k)
    if row.get("sessions", 1) < MIN_SESSIONS:
        raise SystemExit(
            f"{k}'s committed baseline was recorded from {row.get('sessions', 1)} "
            f"session(s), so its spread cannot bound a cross-process comparison. "
            f"Re-record it: a baseline is a measurement.")
    spread = max(row["iqr_ms"], resolution_ms)
    limit = threshold_ms(row["median_ms"], spread)
    got = measured["median_ms"]
    return {"key": k, "verdict": "over_limit" if got > limit else "within_limit",
            "median_ms": got, "limit_ms": limit, "baseline_median_ms": row["median_ms"],
            "baseline_iqr_ms": row["iqr_ms"], "spread_ms": spread,
            "floored_at_resolution": spread > row["iqr_ms"],
            "resolution_ms": resolution_ms, "source": measured["source"]}
