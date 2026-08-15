"""The committed per-cell latency baseline of the CHUNK arm, and the gate that reads it.

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
    if dirty:
        raise SystemExit(
            "REFUSING to write a baseline from a DIRTY tree: the recorded number would "
            "describe edits that are not in any commit, and every later gate would "
            "compare against a tree nobody can check out. Commit first, then record.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance(), "alarm_sigma": ALARM_SIGMA, "sha": sha,
               "build_stamp": build_stamp, "sessions": sessions, "arms": arms}
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def compare(baseline: dict, subject: str, cell: str, measured: dict) -> dict:
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
    limit = threshold_ms(row["median_ms"], row["iqr_ms"])
    got = measured["median_ms"]
    return {"key": k, "verdict": "over_limit" if got > limit else "within_limit",
            "median_ms": got, "limit_ms": limit, "baseline_median_ms": row["median_ms"],
            "baseline_iqr_ms": row["iqr_ms"], "source": measured["source"]}
