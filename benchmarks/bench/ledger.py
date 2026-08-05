"""WHERE a number is kept: the append-only measurement ledger.

`perf/ledger.jsonl`, one JSON object per line, one line per measurement of one arm.
The full schema and the reasoning behind each field are in `docs/measurement.md`;
what this module is responsible for is that no field can be silently omitted and
that nothing but a harness ever writes a line.

Three properties are the reason this file exists rather than a markdown table:

* **RAW SAMPLES ARE STORED, DERIVED SCALARS ARE NOT.** `values` is the per-rep list
  in RUN ORDER, because the order carries the drift signal and because a median
  stored beside its samples is a second copy that can disagree with them. Medians
  and IQRs are computed on read.
* **THE FINGERPRINT CARRIES THE HARDWARE AND THE PROBE VERSION.** Two numbers
  compare only if they answer the same question on the same box under the same
  assembler. That is a hash, not a convention, so a comparison across an arch or
  across a probe redefinition is impossible rather than discouraged.
* **THE STOPWATCH IS A FIELD.** `source` names the instrument
  (:mod:`bench.pairing`); the regression rule filters on it, and legacy rows carry
  `probe_version: 0` so they can never enter a comparison with disciplined ones.

`perf/ledger.jsonl merge=union` in `.gitattributes` is what makes append-only
viable: every conflict between two branches that both appended is resolved by
keeping both lines.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "perf" / "ledger.jsonl"

#: Bump only for a change that makes OLD LINES UNREADABLE. Adding a field does not
#: qualify; readers ignore what they do not know.
SCHEMA = 1

#: The fields no row may omit. A row missing one of these is not a weaker
#: measurement, it is an unusable one -- the writer refuses rather than the reader
#: guessing.
REQUIRED = (
    "schema", "ts", "sha", "dirty", "csrc_sha256", "probe", "probe_version",
    "arm", "fingerprint", "env", "source", "metric", "unit", "values", "tier",
    "run_id",
)

#: Instruments whose numbers are wall-clock durations of a launch. The regression
#: rule applies to these; `ncu` counters and `legacy` imports get their own
#: treatment, which is why the distinction is data rather than a comment.
TIMING_SOURCES = ("cuda_events", "perf_counter")


def git_state() -> tuple[str, bool]:
    """`(sha, dirty)` of the tree being measured.

    A dirty tree does not block a measurement -- most measurements are taken while
    working -- but it marks the row ADVISORY: `baseline_series` drops dirty rows, so
    a number taken over uncommitted edits can never become the thing later numbers
    are compared against.
    """
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    porcelain = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout
    return sha, bool(porcelain.strip())


def fingerprint(*, probe: str, probe_version: int, arm: str, env: dict) -> str:
    """The identity a series is keyed by: the question, its version, and the box.

    `sm`, the driver's CUDA version and the assembler are IN it from day one. They
    cost nothing now and they are what stops an sm_86 number ever being compared to
    an sm_90 one -- a mistake that is unrecoverable after the fact, because the rows
    would already be indistinguishable.
    """
    material = json.dumps({
        "probe": probe, "probe_version": probe_version, "arm": arm,
        "gpu": env.get("gpu"), "sm": env.get("sm"),
        "driver_cuda": env.get("driver_cuda"), "torch": env.get("torch"),
        "ptxas": env.get("ptxas"),
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def make_row(*, probe: str, probe_version: int, arm: str, env: dict, source: str,
             metric: str, unit: str, values, tier: str, run_id: str,
             csrc_sha256: str, sha: str, dirty: bool, **extra) -> dict:
    """Build one row, refusing anything the reader would have to guess at."""
    row = {
        "schema": SCHEMA,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha": sha,
        "dirty": dirty,
        "csrc_sha256": csrc_sha256,
        "probe": probe,
        "probe_version": probe_version,
        "arm": arm,
        "fingerprint": fingerprint(probe=probe, probe_version=probe_version,
                                   arm=arm, env=env),
        "env": env,
        "source": source,
        "metric": metric,
        "unit": unit,
        "values": [float(v) for v in values],
        "tier": tier,
        "run_id": run_id,
    }
    row.update(extra)
    missing = [k for k in REQUIRED if row.get(k) is None]
    if missing:
        raise ValueError(f"ledger row is missing required field(s): {missing}")
    if not row["values"]:
        raise ValueError("a ledger row with no samples records nothing")
    return row


#: The probe version that means "this number predates the harness". It is not a
#: smaller version of the current one; it is a separate namespace, and because it is
#: inside the fingerprint an imported number can never enter a disciplined series.
LEGACY_VERSION = 0


def make_legacy_row(*, probe: str, arm: str, source: str, metric: str, unit: str,
                    values, note: str, env: dict) -> dict:
    """A banked markdown-era number, recorded as what it is.

    The fields a live row is REQUIRED to carry are NULL here rather than invented --
    there is no honest timestamp, no kernel digest and no tree sha for a number whose
    provenance is a line in a document -- and `dirty` is true, so
    :func:`baseline_series` drops it. The row shape is defined once, beside
    :func:`make_row`, so the two cannot drift into two schemas.
    """
    return {
        "schema": SCHEMA,
        "ts": None,
        "sha": None,
        "dirty": True,
        "csrc_sha256": None,
        "probe": probe,
        "probe_version": LEGACY_VERSION,
        "arm": arm,
        "fingerprint": fingerprint(probe=probe, probe_version=LEGACY_VERSION,
                                   arm=arm, env=env),
        "env": dict(env),
        "source": source,
        "metric": metric,
        "unit": unit,
        "values": [float(v) for v in values],
        "tier": "legacy",
        "run_id": "legacy-import",
        "note": note,
    }


def append(rows, path: Path = LEDGER) -> int:
    """Append rows. Never rewrites, never reorders, never de-duplicates.

    A measurement taken twice is two measurements; collapsing them would delete the
    evidence that the box reproduces.
    """
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def load(path: Path = LEDGER) -> list[dict]:
    """Every row, in file order. Missing file reads as empty, a blank line is skipped,
    and a row from a schema this reader does not know is REFUSED rather than
    partially interpreted."""
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != SCHEMA:
            raise ValueError(f"{path}:{n} is schema {row.get('schema')!r}; this "
                             f"reader knows schema {SCHEMA}")
        rows.append(row)
    return rows


def baseline_series(rows, fp: str, *, source: str, window: int = 20) -> list[dict]:
    """The last `window` CLEAN rows on one fingerprint and one instrument, oldest
    first -- the rows a rolling baseline is entitled to be built from."""
    same = [r for r in rows
            if r["fingerprint"] == fp and r["source"] == source and not r["dirty"]]
    return same[-window:]


def median_of(row: dict) -> float:
    import statistics

    return statistics.median(row["values"])
