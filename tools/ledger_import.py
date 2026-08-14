"""Import the markdown-era banked numbers into the ledger, ONCE, marked as legacy.

    python tools/ledger_import.py --write

Two sources, both already in the tree: the committed latency baseline
(`benchmarks/cells/latency_baseline.json`) and the curated docket numbers with their
citations (`perf/legacy_measurements.json`).

**EVERY IMPORTED ROW CARRIES `probe_version: 0`.** That field is inside the ledger
fingerprint, so a legacy row can never enter a series with a disciplined one. This
is the point of the import, not a caveat on it: those numbers have unknown warm-up,
unknown clock state, and -- for the docket rows -- were taken with a stopwatch that
was not in the tree. They are kept because they are the record, not because they are
comparable.

A legacy number that is a TARGET worth beating belongs in a threshold file as a
target, never in the ledger as a measurement of this tree.

Re-running is safe in the sense that nothing is corrupted, and wrong in the sense
that the same measurement would be recorded twice; `--write` therefore refuses once
legacy rows for a fingerprint are already present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench.ledger import LEDGER, LEGACY_VERSION, append, load, make_legacy_row  # noqa: E402

BASELINE = ROOT / "benchmarks" / "cells" / "latency_baseline.json"
CURATED = ROOT / "perf" / "legacy_measurements.json"


def rows() -> list[dict]:
    """Every legacy row, from both sources.

    `ts`, `csrc_sha256` and the run identity are NULL rather than invented, and
    `dirty` is true: an imported number describes a tree nobody can name exactly, so
    it is advisory by construction and `bench.ledger.baseline_series` drops it.
    """
    out = []
    baseline = json.loads(BASELINE.read_text())
    prov = baseline["provenance"]
    for arm, entry in sorted(baseline["arms"].items()):
        out.append(make_legacy_row(
            probe="latency_baseline", arm=arm, source=entry["source"],
            metric="duration", unit="ms", values=[entry["median_ms"]],
            note=f"committed latency baseline, {entry['reps']} reps, "
                 f"IQR {entry['iqr_ms']} ms; benchmarks/cells/latency_baseline.json",
            env=prov))
    curated = json.loads(CURATED.read_text())
    for m in curated["measurements"]:
        out.append(make_legacy_row(
            probe=m["probe"], arm=m["arm"], source=m["source"], metric=m["metric"],
            unit=m["unit"], values=m["values"], note=m["note"], env=m["env"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="append the legacy rows to perf/ledger.jsonl")
    a = ap.parse_args()

    new = rows()
    present = {r["fingerprint"] for r in load() if r["probe_version"] == LEGACY_VERSION}
    clash = sorted({r["fingerprint"] for r in new} & present)
    for r in new:
        print(f"  {r['probe']:20s} {r['arm']:36s} {r['source']:13s} "
              f"{r['values']} {r['unit']}")
    if not a.write:
        print(f"\n{len(new)} legacy row(s); pass --write to append them to {LEDGER}")
        return 0
    if clash:
        raise SystemExit(
            f"{len(clash)} of these fingerprints already have legacy rows in "
            f"{LEDGER}. Importing again would record one measurement twice; the "
            f"import is a one-time act.")
    append(new)
    print(f"\n{len(new)} legacy row(s) appended to {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
