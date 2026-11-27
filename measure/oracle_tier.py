# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ORACLE TIER AS AN INSTRUMENT (`//measure:oracle_tier`, recorded by `bazel run //measure:store_oracle_tier`): the
oracle battery run whole on the device, its output how many cases passed, failed, errored and skipped, WHICH failed (a
red case is a stored fact, not a number re-derived by hand), every per-slot comparison's margins (`--oracle-margins`, so
the clauses' headroom is on the record) and the worst margin per output kind.

    python measure/oracle_tier.py --json tier.json      # in the instrument's tree (fsla_devtools.instrument)

A host with no CUDA device refuses (`{"refused": ...}`); a pytest that writes no report fails. `pytest -k` stays the dev
loop and `bazel test //tests:oracle` the gate; this is the tier's record.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from fsla_devtools.process import run
from fsla_devtools.results import portable

ROOT = Path(__file__).resolve().parents[1]
#: a whole tier is minutes; a hang costs this and fails the run
TIMEOUT_S = 3600
#: the device probe's exit when torch sees no CUDA device
NO_DEVICE = 3


def cases(junit: Path) -> tuple[dict, list[dict]]:
    """The report's counts and each failed or errored case: its id, verdict and the first line of its message."""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failed = []
    for suite in ET.parse(junit).getroot().iter("testsuite"):
        for k in counts:
            counts[k] += int(suite.get(k, 0))
        for case in suite.iter("testcase"):
            verdict = next((child for child in case if child.tag in ("failure", "error")), None)
            if verdict is not None:
                failed.append({"id": f"{case.get('classname')}::{case.get('name')}", "verdict": verdict.tag,
                               "message": portable((verdict.get("message") or "")[:300], ROOT)})
    return counts, failed


def tier() -> dict:
    with tempfile.TemporaryDirectory(prefix="oracle-tier-") as scratch:
        junit, margins = Path(scratch) / "junit.xml", Path(scratch) / "margins.jsonl"
        done = run([sys.executable, "-m", "pytest", "tests/oracle", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
                    f"--oracle-margins={margins}"], cwd=ROOT, timeout=TIMEOUT_S)
        if not junit.exists():
            raise SystemExit(f"pytest exited {done.returncode} without a report:\n"
                             f"{portable((done.stdout + done.stderr)[-2000:], ROOT)}")
        counts, failed = cases(junit)
        rows = [json.loads(line) for line in margins.read_text().splitlines()] if margins.exists() else []
    worst: dict[str, float] = {}
    for row in rows:
        if row.get("worst") is not None:
            worst[row["output"]] = max(worst.get(row["output"], 0.0), row["worst"])
    return {"counts": counts, "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
            "failed": failed, "margins": rows, "worst_margin": worst, "exit": done.returncode}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    #: asked in a process of its own, so this one holds no CUDA context while the tier runs
    asked = f"import sys, torch; sys.exit(0 if torch.cuda.is_available() else {NO_DEVICE})"
    probe = subprocess.run([sys.executable, "-c", asked], check=False)
    if probe.returncode not in (0, NO_DEVICE):
        raise SystemExit(f"oracle tier: the device probe exited {probe.returncode}")
    out = tier() if probe.returncode == 0 else {"refused": "no CUDA device is visible: the oracle tier runs on one"}
    a.json.write_text(json.dumps(out, sort_keys=True) + "\n")
    if "refused" not in out:
        print(f"oracle tier: {out['passed']} passed of {out['counts']['tests']}, {len(out['failed'])} failed, "
              f"{len(out['margins'])} comparisons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
