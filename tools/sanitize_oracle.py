#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K39 -- the compute-sanitizer NAMED gate on the carry/intra/decode oracle cells.

`docs/testing.md`'s "sanitizer lane" section records the shape (a SCHEDULED lane
over a named subset, never a PR gate -- the tools cost one to two orders of
magnitude -- `--error-exitcode=1`, a committed skip-list whose entries carry a
reason) and its one measurement so far: memcheck/initcheck/synccheck/racecheck all
clean on the TILED consumer's two registered probes. It also states the gap this
script closes: "no such measurement exists for the chunk arm yet."

This is that measurement, promoted from an ad-hoc invocation (recorded once, by
hand, in FINDINGS `## PROBE-K35` under "memcheck on paging") to a script anyone
can run and CI can schedule: `racecheck`/`synccheck`/`initcheck` (memcheck already
ran, ad hoc, on paging -- K39 card) against ONE named oracle cell per family
(carry, intra via the combined prefill operator, decode), each run as its own
`compute-sanitizer`-wrapped pytest process so a hang or an error in one tool
cannot block another, under the GPU lock every campaign shares.

Usage:
    flock /tmp/rola_gpu.lock python tools/sanitize_oracle.py [--tool TOOL ...]
        [--family FAMILY ...] [--timeout SECONDS]

Exit code is 0 iff every (family, tool) pair CLEAN. A per-pair timeout (default
600s) turns a hang into a NAMED failure rather than a stuck lane holding the GPU
lock forever (KERNEL_STANDARDS §14: "a hang here blocks the whole flock queue for
every other agent").
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: one cell per family, chosen for being ALREADY a named, gated test -- this script
#: adds no new cell, it reruns an existing one under the sanitizer. `-k` narrows to
#: exactly one parametrization so each (family, tool) pair is one launch, not a
#: whole file's worth serialized under the heaviest instrumentation this repo has.
_FAMILY_TESTS = {
    "carry": "tests/k31/test_carry_vs_oracle.py::test_m3_two_sided_alt_small",
    "intra": "tests/k31/test_prefill_vs_oracle.py::test_the_intra_term_is_actually_present",
    "decode": "tests/oracle/test_decode_vs_oracle.py::test_g1_is_byte_reproducible_against_itself",
}

#: memcheck is NOT re-run here: it already ran (ad hoc) on paging and is recorded
#: in FINDINGS; the K39 card's own text is "memcheck already runs on paging" --
#: this script's job is the three tools that never had ANY committed invocation.
_TOOLS = ("racecheck", "synccheck", "initcheck")

#: entries here are (family, tool) pairs with a recorded reason, mirroring
#: `docs/testing.md`'s "committed suppressions file, and an explicit skip-list
#: whose entries carry their reason" -- empty by design (docs/testing.md: "An
#: empty skip-list is the standard the successor lane inherits").
SKIP: dict[tuple[str, str], str] = {}


def run_one(python: str, family: str, tool: str, timeout: int) -> tuple[bool, str]:
    test_id = _FAMILY_TESTS[family]
    cmd = [
        "compute-sanitizer", f"--tool={tool}", "--error-exitcode=1",
        python, "-m", "pytest", test_id, "-x", "-q",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s -- a hang under {tool}, not a correctness verdict"
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-30:] + proc.stderr.splitlines()[-30:])
    return ok, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", action="append", choices=_TOOLS, default=None)
    ap.add_argument("--family", action="append", choices=tuple(_FAMILY_TESTS), default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tools = args.tool or list(_TOOLS)
    families = args.family or list(_FAMILY_TESTS)

    failures = []
    for family in families:
        for tool in tools:
            reason = SKIP.get((family, tool))
            if reason is not None:
                print(f"SKIP {family}/{tool}: {reason}")
                continue
            print(f"RUN  {family}/{tool} ({_FAMILY_TESTS[family]}) ...", flush=True)
            ok, tail = run_one(args.python, family, tool, args.timeout)
            status = "CLEAN" if ok else "FAIL"
            print(f"{status} {family}/{tool}")
            if not ok:
                print(tail)
                failures.append((family, tool))

    if failures:
        print(f"\n{len(failures)} sanitizer failure(s): {failures}")
        return 1
    print("\nall requested (family, tool) pairs CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
