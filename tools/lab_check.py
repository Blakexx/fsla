# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB'S CHECKS AS A GRAPH ACTION (bazel/lab.bzl's `lab_experiment`, docs/internals/tools/lab.md#graph): the SASS checker, the
toolkit notices' ledger and the experiment's declarations, over the files the experiment's target built -- what `lab run` holds an
experiment to before it measures. A refusal fails the build, in the lab's own words; a pass writes the verdict file.

    lab_check.py --built DIR --experiment NAME --arch sm_90a --verdict OUT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import lab  # noqa: E402 -- path insert must precede this import


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, required=True, help="the directory of the target's outputs")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--verdict", type=Path, required=True)
    a = ap.parse_args()
    built = lab.Built(a.built, a.experiment, a.arch)
    failures = lab.check(built, lab.kernels(built))
    if failures:
        print(f"lab: {a.experiment} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1
    a.verdict.write_text(f"{a.experiment} {a.arch}: the SASS is what it measures; no notice the ledger refuses\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
