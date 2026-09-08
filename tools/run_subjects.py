#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RUN THE SUBJECTS OF ONE TREE: several one-thing benches on the same cells, one
interleaved clock-locked run, printed as a table, recorded durably by the prober.

    python tools/run_subjects.py --cells nl64k-alt-k4,nl64k-dense \\
        --worktree WORKTREE --venv VENV \\
        --subjects prefill_op,carry_forward:sparse-g32,intra_forward,attention_reference \\
        [--ratio-to attention_reference]

The primitives are two: a SUBJECT measures one thing (`benchmarks/bench/subjects.py`), and
the PROBER (`tools/probe_cells.py`) runs lanes -- a binary and a subject each -- correctly:
interleaved with the first position alternating by round, under the host's clock lock with
every row's clock read and refused if off it, recorded with the tree's sha and digest.
This tool is the convenience over it for one tree: a lane per subject (the subject's
schedule after a colon where it takes one), the prober's rows as a table, and a ratio
column only when one is asked for. Across commits the same runner takes lanes on several
worktrees, or the records are read by sha; a comparison is never a bench of its own.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--subjects", required=True,
                    help="comma-separated subjects, each optionally SUBJECT:SCHEDULE")
    ap.add_argument("--ratio-to", default=None, help="a subject to print each row's ratio to")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--state", default="fresh")
    args = ap.parse_args()

    subjects = [s.split(":", 1) for s in args.subjects.split(",")]
    labels = [s[0] if len(s) == 1 else f"{s[0]}:{s[1]}" for s in subjects]
    if args.ratio_to and args.ratio_to not in [s[0] for s in subjects]:
        ap.error(f"{args.ratio_to!r} is not among the subjects")
    cmd = [sys.executable, str(HERE / "probe_cells.py"), "--bench", subjects[0][0],
           "--cells", args.cells, "--rounds", str(args.rounds), "--reps", str(args.reps),
           "--state", args.state]
    for (name, *sched), label in zip(subjects, labels):
        lane = f"worktree:{args.worktree},venv:{args.venv},label:{label},bench:{name}"
        if sched:
            lane += f",schedule:{sched[0]}"
        cmd += ["--binary", lane]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        return proc.returncode

    #: the prober's rows: label, cell, ms; the table is cells by subjects with the ratio.
    rows = {}
    for line in proc.stdout.splitlines():
        for lb in labels:
            if line.startswith(lb):
                m = re.match(r"\s*(\S+)\s+([0-9.]+)\s", line[len(lb):])
                if m:
                    rows[(m.group(1), lb)] = float(m.group(2))
    cells = args.cells.split(",")
    ref_label = (next(lb for (n, *_), lb in zip(subjects, labels) if n == args.ratio_to)
                 if args.ratio_to else None)
    width = max(len(lb) for lb in labels) + 2
    print(f"{'cell':<28}" + "".join(f"{lb:>{width + 8}}" for lb in labels))
    for cell in cells:
        ref = rows.get((cell, ref_label))
        out = f"{cell:<28}"
        for lb in labels:
            ms = rows.get((cell, lb))
            if ms is None:
                out += f"{'refused':>{width + 8}}"
            else:
                ratio = f" ({ms / ref:.2f}x)" if ref else ""
                out += f"{f'{ms:.3f}ms{ratio}':>{width + 8}}"
        print(out)
    for line in proc.stdout.splitlines():
        if line.startswith("CLOCK") or "UNLOCKED" in line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
