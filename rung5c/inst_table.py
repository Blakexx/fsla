"""The ablation ladder, read out of the ncu captures.

    python inst_table.py <outdir> [regime]

One row per (arm, stage): the exact counters, `inst/mma`, the per-pipe split, the
duration and the two utilisations -- and, beside each, the DELTA AGAINST THE
STAGE BELOW IT, which is the mechanism that stage admits.
"""
from __future__ import annotations

import csv
import pathlib
import sys

ARMS = ["16_0_128_64_64", "32_0_128_64_64", "64_0_128_64_64",
        "32_0_64_64_64", "64_0_64_64_64", "32_0_32_64_64"]
STAGES = [("r4", "5A  baseline"), ("a1", "+1 gathered addressing"),
          ("a2", "+2 staircase mask"), ("a3", "+3 per-side gates"),
          ("a4", "+4 the cook  (= 5B)"),
          ("a5", "+5 schedule window")]
KEYS = {
    "smsp__inst_executed.sum": "inst",
    "smsp__inst_executed_pipe_tensor_op_hmma.sum": "mma",
    "smsp__inst_executed_pipe_alu.sum": "alu",
    "smsp__inst_executed_pipe_fma.sum": "fma",
    "smsp__inst_executed_pipe_lsu.sum": "lsu",
    "smsp__inst_executed_pipe_cbu.sum": "cbu",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue",
    "gpu__time_duration.sum": "us",
}


def read(path):
    rows = {}
    #: THE CAPTURE CARRIES THE SCRIPT'S OWN STDOUT AHEAD OF THE TABLE -- the
    #: build identity and the partition, which is why they are printed there.
    #: The table starts at its header row and nothing before it is csv.
    with open(path) as f:
        raw = f.read().splitlines(keepends=True)
    start = next((i for i, ln in enumerate(raw) if ln.startswith('"ID"')), None)
    if start is None:
        return {}
    lines = raw[start:]
    for r in csv.DictReader(lines):
        nm, val = r.get("Metric Name"), r.get("Metric Value")
        if nm in KEYS and val is not None:
            rows[KEYS[nm]] = float(val.replace(",", ""))
    if "us" in rows and rows["us"] > 1e4:      # nsecond capture
        rows["us"] /= 1e3
    return rows


def main():
    out = pathlib.Path(sys.argv[1])
    reg = sys.argv[2] if len(sys.argv) > 2 else "dense"
    for arm in ARMS:
        c, w, bc, w0, w1 = arm.split("_")
        print(f"\n=== C={c} BC={bc} N={int(w0)*int(w1)}  regime={reg}   "
              f"(B=1 H=2 L=4096)")
        print(f"{'stage':24s}{'inst':>12s}{'mma':>10s}{'i/mma':>8s}"
              f"{'ALU':>10s}{'FMA':>10s}{'LSU':>10s}{'CBU':>10s}"
              f"{'tensor%':>9s}{'issue%':>8s}{'us':>9s}")
        prev = None
        for tag, label in STAGES:
            p = out / f"{tag}{arm}_{reg}.csv"
            if not p.exists():
                print(f"{label:24s}  MISSING")
                continue
            r = read(p)
            if "inst" not in r:
                print(f"{label:24s}  EMPTY CAPTURE")
                continue
            print(f"{label:24s}{r['inst']:12,.0f}{r['mma']:10,.0f}"
                  f"{r['inst']/r['mma']:8.2f}{r['alu']:10,.0f}{r['fma']:10,.0f}"
                  f"{r['lsu']:10,.0f}{r['cbu']:10,.0f}"
                  f"{r['tensor']:9.2f}{r['issue']:8.2f}{r['us']:9.1f}")
            if prev is not None:
                d = {k: r[k] - prev[k] for k in ("inst", "mma", "alu", "fma",
                                                 "lsu", "cbu", "us")}
                print(f"{'    delta':24s}{d['inst']:+12,.0f}{d['mma']:+10,.0f}"
                      f"{r['inst']/r['mma'] - prev['inst']/prev['mma']:+8.2f}"
                      f"{d['alu']:+10,.0f}{d['fma']:+10,.0f}{d['lsu']:+10,.0f}"
                      f"{d['cbu']:+10,.0f}{'':>9s}{'':>8s}{d['us']:+9.1f}")
            prev = r


if __name__ == "__main__":
    main()
