#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMPOSER (KERNEL_STANDARDS §22 (9)): a phase's time attributed to its parts by composition.

    python tools/compose_ledger.py --cells flagship-dense,nl64k-alt-k4 \\
        --ladder readout:stream,loads,drain --ladder fold:pool,ring,loads

The carry kernel's MMA phases are built from PARTS (`gen_shards.CARRY_PARTS`); a build may stub
any of them (`ROLA_CARRY_PARTS`): a stub keeps the part's HMMAs -- the same count, atom and
accumulators -- and drops its other work. A LADDER over a phase builds the kernel with none of
that phase's parts real (every other part real), then adds the parts back in the order named, one
rung a build; the last rung is the kernel. Each rung is read in wall time by the phase ledger (a
warm-up launch, then `--launches`), so a part's cost is the step between two rungs -- no constant,
no reading of samples. Each rung is CHECKED: the profiler's HMMA count for the cell must equal the
kernel's (a stub that changed the workload fails its rung), and the build's device code must differ
from the kernel's (a build that did not take the mask fails). The kernel is rebuilt with every part
real at the end and its device code must hash to the value taken before the first rung.
Report: `benchmarks/results/ledgers/<utc>-<sha7>-compose.{json,md}`. Serial builds (the host's
memory watchdog kills parallel ones); measured work takes the GPU lock through the tools it drives.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dev_config  # noqa: E402
import gen_shards  # noqa: E402

PY = sys.executable
NCU = dev_config.get("toolchain.ncu")
CUOBJDUMP = dev_config.cuda_bin("cuobjdump")
NVDISASM = dev_config.cuda_bin("nvdisasm")
LEDGERS = ROOT / "benchmarks" / "results" / "ledgers"


def run(cmd: list[str], env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def extension_so() -> Path:
    return next((ROOT / "rola").glob("_C.*.so"))


def device_hash(arch: str = "sm_86") -> str:
    """The carry arm's device INSTRUCTIONS, disassembled, hashed: a fact about the installed
    binary that no path or timestamp can fake."""
    with tempfile.TemporaryDirectory(prefix="compose_") as tmp:
        subprocess.run([CUOBJDUMP, "-xelf", "all", str(extension_so())], cwd=tmp, capture_output=True, check=True)
        cub = next(p for p in Path(tmp).iterdir() if p.name.startswith("carry_arm") and arch in p.name)
        sass = subprocess.run([NVDISASM, "-gi", str(cub)], capture_output=True, text=True, check=True).stdout
    #: instructions only: the debug line table moves with every source line added above a site
    #: and is not code (a byte-identical kernel differed there alone, 2026-09-12).
    text = "\n".join(re.sub(r"//.*$", "", l).rstrip() for l in sass.splitlines()
                     if re.match(r"\s*/\*[0-9a-f]+\*/\s+[A-Z@]", l))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build(real: tuple[str, ...]) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("ROLA_BUILD_PARTS", None)
    env["ROLA_CARRY_PARTS"] = "all" if set(real) == set(gen_shards.CARRY_PARTS) else (",".join(real) or "none")
    return run([PY, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "--no-deps", "-q"], env=env, timeout=3600)


PHASE_NAMES = ("head", "readout", "fold", "snapshot", "edges", "sweep", "head_words", "head_scans")

#: THE RESOURCE COUNTERS a rung records (launch totals; a part's step in each is its resource cost).
COUNTERS = (
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "smsp__inst_executed.sum",
    "sm__cycles_elapsed.avg", "smsp__cycles_active.avg",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_active.avg.per_cycle_active", "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__sass_inst_executed_op_shared_ld.sum", "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_ldsm.sum", "smsp__sass_inst_executed_op_global_red.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum", "lts__t_sectors_op_red.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
)
COMPONENTS = {"FoldStream", "ReadoutStream", "head", "snapshot_publish", "pool_fill"}


def worker(cell: str) -> list[str]:
    return [PY, "tools/probe_cells.py", "--worker", "--bench", "carry_forward", "--reps", "0", "--warmup", "0",
            "--state", "fresh", "--oneshot", "--schedule", "first", "--cells", cell]


def cta_windows(cell: str) -> int:
    rc, out = run([PY, "-c", "import sys; sys.path.insert(0, 'tools'); import build_ledger as B; "
                   f"print(B.cta_windows({cell!r}))"], timeout=600)
    return int(out.strip().splitlines()[-1])


def phases(cell: str, launches: int) -> dict:
    """Cycles a warp a window per phase (warm-up + `launches`), and every phase's per-warp row."""
    rc, out = run([PY, "tools/phase_ledger.py", cell, "--launches", str(launches), "--warmup", "1",
                   "--per-warp", ",".join(PHASE_NAMES)], timeout=3600)
    m = re.search(r"cycles a warp a window.*?: (.*?)  total ([\d.]+)", out)
    if not m:
        return {"rc": rc, "raw": out[-800:]}
    row = {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", m.group(1))}
    row["total"] = float(m.group(2))
    row["per_warp"] = {g.group(1): [int(x) for x in g.group(2).split()]
                       for g in re.finditer(r"^\s+(\w+)\s+a warp:\s+([\d\s]+)$", out, re.M)}
    return row


def counters(cell: str) -> dict:
    """The rung's resource counters for one launch of the cell, per CTA-window where a sum."""
    rc, out = run([NCU, "--target-processes", "all", "-k", "regex:carry_kernel", "-c", "1",
                   "--metrics", ",".join(COUNTERS), "--csv", *worker(cell)], timeout=3600)
    per = cta_windows(cell)
    vals = {}
    for line in out.splitlines():
        parts = [x.strip().strip('"') for x in line.split('","')]
        if len(parts) > 12 and parts[0].isdigit() and parts[-3] in COUNTERS:
            v = float(parts[-1].replace(",", ""))
            vals[parts[-3]] = v / per if parts[-3].endswith(".sum") else v
    vals["hmma_launch"] = vals.get("sm__inst_executed_pipe_tensor_op_hmma.sum", 0.0) * per
    return vals


def census(cell: str, total_cycles: float | None) -> dict:
    """The rung's stall census: every warp-stall sample of one launch by reason, and by component and
    reason, as cycles a warp a window (the sample share times the phase ledger's total). A part's step
    in a reason is how its un-overlapped time stalls."""
    sys.path.insert(0, str(ROOT / "tools"))
    import region_ledger as R  # noqa: E402
    with tempfile.TemporaryDirectory(prefix="compose_census_") as tmp:
        rep = Path(tmp) / "rep"
        rc, out = run([NCU, "--target-processes", "all", "-k", "regex:carry_kernel", "-c", "1", "--section",
                       "SourceCounters", "--metrics", "group:smsp__pcsamp_warp_stall_reasons", "--import-source",
                       "yes", "-f", "-o", str(rep), *worker(cell)], timeout=3600)
        rc2, out2 = run([NCU, "--import", f"{rep}.ncu-rep", "--page", "source", "--print-source", "sass", "--csv"],
                        timeout=1800)
        if rc2:
            return {"rc": rc2, "raw": (out + out2)[-600:]}
        csvp = Path(tmp) / "src.csv"
        csvp.write_text(out2)
        rows = R.read_source_counters(csvp)
    so = extension_so()
    attr = R.Attribution(so, "carry_arm", ROOT / "csrc/rola/src/carry/carry_kernel.cuh",
                         ROOT / "csrc/rola/src/common/ops.cuh")
    base = rows[0][0]
    by_reason: dict[str, int] = {}
    by_comp: dict[str, dict[str, int]] = {}
    total = 0
    for addr, _n, sm, st in rows:
        fn, spin = attr.of(addr - base, COMPONENTS)
        fn = fn + " [spin]" if spin else fn
        for k, v in st.items():
            by_reason[k] = by_reason.get(k, 0) + v
            by_comp.setdefault(fn, {})[k] = by_comp.setdefault(fn, {}).get(k, 0) + v
        total += sm
    scale = (total_cycles or 0.0) / max(1, total)
    return {"samples": total,
            "by_reason": {k: v * scale for k, v in sorted(by_reason.items(), key=lambda x: -x[1])},
            "by_component": {c: {k: v * scale for k, v in sorted(d.items(), key=lambda x: -x[1])}
                             for c, d in by_comp.items()}}


def registers() -> dict:
    """Peak live registers of the rung's arm (a -lineinfo compile against the rung's parts header)."""
    with tempfile.TemporaryDirectory(prefix="compose_reg_") as tmp:
        cub = Path(tmp) / "arm0.cubin"
        rc, out = run([dev_config.cuda_bin("nvcc"), "-O3", "-std=c++17", "-arch=sm_86", "--expt-relaxed-constexpr",
                       "-lineinfo", "-DROLA_CARRY_BUILD_0", "-I", "csrc/rola/src", "-I", "build/generated", "-cubin",
                       "csrc/rola/src/instantiations/carry_arm_0.cu", "-o", str(cub)], timeout=1800)
        if rc:
            return {"rc": rc, "raw": out[-600:]}
        rc, out = run([PY, "tools/life_ranges.py", "--cubin", str(cub), "--source",
                       "csrc/rola/src/carry/carry_kernel.cuh", "--top", "8"], timeout=900)
    peak = re.search(r"peak live registers (\d+)", out)
    return {"peak_live": int(peak.group(1)) if peak else None,
            "by_region": {k: int(v) for k, v in re.findall(r"^(\S+)\s+(\d+)\s+[\d.]+\s+\d+$", out, re.M)}}


def timeline(cell: str) -> dict:
    """The rung's pipe timeline summary (`tools/pipe_timeline.py`, stored scale)."""
    import pipe_timeline as T
    with tempfile.TemporaryDirectory(prefix="compose_tl_") as tmp:
        out = Path(tmp) / "tl"
        rep = T.capture(cell, out)
        tl = T.parse(*T.export(rep))
    summ = T.summary(tl, T.stored_scale())
    return {"summary": summ, "tensor_5us": T.bins(tl["series"].get("tensor", []), 5.0, tl["duration_us"])}


TIMELINE = False


def measure(cells: list[str], launches: int) -> dict:
    """Everything a rung records: per cell the phases (with per-warp rows), the resource counters, the stall
    census and (with --timeline) the pipe timeline; once, the register life ranges."""
    out = {"phases": {}, "counters": {}, "census": {}, "timeline": {}}
    for c in cells:
        out["phases"][c] = phases(c, launches)
        out["counters"][c] = counters(c)
        out["census"][c] = census(c, out["phases"][c].get("total"))
        if TIMELINE:
            out["timeline"][c] = timeline(c)
    out["registers"] = registers()
    out["hmma"] = {c: int(out["counters"][c].get("hmma_launch", 0)) for c in cells}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="flagship-dense,nl64k-alt-k4")
    ap.add_argument("--ladder", action="append", required=True,
                    help="PHASE:part,part,... -- the phase's parts in the order they are added back")
    ap.add_argument("--launches", type=int, default=2)
    ap.add_argument("--timeline", action="store_true", help="capture every rung's pipe timeline (PM sampling)")
    a = ap.parse_args()
    global TIMELINE
    TIMELINE = a.timeline
    cells = [c for c in a.cells.split(",") if c]
    everything = gen_shards.CARRY_PARTS

    ladders = []
    for spec in a.ladder:
        phase, names = spec.split(":", 1)
        order = [f"{phase}.{n}" for n in names.split(",") if n]
        bad = sorted(set(order) - set(everything))
        if bad or len(set(order)) != len([x for x in everything if x.startswith(phase + ".")]):
            ap.error(f"--ladder {spec}: name every part of {phase} once ({[x for x in everything if x.startswith(phase + '.')]})")
        others = tuple(x for x in everything if not x.startswith(phase + "."))
        rungs = [others + tuple(order[:i]) for i in range(len(order) + 1)]
        ladders.append((phase, order, rungs))

    report = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ"),
              "sha": run(["git", "rev-parse", "--short=7", "HEAD"])[1].strip(), "cells": cells,
              "launches": a.launches, "ladders": [], "kernel": {}}

    rc, out = build(everything)
    if rc:
        print(out[-2000:])
        return 1
    ref = device_hash()
    report["kernel"]["hash"] = ref
    report["kernel"].update(measure(cells, a.launches))
    print(f"kernel {ref}: " + "; ".join(f"{c} {report['kernel']['phases'][c].get('total')}" for c in cells), flush=True)

    for phase, order, rungs in ladders:
        rows = []
        for real in rungs[:-1]:
            rc, out = build(real)
            row = {"real": sorted(real), "stubbed": sorted(set(everything) - set(real)), "build_rc": rc}
            if rc:
                row["raw"] = out[-1500:]
                rows.append(row)
                continue
            row["hash"] = device_hash()
            row["mask_took"] = row["hash"] != ref
            row.update(measure(cells, a.launches))
            row["hmma_ok"] = all(row["hmma"][c] == report["kernel"]["hmma"][c] for c in cells)
            rows.append(row)
            print(f"{phase} rung {len(rows) - 1} real={[x for x in order if x in real]} hash={row['hash']} "
                  f"took={row['mask_took']} hmma_ok={row['hmma_ok']}: "
                  + "; ".join(f"{c} {phase} {row['phases'][c].get(phase)}" for c in cells), flush=True)
        rows.append({"real": sorted(everything), "stubbed": [], "hash": ref, "mask_took": True, "hmma_ok": True,
                     **{k: report["kernel"][k] for k in ("phases", "counters", "census", "registers", "hmma", "timeline")}})
        report["ladders"].append({"phase": phase, "order": order, "rungs": rows})

    rc, out = build(everything)
    report["kernel"]["hash_after"] = device_hash() if rc == 0 else None
    report["kernel"]["restored"] = report["kernel"]["hash_after"] == ref
    #: the kernel read again at the end: the run's drift bound (a rung step smaller than the
    #: kernel's own before/after difference is not a reading).
    report["kernel"]["phases_after"] = {c: phases(c, a.launches) for c in cells} if rc == 0 else {}

    drift = []
    for c in cells:
        b = report["kernel"]["phases"].get(c, {})
        e = report["kernel"]["phases_after"].get(c, {})
        drift.append(f"{c} total {b.get('total')} before, {e.get('total')} after"
                     + "".join(f"; {ph} {b.get(ph)} / {e.get(ph)}" for lad in ladders for ph in [lad[0]]))
    md = [f"# Composer {report['sha']} ({report['utc']})", "",
          f"Kernel device hash {ref}; restored after the ladders: {report['kernel']['restored']}.",
          "Kernel read before and after the ladders (the run's drift bound): " + " | ".join(drift), ""]
    for lad in report["ladders"]:
        phase, order = lad["phase"], lad["order"]
        md.append(f"## {phase}: parts added in order {order}")
        md.append("| rung | real parts of the phase | " + " | ".join(f"{c} {phase} | step | total" for c in cells)
                  + " | HMMA count = kernel | build took the mask | tensor pipe on silicon (cells in order) |")
        md.append("|---|---|" + "---|---|---|" * len(cells) + "---|---|---|")
        prev = None
        for i, r in enumerate(lad["rungs"]):
            here = [x.split(".", 1)[1] for x in order if x in r["real"]]
            cells_md = []
            for c in cells:
                v = (r.get("phases") or {}).get(c, {}).get(phase)
                t = (r.get("phases") or {}).get(c, {}).get("total")
                pv = None if prev is None else (prev.get("phases") or {}).get(c, {}).get(phase)
                step = "" if v is None or pv is None else f"{v - pv:+.0f}"
                cells_md.append(f"{'' if v is None else f'{v:.0f}'} | {step} | {'' if t is None else f'{t:.0f}'}")
            util = " / ".join(
                f"{100 * u:.0f}%" if (u := (((r.get("timeline") or {}).get(c) or {}).get("summary") or {})
                                      .get("utilization_mean_full")) is not None else "-" for c in cells)
            md.append(f"| {i} | {', '.join(here) or '(all stubbed)'} | " + " | ".join(cells_md)
                      + f" | {r.get('hmma_ok')} | {r.get('mask_took')} | {util} |")
            prev = r
        md.append("")
        #: THE PART STEPS: each added part's step in the kernel total, in its stall reasons, in the
        #: resource counters (a CTA-window) and in peak live registers.
        keys = [("shared ld", "smsp__sass_inst_executed_op_shared_ld.sum"),
                ("shared st", "smsp__sass_inst_executed_op_shared_st.sum"),
                ("ldsm", "smsp__sass_inst_executed_op_ldsm.sum"),
                ("global red", "smsp__sass_inst_executed_op_global_red.sum"),
                ("wf ld", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
                ("wf st", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
                ("conflict ld", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
                ("conflict st", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
                ("issue %", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
                ("warps active", "smsp__warps_active.avg.per_cycle_active")]
        for c in cells:
            md.append(f"**{c}, {phase} part steps** (total cycles; stall-reason cycles; counters a CTA-window; peak live):")
            md.append("| part added | total | top stall-reason steps | " + " | ".join(k for k, _ in keys) + " | peak live |")
            md.append("|---|---|---|" + "---|" * len(keys) + "---|")
            rungs = lad["rungs"]
            for i in range(1, len(rungs)):
                a_, b_ = rungs[i - 1], rungs[i]
                added = [x.split(".", 1)[1] for x in order if x in b_["real"] and x not in a_["real"]]
                ta = (a_.get("phases") or {}).get(c, {}).get("total")
                tb = (b_.get("phases") or {}).get(c, {}).get("total")
                ra = ((a_.get("census") or {}).get(c) or {}).get("by_reason", {})
                rb = ((b_.get("census") or {}).get(c) or {}).get("by_reason", {})
                steps = sorted(((k, rb.get(k, 0) - ra.get(k, 0)) for k in set(ra) | set(rb)), key=lambda x: -abs(x[1]))[:4]
                ca = (a_.get("counters") or {}).get(c, {})
                cb = (b_.get("counters") or {}).get(c, {})
                pa = (a_.get("registers") or {}).get("peak_live")
                pb = (b_.get("registers") or {}).get("peak_live")
                md.append(f"| {', '.join(added)} | {'' if ta is None or tb is None else f'{tb - ta:+.0f}'} | "
                          + " ".join(f"{k} {v:+.0f}" for k, v in steps) + " | "
                          + " | ".join(f"{cb.get(m, 0) - ca.get(m, 0):+.1f}" for _, m in keys)
                          + f" | {pa} -> {pb} |")
            md.append("")
    name = f"{report['utc']}-{report['sha']}-compose"
    LEDGERS.mkdir(parents=True, exist_ok=True)
    (LEDGERS / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    (LEDGERS / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"report: {LEDGERS / name}.json")
    rc, text = run([sys.executable, "tools/dashboard.py"])
    print(text.strip() if rc == 0 else f"dashboard: not rendered (rc {rc}): {text[-400:]}")
    ok = report["kernel"]["restored"] and all(r.get("hmma_ok") and r.get("mask_took") for lad in report["ladders"] for r in lad["rungs"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
