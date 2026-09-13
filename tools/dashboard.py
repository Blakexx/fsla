#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DASHBOARD: every stored measurement of the kernel and its variants over the cells, as one static page.

    python tools/dashboard.py            # writes <store>/ledgers/dashboard.html
    python tools/dashboard.py --check    # renders twice and refuses unless the bytes agree

It measures nothing. It reads what the suite already stored and renders it, so the same inputs give the same
bytes (no clock, no randomness, inputs read in sorted order, data serialized with sorted keys):
  build ledger reports   <store>/ledgers/<utc>-<sha7>-<tag>.json        standings, builds
  composer reports       <store>/ledgers/<utc>-<sha7>-compose.json      part ladders
  pipe timelines         <store>/ledgers/<stem>-timeline-<cell>.json    tensor pipe over a launch
  the probe record       <store>/<hw-profile>/<yyyy-mm>/<run_id>.jsonl  variants against a baseline
The page is `tools/dashboard_page.html` with the data inlined; docs/internals/tools/dashboard.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dev_config  # noqa: E402 -- path insert must precede this import

LEDGERS = dev_config.store("ledgers")
RESULTS = dev_config.store()
PAGE = Path(__file__).with_name("dashboard_page.html")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d{4}Z)-([0-9a-f]{7}|unknown)-(.+)$")
PHASES = ("head", "readout", "fold", "snapshot", "edges", "sweep", "head_words", "head_scans")
GATE_CELLS = ("nl64k-alt-k4", "flagship-dense", "flagship-alt-k4", "flagship-cohort-k4")
DEFAULT_ARM = "carry_forward"
SESSION_GAP_S = 2


def stamp(stem: str) -> dict:
    m = STAMP.match(stem)
    return {"utc": m.group(1), "sha": m.group(2), "tag": m.group(3)} if m else {"utc": "", "sha": "", "tag": stem}


def pick(d: dict | None, *keys: str) -> dict | None:
    return {k: d[k] for k in keys if k in d} if d else None


def build_row(stem: str, r: dict) -> dict:
    cells = {}
    for c in r["cells"]:
        ph = r.get("phases", {}).get(c, {})
        ab = pick(r.get("ab", {}).get(c), "master", "this", "ratio", "master_schedule")
        cells[c] = {"phases": ph.get("per_phase"), "total": ph.get("total"),
                    "util_count": r.get("profile", {}).get(c, {}).get("tensor_util_active"),
                    "util_true": r.get("timeline", {}).get(c, {}).get("utilization_mean_full"),
                    "ab": ab if ab and "this" in ab else None}
    return {"name": stem, "utc": r["utc"], "sha": r["sha"], "tag": r["tag"], "cells": cells,
            "oracle": pick(r.get("oracle"), "passed", "failed"), "sass": pick(r.get("sass"), "instr", "hmma"),
            "peak_live": r.get("registers", {}).get("peak_live")}


def compose_row(stem: str, r: dict) -> dict:
    ladders = []
    for lad in r["ladders"]:
        rungs = []
        for g in lad["rungs"]:
            ph = g.get("phases", {})
            tl = g.get("timeline") or {}
            rungs.append({"real": g.get("real", []), "hmma_ok": g.get("hmma_ok"), "mask_took": g.get("mask_took"),
                          "total": {c: ph.get(c, {}).get("total") for c in r["cells"]},
                          "util_true": {c: (tl.get(c) or {}).get("summary", {}).get("utilization_mean_full") for c in r["cells"]},
                          "phase": {c: ph.get(c, {}).get(lad["phase"]) for c in r["cells"]}})
        ladders.append({"phase": lad["phase"], "order": lad["order"], "rungs": rungs})
    kernel = r.get("kernel", {})
    return {"name": stem, "utc": r["utc"], "sha": r["sha"], "cells": r["cells"], "ladders": ladders,
            "restored": kernel.get("restored"),
            "kernel_total": {c: kernel.get("phases", {}).get(c, {}).get("total") for c in r["cells"]}}


def timeline_row(stem: str, r: dict) -> dict:
    cell, scale = r["cell"], r.get("scale")
    suffix = f"-timeline-{cell}"
    series = r["series"]

    def pts(name: str, k: float = 1.0) -> list:
        return [[round(t, 1), round(v * k, 1)] for t, v in series.get(name, [])]

    return {"name": stem, **stamp(stem[: -len(suffix)] if stem.endswith(suffix) else stem), "cell": cell,
            "duration_us": round(r["duration_us"], 1), "scaled": bool(scale),
            "summary": {k: round(v, 4) for k, v in r.get("summary", {}).items() if isinstance(v, (int, float))},
            "tensor": pts("tensor", 100.0 / scale if scale else 1.0), "alu": pts("alu"), "warps": pts("warps")}


def ledger_inputs(ledgers: Path) -> tuple[list, list, list]:
    builds, composes, timelines = [], [], []
    for p in sorted(ledgers.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "series" in r and "cell" in r:
            timelines.append(timeline_row(p.stem, r))
        elif "ladders" in r:
            composes.append(compose_row(p.stem, r))
        elif "tag" in r and "cells" in r:
            builds.append(build_row(p.stem, r))
    order = lambda row: (row["utc"], row["name"])  # noqa: E731
    return sorted(builds, key=order), sorted(composes, key=order), sorted(timelines, key=order)


def expected_schedule(cell: str) -> str:
    """The baseline's own order policy per cell (`tools/build_ledger.py` step ab): box dense, sparse-gN sparse."""
    return "box" if "dense" in cell or "struct" in cell else "sparse-g"


def seconds(ts: str) -> float:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def unit_of(arm: str, calls: int) -> str:
    """A series is a subject at a call count: the same subject over N carried calls is its own row (`bench.subjects.Subject.calls`)."""
    return arm if calls == 1 else f"{arm}@calls={calls}"


def probe_runs(results: Path, hw: str) -> list[dict]:
    runs = []
    for p in sorted((results / hw).rglob("*.jsonl")):
        lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        head = next((x for x in lines if x.get("kind") == "run"), None)
        probes = [x for x in lines if x.get("kind") == "probe" and x.get("value") is not None]
        if not head or not probes:
            continue
        extras = [json.loads(x.get("extra_json") or "{}") for x in probes]
        runs.append({"ts": head.get("ts") or "", "run_id": head["run_id"], "stage": head.get("stage"),
                     "purpose": head.get("purpose"), "branch": head.get("branch"), "worktree": head.get("worktree"),
                     "sha": (head.get("git_sha") or "")[:7], "dirty": bool(head.get("dirty")),
                     "label": extras[0].get("binary") or head.get("worktree") or head["run_id"],
                     "session": extras[0].get("session"),
                     "rows": [{"cell": x["cell"], "arm": unit_of(x.get("arm") or DEFAULT_ARM, e.get("calls", 1)), "ms": x["value"],
                               "schedule": e.get("schedule"), "locked": e.get("clock_locked")}
                              for x, e in zip(probes, extras)]})
    return sorted(runs, key=lambda r: (r["ts"], r["run_id"]))


def sessions(runs: list[dict]) -> list[list[dict]]:
    """One probe invocation's runs. A row written since the session field carries it; an older row joins the previous
    session when it has the same stage, was written within SESSION_GAP_S seconds and names a binary not yet in it
    (`tools/probe_cells.py` records every binary of an invocation back to back)."""
    out: list[list[dict]] = []
    by_key: dict[str, list[dict]] = {}
    for r in runs:
        if r["session"]:
            if r["session"] not in by_key:
                by_key[r["session"]] = []
                out.append(by_key[r["session"]])
            by_key[r["session"]].append(r)
            continue
        last = out[-1] if out else None
        if (last and not last[-1]["session"] and last[-1]["stage"] == r["stage"]
                and seconds(r["ts"]) - seconds(last[-1]["ts"]) <= SESSION_GAP_S
                and r["label"] not in {x["label"] for x in last}):
            last.append(r)
        else:
            out.append([r])
    return out


def baseline_for(group: list[dict], baselines: list[str], arm: str, cell: str) -> tuple[dict, dict] | None:
    """The session's baseline measurement of (arm, cell): a run on the first of `baselines` (git branches) present.
    On `baselines[0]`, whose order policy `expected_schedule` states, a run that followed the policy is preferred."""
    for branch in baselines:
        cands = [(r, x) for r in group if r["branch"] == branch for x in r["rows"] if (x["arm"], x["cell"]) == (arm, cell)]
        if cands:
            if branch == baselines[0]:
                return next((c for c in cands if (c[1]["schedule"] or "").startswith(expected_schedule(cell))), cands[0])
            return cands[0]
    return None


def variants(runs: list[dict], baselines: list[str]) -> list[dict]:
    """Every probed binary, per cell, against the baseline measured in the same interleaved session. A baseline on
    `baselines[0]` that did not run its order policy is flagged: its time, and every ratio against it, is not the
    baseline's."""
    rows = []
    for group in sessions(runs):
        for r in group:
            arms: dict[str, dict] = {}
            for x in r["rows"]:
                found = baseline_for(group, baselines, x["arm"], x["cell"])
                if found and found[0] is r:
                    continue
                entry = {"ms": x["ms"], "schedule": x["schedule"], "flags": []}
                if found:
                    br, b = found
                    entry.update({"base": br["label"], "base_sha": br["sha"], "base_ms": b["ms"],
                                  "base_schedule": b["schedule"], "ratio": x["ms"] / b["ms"] if b["ms"] else None})
                    want = expected_schedule(x["cell"])
                    if br["branch"] == baselines[0] and not (b["schedule"] or "").startswith(want):
                        entry["flags"].append(f"baseline ran {b['schedule'] or 'an unrecorded schedule'}, its policy here is {want}")
                if x["locked"] is False or (found and found[1]["locked"] is False):
                    entry["flags"].append("clock unlocked")
                arms.setdefault(x["arm"], {})[x["cell"]] = entry
            for arm, cells in sorted(arms.items()):
                bases = sorted({f"{e['base']} {e['base_sha']}" for e in cells.values() if "base" in e})
                rows.append({"utc": r["ts"], "stage": r["stage"], "purpose": r["purpose"], "label": r["label"],
                             "arm": arm, "worktree": r["worktree"], "sha": r["sha"], "dirty": r["dirty"],
                             "bases": bases, "cells": cells})
    return rows


def standings(builds: list[dict]) -> dict:
    """Per cell, each metric's newest stored value and the report it came from."""
    out: dict = {}
    for b in builds:
        for c, v in b["cells"].items():
            row = out.setdefault(c, {})
            for k in ("ab", "util_true", "util_count", "total", "phases"):
                if v.get(k) is not None:
                    row[k] = {"value": v[k], "from": b["name"]}
    return out


def cell_order(cells: set[str]) -> list[str]:
    return [c for c in GATE_CELLS if c in cells] + sorted(cells - set(GATE_CELLS))


def data(ledgers: Path, results: Path, hw: str, baselines: list[str]) -> dict:
    builds, composes, timelines = ledger_inputs(ledgers)
    probes = variants(probe_runs(results, hw), baselines)
    scales = sorted((results / hw).glob("pipe-timeline-scale-*.json"))
    cells = {c for b in builds for c in b["cells"]} | {t["cell"] for t in timelines}
    cells |= {c for v in probes for c in v["cells"]}
    newest = max([x["utc"] for x in builds + composes + timelines] + [dt.datetime.strptime(v["utc"], "%Y-%m-%dT%H:%M:%SZ")
                  .strftime("%Y-%m-%dT%H%MZ") for v in probes] or [""])
    return {"hw": hw, "baselines": baselines, "phases": PHASES, "cells": cell_order(cells), "newest": newest,
            "scale": json.loads(scales[-1].read_text()) if scales else None, "standings": standings(builds),
            "builds": builds, "composes": composes, "timelines": timelines, "variants": probes}


def render(d: dict) -> bytes:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    page = PAGE.read_text()
    return page.replace("__DIGEST__", digest).replace("__DATA__", blob).encode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledgers", type=Path, default=LEDGERS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--hw", default="rtx3080ti-sm86", help="the hardware profile of the probe record to read")
    ap.add_argument("--baselines", default="master,k31/carry-clean",
                    help="git branches a session's baseline run may be on, in order of preference")
    ap.add_argument("--out", type=Path, default=LEDGERS / "dashboard.html")
    ap.add_argument("--check", action="store_true", help="render twice from fresh reads; refuse unless the bytes agree")
    a = ap.parse_args()
    baselines = [b for b in a.baselines.split(",") if b]

    page = render(data(a.ledgers, a.results, a.hw, baselines))
    if a.check:
        again = render(data(a.ledgers, a.results, a.hw, baselines))
        leaks = [s for s in (str(ROOT), "/tmp/", "/home/") if s in page.decode()]
        if page != again or leaks:
            print(f"dashboard: REFUSED ({'renders differ' if page != again else 'absolute paths inlined: ' + ', '.join(leaks)})")
            return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(page)
    print(f"dashboard: {a.out} ({len(page)} bytes, sha256 {hashlib.sha256(page).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
