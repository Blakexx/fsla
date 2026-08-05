"""The `counters` subcommand: ncu, driven from a committed section file.

The metric list is NOT a flag string. It lives in `probe/sections/*.section`, which
is a reviewable, diffable statement of how a question is measured; two throwaway
scripts holding the same 24-metric comma-joined string is how the two drifted apart.

Two ncu behaviours are load-bearing and are handled here rather than remembered:

* `--section-folder` REPLACES the default search path instead of adding to it, so
  the stock folder is passed alongside ours or every stock section disappears.
* counter capture runs at `--launch-count 1` with the timing loop disabled. A
  warm-up loop pollutes counters the way it does not pollute a median.

The replay mode is RECORDED in every row. `kernel` (the default) restores only the
memory the kernel wrote; a kernel that carries state across launches -- the paged and
`use_cache` continuation arms -- needs `application` replay, and nothing in the tree
previously said which had been used.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from bench.ledger import LEDGER, append
from probe.registry import arm_id, get

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"
REPRO = HERE / "repro.py"

#: Launches discarded before the profiled one. The kernel's first launches build the
#: schedule and warm the instruction cache; the counters of interest are the steady
#: state's.
LAUNCH_SKIP = 6

REPLAY_MODES = ("kernel", "application", "range", "app-range")


def stock_section_folder() -> Path:
    """Nsight Compute's own section folder, for the version of `ncu` on PATH.

    Found rather than hardcoded, and a miss is a hard error: passing only our folder
    would silently delete every stock section from the search path, which is the
    documented behaviour of `--section-folder` and the exact failure this function
    exists to prevent.
    """
    if os.environ.get("ROLA_NCU_SECTIONS"):
        return Path(os.environ["ROLA_NCU_SECTIONS"])
    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("ncu is not on PATH; counter capture needs Nsight Compute")
    out = subprocess.run([ncu, "--version"], capture_output=True, text=True, check=True)
    m = re.search(r"Version (\d+\.\d+\.\d+)", out.stdout)
    if m:
        candidate = Path("/opt/nvidia/nsight-compute") / m.group(1) / "sections"
        if candidate.is_dir():
            return candidate
    beside = Path(ncu).resolve().parent / "sections"
    if beside.is_dir():
        return beside
    raise SystemExit(
        "could not locate Nsight Compute's stock section folder. Set "
        "ROLA_NCU_SECTIONS to it: --section-folder REPLACES the default search "
        "path, so running without it would silently drop every stock section.")


def validate_sections(folder: Path = SECTIONS) -> list[str]:
    """Ask `ncu` to parse our section files, listing what it registered.

    No GPU work and no launch: `--list-sections` is a parse. A file with a syntax
    error fails here, in a check that costs nothing, instead of at the end of a
    profiling run.
    """
    ncu = shutil.which("ncu")
    if ncu is None:
        raise SystemExit("ncu is not on PATH; the section files cannot be validated")
    out = subprocess.run([ncu, "--section-folder", str(folder), "--list-sections"],
                         capture_output=True, text=True)
    if out.returncode != 0 or "No section files found" in out.stdout + out.stderr:
        raise SystemExit(f"ncu REFUSED the section folder {folder}:\n"
                         f"{out.stdout}\n{out.stderr}")
    ours = {p.stem for p in folder.glob("*.section")}
    listed = {line.split()[0] for line in out.stdout.splitlines()[2:] if line.strip()}
    missing = sorted(ours - listed)
    if missing:
        raise SystemExit(f"ncu parsed {folder} but did not register {missing}; "
                         f"its output was:\n{out.stdout}")
    return sorted(ours)


def _parse_raw_csv(text: str) -> list[tuple[str, str, str]]:
    """`(kernel, metric, value)` from `ncu --csv --page raw` output.

    The raw page is long-format -- one row per metric per launch -- and the header is
    not the first line of the stream, because ncu prints its own progress ahead of
    it. Finding the header rather than assuming its position is what keeps this from
    breaking on a version that prints one more banner line.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if '"Metric Name"' in line or line.startswith("ID,")), None)
    if start is None:
        raise SystemExit(f"no CSV header in ncu's output; it printed:\n{text[:2000]}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    out = []
    for row in rows:
        name = row.get("Metric Name")
        if not name:
            continue
        out.append((row.get("Kernel Name", ""), name, row.get("Metric Value", "")))
    return out


def run(*, run_ctx, probe_name: str, arm: str, section: str, replay_mode: str) -> int:
    probe = get(probe_name)
    if arm not in probe.arms:
        raise SystemExit(f"{probe.name} has no arm {arm!r}")
    if replay_mode not in REPLAY_MODES:
        raise SystemExit(f"--replay-mode must be one of {REPLAY_MODES}")
    validate_sections()

    cmd = [
        shutil.which("ncu"),
        "--section-folder", str(stock_section_folder()),
        "--section-folder", str(SECTIONS),
        "--section", section,
        "--replay-mode", replay_mode,
        "--target-processes", "all",
        "-k", f"regex:{probe.kernel_regex}",
        "--launch-skip", str(LAUNCH_SKIP),
        "--launch-count", "1",
        "--csv", "--page", "raw",
        sys.executable, str(REPRO),
        "--probe", probe.name, "--arm", arm,
        "--launches", str(LAUNCH_SKIP + 2),
    ]
    print(" ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout)
        sys.stderr.write(out.stderr)
        raise SystemExit(f"ncu FAILED (rc={out.returncode})")

    measured = _parse_raw_csv(out.stdout)
    if not measured:
        raise SystemExit("ncu produced no metric rows; a counter capture that "
                         "measured nothing must not be recorded as a measurement")
    rows = [run_ctx.row(probe=probe.name, probe_version=probe.version,
                        arm=arm_id(probe, arm), source="ncu", metric=metric,
                        unit="count", values=[float(value)],
                        replay_mode=replay_mode, section=section,
                        kernel=kernel)
            for kernel, metric, value in measured
            if _is_number(value)]
    append(rows)
    print(f"{len(rows)} counter rows appended to {LEDGER}")
    for _, metric, value in measured[:24]:
        print(f"  {metric:70s} {value}")
    return 0


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
