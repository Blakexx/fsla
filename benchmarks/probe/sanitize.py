"""The `sanitize` subcommand: compute-sanitizer over a named cell subset.

**SOLD HONESTLY.** compute-sanitizer owns memory errors, uninitialized reads and
synchronization misuse. It does NOT own this repository's historical bug class: both
stream races that escaped here were host-side Python ordering, and `racecheck` detects
*"accesses to on-chip shared memory"* only. The fixture in
`tests/tier2_geometry/stream_discipline.py` is what covers that class; this covers
what it owns, which is worth having and is not the same thing.

**A SCHEDULED LANE, NEVER A PR GATE.** This is the shape every surveyed adopter
converged on (cccl, cudf, cuda-python all run it nightly or on dispatch, over a named
subset, with a skip-list): the tools are one to two orders of magnitude slower, and a
gate that takes an hour is a gate that gets disabled. `--tool`, a committed
suppressions file, an explicit skip-list, and `--error-exitcode=1` so a finding is a
failure rather than a line in a log nobody reads.

The four tools, quoted from the Compute Sanitizer manual:

* `memcheck` -- out-of-bounds and misaligned accesses, plus hardware exceptions;
* `initcheck` -- device memory read before it was written;
* `synccheck` -- misuse of `__syncthreads` / `__syncwarp` and their cooperative-groups
  counterparts;
* `racecheck` -- **shared-memory** hazards only, and see above.

**WHAT IT CANNOT SEE HERE, measured.** torch's caching allocator SUBALLOCATES tensors
from large segments, so an overrun that stays inside the owning segment is memory the
allocator legitimately holds and no tool can call it out of bounds. This lane detects
accesses that escape the segment. That hole is stated so a clean run is read for what
it is; `tests/unit/test_probe_harness.py` plants an escaping write and requires the
finding.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bench.ledger import LEDGER, append
from probe.registry import arm_id, get

HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro.py"
SUPPRESSIONS = HERE / "sanitizer_suppressions.xml"

TOOLS = ("memcheck", "initcheck", "synccheck", "racecheck")

#: Launches the child performs under the sanitizer. Small on purpose: every tool here
#: costs one to two orders of magnitude, so the subset is sized to say something about
#: the kernel rather than to cover the grid.
SANITIZED_LAUNCHES = 2

#: `(probe, tool)` pairs this lane REFUSES to run, with the reason. A skip-list is
#: part of the pattern every adopter landed on, and an empty one is a claim about
#: runtimes nobody has measured -- so entries are added when a cell is measured over
#: budget or structurally unanswerable, and the reason is recorded beside it, never a
#: bare name. A skip is preferred to a suppressions entry whenever the cause is
#: STRUCTURAL: a per-source-location suppression for a structural cause rots the
#: moment the code moves, and silently stops covering what it was meant to cover.
SKIP: dict[tuple[str, str], str] = {
    ("consumer_forward", "racecheck"):
        "MEASURED 2026-08-05: 4 hazards, 131072 each, between the `Gm` write "
        "(mma_tile.cuh:69, via consumer_kernel.cuh:729-748) and its read "
        "(mma_tile.cuh:192, via consumer_kernel.cuh:853/867) -- and they are the "
        "tool's blind spot, not the kernel's defect. racecheck orders shared memory "
        "by __syncthreads/__syncwarp and cooperative-groups barriers ONLY. That edge "
        "is published by the item pipeline's own release/acquire protocol instead "
        "(`retire_item`'s __threadfence_block + atomic counter, `await_count`'s "
        "spin-acquire; pipeline.cuh, docs/internals/pipeline.md#release-acquire), "
        "which the kernel uses DELIBERATELY so an early warp takes another work item "
        "rather than stalling on a rendezvous. The tool cannot be taught that "
        "protocol, so every write/read pair across the edge reads as unordered. A "
        "kernel that publishes through a standard barrier is still checked by this "
        "lane; this cell is not, and the reason is structural rather than a budget.",
    ("consumer_forward_deep", "racecheck"):
        "Same kernel and the same publication protocol as consumer_forward above: "
        "the item pipeline's release/acquire edge is invisible to racecheck, which "
        "orders shared memory by standard barriers only, so the hazards it reports "
        "here are the tool's blind spot rather than a defect in this cell.",
}


def tool_flags(tool: str) -> list[str]:
    """The per-tool flags the lane runs with. ONE definition, so the non-vacuity test
    exercises the flags the lane actually uses rather than a restatement of them."""
    if tool != "memcheck":
        return []
    #: `--padding` is what turns a JUST-past-the-end access into a caught one. Without
    #: it the caching allocator's block rounding swallows a one-element overrun and
    #: memcheck reports nothing -- measured, and the reason the non-vacuity test
    #: plants exactly a one-element overrun.
    #:
    #: `--leak-check full` is DELIBERATELY ABSENT, also measured: with it this lane
    #: reports "23068676 bytes leaked in 3 allocations, ERROR SUMMARY: 3 errors" on a
    #: clean run, every one a torch caching-allocator segment still held at exit. The
    #: allocator retains its pools by design, so the flag reports a design property as
    #: a defect -- and a lane whose clean state is three errors is a lane whose
    #: findings nobody reads. Suppressing them would have hidden real leaks in the
    #: same allocations; not asking the question does not.
    return ["--padding", "512"]


def base_command(tool: str) -> list[str]:
    """Everything before the program under test. Shared with the non-vacuity test."""
    return [
        _sanitizer(),
        "--tool", tool,
        #: `application-only` rather than `all`: the documented fix for CI hangs when
        #: multiprocessing children launch slowly under the sanitizer, and this child
        #: launches none.
        "--target-processes", "application-only",
        "--error-exitcode", "1",
        "--suppressions", str(SUPPRESSIONS),
        *tool_flags(tool),
    ]


def _sanitizer() -> str:
    found = shutil.which("compute-sanitizer")
    if found is None:
        raise SystemExit(
            "compute-sanitizer is not on PATH. It ships with the CUDA toolkit and is "
            "supported on WSL2 for Pascal and later; this lane cannot be faked with a "
            "plain run.")
    return found


def run(*, run_ctx, probe_name: str, tool: str) -> int:
    probe = get(probe_name)
    if tool not in TOOLS:
        raise SystemExit(f"--tool must be one of {TOOLS}")
    reason = SKIP.get((probe_name, tool))
    if reason:
        raise SystemExit(f"{probe_name} is skipped under {tool}: {reason}")

    cmd = base_command(tool) + [
        sys.executable, str(REPRO),
        "--probe", probe.name, "--arm", "global",
        "--launches", str(SANITIZED_LAUNCHES),
    ]
    print(" ".join(cmd))

    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    sys.stdout.write(proc.stdout[-4000:])
    if proc.stderr:
        sys.stderr.write(proc.stderr[-4000:])

    clean = proc.returncode == 0
    append([run_ctx.row(
        probe=probe.name, probe_version=probe.version,
        arm=arm_id(probe, "global"), source="sanitizer", metric=f"{tool}_errors",
        unit="count", values=[0.0 if clean else 1.0], tool=tool,
        elapsed_s=round(elapsed, 1), returncode=proc.returncode)])
    print(f"\n{tool}: {'CLEAN' if clean else 'FINDINGS'} in {elapsed:.1f}s "
          f"(rc={proc.returncode}); row appended to {LEDGER}")
    return 0 if clean else 1


def report_skips() -> str:
    return json.dumps({f"{p}/{t}": why for (p, t), why in sorted(SKIP.items())},
                      indent=1)
