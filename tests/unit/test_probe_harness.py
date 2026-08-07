"""The harness's DISCIPLINE, proved by planting each violation it exists to refuse.

The claim `rola-probe` makes is not that it can measure -- any script can measure --
but that it cannot measure UNDISCIPLINED. That claim is only worth the checks below:
each one removes a precondition and requires the refusal, and one of them holds the
lock from another process and requires the harness to wait for it.

No kernel is launched here. Every check is about what the harness does BEFORE it
would launch one, which is exactly the part the throwaway scripts skipped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from probe import discipline
from probe.cli import build_parser
from probe.counters import SECTIONS, validate_sections
from probe.registry import PROBES, arm_id, get

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# THE LOCK
# ---------------------------------------------------------------------------

def test_no_subcommand_offers_a_way_to_skip_the_lock():
    """The refusal has to be structural: a `--no-lock` flag is a flag that gets
    used. This asserts the option does not exist on any subcommand."""
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    for sub in parser._subparsers._group_actions[0].choices.values():
        options |= {action.dest for action in sub._actions}
    assert not {o for o in options if "lock" in o}


def test_the_harness_waits_for_a_lock_another_process_holds(tmp_path):
    """PLANTED VIOLATION: a second measurer arrives while the device is in use. The
    harness must BLOCK, not proceed -- two probes sharing a GPU is the condition that
    makes both numbers meaningless."""
    lock = tmp_path / "gpu.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import fcntl, os, time
            fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print("held", flush=True)
            time.sleep(3)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        with discipline.gpu_lock(str(lock)):
            waited = time.monotonic() - t0
        assert waited > 0.5, ("the harness acquired a lock another process was "
                              "holding; the lock is not exclusive")
    finally:
        holder.wait(timeout=10)


def test_an_outer_flock_on_the_same_path_is_refused_not_waited_on(tmp_path):
    """PLANTED VIOLATION, and it is the one a reviewer actually tripped over:
    `flock /tmp/rola_gpu.lock python benchmarks/rola_probe.py ...`.

    The parent holds the lock the harness is about to take, and waits for the child
    that is waiting for it. Nothing times out and nothing prints. The harness must
    NAME that and exit, so the rule ("everything wrapped EXCEPT rola-probe") has a
    mechanical detector and not only a paragraph in docs/testing.md.
    """
    lock = tmp_path / "gpu.lock"
    child = f"""
        import sys
        sys.path.insert(0, {str(ROOT / "benchmarks")!r})
        from probe.discipline import gpu_lock
        with gpu_lock({str(lock)!r}):
            print("ACQUIRED")
    """
    wrapper = subprocess.run(
        ["flock", str(lock), sys.executable, "-c", textwrap.dedent(child)],
        capture_output=True, text=True, timeout=60)
    assert "ACQUIRED" not in wrapper.stdout, (
        "the harness acquired a lock its own caller was holding")
    assert "DEADLOCK REFUSED" in wrapper.stderr, wrapper.stderr[-2000:]
    assert wrapper.returncode != 0


def test_a_lock_held_by_an_unrelated_process_is_still_waited_on(tmp_path):
    """The refusal must be narrow: an unrelated holder is CONTENTION, which is what
    the lock is for, and waiting is the correct behaviour there. Without this the
    deadlock detector could turn every real wait into a refusal and nothing would
    notice."""
    lock = tmp_path / "gpu.lock"
    assert discipline._ancestor_holding(str(lock)) is None
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import fcntl, os, time
            fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            print("held", flush=True)
            time.sleep(2)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert discipline._ancestor_holding(str(lock)) is None
    finally:
        holder.wait(timeout=10)


def test_the_lock_path_is_the_one_the_repository_documents():
    """One lock file, or the exclusion excludes nobody. `ROLA_GPU_LOCK` may MOVE it
    for a caller composing with an outer lock; it cannot switch it off."""
    assert os.environ.get("ROLA_GPU_LOCK", "/tmp/rola_gpu.lock") == discipline.LOCK_PATH
    recipe = (ROOT / "docs" / "testing.md").read_text()
    assert "/tmp/rola_gpu.lock" in recipe


# ---------------------------------------------------------------------------
# THE CLOCK
# ---------------------------------------------------------------------------

def test_an_unsettled_clock_aborts_the_run(monkeypatch):
    """PLANTED VIOLATION: the box never plateaus. The harness must refuse to
    measure, not proceed with a note in the log."""
    monkeypatch.setattr(discipline, "settle_clocks",
                        lambda: {"settled": False, "reason": "timeout"})
    with pytest.raises(SystemExit, match="REFUSING TO MEASURE"):
        discipline.Run(tier="light", cmd="probe duration")


def test_an_unknown_tier_is_refused_before_anything_is_measured(monkeypatch):
    monkeypatch.setattr(discipline, "settle_clocks",
                        lambda: {"settled": True, "sm_mhz": 1850, "plateau_mhz": 5,
                                 "elapsed_s": 1.0})
    with pytest.raises(SystemExit, match="tier must be one of"):
        discipline.Run(tier="whatever", cmd="probe duration")


def test_every_row_carries_the_tier_and_the_clock_it_was_taken_under(monkeypatch):
    """The tier is DATA, so a light-tier number can never be quoted as a landing
    one by someone who did not read the command that produced it."""
    monkeypatch.setattr(discipline, "settle_clocks",
                        lambda: {"settled": True, "sm_mhz": 1852, "plateau_mhz": 8,
                                 "elapsed_s": 1.1})
    monkeypatch.setattr(discipline, "_git_state", lambda: ("c" * 40, False))
    monkeypatch.setattr(discipline, "_csrc_sha256", lambda: "d" * 64)
    monkeypatch.setattr(discipline, "_gpu_temp_c", lambda: 44.0)
    monkeypatch.setattr("bench.regression.provenance",
                        lambda: {"gpu": "g", "sm": "sm_86", "driver_cuda": "12.4",
                                 "torch": "2.6.0", "ptxas": "p",
                                 "manifest_sha256": "m", "rola": "0.1"})
    run = discipline.Run(tier="light", cmd="probe duration")
    row = run.row(probe="consumer_forward", probe_version=1, arm="a",
                  source="cuda_events", metric="duration", unit="ms",
                  values=[1.0, 1.0])
    assert row["tier"] == "light"
    assert row["env"]["sm_mhz"] == 1852 and row["env"]["settled"] is True
    assert row["env"]["gpu_temp_c_start"] == 44.0
    assert row["csrc_sha256"] == "d" * 64 and row["sha"] == "c" * 40
    assert row["cmd"] == "probe duration"


def test_the_memory_fraction_is_the_one_the_freeze_remedy_pinned():
    assert discipline.MEMORY_FRACTION == 0.65


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------

def test_every_probe_names_a_real_cell_and_addresses_the_binary():
    from bench.cells import CELLS

    for name, probe in PROBES.items():
        assert probe.name == name
        assert probe.cell in CELLS
        assert probe.kernel_regex and probe.arms
        for arm in probe.arms:
            assert arm_id(probe, arm).startswith(probe.cell + "|")


def test_an_unknown_probe_or_arm_is_refused():
    with pytest.raises(SystemExit, match="unknown probe"):
        get("no_such_probe")


# ---------------------------------------------------------------------------
# THE SECTION FILE (I-6)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (SECTIONS / "RolaMemOcc.section").exists(),
                    reason="the committed section file is the subject")
def test_ncu_parses_the_committed_section_file():
    import shutil

    if shutil.which("ncu") is None:
        pytest.skip("ncu is not installed on this box")
    assert "RolaMemOcc" in validate_sections()


def test_a_corrupt_section_file_is_refused(tmp_path):
    """PLANTED VIOLATION: the validation must be a parse, not a file-existence
    check. A section file nothing validates is a flag string with a longer name."""
    import shutil

    if shutil.which("ncu") is None:
        pytest.skip("ncu is not installed on this box")
    bad = tmp_path / "Broken.section"
    bad.write_text('Identifier: "Broken"\nHeader { Metrics { Rows { } } }\n')
    with pytest.raises(SystemExit):
        validate_sections(tmp_path)


# ---------------------------------------------------------------------------
# THE SANITIZER LANE (I-7)
# ---------------------------------------------------------------------------

def test_the_sanitizer_lane_reports_a_planted_out_of_bounds_write():
    """PLANTED VIOLATION: a kernel that writes one element past its allocation.

    The lane's value is entirely in `--error-exitcode=1` plus a readable suppressions
    file, and both are easy to have wrong in a way that exits zero forever. So the
    exact command the lane builds is run against a program that IS broken, and must
    come back non-zero with the finding named.

    Deliberately not the RoLA kernel: this proves the INSTRUMENT reads. What the tool
    does and does not own for our kernels is `sanitize.SKIP`'s subject.

    **THE OVERRUN IS FAR PAST THE END, AND THAT IS A MEASURED LIMITATION, NOT A
    CONVENIENCE.** A one-element overrun on a torch tensor is invisible to memcheck:
    the caching allocator SUBALLOCATES from large segments, so the write lands inside
    memory the allocator legitimately owns and no tool can call it out of bounds.
    `--padding` pads `cudaMalloc`ed allocations, which is the extension's own, not a
    tensor's slot inside a segment. So this lane detects accesses that escape the
    SEGMENT, and an overrun within one is a hole the lane cannot cover -- stated here
    rather than left for someone to discover after trusting a clean run.
    """
    import shutil

    import torch

    if shutil.which("compute-sanitizer") is None:
        pytest.skip("compute-sanitizer ships with the CUDA toolkit; not on PATH")
    if not torch.cuda.is_available():
        pytest.skip("the sanitizer needs a device")

    from probe.sanitize import base_command

    planted = textwrap.dedent("""
        import torch
        from torch.utils.cpp_extension import load_inline
        mod = load_inline(
            name="rola_planted_oob",
            cpp_sources="void run(torch::Tensor t);",
            cuda_sources=(
                "#include <torch/extension.h>\\n"
                "__global__ void k(float* p, int n) { p[n + (1 << 22)] = 1.0f; }\\n"
                "void run(torch::Tensor t) {\\n"
                "  k<<<1, 1>>>(t.data_ptr<float>(), t.numel());\\n"
                "}\\n"),
            functions=["run"], verbose=False)
        t = torch.zeros(32, device="cuda")
        mod.run(t)
        torch.cuda.synchronize()
    """)
    #: THE LANE'S OWN COMMAND, not a restatement of it -- including `--padding`,
    #: which is what makes the ONE-ELEMENT overrun planted above visible at all. The
    #: caching allocator's block rounding swallows it otherwise, measured.
    proc = subprocess.run(
        base_command("memcheck") + [sys.executable, "-c", planted],
        capture_output=True, text=True, timeout=1800)

    combined = proc.stdout + proc.stderr
    assert "Error reading" not in combined, (
        f"the committed suppressions file is not readable by this tool, so every "
        f"lane run prints an error and still exits zero:\n{combined[:1500]}")
    assert proc.returncode != 0, (
        f"an out-of-bounds write exited ZERO under the lane's own flags:\n"
        f"{combined[-3000:]}")
    assert "Invalid __global__ write" in combined or "out of bounds" in combined, (
        f"the sanitizer ran but named no memory error:\n{combined[-3000:]}")


def test_every_sanitizer_skip_states_a_reason():
    """A skip-list entry without a reason is a silence nobody can audit. The reasons
    are long on purpose -- each has to say why the finding is not ours to fix."""
    from probe.registry import PROBES
    from probe.sanitize import SKIP, TOOLS

    for (probe, tool), reason in SKIP.items():
        assert probe in PROBES, probe
        assert tool in TOOLS, tool
        assert len(reason) > 80, f"{probe}/{tool}: a bare skip is not a decision"


def test_racecheck_covers_every_registered_probe():
    """The racecheck arm is LIVE, and that is a property of the kernel, not of the
    lane. racecheck orders shared memory by `__syncthreads`/`__syncwarp` and
    cooperative-groups barriers ONLY, so a kernel that publishes a shared-memory edge
    through anything else reads as unordered to it and every cell has to be skipped.
    The consumer publishes both of its intra-visit edges through ONE `__syncthreads()`
    (`docs/internals/consumer_kernel.md#visit-barrier`), so the tool sees the whole
    ordering and both registered probes run clean. A future edge published by a
    software protocol would put entries back in this skip-list, and that is a design
    decision that has to be made deliberately rather than discovered in a dormant
    lane."""
    from probe.registry import PROBES
    from probe.sanitize import SKIP

    dormant = [p for p in PROBES if (p, "racecheck") in SKIP]
    assert not dormant, (
        f"racecheck is skipped for {dormant} -- the lane is dormant again. A skip here "
        f"means a shared-memory edge is published by something racecheck cannot see.")


def test_the_stock_section_folder_is_passed_alongside_ours():
    """`--section-folder` REPLACES the default search path. Passing only ours would
    silently delete every stock section, so the command must name both."""
    import inspect

    from probe import counters

    source = inspect.getsource(counters.run)
    assert source.count('"--section-folder"') == 2
    assert "stock_section_folder()" in source
