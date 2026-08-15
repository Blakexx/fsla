"""THE PLANTED VIOLATIONS: every discipline gate of the chunk-arm harness, made to fire.

A gate that has never refused anything is a gate nobody has evidence for -- the retired
probe harness's own standard, and the reason its `test_probe_harness.py` existed. So each
row here PLANTS the violation its gate exists to catch and requires the refusal, and the
rows that can have one carry a positive control beside them so a gate that refuses
EVERYTHING cannot pass by refusing.

**No row launches a kernel.** The subjects are the harness's own preconditions, which are
host-side by construction; the one row that needs a real `flock` takes it in a
subprocess.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bench import chunk_baseline
from bench.chunk_cells import CELLS, GROUPS, MANIFEST
from bench.discipline import TIERS, Run, _ancestor_holding, require_gpu_lock
from bench.pairing import WARMUP_LAUNCHES, Arm, PairedResult, measure_arm
from bench.regression import CANONICAL_INSTRUMENT
from bench.subjects import INTEGRATION, SUBJECTS, UNIT, applicable

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"

_SETTLED = {"settled": True, "sm_mhz": 1845.0, "plateau_mhz": 15.0, "elapsed_s": 1.1}


# ---------------------------------------------------------------------------
# VIOLATION 1 -- the lock
# ---------------------------------------------------------------------------

def test_a_bare_invocation_is_refused_because_no_ancestor_holds_the_lock(tmp_path):
    """PLANTED: the harness run without its `flock` wrapper.

    A shared device does not fail a measurement, it inflates it -- sometimes, which is
    the shape of a number nobody can reproduce or defend.
    """
    unheld = tmp_path / "rola_gpu.lock"
    with pytest.raises(SystemExit) as excinfo:
        require_gpu_lock(str(unheld))
    assert "no ancestor of this process holds" in str(excinfo.value)
    assert "flock" in str(excinfo.value)


def test_the_same_check_PASSES_under_a_real_flock_wrapper(tmp_path):
    """The positive control: a gate that refuses everything is not a gate."""
    lock = tmp_path / "rola_gpu.lock"
    lock.touch()
    program = (f"import sys; sys.path.insert(0, {str(BENCHMARKS)!r});\n"
               f"from bench.discipline import require_gpu_lock;\n"
               f"print(require_gpu_lock({str(lock)!r}))")
    out = subprocess.run(["flock", str(lock), sys.executable, "-c", program],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) > 1


def test_the_ancestor_scan_reports_nothing_for_a_path_no_one_opened(tmp_path):
    assert _ancestor_holding(str(tmp_path / "never-opened")) is None


# ---------------------------------------------------------------------------
# VIOLATION 2 -- the clock state
# ---------------------------------------------------------------------------

def test_an_unsettled_clock_aborts_the_run_rather_than_measuring(monkeypatch):
    """PLANTED: the box never reaches a plateau.

    A number taken while the clock is ramping is a number about the clock. The refusal
    happens in the identity step, before any row exists.
    """
    monkeypatch.setattr("bench.discipline._gpu_temp_c", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        Run(tier="light", cmd="planted", settle=lambda: {"settled": False, "reason": "planted"})
    assert "did not settle" in str(excinfo.value)


def test_a_tier_the_record_cannot_carry_is_refused(monkeypatch):
    monkeypatch.setattr("bench.discipline._gpu_temp_c", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        Run(tier="quick", cmd="planted", settle=lambda: dict(_SETTLED))
    assert str(TIERS) in str(excinfo.value)


# ---------------------------------------------------------------------------
# VIOLATION 3 -- the extension's identity
# ---------------------------------------------------------------------------

def test_an_extension_from_another_checkout_aborts_the_run(monkeypatch, tmp_path):
    """PLANTED: `import rola` resolves outside this tree.

    Both arms would then measure the SAME binary and simply agree -- which is also what a
    true negative looks like, so nothing downstream could tell the difference.
    """
    import rola

    monkeypatch.setattr(rola, "__file__", str(tmp_path / "other-tree" / "rola" / "__init__.py"))
    monkeypatch.setattr("bench.discipline._gpu_temp_c", lambda: None)
    with pytest.raises(SystemExit) as excinfo:
        Run(tier="light", cmd="planted", settle=lambda: dict(_SETTLED))
    message = str(excinfo.value)
    assert "NOT inside this checkout" in message and "PYTHONPATH" in message


# ---------------------------------------------------------------------------
# VIOLATION 4 -- the warmup policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("warmup", [0, 1, WARMUP_LAUNCHES - 1])
def test_measuring_without_the_warmup_floor_is_refused(warmup):
    """PLANTED: the first launches -- module load, schedule build, cold icache -- inside
    the recorded median."""
    arm = Arm(name="planted", call=lambda: None)
    with pytest.raises(ValueError) as excinfo:
        measure_arm(arm, rounds=1, reps_per_round=1, warmup=warmup)
    assert f"floor is {WARMUP_LAUNCHES}" in str(excinfo.value)


# ---------------------------------------------------------------------------
# VIOLATION 5 -- unpaired runs
# ---------------------------------------------------------------------------

def _result(order, a_times, b_times, reps=2):
    return PairedResult(a=Arm("a", None), b=Arm("b", None), order=list(order),
                        a_times=list(a_times), b_times=list(b_times), rounds=2,
                        reps_per_round=reps, warmup=WARMUP_LAUNCHES)


def test_block_sequential_samples_are_refused_as_a_paired_statistic():
    """PLANTED: arm A's block, then arm B's block -- the layout that lets a drift step
    land BETWEEN the arms. Measured at ~7x the run-to-run spread of the interleaved one."""
    blocked = _result(["a", "a", "a", "a", "b", "b", "b", "b"],
                      [1.0, 1.1, 1.0, 1.2], [2.0, 2.1, 2.0, 2.2])
    for statistic in (blocked.diffs, blocked.ratios):
        with pytest.raises(ValueError) as excinfo:
            statistic()
        assert "alternating the arms rep by rep" in str(excinfo.value)


def test_unequal_arms_are_refused_rather_than_truncated():
    ragged = _result(["a", "b", "a", "b"], [1.0, 1.1, 1.0], [2.0, 2.1])
    with pytest.raises(ValueError) as excinfo:
        ragged.diffs()
    assert "unpaired samples" in str(excinfo.value)


def test_the_interleaved_order_is_accepted_and_differences_are_per_round():
    """The positive control, and the shape of the quantity the significance test reads:
    one difference per ROUND, taken between the two arms' block medians."""
    paired = _result(["a", "b"] * 4, [1.0, 1.2, 1.0, 1.2], [2.0, 2.2, 3.0, 3.2])
    assert paired.diffs() == pytest.approx([1.0, 2.0])
    assert paired.ratios() == pytest.approx([2.0, 2.2 / 1.2, 3.0, 3.2 / 1.2])


# ---------------------------------------------------------------------------
# VIOLATION 6 -- the baseline's own discipline
# ---------------------------------------------------------------------------

def test_a_baseline_from_a_dirty_tree_is_refused(tmp_path):
    """PLANTED: a baseline recorded over uncommitted edits. Every later gate would then
    compare against a tree nobody can check out."""
    with pytest.raises(SystemExit) as excinfo:
        chunk_baseline.write({}, dirty=True, sha="0" * 40, build_stamp={}, sessions=7,
                             path=tmp_path / "b.json")
    assert "DIRTY tree" in str(excinfo.value)
    assert not (tmp_path / "b.json").exists()


def test_a_baseline_recorded_from_ONE_session_is_refused_on_both_sides(tmp_path):
    """PLANTED: the defect this harness shipped and then measured. A within-session block
    IQR is the spread of eleven reps on one warm fixture -- often exactly zero -- and the
    gate's comparison crosses two PROCESSES. Recorded that way, 17 of 44 arms went over
    their limit on the very next run of an unchanged binary."""
    with pytest.raises(SystemExit) as excinfo:
        chunk_baseline.write({}, dirty=False, sha="0" * 40, build_stamp={}, sessions=1,
                             path=tmp_path / "b.json")
    assert "across PROCESSES" in str(excinfo.value)
    assert not (tmp_path / "b.json").exists()

    measured = {"median_ms": 1.0, "source": CANONICAL_INSTRUMENT}
    with pytest.raises(SystemExit) as reading:
        chunk_baseline.compare(_baseline(CANONICAL_INSTRUMENT, sessions=1),
                               "chunk_forward", "c", measured)
    assert "cannot bound a cross-process comparison" in str(reading.value)


def test_the_committed_baseline_carries_its_session_count_and_clears_the_floor():
    """The file in the tree, not a fixture: the shipped baseline must itself satisfy the
    floor, or every gate run against it is refused."""
    committed = json.loads((BENCHMARKS / "cells" / "chunk_latency_baseline.json").read_text())
    assert committed["sessions"] >= chunk_baseline.MIN_SESSIONS
    roster = {chunk_baseline.key(subject, name)
              for name, cell in CELLS.items() for subject in applicable(cell)}
    assert set(committed["arms"]) == roster
    for name, row in committed["arms"].items():
        assert row["sessions"] >= chunk_baseline.MIN_SESSIONS, name
        assert row["source"] == CANONICAL_INSTRUMENT, name
        assert "within_session_iqr_ms" in row, name


def _baseline(source, sessions=7):
    row = {"median_ms": 1.0, "iqr_ms": 0.01, "reps": 8, "sessions": sessions}
    if source is not None:
        row["source"] = source
    return {"provenance": {}, "alarm_sigma": 3.0, "sessions": sessions,
            "arms": {"chunk_forward|c": row}}


def test_a_gate_across_two_stopwatches_is_refused():
    """PLANTED: a baseline taken with the host timer, gated against a device-event run.
    The gap between them is the launch and sync cost, not noise."""
    measured = {"median_ms": 1.0, "source": CANONICAL_INSTRUMENT}
    with pytest.raises(ValueError) as excinfo:
        chunk_baseline.compare(_baseline("perf_counter"), "chunk_forward", "c", measured)
    assert "different quantities" in str(excinfo.value)


def test_a_baseline_that_does_not_say_which_stopwatch_took_it_is_refused():
    measured = {"median_ms": 1.0, "source": CANONICAL_INSTRUMENT}
    with pytest.raises(ValueError) as excinfo:
        chunk_baseline.compare(_baseline(None), "chunk_forward", "c", measured)
    assert "does not say which stopwatch" in str(excinfo.value)


def test_a_measured_arm_with_no_committed_row_is_refused_not_skipped():
    """PLANTED: a cell the gate measures and the baseline never recorded -- a gate that
    silently checks nothing."""
    measured = {"median_ms": 1.0, "source": CANONICAL_INSTRUMENT}
    with pytest.raises(SystemExit) as excinfo:
        chunk_baseline.compare(_baseline(CANONICAL_INSTRUMENT), "decode_step", "c", measured)
    assert "no committed baseline row" in str(excinfo.value)


def test_the_limit_is_derived_from_the_recorded_cell_own_spread():
    """The positive control: the same comparison decides, and the line it decides at is
    the recorded IQR's, never a flat percentage."""
    baseline = _baseline(CANONICAL_INSTRUMENT)
    inside = chunk_baseline.compare(baseline, "chunk_forward", "c",
                                    {"median_ms": 1.01, "source": CANONICAL_INSTRUMENT})
    outside = chunk_baseline.compare(baseline, "chunk_forward", "c",
                                     {"median_ms": 1.5, "source": CANONICAL_INSTRUMENT})
    assert inside["verdict"] == "within_limit"
    assert outside["verdict"] == "over_limit"
    assert outside["limit_ms"] == pytest.approx(1.0 + 3.0 * 0.01 / 1.349)


# ---------------------------------------------------------------------------
# THE REGISTRY AND THE TAXONOMY
# ---------------------------------------------------------------------------

def test_every_cell_is_named_once_and_every_group_names_real_cells():
    assert len(CELLS) == len({cell.name for cell in CELLS.values()})
    for group, names in GROUPS.items():
        assert names, f"group {group!r} is empty, so citing it cites nothing"
        assert set(names) <= set(CELLS)


def test_the_registry_carries_both_ends_of_the_distribution_and_a_decode_cell():
    """A single-density registry makes every verdict an artifact of that density."""
    assert any(cell.alpha is None for cell in CELLS.values()), "no dense cell"
    assert any(cell.logit_gain > 1.0 for cell in CELLS.values()), "no sparse cell"
    assert any(len(cell.widths) >= 4 for cell in CELLS.values()), "no deep-arm cell"
    assert GROUPS["decode"], "no cell continues into decode"


def test_the_committed_manifest_covers_every_registered_cell():
    """A cell with no manifest row is a cell whose drift nothing announces."""
    committed = json.loads(MANIFEST.read_text())
    assert set(committed) == set(CELLS)
    for name, row in committed.items():
        cell = CELLS[name]
        assert row["widths"] == list(cell.widths) and row["seed"] == cell.seed
        assert row["T"] == cell.T and row["B"] == cell.B


def test_every_subject_has_a_driver_in_its_own_tier():
    """THE TAXONOMY, gated: `benchmarks/unit/` is one driver per custom kernel and
    `benchmarks/integration/` is the compositions, so a subject cannot be added to the
    roster without a way to run it -- or filed under the wrong tier."""
    for tier, roster in (("unit", UNIT), ("integration", INTEGRATION)):
        sources = "\n".join(path.read_text() for path in (BENCHMARKS / tier).glob("bench_*.py"))
        assert sources, f"benchmarks/{tier}/ has no drivers"
        for subject in roster:
            assert f'"{subject}"' in sources, f"{subject} has no driver under benchmarks/{tier}/"


def test_the_decode_subjects_are_offered_only_where_a_state_continues():
    for cell in CELLS.values():
        names = applicable(cell)
        assert set(names) <= set(SUBJECTS)
        assert ("decode_step" in names) == (cell.decode_steps > 0)


def test_the_drivers_state_that_they_are_invoked_wrapped():
    """The locking statement is the successor's owed one, so it is gated rather than
    remembered: every driver's usage line carries the `flock` this harness requires."""
    drivers = list((BENCHMARKS / "unit").glob("bench_*.py"))
    drivers += list((BENCHMARKS / "integration").glob("bench_*.py"))
    drivers.append(BENCHMARKS / "bench_chunk.py")
    for path in drivers:
        assert "flock /tmp/rola_gpu.lock" in path.read_text(), path


def test_the_harness_never_names_a_second_lock_path():
    """`flock` is advisory and keyed on a NAME: a second spelling excludes nobody."""
    for path in BENCHMARKS.rglob("*.py"):
        text = path.read_text()
        for token in ("rola-gpu.lock", "rola.lock", "/tmp/gpu.lock"):
            assert token not in text, f"{path} names a second lock spelling: {token}"


def test_the_environment_variable_moves_the_lock_and_cannot_disable_it():
    """`ROLA_GPU_LOCK` composes with a genuinely different OUTER lock. There is no branch
    behind it that skips the check, which is what makes it a move and not an off switch."""
    source = (BENCHMARKS / "bench" / "discipline.py").read_text()
    assert 'os.environ.get("ROLA_GPU_LOCK"' in source
    assert "def require_gpu_lock" in source
