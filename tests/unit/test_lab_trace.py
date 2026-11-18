# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""`lab trace` (tools/lab_trace.py) reads a coarse row's and an edges row's launch from warpline's native record: each CTA's
entry to exit in cycles and in nanoseconds, the rate between them, the warps' spread -- and an edges row, which holds no
window, for what it holds."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import dev_config  # noqa: E402

if dev_config.warpline_python() is None:
    pytest.skip("warpline's library is handed in by the graph (bazel test //tests:cpu)", allow_module_level=True)
import lab_trace  # noqa: E402

PLAIN = "the carry"
EDGES = PLAIN + ", the launch's two edges stamped"
COARSE = PLAIN + ", every warp's phases stamped"


def record(variant, warps):
    """A native record of one CTA, a warp a lane: each warp is `(entry, exit, windows)`, a point a `(cycles, nanoseconds)`
    -- the launch's edges and each window's start, the stamp and its tie, the window phase closed where the next opens
    and at the exit. A warp with no entry holds no edge, and its last window stays open (a row with no exit)."""
    names, stamps, ties = {}, [], []

    def stamp(lane, name, point, tie=True):
        stamps.append([lane, names.setdefault(name, len(names)), point[0]])
        if tie:
            ties.append([lane, point[0], point[1]])

    for w, (entry, exit_, windows) in enumerate(warps):
        lane = f"cta0/warp{w}"
        if entry:
            stamp(lane, "launch.begin", entry)
        for i, point in enumerate(windows):
            if i:
                stamp(lane, "phase.window.end", point, tie=False)
            stamp(lane, "phase.window.begin", point)
        if entry:
            if windows:
                stamp(lane, "phase.window.end", exit_, tie=False)
            stamp(lane, "launch.end", exit_)
    return lab_trace.wl_record.from_dict({
        "format": "warpline/0", "clock": {"unit": "cycles", "hz": None, "time_unit": "ns", "time_origin": "the test's"},
        "names": {str(i): n for n, i in names.items()},
        "lanes": [{"id": f"cta0/warp{w}", "role": "fold", "group": "fold"} for w in range(len(warps))],
        "stamps": stamps, "dropped": {}, "facts": {"barriers": {}, "operations": {}}, "ties": ties,
        "source": {"variant": variant}})


def test_the_names_of_an_untraced_twin():
    assert lab_trace.untraced_name(EDGES) == PLAIN
    assert lab_trace.untraced_name(COARSE) == PLAIN
    assert lab_trace.untraced_name(PLAIN + ", every warp's timeline stamped") == PLAIN
    assert lab_trace.untraced_name(PLAIN) == PLAIN


def test_an_edges_row_reads_its_launch(capsys):
    # two warps: entries 10 cycles apart, exits 30 apart; 1,000 cycles to 600 ns is 1,666.7 cycles a microsecond
    rec = record(EDGES, [((0, 0), (1000, 600), []), ((10, 32), (1030, 640), [])])
    launch = lab_trace.Cta(rec, "cta0").launch()
    assert launch["cycles"] == 1030 and launch["ns"] == 640
    assert abs(launch["rate"] - 1000.0 * 1030 / 640) < 1e-9
    assert launch["entry_spread"] == 10 and launch["exit_spread"] == 30
    assert lab_trace.report([(EDGES, rec)], None, 3, {EDGES: 5288.0, PLAIN: 5295.0}, None) == 0
    out = capsys.readouterr().out
    assert "launch 1,030 cycles, 640 ns, 1,609.4 cycles a microsecond" in out
    assert "too few windows" not in out
    assert "(-0.1% the trace's cost)" in out


def test_a_coarse_row_keeps_its_steady_span_inside_the_edges(capsys):
    windows = [(1000 * i, 600 * i) for i in range(1, 9)]
    rec = record(COARSE, [((0, 0), (9500, 5700), windows), ((5, 0), (9400, 5600), windows)])
    cta = lab_trace.Cta(rec, "cta0")
    # the edges are not the span's: it runs from the third window to the second last, and every window is one the fold's
    assert cta.steady(3) == (4000, 7000, 3)
    assert cta.launch()["cycles"] == 9500
    lab_trace.report([(COARSE, rec)], None, 3, {COARSE: 1000.0, PLAIN: 1000.0}, None)
    out = capsys.readouterr().out
    assert "windows 3 to 6, 1,000 cycles a window" in out
    assert "launch 9,500 cycles, 5,700 ns, 1,666.7 cycles a microsecond" in out
    # the role's phase and its rate, warpline's: the window phase lasts the window, 1,000 cycles to 600 ns
    assert "1,667 cycles a microsecond" in out


def test_a_row_without_edges_has_no_launch(capsys):
    rec = record(PLAIN, [(None, None, [(0, 0), (100, 60)])])
    assert lab_trace.Cta(rec, "cta0").launch() is None
    lab_trace.report([(PLAIN, rec)], None, 3, {}, None)
    assert "too few windows stamped" in capsys.readouterr().out


def test_a_row_with_no_exit_leaves_its_last_phase_open_and_that_is_no_finding():
    windows = [(1000 * i, 600 * i) for i in range(1, 9)]
    assert lab_trace.findings(record(COARSE, [((0, 0), (9500, 5700), windows)])) == []
    open_row = record(COARSE, [(None, None, windows)])
    assert [f.code for f in lab_trace.check.all_findings(open_row)] == ["region_unclosed"]
    assert lab_trace.findings(open_row) == []
