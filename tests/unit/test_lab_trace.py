# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""`lab trace` (tools/lab_trace.py) reads a coarse row's and an edges row's launch: each CTA's entry to exit in cycles and
in nanoseconds, the rate between them, the warps' spread -- and an edges row, which holds no window, for what it holds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab_trace  # noqa: E402

WINDOW, ENTRY, EXIT = 0xC000, 0xC008, 0xC009
PLAIN = "the carry"
EDGES = PLAIN + ", the launch's two edges stamped"
COARSE = PLAIN + ", every warp's phases stamped"


def mark(site, cycles, ns):
    """A mark: the stamp on the cycle counter, then the timer's low word at the same site."""
    return [cycles, 1 << 16 | site, ns, 6 << 16 | site]


def warp(entry, exit_, windows=()):
    stamps = mark(ENTRY, *entry)
    for cycles, ns in windows:
        stamps += mark(WINDOW, cycles, ns)
    stamps += mark(EXIT, *exit_)
    return {"count": len(stamps) // 2, "stamps": stamps}


def row(variant, warps):
    legend = {str(WINDOW): "window", str(ENTRY): "entry", str(EXIT): "exit"}
    return {"variant": variant, "roles": ["fold"] * len(warps), "legend": legend, "ctas": [{"cta": 0, "warps": warps}]}


def test_the_names_of_an_untraced_twin():
    assert lab_trace.untraced_name(EDGES) == PLAIN
    assert lab_trace.untraced_name(COARSE) == PLAIN
    assert lab_trace.untraced_name(PLAIN + ", every warp's timeline stamped") == PLAIN
    assert lab_trace.untraced_name(PLAIN) == PLAIN


def test_an_edges_row_reads_its_launch(capsys):
    # two warps: entries 10 cycles apart, exits 30 apart; 1,000 cycles to 600 ns is 1,666.7 cycles a microsecond
    warps = [warp((0, 0), (1000, 600)), warp((10, 32), (1030, 640))]
    cta = lab_trace.Cta(row(EDGES, warps), row(EDGES, warps)["ctas"][0])
    launch = cta.launch()
    assert launch["cycles"] == 1030 and launch["ns"] == 640
    assert abs(launch["rate"] - 1000.0 * 1030 / 640) < 1e-9
    assert launch["entry_spread"] == 10 and launch["exit_spread"] == 30
    assert lab_trace.report([row(EDGES, warps)], None, 3, {EDGES: 5288.0, PLAIN: 5295.0}, None) == 0
    out = capsys.readouterr().out
    assert "launch 1,030 cycles, 640 ns, 1,609.4 cycles a microsecond" in out
    assert "too few windows" not in out
    assert "(-0.1% the trace's cost)" in out


def test_the_timer_word_wraps():
    warps = [warp((0, (1 << 32) - 100), (500, 300))]
    assert lab_trace.Cta(row(EDGES, warps), row(EDGES, warps)["ctas"][0]).launch()["ns"] == 400


def test_a_coarse_row_keeps_its_steady_span_inside_the_edges(capsys):
    windows = [(1000 * i, 600 * i) for i in range(1, 9)]
    warps = [warp((0, 0), (9500, 5700), windows), warp((5, 0), (9400, 5600), windows)]
    cta = lab_trace.Cta(row(COARSE, warps), row(COARSE, warps)["ctas"][0])
    # the edges are not the span's: it runs from the third window to the second last, and every window is one the fold's
    t0, t1, n = cta.steady(3)
    assert (t0, t1, n) == (4000, 7000, 3)
    assert cta.launch()["cycles"] == 9500
    lab_trace.report([row(COARSE, warps)], None, 3, {COARSE: 1000.0, PLAIN: 1000.0}, None)
    out = capsys.readouterr().out
    assert "windows 3 to 6, 1,000 cycles a window" in out
    assert "launch 9,500 cycles, 5,700 ns, 1,666.7 cycles a microsecond" in out


def test_a_row_without_edges_has_no_launch(capsys):
    stamps = mark(WINDOW, 0, 0) + mark(WINDOW, 100, 60)
    warps = [{"count": 4, "stamps": stamps}]
    r = row(PLAIN, warps)
    assert lab_trace.Cta(r, r["ctas"][0]).launch() is None
    lab_trace.report([r], None, 3, {}, None)
    assert "too few windows stamped" in capsys.readouterr().out
