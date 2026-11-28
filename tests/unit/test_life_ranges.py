"""The life-range report's registers (`tools/life_ranges.py`, docs/internals/tools/life_ranges.md#liveness): warpline's pressure over
the life-range marks (`flow.pressure`), read through the seam (`tools/checked.py`): a value held across a division's slow-path call
counts before the call -- the shape nvdisasm's own count drops -- on a compiled probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import checked  # noqa: E402
import dev_config  # noqa: E402
import life_ranges  # noqa: E402

#: the probe, compiled by the graph's CUDA rule (tests/BUILD.bazel's `life_ranges_probe`): its one cubin
PROBE = ROOT / "tests" / "_objs" / "life_ranges_probe" / "life_ranges_probe.pic.o"


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    found = checked.checker().runner.cubins(PROBE, tmp_path_factory.mktemp("cubins"), dev_config.cuda_home())
    assert len(found) == 1, found
    return found[0]


def test_values_held_across_a_slow_path_call_count_before_the_call(cubin):
    """nvdisasm's marks have the call write registers the probe holds (its own count drops 16 of them at the row
    before the call): the call's registers are its subroutine's."""
    rows = life_ranges.live_counts(cubin, "division_probe")
    before = [rows[k - 1][2] for k, (_, text, _) in enumerate(rows) if text.startswith("CALL") and "0x" in text]
    assert before, "the probe's division has no slow-path call: the probe no longer has the shape it tests"
    assert min(before) >= 32, rows

