"""THE CENSUS TAKES THE BUILD'S PARAMETERS AND NOTHING ELSE.

`tools/ratify.py` reads the units the graph's CUDA rule compiled (docs/internals/tools/ratify.md#graph) and runs one of
four modes; its scope is the census record's, so it has no flag that names an architecture, a shard or an arm, and it
holds no launch bound of its own to report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ratify  # noqa: E402


def test_the_modes_are_the_census_the_derivation_and_the_self_test(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ratify.py", "--graph", "census.json", "--runfiles", "."])
    with pytest.raises(SystemExit):
        ratify.main()
    assert "one of the arguments --census --derivation --write-derivation --self-test is required" in capsys.readouterr().err


def test_the_scope_is_the_census_records(monkeypatch, capsys):
    for flag in ("--arch", "--shard", "--arm"):
        monkeypatch.setattr(sys, "argv", ["ratify.py", "--census", flag, "x", "--graph", "c.json", "--runfiles", "."])
        with pytest.raises(SystemExit):
            ratify.main()
        assert "unrecognized arguments" in capsys.readouterr().err


def test_the_json_record_is_the_censuss(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ratify.py", "--derivation", "--json", "x", "--graph", "c.json", "--runfiles", "."])
    with pytest.raises(SystemExit, match="--json is the census's"):
        ratify.main()
