# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A checkout's scratch directory is its own (tools/dev_config.py `scratch`): keyed by the checkout's root, stable for one
root, different for two, so two checkouts building one experiment never share an intermediate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import dev_config  # noqa: E402


def test_scratch_is_stable_for_one_root(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_config, "get", lambda key: str(tmp_path))
    assert dev_config.scratch("lab") == dev_config.scratch("lab")
    assert dev_config.scratch("lab").parent == dev_config.scratch("cubins").parent


def test_scratch_differs_for_two_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_config, "get", lambda key: str(tmp_path))
    here = dev_config.scratch("lab")
    monkeypatch.setattr(dev_config, "ROOT", Path("/somewhere/else/rola"))
    there = dev_config.scratch("lab")
    assert here != there
    assert here.parent.parent == there.parent.parent == tmp_path
    assert here.name == there.name == "lab"
