# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""DECODE'S MATCHER AND ITS ARM GENERATOR (`csrc/fsla/src/common/plan/decode_launch.h`, `arms_gen.cpp`, docs/internals/decode/arms_gen.md): the arm list the matcher reaches
over the contract is the sixteen rows the family was declared with, in the same order; a contract context no arm covers fails the generator,
naming the context, and so fails the build."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "decode_arms_gen"
PLAN = ROOT / "csrc" / "fsla" / "decode_plan"

#: the rows `csrc/fsla/src/decode/arms.bzl` declared before the matcher, in the order of their indices
TODAYS_ROWS = [[dv, levels, decay] for dv in (32, 64) for levels in (1, 2, 3, 4) for decay in (0, 1)]


def _run(contexts: str, tmp_path: Path):
    path = tmp_path / "contexts.txt"
    path.write_text(contexts)
    out = tmp_path / "out"
    out.mkdir()
    return subprocess.run([str(TOOL), "--contexts", str(path), "--out-dir", str(out)], capture_output=True, text=True, timeout=60), out


def _contract_with(tmp_path: Path, **call_facts) -> str:
    """The contract's text with the decode call part's facts replaced (`d_v=[32, 64, 128]`)."""
    root = tmp_path / "registry"
    root.mkdir()
    for name in ("schema.json",):
        (root / name).write_text((registry.ROOT / name).read_text())
    for part in registry.parts():
        target = root / "parts" / f"{part}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((registry.ROOT / "parts" / f"{part}.json").read_text())
    (root / "descriptors").mkdir()
    for path in (registry.ROOT / "descriptors").glob("*.json"):
        (root / "descriptors" / path.name).write_text(path.read_text())
    calls = json.loads((root / "parts" / "calls" / "decode.json").read_text())
    for fact, value in call_facts.items():
        calls["facts"][fact]["value"] = value
    (root / "parts" / "calls" / "decode.json").write_text(json.dumps(calls))
    return registry.contexts_text(registry.contexts(root))


def test_the_matcher_over_the_contract_reaches_todays_sixteen_arms_in_todays_order(tmp_path):
    run, out = _run(registry.contexts_text(registry.contexts()), tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == TODAYS_ROWS
    built = (out / "decode_built.h").read_text()
    assert "kBuiltCount = 16" in built and "{32, 1, 0}, {32, 1, 1}" in built and "{64, 4, 1}};" in built


def test_the_graphs_arm_list_is_that_list_and_each_context_reaches_exactly_one_arm():
    assert json.loads((PLAN / "decode_arm_rows.json").read_text()) == {"family": "decode", "key": ["d_v", "levels", "decay"], "arms": TODAYS_ROWS}
    lines = (PLAN / "decode_selection_record.txt").read_text().splitlines()[1:]
    assert len(lines) == 16
    for index, line in enumerate(lines):
        context, reach = line.split("\t-> ")
        d_v, levels, decay = TODAYS_ROWS[index]
        assert context == f"hopper_decode d_v={d_v} levels={levels} decay={decay}"
        assert reach == f"{index}(d_v={d_v}, levels={levels}, decay={decay})"


@pytest.mark.parametrize("fact, value, named", [
    ("d_v", [32, 64, 128], "d_v=128"),
    ("levels", [1, 2, 3, 4, 5], "levels=5"),
    ("decay", [0, 1, 2], "decay=2"),
])
def test_a_contract_context_no_arm_covers_fails_the_generator_naming_it(tmp_path, fact, value, named):
    run, out = _run(_contract_with(tmp_path, **{fact: value}), tmp_path)
    assert run.returncode == 1
    assert "context 'hopper_decode" in run.stderr and named in run.stderr and "maps it to NONE" in run.stderr
    assert "decode_launch.h" in run.stderr
    assert not list(out.iterdir()), "a refused contract writes no arm list"


def test_a_context_missing_a_schema_fact_is_refused(tmp_path):
    line = "hopper_decode d_v=32 levels=1 decay=0\tcc=900:900\td_v=32:32\tlevels=1:1\tdecay=0:0\n"
    run, _ = _run(line, tmp_path)
    assert run.returncode == 1 and "states no value for the schema's fact" in run.stderr


def test_a_range_reaches_every_arm_the_range_can(tmp_path):
    contract = _contract_with(tmp_path, d_v={"lo": 32, "hi": 64}, levels=[1], decay=[0])
    assert "d_v=32:64" in contract
    run, out = _run(contract, tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == [[32, 1, 0], [64, 1, 0]]
    assert "0(d_v=32, levels=1, decay=0) 1(d_v=64, levels=1, decay=0)" in (out / "decode_selection_record.txt").read_text()
