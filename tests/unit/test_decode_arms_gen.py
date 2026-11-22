# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""DECODE'S LAUNCH FUNCTION OVER THE CONTRACT AND THE ARM GENERATOR (`csrc/fsla/src/common/plan/decode_launch.h`, `arms_gen.cpp`,
docs/internals/decode/arms_gen.md): the arms the launch function reaches over the contract are the decode arm list, in canonical order
(`d_v`, then `levels`, then `decay`); a contract context no arm covers fails the generator, naming the context, and so fails the build; a
point reaching other than one arm, a range's point outside the range's arms and a context with too many ranged facts to try are refused."""
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

#: every arm the contract reaches: each built value width at each depth and decay
REACHED = [[dv, levels, decay] for dv in (32, 64) for levels in (1, 2, 3, 4) for decay in (0, 1)]


def _run(contexts: str, tmp_path: Path):
    path = tmp_path / "contexts.txt"
    path.write_text(contexts)
    out = tmp_path / "out"
    out.mkdir()
    return subprocess.run([str(TOOL), "--contexts", str(path), "--out-dir", str(out)], capture_output=True, text=True, timeout=60), out


def _contract_with(tmp_path: Path, parts: dict | None = None, **call_facts) -> str:
    """The contract's text with the decode call part's facts replaced (`d_v=[32, 64, 128]`), and any other part's (`{"device/hopper_sm90a": {...}}`)."""
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
    for part, facts in {"calls/decode": call_facts, **(parts or {})}.items():
        file = root / "parts" / f"{part}.json"
        data = json.loads(file.read_text())
        for fact, value in facts.items():
            data["facts"][fact]["value"] = value
        file.write_text(json.dumps(data))
    return registry.contexts_text(registry.contexts(root))


def test_the_launch_function_over_the_contract_reaches_the_sixteen_arms_in_canonical_order(tmp_path):
    run, out = _run(registry.contexts_text(registry.contexts()), tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == REACHED
    assert sorted(path.name for path in out.iterdir()) == ["decode_arm_rows.json", "decode_selection_record.txt"]


def test_the_graphs_arm_list_is_that_list_and_each_context_reaches_both_value_widths():
    assert json.loads((PLAN / "decode_arm_rows.json").read_text()) == {"family": "decode", "key": ["d_v", "levels", "decay"], "arms": REACHED}
    lines = (PLAN / "decode_selection_record.txt").read_text().splitlines()[1:]
    assert len(lines) == 8
    for line in lines:
        context, reach = line.split("\t-> ")
        levels, decay = (int(cell.split("=")[1]) for cell in context.split()[1:])
        assert context == f"hopper_decode levels={levels} decay={decay}"
        indices = [REACHED.index([dv, levels, decay]) for dv in (32, 64)]
        assert reach == " ".join(f"{i}(d_v={REACHED[i][0]}, levels={levels}, decay={decay})" for i in indices)


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


def test_a_range_wider_than_the_arms_fails_at_the_point_no_arm_serves(tmp_path):
    run, out = _run(_contract_with(tmp_path, d_v={"lo": 2, "hi": 128}), tmp_path)
    assert run.returncode == 1
    assert "context 'hopper_decode" in run.stderr and "d_v=128" in run.stderr and "maps to 0 arms, not one" in run.stderr
    assert not list(out.iterdir())


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


def test_a_range_inside_one_arms_width_reaches_that_arm_alone(tmp_path):
    run, out = _run(_contract_with(tmp_path, d_v={"lo": 33, "hi": 64}, levels=[1], decay=[0]), tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == [[64, 1, 0]]


def test_a_context_with_more_ranged_facts_than_the_property_tries_is_refused_not_passed(tmp_path):
    """Every integer fact a range: nine of them, over the eight whose corners are tried."""
    ranged = {name: {"lo": value, "hi": value + 1} for name, value in
              [("cc", 900), ("regs_per_sm", 65536), ("smem_per_sm", 233472), ("smem_per_cta_optin", 232448), ("max_threads_per_sm", 2048),
               ("sm_count", 114)]}
    contract = _contract_with(tmp_path, parts={"device/hopper_sm90a": ranged}, d_v={"lo": 2, "hi": 64},
                              levels={"lo": 1, "hi": 2}, decay={"lo": 0, "hi": 1})
    run, out = _run(contract, tmp_path)
    assert run.returncode == 1
    assert "has 9 ranged facts" in run.stderr and "refused" in run.stderr
    assert not list(out.iterdir())
