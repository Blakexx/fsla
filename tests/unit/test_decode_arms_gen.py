# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""DECODE'S LAUNCH FUNCTION OVER THE CONTRACT AND THE ARM GENERATOR (`csrc/fsla/src/common/plan/decode_launch.h`, `arms_gen.cpp`,
docs/internals/decode/arms_gen.md): the arms the launch function reaches over the contract are the decode arm list, in canonical order
(`d_v`, then `levels`, then `decay`); a contract context no arm covers fails the generator, naming the context, and so fails the build; a
context whose ranges hold a point no arm covers fails it too, naming such a point, whatever the count of its ranged facts (the coverage check is the rule
language's three-valued walk over the ranges, exact: no corner is sampled)."""
from __future__ import annotations

import itertools
import json
import random
import re
import shutil
import subprocess
import sys
import time
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


def _contract_with(tmp_path: Path, parts: dict | None = None, levels=None, **call_facts) -> str:
    """The decode contract's text with the decode `call` part's facts replaced (`d_v=[32, 64, 128]`), and any other part's (`{"device/hopper_sm90a": {...}}`).
    `levels` replaces the depth: the contract is then the depth-1 descriptor alone, its part's `levels` the given value."""
    root = tmp_path / "registry"
    shutil.copytree(registry.ROOT, root, ignore=shutil.ignore_patterns("__pycache__", "*.py", "BUILD.bazel"))
    if levels is not None:
        for path in (root / "kernels" / "decode" / "descriptors").glob("*.json"):
            if path.stem != "hopper_decode_d1":
                path.unlink()
    depth = {"depth1": {"levels": levels}} if levels is not None else {}
    for part, facts in {"call": call_facts, **depth, **(parts or {})}.items():
        file = (root / "parts" if part.startswith("device/") else root / "kernels" / "decode" / "parts") / f"{part}.json"
        data = json.loads(file.read_text())
        for fact, value in facts.items():
            data["facts"][fact]["value"] = value
        file.write_text(json.dumps(data))
    return registry.contexts_text("decode", registry.contexts("decode", root))


def test_the_launch_function_over_the_contract_reaches_the_sixteen_arms_in_canonical_order(tmp_path):
    run, out = _run(registry.contexts_text("decode", registry.contexts("decode")), tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == REACHED
    assert sorted(path.name for path in out.iterdir()) == ["decode_arm_rows.json", "decode_selection_record.txt"]


def test_the_graphs_arm_list_is_that_list_and_each_context_reaches_both_value_widths():
    assert json.loads((PLAN / "decode_arm_rows.json").read_text()) == {"family": "decode", "key": ["d_v", "levels", "decay"], "arms": REACHED}
    lines = (PLAN / "decode_selection_record.txt").read_text().splitlines()[1:]
    assert len(lines) == 8
    for line in lines:
        context, reach = line.split("\t-> ")
        name, decay_cell = context.split()
        levels, decay = int(name.removeprefix("hopper_decode_d")), int(decay_cell.split("=")[1])
        assert context == f"hopper_decode_d{levels} decay={decay}"
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
    assert "context 'hopper_decode" in run.stderr and named in run.stderr and "to NONE" in run.stderr
    assert "decode_launch.h" in run.stderr
    assert not list(out.iterdir()), "a refused contract writes no arm list"


def test_a_range_wider_than_the_arms_fails_at_a_point_no_arm_serves(tmp_path):
    run, out = _run(_contract_with(tmp_path, d_v={"lo": 2, "hi": 128}), tmp_path)
    assert run.returncode == 1
    assert "context 'hopper_decode" in run.stderr and "maps the point" in run.stderr and "d_v=65" in run.stderr and "to NONE" in run.stderr
    assert not list(out.iterdir())


@pytest.mark.parametrize("fact, value, witness", [
    ("d_v", {"lo": 0, "hi": 40}, "d_v=0"),
    ("levels", {"lo": 3, "hi": 9}, "levels=5"),
    ("decay", {"lo": -1, "hi": 1}, "decay=-1"),
])
def test_a_hole_only_some_points_of_a_range_fall_in_is_found_and_named(tmp_path, fact, value, witness):
    run, out = _run(_contract_with(tmp_path, **{fact: value}), tmp_path)
    assert run.returncode == 1
    assert "the point" in run.stderr and witness in run.stderr and "to NONE" in run.stderr
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


NINE = list(registry.device_point().items())


def test_a_context_with_nine_ranged_facts_is_checked_exactly_and_fast(tmp_path):
    """All nine integer facts ranged: the walk is exact at any number of ranged facts and fast, and finds the gap at the cc the contract does not serve."""
    ranged = {name: {"lo": value - 1, "hi": value + 1} for name, value in NINE}
    contract = _contract_with(tmp_path, parts={"device/hopper_sm90a": ranged}, d_v={"lo": 2, "hi": 64}, levels={"lo": 1, "hi": 2},
                              decay={"lo": 0, "hi": 1})
    assert len(re.findall(r"\w+=\d+:\d+", contract.splitlines()[-1])) >= 9
    start = time.monotonic()
    run, out = _run(contract, tmp_path)
    assert time.monotonic() - start < 5
    assert run.returncode == 1
    assert "ranged facts" not in run.stderr and "cc=899" in run.stderr and "to NONE" in run.stderr
    assert not list(out.iterdir())


def test_eight_ranged_facts_the_launch_function_serves_pass(tmp_path):
    ranged = {name: {"lo": value - 1, "hi": value + 1} for name, value in NINE[1:]}
    contract = _contract_with(tmp_path, parts={"device/hopper_sm90a": ranged}, d_v={"lo": 2, "hi": 64}, levels={"lo": 1, "hi": 4},
                              decay={"lo": 0, "hi": 1})
    run, out = _run(contract, tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == REACHED


def _expected(d_v, levels, decay):
    """The arms the ranges reach and whether some point of them reaches none, by enumerating every point (the oracle of the exact walk)."""
    arms, gap = set(), False
    for dv, lv, dc in itertools.product(range(d_v[0], d_v[1] + 1), range(levels[0], levels[1] + 1), range(decay[0], decay[1] + 1)):
        if 1 <= dv <= 64 and 1 <= lv <= 4 and dc in (0, 1):
            arms.add((32 if dv <= 32 else 64, lv, dc))
        else:
            gap = True
    return sorted(arms), gap


def test_the_walk_is_exact_against_every_point_of_seeded_ranges(tmp_path):
    """The walk over a context's ranges reaches the arms and the NONE leaf of exactly the points inside: seeded random ranges, many straddling an arm width."""
    rng = random.Random(23)

    def span(low, high, reach):
        lo = rng.randint(low, high)
        return lo, rng.randint(lo, lo + reach)

    for case in range(40):
        d_v, levels, decay = span(0, 70, rng.choice([0, 3, 40])), span(0, 6, 4), span(-1, 2, 2)
        sub = tmp_path / f"c{case}"
        sub.mkdir()
        run, out = _run(_contract_with(sub, d_v={"lo": d_v[0], "hi": d_v[1]}, levels={"lo": levels[0], "hi": levels[1]},
                                       decay={"lo": decay[0], "hi": decay[1]}), sub)
        arms, gap = _expected(d_v, levels, decay)
        assert (run.returncode == 1) == gap, (d_v, levels, decay, run.stderr)
        if gap:
            assert "to NONE" in run.stderr
        else:
            assert json.loads((out / "decode_arm_rows.json").read_text())["arms"] == [list(a) for a in arms], (d_v, levels, decay)


def test_sampled_points_of_every_context_reach_an_arm_among_the_contexts(tmp_path):
    """The sanity check of the argument, not its guarantee: seeded random points inside each context of the contract and of a wide one."""
    wide = _contract_with(tmp_path, d_v={"lo": 1, "hi": 64}, levels={"lo": 1, "hi": 4}, decay={"lo": 0, "hi": 1})
    for n, contexts in enumerate((registry.contexts_text("decode", registry.contexts("decode")), wide)):
        sub = tmp_path / f"s{n}"
        sub.mkdir()
        (sub / "contexts.txt").write_text(contexts)
        (sub / "out").mkdir()
        run = subprocess.run([str(TOOL), "--contexts", str(sub / "contexts.txt"), "--out-dir", str(sub / "out"), "--sample", "300", "--seed", "93"],
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr
