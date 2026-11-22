# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A DEVELOPMENT BUILD'S NARROWED CONTRACT (`hardware/descriptors/dev/`, `--//:contract`, docs/build.md#narrowed): a narrowed contract is a descriptor of its
own, marked narrowed and inside the full contract; the arm generator built under it compiles exactly the arms it reaches (fewer than the full
contract's sixteen); the run time built under it refuses a call outside it, naming it; the full contract is the default. No GPU."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402

NARROWED = "decode_d32_l2"
PLAN = ROOT / "csrc" / "fsla" / "decode_plan_narrowed"
FULL_PLAN = ROOT / "csrc" / "fsla" / "decode_plan"
TOOL = ROOT / "csrc" / "fsla" / "plan_tool_narrowed" / "plan_tool"
HOPPER = "cc=900,regs_per_sm=65536,smem_per_sm=233472,smem_per_cta_optin=232448,max_threads_per_sm=2048,sm_count=132"
#: the call's facts beside `d_v`, `levels` and `decay`: the widths of a depth-2 call and the step's action
REST = "write_dense=1,width_bits_1=4,width_bits_2=5,width_bits_3=0,width_bits_4=0"


def _plan(call: str):
    return subprocess.run([str(TOOL), "--device", HOPPER, "--call", f"{call},{REST}"], capture_output=True, text=True, timeout=60)


@pytest.fixture()
def tree(tmp_path):
    shutil.copytree(registry.ROOT, tmp_path / "hardware", ignore=shutil.ignore_patterns("__pycache__", "*.py", "BUILD.bazel"))
    return tmp_path / "hardware"


def test_the_narrowed_contract_is_a_declared_descriptor_inside_the_full_one():
    registry.check()
    assert registry.narrowed(NARROWED)["narrowed"] is True
    full, narrowed = registry.contexts(), registry.contexts(contract=NARROWED)
    assert len(narrowed) == 2 < len(full) == 8
    assert {c.descriptor for c in narrowed} == {NARROWED}
    assert {c.descriptor for c in full} == {f"hopper_decode_d{depth}" for depth in (1, 2, 3, 4)}, "a narrowed contract is never part of the full one"


def test_a_name_no_narrowed_contract_declares_is_refused_and_the_full_contract_is_the_default():
    with pytest.raises(registry.RegistryError, match="no narrowed contract 'nope'.*decode_d32_l2"):
        registry.contexts(contract="nope")
    assert registry.contexts(contract=registry.FULL) == registry.contexts()


def test_a_descriptor_not_marked_narrowed_may_not_narrow_a_build(tree):
    path = tree / "descriptors" / "dev" / f"{NARROWED}.json"
    data = json.loads(path.read_text())
    del data["narrowed"]
    path.write_text(json.dumps(data))
    with pytest.raises(registry.RegistryError, match="not marked narrowed"):
        registry.check(tree)


def test_a_narrowed_context_outside_the_full_contract_is_refused(tree):
    path = tree / "parts" / "calls" / "decode_narrow_d32.json"
    data = json.loads(path.read_text())
    data["facts"]["d_v"]["value"] = {"lo": 2, "hi": 128}
    path.write_text(json.dumps(data))
    with pytest.raises(registry.RegistryError, match="lies outside the full contract"):
        registry.check(tree)


def test_the_contexts_file_and_the_header_name_the_contract():
    text = registry.contexts_text(registry.contexts(contract=NARROWED), NARROWED)
    assert text.splitlines()[0] == f"# contract: {NARROWED}"
    facts = registry.schema()
    header = registry.contract_header(registry.contexts(contract=NARROWED), facts, NARROWED)
    assert f'kContractName = "{NARROWED}"' in header and "kContractNarrowed = true" in header
    assert "kContractNarrowed = false" in registry.contract_header(registry.contexts(), facts)


def test_the_arm_generator_under_the_narrowed_contract_compiles_exactly_the_arms_it_reaches():
    narrowed = json.loads((PLAN / "decode_arm_rows.json").read_text())["arms"]
    full = json.loads((FULL_PLAN / "decode_arm_rows.json").read_text())["arms"]
    assert narrowed == [[32, 2, decay] for decay in (0, 1)]
    assert len(narrowed) == 2 < len(full) == 16
    assert all(arm in full for arm in narrowed)
    record = (PLAN / "decode_selection_record.txt").read_text()
    assert f"from the {NARROWED} contract" in record and "levels=3" not in record


def test_the_run_time_under_the_narrowed_contract_plans_inside_it():
    run = _plan("d_v=32,levels=2,decay=1")
    assert run.returncode == 0, run.stderr
    assert "arm: d_v=32 levels=2 decay=1" in run.stdout


@pytest.mark.parametrize("call, fact", [("d_v=64,levels=2,decay=1", "d_v=64"), ("d_v=32,levels=3,decay=1", "levels=3"),
                                         ("d_v=32,levels=1,decay=1", "levels=1")])
def test_the_run_time_refuses_a_call_outside_the_narrowed_contract_naming_it(call, fact):
    run = _plan(call)
    assert run.returncode == 1
    assert f"NARROWED contract '{NARROWED}'" in run.stderr and fact in run.stderr
