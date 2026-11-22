# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT REGISTRY (`hardware/`, docs/internals/hardware/registry.md): every file of the registry holds the schema, composition expands to
full contexts, a duplicate context and every malformed part is refused, and the Hopper device part is the hardware table's own values."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402
from measure import hardware as table  # noqa: E402


@pytest.fixture()
def tree(tmp_path):
    """A copy of the registry to break."""
    shutil.copytree(registry.ROOT, tmp_path / "hardware", ignore=shutil.ignore_patterns("__pycache__", "*.py", "BUILD.bazel"))
    return tmp_path / "hardware"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_every_file_of_the_registry_holds_the_schema():
    registry.check()
    facts = registry.schema()
    assert {f.read for f in facts.values()} == {"DEVICE", "CALL"}
    for name in registry.parts():
        for fact, record in registry.part(name)["facts"].items():
            assert fact in facts
            assert sum(k in record for k in registry.PROVENANCE) == 1


def test_the_decode_contract_is_the_value_widths_two_to_sixty_four_at_each_depth_and_decay():
    contexts = registry.contexts()
    assert len(contexts) == 8
    assert {tuple(c.facts[n] for n in ("d_v", "levels", "decay")) for c in contexts} == {
        ((2, 64), (lv, lv), (dc, dc)) for lv in (1, 2, 3, 4) for dc in (0, 1)}
    for c in contexts:
        assert set(c.facts) == set(registry.schema())
        assert c.alternatives == ("levels", "decay")
        assert c.facts["sm_count"] == (114, 132)
    assert len({c.id for c in contexts}) == 8


def test_the_hopper_part_is_the_hardware_tables_values():
    caps = table.load()["arch"]["sm_90a"]["caps"]
    part = registry.part("device/hopper_sm90a")["facts"]
    assert part["cc"]["value"] == 900
    for fact, cap in [("regs_per_sm", "regs_per_sm"), ("smem_per_sm", "smem_per_sm"), ("smem_per_cta_optin", "smem_per_cta_max"),
                      ("max_threads_per_sm", "max_threads_per_sm")]:
        assert part[fact]["value"] == caps[cap]
    assert set(part["instructions"]["value"]["set"]) == {c for c, v in caps.items() if v is True}
    assert set(part["tensor"]["value"]["set"]) == set(table.load()["arch"]["sm_90a"]["rates"]["mma"])


def test_the_contexts_text_is_a_line_a_context_the_matchers_tool_reads():
    lines = registry.contexts_text(registry.contexts()).splitlines()
    assert len(lines) == 8
    cells = lines[0].split("\t")
    assert cells[0] == "hopper_decode levels=1 decay=0"
    assert "cc=900:900" in cells and "d_v=2:64" in cells and "sm_count=114:132" in cells and "tensor={hmma_bf16_f32,hmma_f16_f32,imma_u8_s8_k32}" in cells


def test_the_facts_header_has_a_member_for_each_fact():
    text = registry.facts_header(registry.schema())
    for name in registry.schema():
        assert f" {name}_{{}};" in text and "Reader" in text and f"{name}(const Facts& f)" in text
    assert "enum class Tensor : int { hmma_f16_f32, hmma_bf16_f32, imma_u8_s8_k32 };" in text


def test_the_contract_header_holds_each_context_and_the_call_facts_in_the_schemas_order():
    text = registry.contract_header(registry.contexts(), registry.schema())
    assert text.count("out.push_back({\"hopper_decode ") == 8
    assert "f.assign(\"d_v\", Interval{2, 64});" in text and "f.assign(\"sm_count\", Interval{114, 132});" in text
    assert "f.assign(\"tensor\", EnumSet{7});" in text
    assert "kCallFactNames[] = {\"d_v\", \"levels\", \"decay\"}" in registry.facts_header(registry.schema())


def test_two_contexts_identical_in_every_fact_are_refused(tree):
    _write(tree / "descriptors" / "twin.json", {"name": "twin", "gpus": "H100", "arch": "sm_90a", "compose": ["device/hopper_sm90a", "calls/decode"]})
    with pytest.raises(registry.RegistryError, match="identical in every fact"):
        registry.contexts(tree)


def test_a_fact_stated_by_two_parts_is_refused(tree):
    _write(tree / "parts" / "calls" / "also_dv.json", {"name": "calls/also_dv", "facts": {
        "d_v": {"value": 128, "declared_by": "a test"}}})
    _write(tree / "descriptors" / "hopper_decode.json", {"name": "hopper_decode", "gpus": "H100", "arch": "sm_90a",
                                                       "compose": ["device/hopper_sm90a", "calls/decode", "calls/also_dv"]})
    with pytest.raises(registry.RegistryError, match="d_v is stated by another part"):
        registry.contexts(tree)


def test_a_descriptor_leaving_a_fact_out_is_refused(tree):
    _write(tree / "descriptors" / "hopper_decode.json", {"name": "hopper_decode", "gpus": "H100", "arch": "sm_90a", "compose": ["calls/decode"]})
    with pytest.raises(registry.RegistryError, match="states no value for"):
        registry.contexts(tree)


@pytest.mark.parametrize("edit, match", [
    (lambda p: p["facts"]["cc"].pop("validated_by"), "cites exactly one of"),
    (lambda p: p["facts"]["cc"].update(measured_by="a lab row"), "cites exactly one of"),
    (lambda p: p["facts"]["cc"].update(value="nine hundred"), "is an int"),
    (lambda p: p["facts"]["cc"].update(value={"lo": 9, "hi": 1}), "lo above hi"),
    (lambda p: p["facts"]["tensor"].update(value={"set": ["fp4"]}), "has no member"),
    (lambda p: p["facts"].update(warps={"value": 4, "declared_by": "a test"}), "is no fact of the schema"),
])
def test_a_malformed_part_is_refused(tree, edit, match):
    path = tree / "parts" / "device" / "hopper_sm90a.json"
    data = json.loads(path.read_text())
    edit(data)
    _write(path, data)
    with pytest.raises(registry.RegistryError, match=match):
        registry.check(tree)


@pytest.mark.parametrize("edit, match", [
    (lambda s: s["facts"][0].update(type="float"), "has type"),
    (lambda s: s["facts"][0].update(read="HOST"), "is read from"),
    (lambda s: s["facts"][0].pop("unit"), "names no unit"),
    (lambda s: s["facts"].append(dict(s["facts"][0])), "declared twice"),
    (lambda s: s["facts"][0].update(members=["a"]), "set<enum> fact names"),
])
def test_a_malformed_schema_is_refused(tree, edit, match):
    data = json.loads((tree / "schema.json").read_text())
    edit(data)
    _write(tree / "schema.json", data)
    with pytest.raises(registry.RegistryError, match=match):
        registry.schema(tree)


def test_ranges_and_sets_expand_without_crossing(tree):
    _write(tree / "parts" / "calls" / "decode.json", {"name": "calls/decode", "facts": {
        "d_v": {"value": {"lo": 65, "hi": 128}, "declared_by": "a test"},
        "levels": {"value": 2, "declared_by": "a test"},
        "decay": {"value": [0, 1], "declared_by": "a test"}}})
    contexts = registry.contexts(tree)
    assert len(contexts) == 2 and contexts[0].facts["d_v"] == (65, 128)
    assert contexts[0].id == "hopper_decode decay=0"
