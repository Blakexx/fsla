# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT REGISTRY (`hardware/`, docs/internals/hardware/registry.md): every file of the registry holds the schema, composition expands to
full contexts, a duplicate context and every malformed part is refused, and the Hopper device part states its architecture whole."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402

#: the largest B times H a decode step serves (docs/api.md#batch-heads)
BATCH_HEADS_MAX = 536870911


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
        assert c.alternatives == ("decay",)
        assert c.facts["sm_count"] == (114, 132)
        assert c.facts["prefill"] == (0, 0) and c.facts["tokens"] == (1, 1) and c.facts["stateful"] == (1, 1)
        assert c.facts["batch_heads"] == (1, BATCH_HEADS_MAX)
        levels = c.facts["levels"][0]
        for side in ("read", "write"):
            assert [c.facts[f"{side}_listed_{k}"] for k in (1, 2, 3, 4)] == [(0, 1)] * levels + [(0, 0)] * (4 - levels)
    assert len({c.id for c in contexts}) == 8
    assert {c.descriptor for c in contexts} == {f"hopper_decode_d{lv}" for lv in (1, 2, 3, 4)}


def test_a_level_beyond_the_depth_has_no_width_and_a_level_it_has_is_a_width_from_sixteen_to_two_hundred_fifty_six():
    """The depth and the widths are correlated, so each depth is a part of its own: width_bits k is the log2 range 4..8 (16 to 256) at a level the call
    has and the point 0 beyond (the floor is `fsla._state.MIN_LEVEL_WIDTH`)."""
    from fsla._state import MIN_LEVEL_WIDTH

    for c in registry.contexts():
        depth = c.facts["levels"][0]
        for level in (1, 2, 3, 4):
            want = (MIN_LEVEL_WIDTH.bit_length() - 1, 8) if level <= depth else (0, 0)
            assert c.facts[f"width_bits_{level}"] == want, (c.id, level)


def test_the_hopper_part_states_its_architecture_whole_and_the_registry_builds_it():
    arch = registry.architecture("device/hopper_sm90a")
    assert (arch.name, arch.cc, arch.family, arch.regs_per_sm) == ("sm_90a", 900, "hopper", 65536)
    assert set(arch.instructions) == set(registry.part("device/hopper_sm90a")["facts"]["instructions"]["value"]["set"])
    assert registry.built() == ["sm_90a"] and registry.arch_of(900) == "sm_90a" and registry.device_arch(9, 0) == "sm_90a" and registry.cc_of("sm_100a") == 1000
    assert {c.arch for c in registry.contexts()} == {"sm_90a"}
    with pytest.raises(SystemExit, match="no device part"):
        registry.arch_of(1000)


def test_an_architecture_no_part_states_is_neither_tabulated_nor_built():
    text = registry.arch_header(registry.architectures())
    assert "constexpr bool tabulated(int cc) {\n  return cc == 900;\n}" in text and "1000" not in text
    with pytest.raises(SystemExit, match="sm_100a has no device part"):
        registry.builds_header(["100a"])
    assert "#define FSLA_BUILDS(F) F(900)" in registry.builds_header(["90a"])


def test_the_architecture_header_is_the_device_parts_one_row_an_architecture():
    text = registry.arch_header(registry.architectures())
    assert "if (cc == 900)" in text and "if (b.cc == 900)" in text and "Traits{Family::kHopper, caps_of(900)}" in text
    #: a family is one a device part states: the header names no other
    assert "enum class Family : int { kHopper };" in text and 'kFamilyNames[] = {"hopper"};' in text
    for gone in ("Mma", "kMmaRows", "smem_per_sm", "smem_per_cta_max", "max_threads_per_sm", "reg_transpose_b16", "window"):
        assert gone not in text
    for kept in ("cc", "regs_per_sm", "bf16_cvt_packed", "bf16_simd_arith", "bf16_simd_fma"):
        assert f" {kept};" in text
    assert ".bf16_simd_fma = true" in text and ".regs_per_sm = 65536" in text


def test_the_device_point_is_the_part_with_each_range_at_its_upper_end():
    assert registry.device_point() == dict(cc=900, regs_per_sm=65536, smem_per_sm=233472, smem_per_cta_optin=232448, max_threads_per_sm=2048, sm_count=132)


def test_a_device_part_names_its_architecture_family_and_compute_capability_and_an_architecture_is_stated_once(tree):
    path = tree / "parts" / "device" / "hopper_sm90a.json"
    original = json.loads(path.read_text())
    for edit, match in [(lambda d: d.pop("arch"), "names the architecture it is built as"),
                        (lambda d: d.pop("family"), "names its family"),
                        (lambda d: d.update(family="Hopper 2"), "names its family"),
                        (lambda d: d.update(arch="sm_80a"), "not the compute capability sm_80a names")]:
        data = json.loads(json.dumps(original))
        edit(data)
        _write(path, data)
        with pytest.raises(registry.RegistryError, match=match):
            registry.check(tree)
    _write(path, original)
    _write(tree / "parts" / "device" / "twin.json", {**original, "name": "device/twin"})
    with pytest.raises(registry.RegistryError, match="an architecture is stated once"):
        registry.check(tree)


def test_a_descriptor_composes_exactly_one_device_part(tree):
    hopper = json.loads((tree / "parts" / "device" / "hopper_sm90a.json").read_text())
    _write(tree / "parts" / "other" / "elsewhere.json", {**hopper, "name": "other/elsewhere"})
    path = tree / "descriptors" / "hopper_decode_d1.json"
    data = json.loads(path.read_text())
    _write(path, {**data, "compose": ["other/elsewhere", *data["compose"][1:]]})
    with pytest.raises(registry.RegistryError, match="composes 0 device parts"):
        registry.check(tree)


def test_the_contexts_text_is_a_line_a_context_the_matchers_tool_reads():
    lines = registry.contexts_text(registry.contexts()).splitlines()
    assert lines.pop(0) == "# contract: full"
    assert len(lines) == 8
    cells = lines[0].split("\t")
    assert cells[0] == "hopper_decode_d1 decay=0"
    assert "cc=900:900" in cells and "d_v=2:64" in cells and "sm_count=114:132" in cells and "tensor={hmma_bf16_f32,hmma_f16_f32,imma_u8_s8_k32}" in cells


def test_the_facts_header_has_a_member_for_each_fact():
    text = registry.facts_header(registry.schema())
    for name in registry.schema():
        assert f" {name}_{{}};" in text and "Reader" in text and f"{name}(const Facts& f)" in text
    assert "enum class Tensor : int { hmma_f16_f32, hmma_bf16_f32, imma_u8_s8_k32 };" in text


def test_the_contract_header_holds_each_context_and_the_call_facts_in_the_schemas_order():
    text = registry.contract_header(registry.contexts(), registry.schema())
    assert text.count("out.push_back({\"hopper_decode_d") == 8
    assert "f.assign(\"d_v\", Interval{2, 64});" in text and "f.assign(\"sm_count\", Interval{114, 132});" in text
    assert "f.assign(\"tensor\", EnumSet{7});" in text
    names = ["d_v", "levels", "decay", "prefill", "tokens", "batch_heads", "stateful", "width_bits_1", "width_bits_2",
             "width_bits_3", "width_bits_4", *(f"{side}_listed_{k}" for side in ("read", "write") for k in (1, 2, 3, 4))]
    assert "kCallFactNames[] = {" + ", ".join(f'"{n}"' for n in names) + "}" in registry.facts_header(registry.schema())


def test_two_contexts_identical_in_every_fact_are_refused(tree):
    _write(tree / "descriptors" / "twin.json", {"name": "twin", "gpus": "H100",
                                                 "compose": ["device/hopper_sm90a", "calls/decode", "calls/depth1"]})
    with pytest.raises(registry.RegistryError, match="identical in every fact"):
        registry.contexts(tree)


def test_a_fact_stated_by_two_parts_is_refused(tree):
    _write(tree / "parts" / "calls" / "also_dv.json", {"name": "calls/also_dv", "facts": {
        "d_v": {"value": 128, "declared_by": "a test"}}})
    _write(tree / "descriptors" / "hopper_decode_d1.json", {"name": "hopper_decode_d1", "gpus": "H100",
                                                          "compose": ["device/hopper_sm90a", "calls/decode", "calls/depth1", "calls/also_dv"]})
    with pytest.raises(registry.RegistryError, match="d_v is stated by another part"):
        registry.contexts(tree)


def test_a_descriptor_leaving_a_fact_out_is_refused(tree):
    _write(tree / "descriptors" / "hopper_decode_d1.json", {"name": "hopper_decode_d1", "gpus": "H100", "compose": ["calls/decode"]})
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
    data = json.loads((tree / "parts" / "calls" / "decode.json").read_text())
    data["facts"]["d_v"] = {"value": {"lo": 65, "hi": 128}, "declared_by": "a test"}
    _write(tree / "parts" / "calls" / "decode.json", data)
    contexts = registry.contexts(tree)
    assert len(contexts) == 8 and contexts[0].facts["d_v"] == (65, 128)
    assert contexts[0].id == "hopper_decode_d1 decay=0"


def test_a_part_lacking_a_schema_fact_is_refused_by_name(tree):
    """The schema gains `l2_bytes` with the first reviewed probe part, never before; the refusal is tested on a schema with an extra device fact the parts do not state."""
    data = json.loads((tree / "schema.json").read_text())
    data["facts"].append({"name": "l2_bytes", "type": "int", "unit": "bytes", "read": "DEVICE", "doc": "a test"})
    _write(tree / "schema.json", data)
    with pytest.raises(registry.RegistryError, match=r"states no value for \['l2_bytes'\]"):
        registry.contexts(tree)


def test_a_candidate_file_is_refused_until_it_is_renamed_to_a_part(tree):
    """A candidate part the probe wrote is no part: the registry refuses it by name (in `check` and in `contexts`), and accepts the same file renamed."""
    admitted = tree
    hopper = json.loads((admitted / "parts" / "device" / "hopper_sm90a.json").read_text())
    candidate = {**hopper, "name": "device/cc900_x", "arch": "sm_100a", "family": "blackwell", "model": {"cc": 1000, "name": "X"}, "unestablished": [],
                 "facts": {**hopper["facts"], "cc": {"value": 1000, "measured_by": "a record"}}}
    _write(admitted / "parts" / "device" / "cc900_x.candidate.json", candidate)
    for call in (registry.check, registry.contexts):
        with pytest.raises(registry.RegistryError, match=r"CANDIDATE part.*renames it to device/cc900_x\.json"):
            call(admitted)
    (admitted / "parts" / "device" / "cc900_x.candidate.json").rename(admitted / "parts" / "device" / "cc900_x.json")
    registry.check(admitted)
    assert registry.part("device/cc900_x", admitted)["facts"]["cc"]["value"] == 1000
    assert registry.built(admitted) == ["sm_90a", "sm_100a"]


def test_a_descriptor_composing_a_candidate_is_refused(tree):
    _write(tree / "parts" / "device" / "cc900_x.candidate.json", {"name": "device/cc900_x", "facts": {"cc": {"value": 900, "measured_by": "a record"}}})
    _write(tree / "descriptors" / "hopper_decode_d1.json", {"name": "hopper_decode_d1", "gpus": "H100",
                                                              "compose": ["device/cc900_x.candidate", "calls/decode"]})
    with pytest.raises(registry.RegistryError, match="CANDIDATE part"):
        registry.contexts(tree)


def test_the_owed_descriptors_are_the_prefills_calls_and_none_of_them_is_in_the_contract():
    """`descriptors/owed` holds the prefill's four depths, each waiting on the prefill kernel: their contexts are held apart from the contract, the contexts file
    the matchers read and the narrowed contracts."""
    owed = registry.owed()
    assert len(owed) == 4  # a depth each: a prefill states no decay
    assert {c.descriptor for c, _ in owed} == {f"hopper_prefill_d{lv}" for lv in (1, 2, 3, 4)}
    assert all(kernels == ("prefill",) and c.facts["prefill"] == (1, 1) for c, kernels in owed)
    assert all(c.facts["stateful"] == (0, 1) and c.facts["tokens"][0] == 1 for c, _ in owed)
    assert all(c.facts["prefill"] == (0, 0) for c in registry.contexts())
    assert "prefill=1:1" not in registry.contexts_text(registry.contexts())
    assert not any(name.startswith("hopper_prefill") for name in registry.descriptors())
    assert not any(name.startswith("hopper_prefill") for name in registry.descriptors(folder="descriptors/dev"))


def test_the_owed_header_holds_each_owed_context_and_the_kernels_it_waits_on():
    text = registry.owed_header(registry.owed(), registry.schema())
    assert text.count("out.push_back({\"hopper_prefill_d") == 4 and text.count("{\"prefill\"}}") == 4
    assert "f.assign(\"prefill\", Interval{1, 1});" in text and "struct OwedContext" in text


def test_an_owed_descriptor_reachable_from_the_full_contract_is_refused(tree):
    """A call is guaranteed or owed, never both: an owed descriptor composing the decode call (a context the contract holds) is refused naming both."""
    _write(tree / "descriptors" / "owed" / "hopper_decode_d1_again.json",
           {"name": "hopper_decode_d1_again", "owed": ["decode"], "gpus": "H100",
            "compose": ["device/hopper_sm90a", "calls/decode", "calls/depth1"]})
    with pytest.raises(registry.RegistryError, match=r"owed descriptor hopper_decode_d1_again: .*reachable from the full contract's 'hopper_decode_d1"):
        registry.check(tree)
    with pytest.raises(registry.RegistryError, match="reachable from the full contract"):
        registry.owed(tree)


@pytest.mark.parametrize("owed", [None, [], "prefill", ["prefill", "prefill"], ["not a name"], [3]])
def test_an_owed_descriptor_names_the_kernels_it_waits_on(tree, owed):
    path = tree / "descriptors" / "owed" / "hopper_prefill_d1.json"
    data = json.loads(path.read_text())
    data.pop("owed")
    if owed is not None:
        data["owed"] = owed
    _write(path, data)
    with pytest.raises(registry.RegistryError, match="names the kernels its call waits on"):
        registry.check(tree)


def test_check_validates_the_owed_parts_and_descriptors_as_any_others(tree):
    """An owed part is held to the schema as every part is, and an owed descriptor leaving a fact out is refused."""
    path = tree / "parts" / "calls" / "prefill.json"
    data = json.loads(path.read_text())
    data["facts"]["warps"] = {"value": 4, "declared_by": "a test"}
    _write(path, data)
    with pytest.raises(registry.RegistryError, match="is no fact of the schema"):
        registry.check(tree)
    del data["facts"]["warps"]
    del data["facts"]["tokens"]
    _write(path, data)
    with pytest.raises(registry.RegistryError, match=r"states no value for \['tokens'\]"):
        registry.check(tree)
