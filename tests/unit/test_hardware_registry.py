# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT REGISTRY (`hardware/`, docs/internals/hardware/registry.md): the shared device table and a registry a kernel (`hardware/kernels/<kernel>/`) hold their schemas,
composition expands to full contexts, a duplicate context and every malformed part is refused, and the Hopper device part states its architecture whole."""
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

DECODE_CALL = ("d_v", "levels", "decay", "batch_heads", "width_bits_1", "width_bits_2", "width_bits_3", "width_bits_4",
               "write_listed_1", "write_listed_2", "write_listed_3", "write_listed_4")
PREFILL_CALL = ("d_v", "levels", "decay", "tokens", "batch_heads", "stateful", "width_bits_1", "width_bits_2", "width_bits_3", "width_bits_4")


@pytest.fixture()
def tree(tmp_path):
    """A copy of the registry to break."""
    shutil.copytree(registry.ROOT, tmp_path / "hardware", ignore=shutil.ignore_patterns("__pycache__", "*.py", "BUILD.bazel"))
    return tmp_path / "hardware"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _decode(tree: Path, *path: str) -> Path:
    return tree.joinpath("kernels", "decode", *path)


def test_every_file_of_the_registry_holds_its_schema():
    registry.check()
    device = registry.schema()
    assert {f.read for f in device.values()} == {"DEVICE"}
    for name in registry.parts():
        for fact, record in registry.part(name)["facts"].items():
            assert fact in device
            assert sum(k in record for k in registry.PROVENANCE) == 1
    for kernel in registry.kernels():
        call = registry.kernel_schema(kernel)
        assert {f.read for f in call.values()} == {"CALL"} and not set(call) & set(device)
        assert list(registry.facts(kernel)) == [*device, *call]
        for name in registry.parts(kernel=kernel):
            for fact, record in registry.part(name, kernel=kernel)["facts"].items():
                assert fact in call
                assert sum(k in record for k in registry.PROVENANCE) == 1


def test_the_registry_holds_a_registry_each_kernel_and_the_prefill_fact_is_nowhere():
    assert registry.kernels() == ["decode", "factor_backward", "factor_forward", "prefill", "softmax_backward", "softmax_forward", "union_backward", "union_forward"]
    assert tuple(registry.kernel_schema("decode")) == DECODE_CALL
    assert tuple(registry.kernel_schema("prefill")) == PREFILL_CALL
    assert "prefill" not in registry.schema() and not any("prefill" in registry.kernel_schema(k) for k in registry.kernels())


def test_the_hopper_instructions_name_only_what_a_kernel_reads():
    assert "reg_transpose_b16" not in registry.schema()["instructions"].members
    assert "reg_transpose_b16" not in registry.part("device/hopper_sm90a")["facts"]["instructions"]["value"]["set"]


def test_the_decode_contract_is_the_value_widths_two_to_sixty_four_at_each_depth_and_decay():
    contexts = registry.contexts("decode")
    assert len(contexts) == 8
    assert {tuple(c.facts[n] for n in ("d_v", "levels", "decay")) for c in contexts} == {
        ((2, 64), (lv, lv), (dc, dc)) for lv in (1, 2, 3, 4) for dc in (0, 1)}
    for c in contexts:
        assert set(c.facts) == set(registry.facts("decode")) and c.kernel == "decode"
        assert c.alternatives == ("decay",)
        assert c.facts["sm_count"] == (114, 132)
        assert c.facts["batch_heads"] == (1, BATCH_HEADS_MAX)
        levels = c.facts["levels"][0]
        assert [c.facts[f"write_listed_{k}"] for k in (1, 2, 3, 4)] == [(0, 1)] * levels + [(0, 0)] * (4 - levels)
    assert len({c.id for c in contexts}) == 8
    assert {c.descriptor for c in contexts} == {f"hopper_decode_d{lv}" for lv in (1, 2, 3, 4)}


def test_a_level_beyond_the_depth_has_no_width_and_a_level_it_has_is_a_width_from_sixteen_to_two_hundred_fifty_six():
    """The depth and the widths are correlated, so each depth is a part of its own: width_bits k is the log2 range 4..8 (16 to 256) at a level the call
    has and the point 0 beyond (the floor is `fsla._state.MIN_LEVEL_WIDTH`)."""
    from fsla._state import MIN_LEVEL_WIDTH

    for c in registry.contexts("decode"):
        depth = c.facts["levels"][0]
        for level in (1, 2, 3, 4):
            want = (MIN_LEVEL_WIDTH.bit_length() - 1, 8) if level <= depth else (0, 0)
            assert c.facts[f"width_bits_{level}"] == want, (c.id, level)


def test_the_hopper_part_states_its_architecture_whole_and_the_registry_builds_it():
    arch = registry.architecture("device/hopper_sm90a")
    assert (arch.name, arch.cc, arch.family, arch.regs_per_sm) == ("sm_90a", 900, "hopper", 65536)
    assert set(arch.instructions) == set(registry.part("device/hopper_sm90a")["facts"]["instructions"]["value"]["set"])
    assert registry.built() == ["sm_90a"] and registry.arch_of(900) == "sm_90a" and registry.device_arch(9, 0) == "sm_90a" and registry.cc_of("sm_100a") == 1000
    assert {c.arch for c in registry.contexts("decode")} == {"sm_90a"}
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
    path = _decode(tree, "descriptors", "hopper_decode_d1.json")
    data = json.loads(path.read_text())
    _write(path, {**data, "compose": data["compose"][1:]})
    with pytest.raises(registry.RegistryError, match="composes 0 device parts"):
        registry.check(tree)


def test_the_contexts_text_is_a_line_a_context_the_matchers_tool_reads():
    lines = registry.contexts_text("decode", registry.contexts("decode")).splitlines()
    assert lines.pop(0) == "# contract: full" and lines.pop(0) == "# kernel: decode"
    assert len(lines) == 8
    cells = lines[0].split("\t")
    assert cells[0] == "hopper_decode_d1 decay=0"
    assert "cc=900:900" in cells and "d_v=2:64" in cells and "sm_count=114:132" in cells and "tensor={hmma_bf16_f32,hmma_f16_f32,imma_u8_s8_k32}" in cells
    assert not any(cell.startswith(("prefill=", "tokens=", "stateful=", "read_listed_")) for cell in cells)


@pytest.mark.parametrize("kernel", ["decode", "prefill"])
def test_the_facts_header_has_a_member_for_each_fact_of_the_kernel(kernel):
    facts = registry.facts(kernel)
    text = registry.facts_header(kernel, facts)
    for name in facts:
        assert f" {name}_{{}};" in text and "Reader" in text and f"{name}(const Facts& f)" in text
    assert f"namespace fsla::plan::{kernel} {{" in text and "using Read = Reader;" in text
    assert "enum class Tensor : int { hmma_f16_f32, hmma_bf16_f32, imma_u8_s8_k32 };" in text


def test_the_facts_header_names_each_kernels_call_facts_in_its_schemas_order():
    for kernel, names in (("decode", DECODE_CALL), ("prefill", PREFILL_CALL)):
        assert "kCallFactNames[] = {" + ", ".join(f'"{n}"' for n in names) + "}" in registry.facts_header(kernel, registry.facts(kernel))


def test_the_contract_header_holds_each_context_and_the_kernels_definition():
    text = registry.contract_header("decode", registry.contexts("decode"), registry.owed("decode"), registry.facts("decode"))
    assert text.count("out.push_back({\"hopper_decode_d") == 8
    assert "f.assign(\"d_v\", Interval{2, 64});" in text and "f.assign(\"sm_count\", Interval{114, 132});" in text
    assert "f.assign(\"tensor\", EnumSet{7});" in text
    assert "struct Kernel {" in text and 'name = "decode"' in text and "OwedContext" in text
    assert "prefill" not in text.replace("fsla::plan::decode", "")


def test_two_contexts_identical_in_every_fact_are_refused(tree):
    _write(_decode(tree, "descriptors", "twin.json"), {"name": "twin", "gpus": "H100", "compose": ["device/hopper_sm90a", "call", "depth1"]})
    with pytest.raises(registry.RegistryError, match="identical in every fact"):
        registry.contexts("decode", tree)


def test_a_fact_stated_by_two_parts_is_refused(tree):
    _write(_decode(tree, "parts", "also_dv.json"), {"name": "also_dv", "facts": {"d_v": {"value": 128, "declared_by": "a test"}}})
    _write(_decode(tree, "descriptors", "hopper_decode_d1.json"), {"name": "hopper_decode_d1", "gpus": "H100",
                                                                   "compose": ["device/hopper_sm90a", "call", "depth1", "also_dv"]})
    with pytest.raises(registry.RegistryError, match="d_v is stated by another part"):
        registry.contexts("decode", tree)


def test_a_part_of_the_wrong_side_is_refused(tree):
    """A device part states device facts and a kernel part its call's: a device fact in a kernel part is refused (a call fact in a device part is no fact of the device table)."""
    _write(_decode(tree, "parts", "device_fact.json"), {"name": "device_fact", "facts": {"cc": {"value": 900, "declared_by": "a test"}}})
    _write(_decode(tree, "descriptors", "hopper_decode_d1.json"), {"name": "hopper_decode_d1", "gpus": "H100",
                                                                   "compose": ["device/hopper_sm90a", "call", "depth1", "device_fact"]})
    with pytest.raises(registry.RegistryError, match="cc is a fact of the device, which a kernel part does not state"):
        registry.contexts("decode", tree)


def test_a_descriptor_leaving_a_fact_out_is_refused(tree):
    _write(_decode(tree, "descriptors", "hopper_decode_d1.json"), {"name": "hopper_decode_d1", "gpus": "H100", "compose": ["device/hopper_sm90a", "call"]})
    with pytest.raises(registry.RegistryError, match="states no value for"):
        registry.contexts("decode", tree)


@pytest.mark.parametrize("edit, match", [
    (lambda p: p["facts"]["cc"].pop("validated_by"), "cites exactly one of"),
    (lambda p: p["facts"]["cc"].update(measured_by="a lab row"), "cites exactly one of"),
    (lambda p: p["facts"]["cc"].update(value="nine hundred"), "is an int"),
    (lambda p: p["facts"]["cc"].update(value={"lo": 9, "hi": 1}), "lo above hi"),
    (lambda p: p["facts"]["tensor"].update(value={"set": ["fp4"]}), "has no member"),
    (lambda p: p["facts"].update(warps={"value": 4, "declared_by": "a test"}), "is no fact of the device table"),
    (lambda p: p["facts"].update(d_v={"value": 4, "declared_by": "a test"}), "is no fact of the device table"),
])
def test_a_malformed_device_part_is_refused(tree, edit, match):
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
    for path, read in ((tree / "schema.json", lambda: registry.schema(tree)), (_decode(tree, "schema.json"), lambda: registry.kernel_schema("decode", tree))):
        data = json.loads(path.read_text())
        edit(data)
        _write(path, data)
        with pytest.raises(registry.RegistryError, match=match):
            read()


def test_a_schema_holds_the_facts_of_its_own_side_alone(tree):
    """The device table holds DEVICE facts and a kernel's schema CALL facts: a call fact in the device table, or a device fact in a kernel's schema, is refused."""
    data = json.loads(_decode(tree, "schema.json").read_text())
    data["facts"].append({"name": "l2_bytes", "type": "int", "unit": "bytes", "read": "DEVICE", "doc": "a test"})
    _write(_decode(tree, "schema.json"), data)
    with pytest.raises(registry.RegistryError, match="holds the CALL facts alone"):
        registry.kernel_schema("decode", tree)
    device = json.loads((tree / "schema.json").read_text())
    device["facts"].append({"name": "tokens", "type": "int", "unit": "tokens", "read": "CALL", "doc": "a test"})
    _write(tree / "schema.json", device)
    with pytest.raises(registry.RegistryError, match="holds the DEVICE facts alone"):
        registry.schema(tree)


def test_a_fact_of_both_the_device_and_a_kernels_call_is_refused(tree):
    data = json.loads(_decode(tree, "schema.json").read_text())
    data["facts"].append({"name": "cc", "type": "int", "unit": "x", "read": "CALL", "doc": "a test"})
    _write(_decode(tree, "schema.json"), data)
    with pytest.raises(registry.RegistryError, match="a fact is one or the other"):
        registry.facts("decode", tree)


def test_ranges_and_sets_expand_without_crossing(tree):
    data = json.loads(_decode(tree, "parts", "call.json").read_text())
    data["facts"]["d_v"] = {"value": {"lo": 65, "hi": 128}, "declared_by": "a test"}
    _write(_decode(tree, "parts", "call.json"), data)
    contexts = registry.contexts("decode", tree)
    assert len(contexts) == 8 and contexts[0].facts["d_v"] == (65, 128)
    assert contexts[0].id == "hopper_decode_d1 decay=0"


def test_a_part_lacking_a_schema_fact_is_refused_by_name(tree):
    """The device table gains `l2_bytes` with the first reviewed probe part, never before; the refusal is tested on a table with an extra device fact the parts do not state."""
    data = json.loads((tree / "schema.json").read_text())
    data["facts"].append({"name": "l2_bytes", "type": "int", "unit": "bytes", "read": "DEVICE", "doc": "a test"})
    _write(tree / "schema.json", data)
    with pytest.raises(registry.RegistryError, match=r"states no value for \['l2_bytes'\]"):
        registry.contexts("decode", tree)


def test_a_candidate_file_is_refused_until_it_is_renamed_to_a_part(tree):
    """A candidate part the probe wrote is no part: the registry refuses it by name (in `check` and in `contexts`), and accepts the same file renamed."""
    admitted = tree
    hopper = json.loads((admitted / "parts" / "device" / "hopper_sm90a.json").read_text())
    candidate = {**hopper, "name": "device/cc900_x", "arch": "sm_100a", "family": "blackwell", "model": {"cc": 1000, "name": "X"}, "unestablished": [],
                 "facts": {**hopper["facts"], "cc": {"value": 1000, "measured_by": "a record"}}}
    _write(admitted / "parts" / "device" / "cc900_x.candidate.json", candidate)
    for call in (registry.check, lambda root: registry.contexts("decode", root)):
        with pytest.raises(registry.RegistryError, match=r"CANDIDATE part.*renames it to device/cc900_x\.json"):
            call(admitted)
    (admitted / "parts" / "device" / "cc900_x.candidate.json").rename(admitted / "parts" / "device" / "cc900_x.json")
    registry.check(admitted)
    assert registry.part("device/cc900_x", admitted)["facts"]["cc"]["value"] == 1000
    assert registry.built(admitted) == ["sm_90a", "sm_100a"]


def test_a_descriptor_composing_a_candidate_is_refused(tree):
    _write(tree / "parts" / "device" / "cc900_x.candidate.json", {"name": "device/cc900_x", "facts": {"cc": {"value": 900, "measured_by": "a record"}}})
    _write(_decode(tree, "descriptors", "hopper_decode_d1.json"), {"name": "hopper_decode_d1", "gpus": "H100",
                                                                   "compose": ["device/cc900_x.candidate", "call"]})
    with pytest.raises(registry.RegistryError, match="CANDIDATE part"):
        registry.contexts("decode", tree)


def test_the_owed_descriptors_are_the_prefills_calls_and_none_of_them_is_in_any_contract():
    """`kernels/prefill/descriptors/owed` holds the prefill's four depths, each waiting on the prefill kernel: the prefill holds no built context, and the owed
    contexts are held apart from the contract, the contexts file the matchers read and the narrowed contracts."""
    owed = registry.owed("prefill")
    assert len(owed) == 4  # a depth each: a prefill states no decay
    assert {c.descriptor for c, _ in owed} == {f"hopper_prefill_d{lv}" for lv in (1, 2, 3, 4)}
    assert all(kernels == ("prefill",) and c.kernel == "prefill" for c, kernels in owed)
    assert all(c.facts["stateful"] == (0, 1) and c.facts["tokens"][0] == 1 for c, _ in owed)
    assert not any(name.startswith(("read_listed_", "write_listed_", "read_density_", "write_density_")) for c, _ in owed for name in c.facts)
    assert registry.contexts("prefill") == [] and registry.owed("decode") == []
    assert not registry.descriptors(kernel="prefill") and not registry.descriptors(kernel="prefill", folder=registry.DEV)
    assert not any(name.startswith("hopper_prefill") for name in registry.descriptors(kernel="decode"))


def test_the_owed_contexts_are_in_the_contract_header_beside_the_kernel_they_wait_on():
    text = registry.contract_header("prefill", registry.contexts("prefill"), registry.owed("prefill"), registry.facts("prefill"))
    assert text.count("out.push_back({\"hopper_prefill_d") == 4 and text.count("{\"prefill\"}}") == 4
    assert "f.assign(\"tokens\", Interval{1, 2147483647});" in text and "OwedContext<Facts>" in text and "struct Kernel {" in text
    assert "kContractNarrowed = false" in text


def test_an_owed_descriptor_reachable_from_the_full_contract_is_refused(tree):
    """A call is guaranteed or owed, never both: an owed descriptor composing the decode call (a context the contract holds) is refused naming both."""
    _write(_decode(tree, "descriptors", "owed", "hopper_decode_d1_again.json"),
           {"name": "hopper_decode_d1_again", "owed": ["decode"], "gpus": "H100",
            "compose": ["device/hopper_sm90a", "call", "depth1"]})
    with pytest.raises(registry.RegistryError, match=r"owed descriptor hopper_decode_d1_again: .*reachable from the full contract's 'hopper_decode_d1"):
        registry.check(tree)
    with pytest.raises(registry.RegistryError, match="reachable from the full contract"):
        registry.owed("decode", tree)


@pytest.mark.parametrize("owed", [None, [], "prefill", ["prefill", "prefill"], ["not a name"], [3]])
def test_an_owed_descriptor_names_the_kernels_it_waits_on(tree, owed):
    path = tree / "kernels" / "prefill" / "descriptors" / "owed" / "hopper_prefill_d1.json"
    data = json.loads(path.read_text())
    data.pop("owed")
    if owed is not None:
        data["owed"] = owed
    _write(path, data)
    with pytest.raises(registry.RegistryError, match="names the kernels its call waits on"):
        registry.check(tree)


def test_check_validates_the_owed_parts_and_descriptors_as_any_others(tree):
    """An owed part is held to its kernel's schema as every part is, and an owed descriptor leaving a fact out is refused."""
    path = tree / "kernels" / "prefill" / "parts" / "call.json"
    data = json.loads(path.read_text())
    data["facts"]["warps"] = {"value": 4, "declared_by": "a test"}
    _write(path, data)
    with pytest.raises(registry.RegistryError, match="is no call fact of the kernel"):
        registry.check(tree)
    del data["facts"]["warps"]
    del data["facts"]["tokens"]
    _write(path, data)
    with pytest.raises(registry.RegistryError, match=r"states no value for \['tokens'\]"):
        registry.check(tree)


def test_a_kernel_with_no_descriptor_at_all_holds_nothing_and_a_registry_with_none_supports_nothing(tree):
    for kernel in registry.kernels(tree):
        for path in (tree / "kernels" / kernel / "descriptors").glob("*.json"):
            path.unlink()
        for path in (tree / "kernels" / kernel / "descriptors" / "dev").glob("*.json"):
            path.unlink()
    with pytest.raises(registry.RegistryError, match="nothing is supported"):
        registry.check(tree)
