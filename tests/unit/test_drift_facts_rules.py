"""The three facts-in-the-template rules of tools/lint/facts_in_template.py (docs/CONSTITUTION.md#r8), run on their fixtures:
each fires with its exact text and stays quiet on its exemptions. No GPU."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "lint"))

import facts_in_template as dg  # noqa: E402

FIXTURES = dg.FIXTURES
BAD, OK = FIXTURES / "drift_facts_bad.cuh", FIXTURES / "drift_facts_ok.cuh"
REL = "tools/lint/fixtures/"
FACTS = "derive it from the build's facts"


def test_a_shared_declaration_outside_the_layout_is_a_finding_a_line():
    assert dg.rule_shared_memory_through_the_layout(files=[BAD]) == [
        f"{REL}drift_facts_bad.cuh:8: shared memory declared outside SmemLayout (common/smem_layout.cuh): declare a tile"]


def test_comments_and_the_layout_file_do_not_count_as_shared_declarations(tmp_path):
    assert dg.rule_shared_memory_through_the_layout(files=[OK]) == []
    layout = tmp_path / "common" / "smem_layout.cuh"
    layout.parent.mkdir()
    layout.write_text("__shared__ char pool[1];\n")
    assert dg.rule_shared_memory_through_the_layout(files=[layout]) == []
    other = tmp_path / "kernel.cuh"
    other.write_text("/* __shared__ int a; */\n// __shared__ int b;\n__shared__ int c;\n")
    assert [f.split(": ")[0] for f in dg.rule_shared_memory_through_the_layout(files=[other])] == [f"{other}:3"]


def test_a_literal_count_or_least_is_a_finding_across_line_breaks_and_a_weight_is_not():
    found = dg.rule_layout_sizes_are_expressions(files=[BAD])
    assert found == [
        f"{REL}drift_facts_bad.cuh:12: a tile's size is the number 100: derive it from the build's facts or the arm's "
        f"parameters",
        f"{REL}drift_facts_bad.cuh:14: a tile's size is the number 0x40u: derive it from the build's facts or the arm's "
        f"parameters"]


def test_a_tile_sized_by_an_expression_or_a_literal_weight_is_quiet(tmp_path):
    assert dg.rule_layout_sizes_are_expressions(files=[OK]) == []
    proof = tmp_path / "common" / "smem_layout.cuh"
    proof.parent.mkdir()
    proof.write_text("using Proof = Fixed<Table, std::int32_t, 100>;\n")
    assert dg.rule_layout_sizes_are_expressions(files=[proof]) == []
    nested = tmp_path / "k.cuh"
    nested.write_text("using A = Share<Ring, T, f<1, 2>(3), 7>;\nusing B = Share<Ring, std::array<int, 2>, 8ul, 3>;\n")
    assert dg.rule_layout_sizes_are_expressions(files=[nested]) == [
        f"{nested}:2: a tile's size is the number 8ul: derive it from the build's facts or the arm's parameters"]


def test_a_number_at_namespace_or_class_scope_is_a_finding_unless_its_block_says_fixed_and_why():
    note = f"{FACTS}, or state in its block why the card does not decide it (FIXED ...)"
    assert dg.rule_card_quantities_are_derived(files=[BAD]) == [
        f"{REL}drift_facts_bad.cuh:16: kWindow is the number 64: {note}",
        f"{REL}drift_facts_bad.cuh:17: kStride is the number 128ul: {note}",
        f"{REL}drift_facts_bad.cuh:21: kLowercase is the number 8: {note}",
        f"{REL}drift_facts_bad.cuh:24: kNoReason is the number 1'024: {note}",
        f"{REL}drift_facts_bad.cuh:27: kLeaves is the number 0x20u: {note}"]


def test_a_stated_reason_an_expression_a_function_scope_constant_or_another_type_is_quiet():
    assert dg.rule_card_quantities_are_derived(files=[OK]) == []


def test_the_scope_is_the_kernels_and_the_planners_headers_only():
    names = {str(p.relative_to(dg.CSRC / "src")) for p in dg.facts_files()}
    assert "common/plan/planner.h" in names and "common/smem_layout.cuh" in names
    assert not any(n.endswith((".cpp", ".py")) or "lab" in n.split("/") for n in names)
    assert not any(p.suffix == ".h" and p.parent != dg.CSRC / "src" / "common" / "plan" for p in dg.facts_files())
