# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""How the lab calls warpline's checker (tools/lab.py, docs/internals/tools/lab.md#hazards): an experiment's `Bench::expect` declarations are
translated to the checker's (kernel regex, rule, allowed count, reason), and a refusal of a rule is shown under the hazard's name the
experiment declares. The rules themselves -- what is a loop convergence, a load in its trip, a step in a rolled loop -- are the module's,
covered by its own tests on fixtures; nothing here recomputes them."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import dev_config  # noqa: E402
import lab  # noqa: E402

needs_checker = pytest.mark.skipif(dev_config.warpline_python() is None,
                                   reason="warpline's checker is handed in by the graph (bazel test //tests:cpu)")


def expect(kernel, op, cmp, count):
    return {"lab": "expect", "kernel": kernel, "op": op, "cmp": cmp, "count": count}


def test_a_hazard_is_declared_as_its_rule_with_its_count_and_the_comment_above_it():
    source = ("  //: one divergent store a trip, kept on purpose:\n"
              "  //: the form is the point.\n"
              '  b.expect(lab::kernel("add", {1}), "LOOP_CONVERGENCE", "<=", 8);\n')
    entries = lab.declaration_entries([expect("add", "LOOP_CONVERGENCE", "<=", 8)], lab.declared_reasons(source))
    assert entries == [{"kernel": "add", "rule": "loop_convergence", "allow": 8,
                        "reason": "one divergent store a trip, kept on purpose: the form is the point."}]


def test_a_declaration_with_no_comment_or_with_two_different_ones_says_where_its_reason_is():
    none = '  b.expect(k, "LOAD_IN_TRIP", "<=", 2);\n'
    two = '  //: a\n  b.expect(k, "LOAD_IN_TRIP", "<=", 2);\n  //: b\n  b.expect(k, "LOAD_IN_TRIP", "<=", 3);\n'
    same = '  //: a\n  b.expect(k, "LOAD_IN_TRIP", "<=", 2);\n\n  //: a\n  b.expect(j, "LOAD_IN_TRIP", "<=", 3);\n'
    assert lab.declared_reasons(none) == {} and lab.declared_reasons(two) == {}
    assert lab.declared_reasons(same) == {"LOAD_IN_TRIP": "a"}
    entries = lab.declaration_entries([expect("k", "LOAD_IN_TRIP", "<=", 2)], lab.declared_reasons(two))
    assert entries[0]["reason"] == lab.UNSTATED


def test_a_reason_is_read_where_the_declaration_is_one_line_with_a_condition():
    source = '  //: the tile forms keep the loop.\n  if (tiled) b.expect(k, "STEP_ROLLED", "<=", 16);\n'
    assert lab.declared_reasons(source) == {"STEP_ROLLED": "the tile forms keep the loop."}


def test_every_hazard_has_a_rule_and_a_refusal_text_and_the_notice_ones_a_statement():
    for hazard, rule in lab.HAZARD_RULES.items():
        assert rule in lab.REFUSAL_TEXT or hazard in lab.HAZARD_STATES, hazard
    for rule in lab.PRESENCE_RULES.values():
        assert rule in lab.REFUSAL_TEXT
    assert "convergence_call" in lab.REFUSAL_TEXT
    assert set(lab.HAZARD_STATES) <= set(lab.HAZARD_RULES)


def test_a_permit_is_any_number_of_its_rule_and_an_opcode_count_is_no_declaration():
    entries = lab.declaration_entries([expect("k", "BSSY", ">=", 0), expect("k", "STEP_XU", "<=", 64),
                                       expect("k", "STEP_GLOBAL", ">=", 0), expect("k", "CALL", ">=", 8),
                                       expect("k", "HMMA", ">=", 1)], {})
    assert [(e["rule"], e["allow"]) for e in entries] == [(r, lab.PRESENCE_ALLOWANCE) for r in
                                                         ("step_branches", "step_conversion", "step_global", "convergence_call")]


def test_the_declarations_of_a_rule_are_ordered_largest_first_so_the_first_match_is_the_old_maximum():
    entries = lab.declaration_entries([expect("a", "LOOP_CONVERGENCE", "<=", 2), expect(".*", "LOOP_CONVERGENCE", "<=", 9),
                                       expect("a", "LOAD_IN_TRIP", "<=", 1)], {})
    assert [(e["rule"], e["allow"]) for e in entries if e["rule"] == "loop_convergence"] == [("loop_convergence", 9), ("loop_convergence", 2)]


def test_a_bound_is_declared_less_or_equal_and_bounds_a_kernel_with_steps():
    with_steps = SimpleNamespace(regions=(object(),), demangled="tile<1>(int)")
    without = SimpleNamespace(regions=(), demangled="plain(int)")
    assert lab.bound_failures([expect("tile", "STEP_PORT", "<=", 4096)], [with_steps, without]) == []
    assert "declared \"<=\" (not '==')" in lab.bound_failures([expect("tile", "STEP_WORK", "==", 40)], [with_steps])[0]
    assert "no kernel with tensor steps matches 'plain'" in lab.bound_failures([expect("plain", "STEP_WORK", "<=", 40)], [with_steps, without])[0]
    assert lab.declaration_entries([expect("tile", "STEP_WORK", "==", 40)], {}) == []


def refusal(rule, count, allowed, *where):
    findings = tuple(SimpleNamespace(where=w, count=1, what=f"at {w}") for w in where)
    return SimpleNamespace(rule=rule, kernel="k", count=count, allowed=allowed, findings=findings, message="", fix="")


def test_a_refusal_names_the_hazard_its_experiment_declares_the_counts_and_the_rule():
    line, = lab.render_refusal(refusal("load_in_trip", 3, 1, "0x0100", "0x0120"))
    assert line.startswith("k: 3 global loads read in the trip that issued them") and "1 declared (LOAD_IN_TRIP)" in line
    assert line.endswith("[checker rule load_in_trip: at 0x0100, 0x0120]")
    line, = lab.render_refusal(refusal("step_rolled", 4, 0, "0x57d0"))
    assert "4 tensor steps in loops that hold no wait on them (loops at 0x57d0)" in line and "0 declared (STEP_ROLLED)" in line
    line, = lab.render_refusal(refusal("step_port", 5120, 4096, "0x0680"))
    assert "5,120 bytes a step through the shared-memory port, over the 4,096 declared (STEP_PORT" in line
    line, = lab.render_refusal(refusal("step_conversion", 2, 0, "0x0010"))
    assert "nobody declared" in line and "STEP_XU" in line


def test_a_declared_notice_count_is_shown_under_its_hazard():
    notice = SimpleNamespace(rule="mma_serialized", kernel="k", count=2, allowed=1, message="", fix="state the count at entry",
                             findings=(SimpleNamespace(where="k", count=2, what="ptxas C7512: the pipeline"),))
    line, = lab.render_refusal(notice)
    assert line.startswith("k: the toolkit serialized its asynchronous MMAs") and "2 x ptxas C7512: the pipeline" in line
    assert "1 declared (STEPS_SERIALIZED). state the count at entry" in line


def test_a_refused_notice_is_shown_with_its_row_and_an_unknown_one_quoted_whole():
    refused = SimpleNamespace(rule="notice ptxas C7505", kernel="k", count=1, allowed=0, message="the reason", fix="the fix",
                              findings=(SimpleNamespace(where="k", count=1, what="'setmaxnreg' ignored"),))
    assert lab.render_refusal(refused) == ["k: 'setmaxnreg' ignored (1 x, ptxas C7505): the reason the fix"]
    unknown = SimpleNamespace(rule="notice_without_row", kernel="k", count=2, allowed=0, fix="",
                              message="the toolkit printed a notice the ledger has no row for (ptxas C7777)",
                              findings=(SimpleNamespace(where="k", count=2, what="ptxas info : (C7777) new"),))
    line, = lab.render_refusal(unknown)
    assert line.startswith("k: 2 x the toolkit printed a notice the ledger has no row for (ptxas C7777):\n      ptxas info : (C7777) new")
    assert "add a row to tools/notice_ledger.json" in line


@needs_checker
def test_every_hazard_names_a_rule_the_checker_has():
    c = lab.checker()
    known = set(c.rules.registry()) | set(c.ledgers.declared_rules(lab.notice_ledger()))
    assert set(lab.HAZARD_RULES.values()) | set(lab.PRESENCE_RULES.values()) | {"convergence_call"} <= known


@needs_checker
def test_the_declarations_a_translation_makes_are_the_checkers_own_to_load():
    c = lab.checker()
    entries = lab.declaration_entries([expect("k", "STEP_PORT", "<=", 2048), expect("k", "STEPS_SERIALIZED", "<=", 1),
                                       expect("k", "BSSY", ">=", 0)], {})
    loaded = c.declare.load_declarations(lab.json.dumps({"declarations": entries}), c.rules.registry(),
                                         c.ledgers.declared_rules(lab.notice_ledger()))
    assert {d.rule for d in loaded} == {"step_port", "mma_serialized", "step_branches"}


def test_the_debts_decide_which_rules_an_experiment_is_held_to():
    new = lab.held_rules("a_new_experiment")
    assert {"load_in_trip", "store_reload", "store_unfenced", "clock_uniform", "step_rolled", "wait_after_issue",
            "loop_convergence"} <= new
    always = {"step_port", "step_work", "step_branches", "step_conversion", "step_global", "convergence_call", "loop_convergence"}
    assert lab.held_rules("tensor_tax") == always
    assert lab.held_rules("clock_time") == new - {"wait_after_issue"}


@needs_checker
def test_the_steps_advisory_prints_the_checkers_figures_and_groups_kernels_with_the_same_lines(monkeypatch):
    def instruction(klass):
        return SimpleNamespace(klass=klass)

    step = SimpleNamespace(operand_bytes=2048, null=False)
    load = SimpleNamespace(space="shared", kind="load", warp_bytes=128)
    region = SimpleNamespace(kind="loop", address=0x100, steps=2, tensor=(step, step), accesses=(load,),
                             instructions=(instruction("tensor"), instruction("tensor"), instruction("integer/address"),
                                           instruction("control")))
    kernels = [SimpleNamespace(demangled=f"k<{i}>", regions=(region,)) for i in range(3)]
    monkeypatch.setattr(lab, "checked_kernels", lambda binary: [*kernels, SimpleNamespace(demangled="plain", regions=())])
    out = lab.steps_report(None)
    assert out[0] == "  k<0>  (and 2 kernels with the same lines)" and len(out) == 3
    assert out[1] == "    port: 2,304 B a step (18 wavefronts) operands 2,048 + shared 256 (x 4 warps)"
    assert out[2] == "    work: 3 instructions a step tensor 1, integer/address 1, control 1"
