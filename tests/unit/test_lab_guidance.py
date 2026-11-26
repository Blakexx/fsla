# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Fsla's guidance on the checker's rules (the `checker` section of docs/rules.json: an entry a checker rule, with `guidance`, `triggers`, `rules`): the lab prints a rule's line
after the checker's fix in a refusal (tools/lab.py `guidance`), and the edit hook (.claude/hooks/guidance.py) injects the lines whose
triggers an edit's new text matches into a kernel source. The two-way lint of the table against the checker's rules is
tools/lint/rules_table.py's, with its own fixtures; these tests cover the lab's call and the hook."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import checked  # noqa: E402
import dev_config  # noqa: E402
import lab  # noqa: E402

HOOK = ROOT / ".claude" / "hooks" / "guidance.py"
needs_checker = pytest.mark.skipif(dev_config.warpline_python() is None,
                                   reason="warpline's checker is handed in by the graph (bazel test //tests:lab)")

#: a line of kernel source for each checker rule that its row's triggers must match: a trigger no source shows is dead
SAMPLES = {
    "step_rolled": "for (int r = 0; r < steps; ++r) {\n  wg_bf16_n16_ss(acc, a, b);\n}",
    "step_branches": "if (live) {\n  wg_bf16_n16_ss(acc, a, b);\n}",
    "step_predicated": "if (live) wg_bf16_n16_ss(acc, a, b);",
    "step_conversion": "int q = __float2int_rn(x);",
    "step_global": "atomicAdd(out + i, v);",
    "step_port": "auto p = reinterpret_cast<uint4*>(smem + base);",
    "step_work": "wg_bf16_n64_rs(acc, a, b);",
    "load_in_trip": "for (int t = 0; t < n; ++t) {\n  auto v = __ldg(src + t);\n}",
    "store_reload": "store_shared_u32(p, v);\nauto w = load_shared_u32(p);",
    "store_unfenced": "mbar_arrive(bar);",
    "clock_uniform": "auto t0 = cycles();",
    "convergence_call": "auto m = __ballot_sync(full, p);",
    "loop_convergence": "while (more) {\n  v = __shfl_sync(full, v, 0);\n}",
    "stack_memory": "  float row[16];",
    "wait_after_issue": "wg_commit();\nwg_wait<0>();",
    "mma_serialized": "asm volatile(\"setmaxnreg.inc.sync.aligned.u32 232;\");",
    "wait_injected": "wg_commit();",
    "shared_wavefronts_excess": "__shared__ float tile[64][65];",
}


def entries():
    return json.loads((ROOT / "docs" / "rules.json").read_text())["checker"]


def hook(path, **args):
    call = {"hook_event_name": "PreToolUse", "tool_name": "Edit", "cwd": str(ROOT), "tool_input": {"file_path": str(ROOT / path), **args}}
    ran = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(call), capture_output=True, text=True, cwd=ROOT,
                         env={"PATH": ""}, check=True)
    return json.loads(ran.stdout)["hookSpecificOutput"] if ran.stdout.strip() else None


def test_an_edit_that_adds_a_branch_around_a_tensor_step_is_given_the_step_rules_guidance():
    out = hook("measure/lab/experiments/carry_token.cu", old_string="wg_bf16_n16_ss(acc, a, b);",
               new_string="if (live) {\n  wg_bf16_n16_ss(acc, a, b);\n}")
    assert out["hookEventName"] == "PreToolUse" and "permissionDecision" not in out
    assert "- step_branches (KERNEL_STANDARDS#s27.1)" in out["additionalContext"] and "- step_predicated" in out["additionalContext"]
    assert "KERNEL_STANDARDS#s27.1" in out["additionalContext"]


def test_an_edit_that_only_renames_a_variable_is_given_nothing():
    assert hook("csrc/fsla/src/common/ops.cuh", old_string="int slot = base;", new_string="int lane_slot = base;") is None
    assert hook("measure/lab/experiments/carry_token.cu", old_string="acc_a", new_string="acc_b") is None


def test_a_file_that_is_no_kernel_source_is_given_nothing_whatever_it_holds():
    for path in ("docs/KERNEL_STANDARDS.md", "tools/lab.py", "tests/unit/test_lab_guidance.py"):
        assert hook(path, new_string="if (live) {\n  wg_bf16_n16_ss(acc, a, b);\n}") is None


def test_a_write_and_a_multi_edit_are_read_by_their_new_text():
    assert "step_rolled" in hook("measure/lab/experiments/new.cu", content=SAMPLES["step_rolled"])["additionalContext"]
    out = hook("csrc/fsla/src/common/ops.cuh", edits=[{"old_string": "a", "new_string": "int b;"}, {"old_string": "c", "new_string": SAMPLES["clock_uniform"]}])
    assert "- clock_uniform" in out["additionalContext"]


@pytest.mark.parametrize("rule", sorted(SAMPLES))
def test_every_rule_with_triggers_fires_on_a_line_of_source_that_shows_it(rule):
    assert entries()[rule]["triggers"], f"{rule} has an entry with no trigger, and a sample of source that shows it"
    assert f"- {rule}" in hook("measure/lab/experiments/x.cu", new_string=SAMPLES[rule])["additionalContext"]


def test_a_rule_whose_entry_has_no_triggers_names_no_source_text_for_it_and_is_not_in_the_samples():
    quiet = {rule for rule, entry in entries().items() if not entry["triggers"]}
    #: the checker's own rules a source text never shows, and the notice rules (the toolkit's messages, which no source line triggers)
    assert quiet == {"step_density", "step_gap", "profiler_verdicts", "function_size", "loop_held", "load_behind_load", "build_info", "local_memory_used",
                     "register_spill",
                     "setmaxnreg_ignored", "unreachable_loop", "unused_variable", "variable_unset", "wgmma_arrive_injected"}
    assert not quiet & set(SAMPLES)


@needs_checker
def test_the_lab_prints_the_checkers_fix_then_the_fsla_note_after_a_refusal():
    notes = lab.guidance()
    assert set(notes) == set(checked.checker().declare.rule_names(checked.project_ledger()))
    refusal = SimpleNamespace(rule="step_rolled", kernel="k", count=4, allowed=0, message="", fix="write the steps a straight block",
                              note=notes["step_rolled"], findings=(SimpleNamespace(where="0x57d0", count=4, what="x"),))
    line, = lab.render_refusal(refusal)
    assert line.index("[checker rule step_rolled") < line.index("\n    fix: write the steps a straight block") < line.index("\n    note: Write steps")


@needs_checker
def test_the_checkers_own_refusal_carries_the_note_the_lab_passes():
    c = checked.checker()
    refusal = c.declare.judge("k", "step_rolled", [SimpleNamespace(where="0x57d0", count=4, what="x")], "m", "the fix", [], lab.guidance())
    assert refusal.note == lab.guidance()["step_rolled"] and refusal.lines()[-1] == f"  note: {refusal.note}"
