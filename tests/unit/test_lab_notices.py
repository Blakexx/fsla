# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's ledger of the toolkit's notices: every notice of a build is read by warpline's checker against ITS base ledger with the
project's rows (tools/notice_ledger.json, passed to the checker), and the lab shows the refusals under its own hazards' names -- a notice
with no row refuses, `refuse` fails, `declare` allows the count an experiment declares of its hazard (`STEPS_SERIALIZED`,
`WAIT_INJECTED`: the checker's rules `mma_serialized` and `wait_injected`), `report` prints, `ignore` is silent. The checker's own
tests cover the reader and the ledger's mechanics (the module's tests/test_checker_notices.py); these cover how the lab calls them and the
project's rows."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import checked  # noqa: E402
import dev_config  # noqa: E402
import lab  # noqa: E402

needs_checker = pytest.mark.skipif(dev_config.warpline_python() is None,
                                   reason="warpline's checker is handed in by the graph (bazel test //tests:lab)")

C7512 = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "insufficient register resources for the wgmma pipeline in the function '{}'")
C7520 = ("ptxas info    : (C7520) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "program dependence on compiler-inserted WG.AR in divergent path in the function '{}'")
#: a code no build has emitted yet, in the toolkit's template for extern calls (a row with no code)
C7599 = ("ptxas info    : (C7599) Potential Performance Loss: wgmma.mma_async instructions are serialized due to the "
         "presence of Extern calls in the function '{}'")
C7511 = ("ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "insufficient register resources for the function '{}'")
C7517 = ("ptxas info    : (C7517) warpgroup.wait is injected in around line 3928 by compiler to allow use of registers "
         "defined by GMMA in function '{}'")
C7519 = ("ptxas info    : (C7519) warpgroup.arrive is injected in around line 9191 by compiler to allow use of "
         "registers in GMMA in function '{}'")
SETMAXNREG = "ptxas info    : (C7505) Potential Performance Loss: 'setmaxnreg' ignored to allow debugging."
#: a template the toolkit prints with no code in front of it
BARE = "Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility into 'extern' call."
UNSET = ('/work/measure/lab/experiments/x.cu(1706): warning #549-D: variable "far" is used before its value is set\n'
         "          detected during:\n"
         '            instantiation of "void x::f<ROLES>()" [with ROLES=1LL]\n')
LOOP = ("/work/measure/lab/experiments/x.cu(354): warning #128-D: loop is not reachable\n"
        "      for (int half = 0; half < 2; ++half) {\n"
        "      ^\n"
        "\n"
        'Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"\n')
SPILL = ("ptxas warning : Registers are spilled to local memory in function '{}', 340 bytes spill stores, "
         "436 bytes spill loads")
LOCAL = "ptxas warning : Local memory used for function '{}', size of stack frame: 328 bytes"
SUMMARY = ("ptxas info    : Compiling entry function '{0}' for 'sm_90a'\n"
           "ptxas info    : Function properties for {0}\n"
           "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
           "ptxas info    : Used 90 registers, used 1 barriers, 4096 bytes smem\n"
           "ptxas info    : Compile time = 1.044 ms\n"
           "ptxas info    : 0 bytes gmem")
AR = "program dependence on compiler-inserted WG.AR in divergent path"


def binary_with(tmp_path, *lines, declared=()):
    """A built experiment whose binary's `--expect` prints the declarations, with the notices of its build beside it."""
    built = lab.Built(tmp_path, "x", "sm_90a")
    built.binary.write_text("#!/bin/sh\n" + "".join(f"echo '{d}'\n" for d in declared))
    built.binary.chmod(0o755)
    lab.notices(built).write_text("\n".join(lines) + "\n")
    return built


def declare(kernel, hazard, count):
    return f'{{"lab": "expect", "kernel": "{kernel}", "op": "{hazard}", "cmp": "<=", "count": {count}}}'


def failures_of(tmp_path, monkeypatch, *lines, declared=()):
    """The lab's failures for a build of `x` with no kernels whose notices are `lines` (names as given: no demangling)."""
    monkeypatch.setattr(lab.Built, "demangle", lambda self, names: {n: n for n in names})
    monkeypatch.setattr(lab, "model_of", lambda built: [])
    monkeypatch.setattr(lab, "experiments", lambda: {})
    return lab.checker_failures(binary_with(tmp_path, *lines, declared=declared), [json.loads(d) for d in declared], set())


@needs_checker
def test_every_line_a_build_prints_is_a_notice_with_its_tool_code_and_text(tmp_path):
    read = lab.read_notices(binary_with(tmp_path, C7520.format("k"), SETMAXNREG, BARE, UNSET, LOOP, SUMMARY.format("k")))
    assert [(n.tool, n.code) for n in read] == [
        ("ptxas", "C7520"), ("ptxas", "C7505"), ("ptxas", None), ("nvcc", "#549-D"), ("nvcc", "#128-D"),
        ("nvcc", None), ("ptxas", None), ("ptxas", None), ("ptxas", None), ("ptxas", None), ("ptxas", None)]
    assert read[0].function == "k" and read[0].text.endswith("WG.AR in divergent path in the function '{}'")
    assert read[3].where == "x.cu(1706)" and read[3].function is None


@needs_checker
def test_a_notice_the_ledger_has_no_row_for_refuses_and_is_quoted_whole(tmp_path, monkeypatch):
    new = "ptxas warning : (C7777) something the toolkit never printed before in the function 'k'"
    failures = failures_of(tmp_path, monkeypatch, new, UNSET.replace("549", "999"))
    assert len(failures) == 2
    assert new in failures[0] and "add a row to tools/notice_ledger.json" in failures[0] and "disposition" in failures[0]
    assert "warning #999-D" in failures[1] and "instantiation of" in failures[1]
    assert "some line that is no notice at all" in failures_of(tmp_path, monkeypatch, "some line that is no notice at all")[0]


@needs_checker
def test_a_refuse_row_fails_with_its_reason_whatever_is_declared(tmp_path, monkeypatch):
    declared = [declare(".*", "STEPS_SERIALIZED", 9)]
    failures = failures_of(tmp_path, monkeypatch, SETMAXNREG, declared=declared)
    assert len(failures) == 1 and failures[0].startswith("x: Potential Performance Loss: 'setmaxnreg' ignored")
    assert "(1 x, ptxas C7505)" in failures[0] and "register reallocation" in failures[0]


@needs_checker
def test_a_variable_used_before_its_value_is_set_refuses_naming_it_and_its_line(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, UNSET, UNSET)
    assert len(failures) == 1 and failures[0].startswith('x.cu(1706): variable "far" is used before its value is set')
    assert "(2 x, nvcc #549-D)" in failures[0] and "no write reaches" in failures[0]


@needs_checker
def test_a_declare_row_allows_the_count_its_experiment_declares_a_kernel_summed_over_its_codes(tmp_path, monkeypatch):
    lines = (C7599.format("k"), C7520.format("k"))
    failures = failures_of(tmp_path, monkeypatch, *lines)
    assert len(failures) == 1 and "k: the toolkit serialized" in failures[0]
    assert "the presence of Extern calls" in failures[0] and AR in failures[0]
    #: the project's own row for C7520 words the fix: the lab's page, not the module's text
    assert "forward branch" in failures[0] and "STEP_ROLLED" in failures[0] and "0 declared (STEPS_SERIALIZED)" in failures[0]
    assert failures_of(tmp_path, monkeypatch, *lines, declared=[declare("k", "STEPS_SERIALIZED", 2)]) == []
    assert failures_of(tmp_path, monkeypatch, *lines, C7520.format("k"),
                       declared=[declare("k", "STEPS_SERIALIZED", 2)]) != []
    #: a declaration is for the kernels its regex names
    assert failures_of(tmp_path, monkeypatch, *lines, declared=[declare("other", "STEPS_SERIALIZED", 9)]) != []


@needs_checker
def test_each_serializing_notice_of_a_kernel_is_counted_by_its_code_and_cause(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7512.format("a"), C7520.format("a"), C7520.format("a"),
                           C7520.format("b"))
    assert len(failures) == 2
    assert "a: the toolkit serialized" in failures[0] and "1 x ptxas C7512" in failures[0]
    assert "2 x ptxas C7520" in failures[0] and "b: " in failures[1] and "1 x ptxas C7520" in failures[1]


@needs_checker
def test_a_register_notice_of_c7511_is_the_known_code_it_is(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7511.format("k"))
    assert len(failures) == 1 and "each role's whole body inside its own branch" in failures[0]


@needs_checker
def test_an_injected_wait_is_counted_a_kernel_unless_declared_and_an_injected_arrive_never(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"))
    assert len(failures) == 1 and "injected a `warpgroup.wait`" in failures[0] and "1 x ptxas C7517" in failures[0]
    assert failures_of(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"),
                       declared=[declare("k", "WAIT_INJECTED", 1)]) == []


@needs_checker
def test_a_report_row_prints_with_its_count_and_blocks_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(lab.Built, "demangle", lambda self, names: {n: n for n in names})
    binary = binary_with(tmp_path, C7519.format("a"), C7519.format("a"), C7519.format("b"), LOOP)
    monkeypatch.setattr(lab, "model_of", lambda built: [])
    monkeypatch.setattr(lab, "experiments", lambda: {})
    assert lab.checker_failures(binary, [], set()) == []
    report = lab.notice_report(binary)
    assert len(report) == 2
    assert "ptxas C7519: 3 x in 2 place(s)" in report[0] and "costs a few cycles" in report[0]
    assert "nvcc #128-D: 1 x in 1 place(s)" in report[1]


@needs_checker
def test_an_ignore_row_is_silent_and_so_is_the_verbose_summary(tmp_path, monkeypatch):
    unused = ('/work/x.cu(308): warning #177-D: variable "tid" was declared but never referenced\n'
              "    const int tid = 0;\n    ^\n")
    assert failures_of(tmp_path, monkeypatch, unused, SUMMARY.format("k")) == []
    assert lab.notice_report(binary_with(tmp_path, unused, SUMMARY.format("k"))) == []


@needs_checker
def test_a_spill_and_a_stack_frame_are_the_toolkits_own_warnings_naming_the_kernel(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, SPILL.format("k"), LOCAL.format("k"), SUMMARY.format("clean"))
    assert len(failures) == 2
    assert any(f.startswith("k: Registers are spilled to local memory in function 'k', 340 bytes spill stores")
               for f in failures)
    assert any("size of stack frame: 328 bytes" in f for f in failures)


@needs_checker
def test_check_refuses_what_the_ledger_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "timeline", lambda path: True)
    monkeypatch.setattr(lab, "collective_selected", lambda path: [])
    failures = failures_of(tmp_path, monkeypatch, UNSET, C7520.format("k"))
    assert len(failures) == 2 and any("#549-D" in f for f in failures) and any("k: the toolkit serialized" in f for f in failures)
    (tmp_path / "x.cu").write_text("// TIMELINE: none\n")
    monkeypatch.setattr(lab, "experiments", lambda: {"x": tmp_path / "x.cu"})
    failures = lab.check(binary_with(tmp_path, UNSET, C7520.format("k")), {})
    assert len(failures) == 2 and any("#549-D" in f for f in failures) and any("k: the toolkit serialized" in f for f in failures)


@needs_checker
def test_a_build_with_no_notices_file_has_none(tmp_path):
    built = lab.Built(tmp_path, "carried", "sm_90a")
    assert lab.read_notices(built) == [] and lab.notice_report(built) == []


def test_the_graphs_compile_switches_on_the_options_ptxas_names():
    """The lab's experiments are compiled by the graph's rule (bazel/lab.bzl), which passes ptxas the options the notices read."""
    rule = (Path(__file__).resolve().parents[2] / "bazel" / "lab.bzl").read_text()
    assert '"--warn-on-spills,--warn-on-local-memory-usage,--verbose"' in rule


@needs_checker
def test_the_projects_ledger_overrides_or_adds_to_the_checkers_base_and_repeats_none_of_it():
    ledgers = checked.checker().ledgers
    project = ledgers.load_ledger(lab.PROJECT_LEDGER.read_text())
    base = {row.key: row for row in ledgers.base_ledger()}
    assert project
    for row in project:
        assert row.reason.endswith(".") and "\n" not in row.reason and row.disposition in ledgers.DISPOSITIONS
        if row.key in base:
            same = (row.disposition, row.rule, row.reason, row.fix) == (base[row.key].disposition, base[row.key].rule,
                                                                           base[row.key].reason, base[row.key].fix)
            assert not same, f"{row.text!r}: only repeats the base row"
    #: the merged ledger is the checker's own check: no key twice, no two rows matching one notice
    assert len(lab.notice_ledger()) >= len(base)


@needs_checker
def test_the_hazards_the_ledger_declares_are_the_ones_an_experiment_may_declare():
    declared = set(checked.checker().ledgers.declared_rules(lab.notice_ledger()))
    assert declared == {lab.HAZARD_RULES[h] for h in lab.HAZARD_STATES} == {"mma_serialized", "wait_injected"}
    assert set(lab.HAZARD_STATES) <= set(lab.HAZARDS)
