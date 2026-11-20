# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's ledger of the toolkit's notices (tools/lab.py, tools/notice_ledger.json): every notice of a build, read by
its tool, code and text, is looked up in the ledger and handled by its row's disposition -- a notice with no row
refuses, `refuse` fails, `declare` allows the count an experiment declares of its hazard, `report` prints, `ignore`
is silent; and the ledger file itself is well formed."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab  # noqa: E402

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
    """A binary whose `--expect` prints the declarations, with the notices of its build beside it."""
    binary = tmp_path / "x.sm_90a"
    binary.write_text("#!/bin/sh\n" + "".join(f"echo '{d}'\n" for d in declared))
    binary.chmod(0o755)
    lab.notices(binary).write_text("\n".join(lines) + "\n")
    return binary


def declare(kernel, hazard, count):
    return f'{{"lab": "expect", "kernel": "{kernel}", "op": "{hazard}", "cmp": "<=", "count": {count}}}'


def failures_of(tmp_path, monkeypatch, *lines, declared=()):
    """`lab.notice_failures` of a build of `x` whose notices are `lines` (kernel names as given: no demangling)."""
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    return lab.notice_failures(binary_with(tmp_path, *lines, declared=declared), [json.loads(d) for d in declared])


def test_every_line_a_build_prints_is_a_notice_with_its_tool_code_and_text(tmp_path):
    read = lab.read_notices(binary_with(tmp_path, C7520.format("k"), SETMAXNREG, BARE, UNSET, LOOP, SUMMARY.format("k")))
    assert [(n.tool, n.code) for n in read] == [
        ("ptxas", "C7520"), ("ptxas", "C7505"), ("ptxas", None), ("nvcc", "#549-D"), ("nvcc", "#128-D"),
        ("nvcc", None), ("ptxas", None), ("ptxas", None), ("ptxas", None), ("ptxas", None), ("ptxas", None)]
    assert read[0].function == "k" and read[0].text.endswith("WG.AR in divergent path in the function '{}'")
    assert read[3].where == "x.cu(1706)" and read[3].function is None
    assert "instantiation of" in read[3].raw and "for (int half" in read[4].raw
    assert "0 bytes stack frame" in read[7].raw


def test_a_coded_notice_matches_by_code_and_text_and_a_code_less_one_by_text(tmp_path):
    coded, bare = lab.read_notices(binary_with(tmp_path, SETMAXNREG, BARE))
    assert lab.row_of(coded)["code"] == "C7505" and lab.row_of(bare)["code"] is None
    assert lab.row_of(coded._replace(code="C7599")) is None
    assert lab.row_of(bare._replace(code="C7599"))["disposition"] == "refuse"
    c7520, = lab.read_notices(binary_with(tmp_path, C7520.format("k")))
    assert lab.row_of(c7520)["code"] == "C7520"
    assert lab.row_of(c7520._replace(code="C7512")) is None


def test_a_notice_the_ledger_has_no_row_for_refuses_and_is_quoted_whole(tmp_path, monkeypatch):
    new = "ptxas warning : (C7777) something the toolkit never printed before in the function 'k'"
    failures = failures_of(tmp_path, monkeypatch, new, UNSET.replace("549", "999"))
    assert len(failures) == 2
    assert new in failures[0] and "add a row to tools/notice_ledger.json" in failures[0] and "disposition" in failures[0]
    assert "warning #999-D" in failures[1] and "instantiation of" in failures[1]
    assert "some line that is no notice at all" in failures_of(tmp_path, monkeypatch, "some line that is no notice at all")[0]


def test_a_refuse_row_fails_with_its_reason_whatever_is_declared(tmp_path, monkeypatch):
    declared = [declare(".*", "SETMAXNREG_IGNORED", 9)]
    failures = failures_of(tmp_path, monkeypatch, SETMAXNREG, declared=declared)
    assert len(failures) == 1 and failures[0].startswith("x: Potential Performance Loss: 'setmaxnreg' ignored")
    assert "(1 x, ptxas C7505)" in failures[0] and "register redistribution" in failures[0]


def test_a_variable_used_before_its_value_is_set_refuses_naming_it_and_its_line(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, UNSET, UNSET)
    assert len(failures) == 1 and failures[0].startswith('x.cu(1706): variable "far" is used before its value is set')
    assert "(2 x, nvcc #549-D)" in failures[0] and "unset value" in failures[0]


def test_a_declare_row_allows_the_count_its_experiment_declares_a_kernel_summed_over_its_codes(tmp_path, monkeypatch):
    lines = (C7599.format("k"), C7520.format("k"))
    failures = failures_of(tmp_path, monkeypatch, *lines)
    assert len(failures) == 1 and "k: the toolkit serialized" in failures[0]
    assert "(C7599) the presence of Extern calls" in failures[0] and f"(C7520) {AR}" in failures[0]
    assert "forward branch" in failures[0] and "STEP_ROLLED" in failures[0] and "0 declared" in failures[0]
    assert failures_of(tmp_path, monkeypatch, *lines, declared=[declare("k", "STEPS_SERIALIZED", 2)]) == []
    assert failures_of(tmp_path, monkeypatch, *lines, C7520.format("k"),
                       declared=[declare("k", "STEPS_SERIALIZED", 2)]) != []
    #: a declaration is for the kernels its regex names
    assert failures_of(tmp_path, monkeypatch, *lines, declared=[declare("other", "STEPS_SERIALIZED", 9)]) != []


def test_each_serializing_notice_of_a_kernel_is_counted_by_its_code_and_cause(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7512.format("a"), C7520.format("a"), C7520.format("a"),
                           C7520.format("b"))
    assert len(failures) == 2
    assert "a: the toolkit serialized" in failures[0] and "1 x (C7512) insufficient register resources" in failures[0]
    assert f"2 x (C7520) {AR}" in failures[0] and "b: " in failures[1] and f"1 x (C7520) {AR}" in failures[1]


def test_a_register_notice_of_c7511_is_the_known_code_it_is(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7511.format("k"))
    assert len(failures) == 1 and "each role's whole body inside its own branch" in failures[0]
    assert "seen here first" not in failures[0]


def test_an_injected_wait_is_counted_a_kernel_unless_declared_and_an_injected_arrive_never(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"))
    assert len(failures) == 1 and "injected a `warpgroup.wait`" in failures[0] and "1 x (C7517)" in failures[0]
    assert failures_of(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"),
                       declared=[declare("k", "WAIT_INJECTED", 1)]) == []


def test_a_report_row_prints_with_its_count_and_blocks_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    binary = binary_with(tmp_path, C7519.format("a"), C7519.format("a"), C7519.format("b"), LOOP)
    assert lab.notice_failures(binary, []) == []
    report = lab.notice_report(binary)
    assert len(report) == 2
    assert "ptxas C7519: 3 x in 2 places" in report[0] and "cost is unmeasured" in report[0]
    assert "nvcc #128-D: 1 x in 1 places, at x.cu(354)" in report[1]


def test_an_ignore_row_is_silent_and_so_is_the_verbose_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    unused = ('/work/x.cu(308): warning #177-D: variable "tid" was declared but never referenced\n'
              "    const int tid = 0;\n    ^\n")
    binary = binary_with(tmp_path, unused, SUMMARY.format("k"))
    assert lab.notice_failures(binary, []) == [] and lab.notice_report(binary) == []


def test_a_spill_and_a_stack_frame_are_the_toolkits_own_warnings_naming_the_kernel(tmp_path, monkeypatch):
    failures = failures_of(tmp_path, monkeypatch, SPILL.format("k"), LOCAL.format("k"), SUMMARY.format("clean"))
    assert len(failures) == 2
    assert any(f.startswith("k: Registers are spilled to local memory in function 'k', 340 bytes spill stores")
               for f in failures)
    assert any("size of stack frame: 328 bytes" in f for f in failures)


def test_check_refuses_what_the_ledger_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    monkeypatch.setattr(lab, "step_bounds", lambda binary, expected: [])
    for name in ("loop_convergence", "clocks_summed_uniform", "loop_hazards", "waits_after_issue", "rolled_steps",
                 "signatures"):
        monkeypatch.setattr(lab, name, lambda binary: {})
    monkeypatch.setattr(lab, "timeline", lambda path: True)
    monkeypatch.setattr(lab, "collective_selected", lambda path: [])
    monkeypatch.setattr(lab, "experiments", lambda: {"x": tmp_path / "x.cu"})
    failures = lab.check(binary_with(tmp_path, UNSET, C7520.format("k")), {})
    assert len(failures) == 2 and any("#549-D" in f for f in failures) and any("k: the toolkit serialized" in f for f in failures)


def test_a_build_with_no_notices_file_has_none(tmp_path):
    binary = tmp_path / "carried.sm_90a"
    assert lab.read_notices(binary) == [] and lab.notice_failures(binary, []) == [] and lab.notice_report(binary) == []


def test_the_toolkits_options_are_the_ones_ptxas_names():
    assert lab.PTXAS_OPTIONS == ("--warn-on-spills", "--warn-on-local-memory-usage", "--verbose")


def sample(row):
    """A notice of `row`'s own template, each `{}` filled with a word, and a code (a made-up one for a row that
    names none)."""
    text = " ".join(row["text"].replace("{}", "x").split()).replace("function 'x'", "function '{}'")
    return lab.Notice(row["tool"], row["code"] or "C7000", text, None, None, text, text)


def matching(notice):
    return [r for r in lab.notice_ledger() if r["tool"] == notice.tool and r["code"] in (None, notice.code)
            and lab._template(r["text"]).fullmatch(notice.text)]


@pytest.mark.parametrize("row", lab.notice_ledger(), ids=lambda r: f"{r['tool']}-{r['code'] or 'nocode'}-{r['text'][:40]}")
def test_every_ledger_row_is_well_formed(row):
    assert row["tool"] in ("ptxas", "nvcc") and row["text"].strip()
    assert row["disposition"] in lab.DISPOSITIONS
    assert row["reason"].strip().endswith(".") and "\n" not in row["reason"]
    assert (row["disposition"] == "declare") == bool(row["hazard"])
    if row["disposition"] == "declare":
        assert row["cause"].strip() and row["hazard"] in lab.HAZARD_STATES
    if row["evidence"]:
        assert (lab.ROOT / row["evidence"]).is_file()
    assert row["code"] is None or row["code"][0] in "C#"
    assert row["fix"] is None or row["fix"].strip()


def test_no_two_rows_match_one_notice():
    for row in lab.notice_ledger():
        assert matching(sample(row)) == [row], f"{row['text']!r}: more than one row matches"


def test_the_hazards_the_ledger_declares_are_the_ones_an_experiment_may_declare():
    assert set(lab.hazard_names()) == set(lab.HAZARD_STATES) == {"STEPS_SERIALIZED", "WAIT_INJECTED"}
    assert set(lab.hazard_names()) <= set(lab.HAZARDS)
