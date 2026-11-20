# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's compile-notice checks (tools/lab.py), read from notice text in the toolkit's form, kept beside a build's
binary: `STEPS_SERIALIZED` (any notice that the toolkit serialized a kernel's asynchronous MMAs, counted by what it
says), `WAIT_INJECTED` (a `warpgroup.wait` the compiler wrote), a `setmaxnreg` the toolkit ignored (declarable by no
one), and nvcc's #549-D (a variable used before its value is set)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab  # noqa: E402

C7512 = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "insufficient register resources for the wgmma pipeline in the function '{}'")
C7520 = ("ptxas info    : (C7520) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "program dependence on compiler-inserted WG.AR in divergent path in the function '{}'")
#: a code no build has emitted yet, in the toolkit's template for extern calls
C7599 = ("ptxas info    : (C7599) Potential Performance Loss: wgmma.mma_async instructions are serialized due to the "
         "presence of Extern calls in the function '{}'")
C7517 = ("ptxas info    : (C7517) warpgroup.wait is injected in around line 3928 by compiler to allow use of registers "
         "defined by GMMA in function '{}'")
C7519 = ("ptxas info    : (C7519) warpgroup.arrive is injected in around line 9191 by compiler to allow use of "
         "registers in GMMA in function '{}'")
SETMAXNREG = "ptxas info    : (C7505) Potential Performance Loss: 'setmaxnreg' ignored to allow debugging."
UNSET = ('/work/measure/lab/experiments/x.cu(1706): warning #549-D: variable "far" is used before its value is set\n'
         "          detected during:\n"
         '            instantiation of "void x::f<ROLES>()" [with ROLES=1LL]')
NOISE = "ptxas info    : Used 90 registers, used 1 barriers, 4096 bytes smem in the function '{}'"
AR = "program dependence on compiler-inserted WG.AR in divergent path"


def binary_with(tmp_path, *lines, declared=()):
    """A binary whose `--expect` prints the declarations, with the notices of its build beside it."""
    binary = tmp_path / "x.sm_90a"
    binary.write_text("#!/bin/sh\n" + "".join(f"echo '{d}'\n" for d in declared))
    binary.chmod(0o755)
    lab.notices(binary).write_text("\n".join(lines) + "\n")
    return binary


def test_each_serializing_notice_is_counted_a_kernel_by_its_code_and_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n.upper() for n in names})
    binary = binary_with(tmp_path, C7512.format("a"), C7520.format("a"), C7520.format("a"), C7520.format("b"),
                         NOISE.format("a"))
    assert lab.serialized(binary) == {
        "A": Counter({("C7512", "insufficient register resources for the wgmma pipeline"): 1, ("C7520", AR): 2}),
        "B": Counter({("C7520", AR): 1})}


def test_a_serialization_notice_of_a_code_nobody_has_seen_is_counted_by_its_text(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    assert lab.serialized(binary_with(tmp_path, C7599.format("k"))) == {
        "k": Counter({("C7599", "the presence of Extern calls"): 1})}


def test_other_notices_are_not_counted_as_serialized(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    binary = binary_with(tmp_path, NOISE.format("a"), C7517.format("a"), C7519.format("a"),
                         "ptxas info    : (C7510) something else in the function 'a'")
    assert lab.serialized(binary) == {}


def test_a_build_with_no_notices_file_has_none(tmp_path):
    binary = tmp_path / "carried.sm_90a"
    assert lab.serialized(binary) == {}
    assert lab.injected_waits(binary) == {}
    assert lab.setmaxnreg_ignored(binary) == []
    assert lab.unset_reads(binary) == {}


def test_an_injected_wait_is_counted_a_kernel_and_an_injected_arrive_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n.upper() for n in names})
    binary = binary_with(tmp_path, C7517.format("a"), C7517.format("a"), C7517.format("b"), C7519.format("a"))
    assert lab.injected_waits(binary) == {"A": Counter({"C7517": 2}), "B": Counter({"C7517": 1})}


def test_an_ignored_setmaxnreg_is_read_by_its_text_and_names_no_kernel(tmp_path):
    other = ("ptxas info    : (C7506) Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum register "
             "requirements.")
    binary = binary_with(tmp_path, SETMAXNREG, NOISE.format("a"), other)
    assert lab.setmaxnreg_ignored(binary) == ["C7505: to allow debugging",
                                              "C7506: to maintain minimum register requirements"]
    assert lab.setmaxnreg_ignored(binary_with(tmp_path, NOISE.format("a"))) == []


def test_a_variable_used_before_its_value_is_set_is_read_by_file_line_and_name(tmp_path):
    assert lab.unset_reads(binary_with(tmp_path, UNSET, UNSET, NOISE.format("a"))) == {("x.cu", 1706, "far"): 2}


def checked(tmp_path, monkeypatch, *lines, declared=()):
    """`lab.check`'s refusals of an experiment `x` whose build left `lines`, no other check firing."""
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    monkeypatch.setattr(lab, "step_bounds", lambda binary, expected: [])
    for name in ("loop_convergence", "clocks_summed_uniform", "loop_hazards", "waits_after_issue", "rolled_steps", "signatures"):
        monkeypatch.setattr(lab, name, lambda binary: {})
    monkeypatch.setattr(lab, "timeline", lambda path: True)
    monkeypatch.setattr(lab, "collective_selected", lambda path: [])
    monkeypatch.setattr(lab, "experiments", lambda: {"x": tmp_path / "x.cu"})
    return lab.check(binary_with(tmp_path, *lines, declared=declared), {})


def test_check_refuses_a_serialization_notice_of_any_code_quoting_its_reason(tmp_path, monkeypatch):
    failures = checked(tmp_path, monkeypatch, C7599.format("k"), C7520.format("k"))
    assert len(failures) == 1 and "k: the toolkit serialized" in failures[0]
    assert "(C7599) the presence of Extern calls" in failures[0] and f"(C7520) {AR}" in failures[0]
    assert "a loop's trip count" in failures[0]


def test_check_lets_a_declared_count_of_serialization_through(tmp_path, monkeypatch):
    declared = ['{"lab": "expect", "kernel": "k", "op": "STEPS_SERIALIZED", "cmp": "<=", "count": 2}']
    assert checked(tmp_path, monkeypatch, C7599.format("k"), C7520.format("k"), declared=declared) == []
    assert checked(tmp_path, monkeypatch, C7599.format("k"), C7520.format("k"), C7520.format("k"),
                   declared=declared) != []


def test_check_refuses_an_injected_wait_unless_declared_and_never_an_injected_arrive(tmp_path, monkeypatch):
    failures = checked(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"))
    assert len(failures) == 1 and "injected a `warpgroup.wait`" in failures[0] and "1 x C7517" in failures[0]
    declared = ['{"lab": "expect", "kernel": "k", "op": "WAIT_INJECTED", "cmp": "<=", "count": 1}']
    assert checked(tmp_path, monkeypatch, C7517.format("k"), C7519.format("k"), declared=declared) == []


def test_check_refuses_an_ignored_setmaxnreg_whatever_is_declared(tmp_path, monkeypatch):
    declared = ['{"lab": "expect", "kernel": ".*", "op": "SETMAXNREG_IGNORED", "cmp": "<=", "count": 9}']
    failures = checked(tmp_path, monkeypatch, SETMAXNREG, declared=declared)
    assert any("x: the toolkit ignored a `setmaxnreg` (C7505: to allow debugging)" in f for f in failures)


def test_check_refuses_a_variable_used_before_its_value_is_set_naming_it_and_its_line(tmp_path, monkeypatch):
    failures = checked(tmp_path, monkeypatch, UNSET)
    assert failures == ['x.cu(1706): variable "far" is used before its value is set (1 instantiations, nvcc #549-D): '
                        "a read of an unset value"]


def test_check_names_a_register_notice_of_c7511_as_the_known_code_it_is(tmp_path, monkeypatch):
    c7511 = ("ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
             "insufficient register resources for the function in the function '{}'")
    failures = checked(tmp_path, monkeypatch, c7511.format("k"))
    assert len(failures) == 1 and "each role's whole body inside its own branch" in failures[0]
    assert "seen here first" not in failures[0]
