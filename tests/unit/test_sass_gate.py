"""The SASS gate (`tools/sass_gate.py`) on compiled probes: a clean kernel, one whose runtime-indexed local array must
go to local memory, one whose division calls its slow path (a call that is not one), and two whose MMA loops convert on the conversion unit
or read global memory (§29's step lints) beside one that converts on the FMA pipe and stages by `cp.async` -- the
gate's reds are what it reads, proven on a cubin that has them. The step lints are the checker's rules (`mma.sync` read as
steps of kind `sync`), and their cases on the lines -- the loops that hold steps, the branches in a step, the exit laid out as
its latch, the convergence call's hot and guarded forms -- are the checker's tests. The gate's declarations are the checker's form
(`kernel` regex, `rule`, `allow`, `reason`), stated beside the kernels in the graph; the probes' are stated here.

See KERNEL_STANDARDS §20, §29.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import checked  # noqa: E402
import sass  # noqa: E402
import sass_gate  # noqa: E402

#: the probe, compiled by the graph's CUDA rule (tests/BUILD.bazel's `sass_gate_probe`): its one cubin
PROBE = ROOT / "tests" / "_objs" / "sass_gate_probe" / "sass_gate_probe.pic.o"


@pytest.fixture(scope="module")
def cubin():
    found = sass.cubins(PROBE)
    assert len(found) == 1, found
    return found[0]


def declare(kernel, rule, allow, reason="the cost the probe measures"):
    return {"kernel": kernel, "rule": rule, "allow": allow, "reason": reason}


def red_rules(stats):
    """The rules a function is red on (a red is `<rule>: <count> found, <allowed> declared (<reason>)`)."""
    return {red.split(":")[0] for red in stats["reds"]}


def test_the_gate_passes_the_clean_kernel_and_reds_local_memory(cubin):
    ok, report = sass_gate.gate(cubin, None, [])
    #: the bare probe by its mangled name: `clean_probe` alone also matches the step probe `step_clean_probe`
    clean = sass_gate.STATS["_Z11clean_probePfPKf"]
    local = next(v for k, v in sass_gate.STATS.items() if "local_probe" in k)
    assert not clean["reds"], report
    assert local["stack"] > 0 and local["ldl"] > 0, report
    assert red_rules(local) == {"stack_memory"} and not ok
    assert "stack_memory: 2048 found, 0 declared (undeclared)" in local["reds"] and "fix:" in report


def test_the_step_lints_red_the_conversion_unit_and_global_memory_in_an_mma_loop(cubin):
    """§29: F2I inside an MMA loop and a global load inside one are red; the FMA-pipe conversion over shared memory is
    clean of both. The probes' step cover is §32's reading, not this test's subject: one dependent MMA a step leaves
    the clean probe under the line."""
    ok, report = sass_gate.gate(cubin, None, [])
    xu = next(v for k, v in sass_gate.STATS.items() if "step_xu_probe" in k)
    memory = next(v for k, v in sass_gate.STATS.items() if "step_global_probe" in k)
    clean = next(v for k, v in sass_gate.STATS.items() if "step_clean_probe" in k)
    assert xu["step_xu"] > 0 and red_rules(xu) == {"step_conversion"}, report
    #: the probe's global load is read by the trip that issued it, which `load_in_trip` (applied with every rule) refuses too
    assert memory["step_global"] > 0 and red_rules(memory) == {"step_global", "load_in_trip", "load_behind_load"}, report
    assert clean["step_xu"] == 0 and clean["step_global"] == 0 and not clean["reds"], report
    assert not ok


def test_a_narrowing_that_matches_nothing_is_red(cubin, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sass_gate.py", str(cubin), "--kernel", "no_such_kernel"])
    assert sass_gate.main() == 1
    assert "no function matched" in capsys.readouterr().out


def test_a_slow_path_call_is_not_red(cubin):
    """The division's slow path is a `CALL` the listing holds and the gate does not count: it is no convergence subroutine. (The
    guarded convergence fallback, `BRA.DIV` into one, is an Ampere shape: a Hopper shuffle is inline after a `WARPSYNC`; its
    hot and guarded forms are the checker's tests, on lines.)"""
    ok, report = sass_gate.gate(cubin, None, [])
    divide = next(v for k, v in sass_gate.STATS.items() if "divide_probe" in k)
    assert "slowpath" in sass.disassemble(cubin), "the probe's division has no slow-path call: it no longer has the shape it tests"
    assert divide["calls"] == 0 and divide["guarded"] == 0 and not divide["reds"], report


def test_a_declaration_permits_the_step_signature_it_names(cubin):
    """A probe keeps a step signature on purpose where its declaration names it (docs/internals/tools/sass_gate.md#probes): the checker
    matches the declaration's regex to the function's demangled name, so the conversion probe is clean and the global-memory probe,
    undeclared, stays red."""
    ok, report = sass_gate.gate(cubin, None, [declare("step_xu_probe", "step_conversion", checked.PERMIT)])
    xu = next(v for k, v in sass_gate.STATS.items() if "step_xu_probe" in k)
    memory = next(v for k, v in sass_gate.STATS.items() if "step_global_probe" in k)
    assert not xu["reds"] and xu["step_xu"] > 0, report
    assert memory["reds"] and not ok
    shown = [(name, counts) for name, counts in sass_gate.PLACES]
    assert sass_gate.unshown({"declarations": [declare("step_xu_probe", "step_conversion", checked.PERMIT)]}, shown, "sm_90a", "d.json") == []


def test_the_instructions_per_step_ceiling_is_the_checkers_density_bound(cubin):
    """A `step_density` declaration is the arm's ceiling: a kernel over it is red, one under it is not."""
    ok, report = sass_gate.gate(cubin, "step_clean_probe", [declare("", "step_density", 1, "the arm's stated ceiling")])
    assert not ok and "step_density: 23 found, 1 declared (the arm's stated ceiling)" in report
    ok, report = sass_gate.gate(cubin, "step_clean_probe", [declare("", "step_density", 10**6, "the arm's stated ceiling")])
    assert ok, report


def test_the_figures_are_shown_where_no_bound_is_declared(cubin):
    """The gate's own counts are the checker's rules: a function's instructions (padding excluded) and the gap in a step."""
    ok, report = sass_gate.gate(cubin, "step_clean_probe", [])
    step = next(iter(sass_gate.STATS.values()))
    assert ok and step["instr"] > 0 and step["mma_gap"] > 0 and "mma_gap_around" not in step, report


def test_a_declaration_the_sass_does_not_show_is_red_and_so_is_an_arm_it_does_not_carry():
    """The declarations say only what is so: an allowance where no function it names has the count (or names none), and an arm whose exact
    kernel pattern does not match one function of the architecture's SASS."""
    places = [("void k<(int)1>(int)", {"step_conversion": 2}), ("void k<(int)2>(int)", {"step_conversion": 0})]
    spec = {"declarations": [declare("k<\\(int\\)1>", "step_conversion", 2), declare("k<\\(int\\)2>", "step_conversion", 2),
                             declare("absent", "step_conversion", 1), declare("k", "step_conversion", 0, "tightened"),
                             declare("k", "step_density", 9, "a bound is a ceiling, not a kept signature")],
            "arms": [{"family": "f", "arm": 0, "kernel": "^void k<\\(int\\)1>"}, {"family": "f", "arm": 1, "kernel": "^void k<\\(int\\)3>"},
                     {"family": "f", "arm": 2, "kernel": "^void k"}]}
    reds = sass_gate.unshown(spec, places, "sm_90a", "d.json")
    assert len(reds) == 4, reds
    assert "`k<\\(int\\)2>` declares step_conversion (allow 2), which no function of sm_90a's SASS shows" in reds[0]
    assert "`absent`" in reds[1] and "names no function there" in reds[1]
    assert "arm f[1]" in reds[2] and "matches 0 functions" in reds[2] and "arm f[2]" in reds[3] and "matches 2 functions" in reds[3]


def test_the_notices_of_a_functions_build_are_read_against_the_ledger_and_the_declarations(cubin):
    """The units' ptxas reports are the gate's notices: a spill the toolkit warns of in a function is the rule `register_spill` at it,
    refused as the lab refuses it and allowed by a declaration; the toolkit's per-function summaries are accepted by default."""
    c = checked.checker()
    summary = ("ptxas info    : Compiling entry function '_Z11clean_probePfPKf' for 'sm_90a'\n"
               "ptxas info    : Function properties for _Z11clean_probePfPKf\n"
               "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
               "ptxas info    : Used 8 registers, used 0 barriers\nptxas info    : Compile time = 1.0 ms\n")
    spill = "ptxas warning : Registers are spilled to local memory in function '_Z11clean_probePfPKf', 8 bytes spill stores, 8 bytes spill loads\n"
    quiet = c.ledgers.read_notices(summary)
    assert len(quiet) == 4
    ok, report = sass_gate.gate(cubin, "^_Z11clean_probe", [], quiet)
    assert ok, report
    clean = sass_gate.STATS["_Z11clean_probePfPKf"]
    assert not clean["reds"]
    spilled = c.ledgers.read_notices(summary + spill)
    sass_gate.gate(cubin, "^_Z11clean_probe", [], spilled)
    assert red_rules(sass_gate.STATS["_Z11clean_probePfPKf"]) == {"register_spill"}
    sass_gate.gate(cubin, "^_Z11clean_probe", [declare("^clean_probe\\(", "register_spill", 1, "a probe that spills on purpose")], spilled)
    assert not sass_gate.STATS["_Z11clean_probePfPKf"]["reds"]
    sass_gate.gate(cubin, "^_Z11clean_probe", [declare("^clean_probe\\(", "register_spill", 0, "tightened")], spilled)
    assert red_rules(sass_gate.STATS["_Z11clean_probePfPKf"]) == {"register_spill"}
