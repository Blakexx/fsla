"""The SASS gate on compiled probes, through the command the library's gate actions run (warpline's `check`: its manifest and its
comparison) with the declarations fsla writes (`bazel/gate_declarations.py`): a clean kernel, one whose
runtime-indexed local array must go to local memory, one whose division calls its slow path (a call that is not one), and two
whose MMA loops convert on the conversion unit or read global memory (§29's step lints) beside one that converts on the FMA pipe
and stages by `cp.async` -- the gate's refusals are what it reads, proven on a cubin that has them. The step lints are the
checker's rules (`mma.sync` read as steps of kind `sync`), and their cases on the lines are the checker's tests. What is fsla's
is the declarations: the ones beside the kernels (the checker's form, `kernel` regex, `rule`, `allow`, `reason`), each arm and each
source kernel as a presence, the notes, strict; the probes' are stated here.

See KERNEL_STANDARDS §20, §29; docs/internals/tools/sass_gate.md.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bazel"))

import checked  # noqa: E402
import dev_config  # noqa: E402
import gate_declarations  # noqa: E402
import kernels  # noqa: E402

#: the probe, compiled by the graph's CUDA rule (tests/BUILD.bazel's `sass_gate_probe`): its one cubin
PROBE = ROOT / "tests" / "_objs" / "sass_gate_probe" / "sass_gate_probe.pic.o"
RULES = json.loads((ROOT / "docs" / "rules.json").read_text())


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    found = checked.checker().runner.cubins(PROBE, tmp_path_factory.mktemp("cubins"), dev_config.cuda_home())
    assert len(found) == 1, found
    return found[0]


def declare(kernel, rule, allow, reason="the cost the probe measures"):
    return {"kernel": kernel, "rule": rule, "allow": allow, "reason": reason}


def written(cubin, *stated, arms=(), source=()):
    """The probe architecture's declarations file as the library rule writes it: the stated entries, the arms (`(family, arm,
    pattern)`) as an arm family's file, the source kernels (none: a library of probes), fsla's notes, strict."""
    #: the image's architecture, as `cuobjdump -xelf` names the cubin it writes (`<object>.<n>.sm_90a.cubin`)
    arch = re.search(r"sm_\d+a?", cubin.name).group()
    files = [{"declarations": [], "arms": [{"arch": arch, "family": f, "arm": i, "kernel": k} for f, i, k in arms]}]
    return gate_declarations.declarations({"architectures": {arch: list(stated)}}, files, RULES, not source, list(source))[arch]


def check(cubin, tmp_path, doc, notices=""):
    """`warpline check` on the cubin, as the gate's actions run it (the graph's toolkit, the build's notices):
    its exit, its printed text and the manifest it wrote."""
    checked.checker()
    from warpline.checker import cli  # noqa: PLC0415 -- `checked.checker()` puts the pinned checker on the path

    decl, manifest, log = tmp_path / "d.json", tmp_path / "m.json", tmp_path / "ptxas.txt"
    decl.write_text(json.dumps(doc))
    log.write_text(notices)
    argv = ["check", str(cubin), "--toolkit", dev_config.cuda_home(), "--declare", str(decl),
            "--notices", str(log), "--manifest", str(manifest)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue(), json.loads(manifest.read_text())


def counts(manifest, name):
    """The rule counts at the kernel whose demangled name starts with `name(`."""
    place, = (p for p in manifest["places"] if p["place"].startswith(name + "("))
    return {rule: r["count"] for rule, r in place["rules"].items()}


def refused(text):
    """The (kernel, rule) of every refusal head line the comparison printed."""
    return {(m.group(1).split("(")[0], m.group(2)) for m in re.finditer(r"^(\S[^\n]*?): ([a-z_]+): \d+ found, \d+ declared", text, re.M)}


def test_the_gate_passes_the_clean_kernel_and_refuses_local_memory(cubin, tmp_path):
    code, text, manifest = check(cubin, tmp_path, written(cubin))
    assert code == 1
    assert not any(k == "clean_probe" for k, _ in refused(text)), text
    assert counts(manifest, "local_probe")["stack_memory"] > 0
    assert {r for k, r in refused(text) if k == "local_probe"} == {"stack_memory"}
    assert "local_probe(float *, const int *): stack_memory: 2048 found, 0 declared (undeclared)" in text and "fix:" in text
    #: fsla's note on the rule, the `checker` section of docs/rules.json, printed after the checker's fix
    assert f"  note: {RULES['checker']['stack_memory']['guidance']}" in text


def test_the_step_lints_refuse_the_conversion_unit_and_global_memory_in_an_mma_loop(cubin, tmp_path):
    """§29: F2I inside an MMA loop and a global load inside one are refused; the FMA-pipe conversion over shared memory is clean of
    both."""
    code, text, manifest = check(cubin, tmp_path, written(cubin))
    found = refused(text)
    assert counts(manifest, "step_xu_probe")["step_conversion"] > 0 and {r for k, r in found if k == "step_xu_probe"} == {"step_conversion"}
    #: the probe's global load is read by the trip that issued it, which `load_in_trip` (applied with every rule) refuses too
    assert counts(manifest, "step_global_probe")["step_global"] > 0
    assert {r for k, r in found if k == "step_global_probe"} == {"step_global", "load_in_trip", "load_behind_load"}
    clean = counts(manifest, "step_clean_probe")
    assert clean["step_conversion"] == 0 and clean["step_global"] == 0 and not any(k == "step_clean_probe" for k, _ in found)


def test_a_slow_path_call_is_not_refused(cubin, tmp_path):
    """The division's slow path is a `CALL` the listing holds and the checker does not count: it is no convergence subroutine."""
    _, text, manifest = check(cubin, tmp_path, written(cubin))
    assert "slowpath" in checked.inline_listing(cubin), "the probe's division has no slow-path call: it no longer has the shape it tests"
    assert counts(manifest, "divide_probe")["convergence_call"] == 0 and not any(k == "divide_probe" for k, _ in refused(text))


def test_a_declaration_permits_the_step_signature_it_names(cubin, tmp_path):
    """A probe keeps a step signature on purpose where its declaration names it (docs/internals/tools/sass_gate.md#probes): the
    conversion probe is clean, the global-memory probe, undeclared, stays refused, and the permit, shown, is no strict refusal."""
    _, text, _ = check(cubin, tmp_path, written(cubin, declare("step_xu_probe", "step_conversion", checked.PERMIT)))
    found = refused(text)
    assert not any(k == "step_xu_probe" for k, _ in found) and ("step_global_probe", "step_global") in found
    assert ": strict: " not in text


def test_the_instructions_per_step_ceiling_is_the_checkers_density_bound(cubin, tmp_path):
    """A `step_density` declaration is the arm's ceiling: a kernel over it is refused, one under it is not."""
    _, text, _ = check(cubin, tmp_path, written(cubin, declare("^step_clean_probe", "step_density", 1, "the arm's stated ceiling")))
    assert "step_clean_probe(int *, const unsigned int *, int): step_density: 23 found, 1 declared (the arm's stated ceiling)" in text
    _, text, _ = check(cubin, tmp_path, written(cubin, declare("^step_clean_probe", "step_density", 10**6, "the arm's stated ceiling")))
    assert ("step_clean_probe", "step_density") not in refused(text)


def test_the_figures_are_shown_where_no_bound_is_declared(cubin, tmp_path):
    """The checker's bound rules are figures: a function's instructions (padding excluded) and the gap in a step, shown, not refused."""
    _, text, manifest = check(cubin, tmp_path, written(cubin))
    clean = counts(manifest, "step_clean_probe")
    assert clean["function_size"] > 0 and clean["step_gap"] > 0
    assert re.search(r"^  step_clean_probe\([^\n]*: step_gap: \d+ \(no bound declared\)", text, re.M), text


def test_a_declaration_the_sass_does_not_show_is_refused_and_so_is_an_arm_or_a_kernel_it_does_not_carry(cubin, tmp_path):
    """The declarations say only what is so: an allowance no function it names has (strict), an arm whose exact pattern does not
    name one function, and a source kernel no function carries (presences)."""
    doc = written(cubin, declare("^clean_probe\\(", "step_conversion", 2), declare("^absent", "step_conversion", 1),
                  declare("^clean_probe\\(", "stack_memory", 0, "tightened"), declare("^clean_probe\\(", "step_density", 9, "a bound"),
                  arms=[("f", 0, "^clean_probe\\("), ("f", 1, "^no_probe\\("), ("f", 2, "_probe\\(")],
                  source=[kernels.Kernel("clean_probe", "probe", kernels.SRC / "x.cu", 1, False),
                          kernels.Kernel("ghost_kernel", "probe", kernels.SRC / "x.cu", 2, False)])
    _, text, _ = check(cubin, tmp_path, doc)
    strict = [line for line in text.splitlines() if ": strict: " in line]
    present = [line for line in text.splitlines() if ": present: " in line]
    assert len(strict) == 2, strict
    assert strict[0].startswith("^clean_probe\\(: strict: declares step_conversion (allow 2), which no place it names has")
    assert strict[1].endswith("; it names no place") and strict[1].startswith("^absent")
    assert len(present) == 3, present
    assert present[0].startswith("^no_probe\\(: present: names 0 kernel(s), exactly 1 declared (arm f[1]")
    assert re.match(r"_probe\\\(: present: names [2-9] kernel\(s\), exactly 1 declared \(arm f\[2\]", present[1]), present[1]
    assert "ghost_kernel" in present[2] and "names 0 kernel(s), at least 1 declared" in present[2]


def test_the_declarations_file_holds_the_stated_then_the_arm_families_then_the_presences():
    """The writer's order is the judgment's: the first declaration of a rule that names a kernel applies, so the stated come first;
    an arm family's entries of another architecture are not this one's; the notes are the rules table's guidance and strict is on."""
    stated = {"architectures": {"sm_90a": [declare("^k\\(", "stack_memory", 8, "stated")]}}
    files = [{"declarations": [{"arch": "sm_90a", **declare("^k\\(", "stack_memory", 16, "the family's")},
                               {"arch": "sm_86", **declare("^k\\(", "stack_memory", 32, "another card's")}],
              "arms": [{"arch": "sm_90a", "family": "f", "arm": 3, "kernel": "^void k<3>\\("},
                       {"arch": "sm_86", "family": "f", "arm": 3, "kernel": "^void k<3>\\(.86"}]}]
    source = [kernels.Kernel("k", "f", kernels.SRC / "f/k.cu", 7, True)]
    doc = gate_declarations.declarations(stated, files, RULES, False, source)["sm_90a"]
    assert [d.get("reason")[:6] for d in doc["declarations"][:2]] == ["stated", "the fa"]
    assert doc["declarations"][2] == {"kernel": "^void k<3>\\(", "check": "present", "exactly": 1,
                                      "reason": "arm f[3], which the build carries: its exact kernel pattern names one function"}
    assert doc["declarations"][3]["check"] == "present" and doc["declarations"][3]["at_least"] == 1 and len(doc["declarations"]) == 4
    assert doc["strict"] is True and doc["notes"] == {rule: e["guidance"] for rule, e in RULES["checker"].items()}
    assert len(gate_declarations.declarations(stated, files, RULES, True, [])["sm_90a"]["declarations"]) == 3   # probes owe no kernel


def test_a_source_kernels_pattern_names_the_function_the_demangler_names():
    """`source_pattern` names the functions `tools/kernels.py`'s `function_name` gives the kernel's name: namespaces, the return type
    and template arguments around it, never a kernel whose name only holds it."""
    pattern = gate_declarations.source_pattern("step_kernel")
    for name in ("void fsla::decode::detail::step_kernel<denseref::arch::BuildTag<(int)900>, (int)32>(fsla::decode::DecodeParams)",
                 "fsla::step_kernel(long *)", "step_kernel"):
        assert re.search(pattern, name) and kernels.function_name(name) == "step_kernel", name
    for name in ("void fsla::decode_step_kernel<1>(int)", "void fsla::other<fsla::step_kernel<1>>(int)", "void fsla::step_kernel_two(int)"):
        assert not re.search(pattern, name) and kernels.function_name(name) != "step_kernel", name


def test_the_notices_of_a_functions_build_are_read_against_the_ledger_and_the_declarations(cubin, tmp_path):
    """The units' ptxas reports are the gate's notices: a spill the toolkit warns of in a function is the rule `register_spill` at it,
    refused as the lab refuses it and allowed by a declaration; the toolkit's per-function summaries are accepted by default."""
    summary = ("ptxas info    : Compiling entry function '_Z11clean_probePfPKf' for 'sm_90a'\n"
               "ptxas info    : Function properties for _Z11clean_probePfPKf\n"
               "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
               "ptxas info    : Used 8 registers, used 0 barriers\nptxas info    : Compile time = 1.0 ms\n")
    spill = "ptxas warning : Registers are spilled to local memory in function '_Z11clean_probePfPKf', 8 bytes spill stores, 8 bytes spill loads\n"
    _, text, _ = check(cubin, tmp_path, written(cubin), summary)
    assert not any(k == "clean_probe" for k, _ in refused(text)) and "4 notice(s)" in text
    _, text, _ = check(cubin, tmp_path, written(cubin), summary + spill)
    assert {r for k, r in refused(text) if k == "clean_probe"} == {"register_spill"}
    _, text, _ = check(cubin, tmp_path, written(cubin, declare("^clean_probe\\(", "register_spill", 1, "a probe that spills on purpose")),
                       summary + spill)
    assert not any(k == "clean_probe" for k, _ in refused(text))
    _, text, _ = check(cubin, tmp_path, written(cubin, declare("^clean_probe\\(", "register_spill", 0, "tightened")), summary + spill)
    assert {r for k, r in refused(text) if k == "clean_probe"} == {"register_spill"}
