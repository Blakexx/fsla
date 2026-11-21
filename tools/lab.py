# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB: small CUDA experiments whose measurements become kernel rules (docs/internals/tools/lab.md).

An experiment is one `measure/lab/experiments/<name>.cu` built on `measure/lab/lab.cuh`. This driver compiles it with
the pinned toolkit for the running card, reads its SASS and refuses to measure unless every expectation the experiment
declares holds (an instruction the experiment means to time is really there -- ptxas deletes work whose result is dead)
and every notice the toolkit printed is allowed by the ledger (a kernel that touches local memory is the toolkit's
own warning); then it checks the GPU is quiet, runs the repetitions, flags a result whose samples
spread, and writes a record keyed by the architecture.

    bazel run //tools:lab -- list
    bazel run //tools:lab -- sass tensor_tax [KERNEL_REGEX] [--full]
    bazel run //tools:lab -- run tensor_tax [--reps 7] [--record] [--force]
    bazel run //tools:lab -- ncu smem_rates --metric lts__t_sectors_srcunit_tex_op_read.sum [--record]
    bazel run //tools:lab -- page carry_window --instance LABEL

`--record` writes `measure/lab/results/<name>/<sm_XX>.json` (or `<sm_XX>.counters.json` for `ncu`), the file a
KERNEL_STANDARDS rule cites; without it the record goes to the scratch space. `ncu` reads hardware counters per kernel
(the median over its launches) where a question is about traffic rather than time. An experiment that reads data (a
cell's routing, say) has a `<name>.inputs.py` beside it whose `write(directory)` writes the files; the driver runs it
first, passes `--inputs <directory>`, and records each file's digest.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import functools
import hashlib
import importlib.util
import json
import pipes
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import build_flags
import checked
import dev_config
import region_ledger
import sass

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "measure" / "lab"
EXPERIMENTS = LAB / "experiments"
#: the kernels' include root: a part harness runs a component's own code (docs/internals/tools/lab.md#parts)
KERNELS = ROOT / "csrc" / "rola" / "src"
RESULTS = LAB / "results"

#: a result whose largest sample exceeds its smallest by more than this is flagged unstable
SPREAD = 1.05
#: GPU utilization above this percent before a run means another consumer shares the card
QUIET_PERCENT = 5


def experiments() -> dict[str, Path]:
    return {p.stem: p for p in sorted(EXPERIMENTS.glob("*.cu"))}


def question(path: Path) -> str:
    """The experiment's QUESTION, from its header comment."""
    text = " ".join(line.lstrip("/ ").strip() for line in path.read_text().splitlines()[:6])
    found = re.search(r"QUESTION:\s*(.*?)\s*METHOD:", text)
    return found.group(1) if found else "(no QUESTION in the header)"


def smi(*query: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [field.strip() for field in out.splitlines()[0].split(",")]


def build(path: Path, arch: str) -> Path:
    """The experiment's binary for `arch`, compiled with the pinned toolkit."""
    out = dev_config.scratch("lab") / f"{path.stem}.{arch}"
    #: the lab's build, its one architecture's own in both passes (`lab.cuh`'s `kLabBuild`)
    sys.path.insert(0, str(ROOT))
    from measure.hardware import cc_of

    cc = cc_of(arch)
    #: the extension's own target form: an architecture-specific row (`sm_90a`) names its own virtual architecture
    target, = build_flags.gencodes([arch.removeprefix("sm_")])
    done = subprocess.run([dev_config.cuda_bin("nvcc"), "-O3", "-std=c++20", target, f"-DROLA_LAB_CC={cc}",
                           "-lineinfo", "-Xptxas", ",".join(PTXAS_OPTIONS), f"-I{LAB}", f"-I{KERNELS}", f"-I{dev_config.warpline_include()}", "--expt-relaxed-constexpr", "-o", str(out),
                           str(path)], capture_output=True, text=True, check=False)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    done.check_returncode()
    #: the toolkit's notices of this build, beside its binary: `serialized` reads them
    notices(out).write_text(done.stderr)
    return out


def notices(binary: Path) -> Path:
    return binary.with_name(binary.name + ".ptxas")


#: THE CHECKER, warpline's (`warpline.checker`, docs/internals/tools/lab.md#hazards): it computes every hazard below from the code the
#: toolkit made -- the lab computes none of them. `bazel run //tools:lab` hands in the module's directory with the stamp's headers
#: (`dev_config.warpline_python()`; the pin is `MODULE.bazel`'s). A hazard is the name an experiment declares a count of
#: (`Bench::expect(kernel, NAME, "<=", n)`); `HAZARD_RULES` says which rule computes it. What stays here is the project's own: which
#: experiments are held to which hazard (the `_PRE_*` debts), the refusals' texts and the declarations' translation.
HAZARD_RULES = {
    "LOOP_CONVERGENCE": "loop_convergence", "LOAD_IN_TRIP": "load_in_trip", "STORE_RELOAD": "store_reload",
    "STORE_UNFENCED": "store_unfenced", "CLOCK_UNIFORM": "clock_uniform", "WAIT_AFTER_ISSUE": "wait_after_issue",
    "STEP_ROLLED": "step_rolled", "STEP_PORT": "step_port", "STEP_WORK": "step_work",
    "STEPS_SERIALIZED": "mma_serialized", "WAIT_INJECTED": "wait_injected",
    "BANK_CONFLICT": "shared_wavefronts_excess",
}
#: the hazards a PROFILE computes (`ledger` runs them on a kernel's dump, `profile_failures`): the checker's profile rules, which `check`
#: has no profile to run and only translates the declarations of. The count is the excess shared wavefronts of a CTA-window of the
#: profiled launch (`--per`), never of the launch: a launch's total moves with the grid and the windows a row times.
PROFILE_HAZARDS = ("BANK_CONFLICT",)
#: the hazards an experiment may declare a count of, beside its opcodes
HAZARDS = tuple(HAZARD_RULES)
#: the two a `declare` row of the notice ledger counts (`mma_serialized`, `wait_injected`), and what each says of a kernel past its count
HAZARD_STATES = {
    "STEPS_SERIALIZED": "the toolkit serialized its asynchronous MMAs, each step waiting out the one before",
    "WAIT_INJECTED": "the toolkit injected a `warpgroup.wait` into the step stream, to let code read registers a tensor step defines",
}
#: an expectation that is a permit, no count: the opcode an experiment means to keep (`BSSY`, `CALL...`) or the form it means
#: (`STEP_XU`, `STEP_GLOBAL`) allows the rule it names any number of findings in the kernels it matches
PRESENCE_RULES = {"STEP_XU": "step_conversion", "STEP_GLOBAL": "step_global", "BSSY": "step_branches"}
PRESENCE_ALLOWANCE = 10**9
#: the hazards that bound a figure, declared `<=`
BOUND_HAZARDS = ("STEP_PORT", "STEP_WORK")
#: the reason of a declaration whose experiment states none in a `//:` comment above it
UNSTATED = "stated in the experiment's source, where it declares the count"
_DECLARED = re.compile(r'"([A-Z][A-Z_]*)"\s*,\s*"(?:<=|>=|==)"')
#: the project's ledger of the toolkit's notices: its rows override or add to the checker's base ledger (the rows that only repeat the
#: base are not here); the hazards a `declare` row counts are the checker's rules `mma_serialized` and `wait_injected`
PROJECT_LEDGER = Path(__file__).resolve().with_name("notice_ledger.json")
#: ptxas's own options the build switches on (`ptxas --help`, passed through nvcc's `-Xptxas`): the warning where a
#: function spills registers to local memory, the warning where a function uses local memory (its stack frame's size in
#: the text) and the verbose per-function summary
PTXAS_OPTIONS = ("--warn-on-spills", "--warn-on-local-memory-usage", "--verbose")


@functools.cache
def notice_ledger() -> list:
    """The ledger the checker reads the notices against: its base ledger with the project's rows (`PROJECT_LEDGER`)."""
    ledgers = checked.checker().ledgers
    return ledgers.merge(ledgers.base_ledger(), ledgers.load_ledger(PROJECT_LEDGER.read_text()))


def load_declarations(expected: list[dict], reasons: dict[str, str]) -> list:
    """The experiment's expectations as the checker's loaded declarations: the model rules', the notice ledger's and the profile rules'."""
    c = checked.checker()
    names = {*c.ledgers.declared_rules(notice_ledger()), *c.rules.profile_registry()}
    return c.declare.load_declarations(json.dumps({"declarations": declaration_entries(expected, reasons)}), c.rules.registry(), names)


def expectations(binary: Path) -> list[dict]:
    """The expectations the experiment declares (`Bench::expect`), as its binary prints them (`--expect`, which runs nothing)."""
    lines = subprocess.run([str(binary), "--expect"], capture_output=True, text=True, check=True).stdout
    return [e for e in map(json.loads, lines.splitlines()) if e["lab"] == "expect"]


def profile_failures(csv_path: Path, kernel: str, declarations: list, lines_at: dict[int, int], source: list[str], div: float, unit: str) -> list[str]:
    """The checker's profile rules on a kernel's SourceCounters dump (`ledger`), under the experiment's declarations (a count of a
    CTA-window of the profiled launch, zero where none names the kernel; where the `unit` is no CTA-window none can be read): a
    refusal names the kernel, then each source line that issues the excess with its wavefronts a `unit` (`div` the launch's units),
    the checker's fix and rola's note. `lines_at` maps an instruction's
    offset from the kernel's first to its line in the experiment's source."""
    lines = csv_path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"Kernel Name"')), None)
    if start is None:
        return [f"{kernel}: {csv_path.name} is no SourceCounters dump: no `Kernel Name` line"]
    text = "\n".join(line for i, line in enumerate(lines[start:]) if i < 2 or line.startswith('"0x'))
    windows = div if unit == "CTA-window" else None
    refusal = checked.profile_refusal(text, kernel, declarations, guidance(), windows)
    if refusal is None:
        return []
    by_line: dict[int, int] = {}
    for f in refusal.findings:
        line = lines_at.get(int(f.where, 16), 0)
        by_line[line] = by_line.get(line, 0) + f.count
    shown = {"per": f"{refusal.count / div:,.1f}", "declared_per": f"{refusal.allowed / div:,.1f}"} if windows else \
        {"per": "(CTA-windows not stated: `--per`)", "declared_per": "none readable"}
    out = [f"{kernel}: " + REFUSAL_TEXT[refusal.rule].format(n=refusal.count, declared=refusal.allowed, **shown)
           + f" [checker rule {refusal.rule}; {csv_path.name}]"]
    for line, n in sorted(by_line.items(), key=lambda x: -x[1]):
        text = source[line - 1].strip()[:90] if 0 < line <= len(source) else "(outside the source)"
        out.append(f"    {n / div:,.1f} a {unit} ({n:,} in the launch): line {line}: {text}")
    out.append(f"    fix: {refusal.fix}")
    if refusal.note:
        out.append(f"    note: {refusal.note}")
    return out


def declared_reasons(source: str) -> dict[str, str]:
    """Per hazard an experiment's source declares: the `//:` comment block directly above its `b.expect(..., "NAME", ...)` calls, where
    exactly one distinct block stands above them (the declaration's reason); a name with none, or with two, has none."""
    lines = source.splitlines()
    found: dict[str, set[str]] = {}
    for i, line in enumerate(lines):
        if "expect(" not in line or not (m := _DECLARED.search(line)):
            continue
        block, j = [], i - 1
        while j >= 0 and lines[j].lstrip().startswith("//:"):
            block.insert(0, lines[j].lstrip()[3:].strip())
            j -= 1
        found.setdefault(m.group(1), set()).add(" ".join(block))
    return {name: next(iter(blocks)) for name, blocks in found.items() if len(blocks) == 1 and "" not in blocks}


def declaration_entries(expected: list[dict], reasons: dict[str, str]) -> list[dict]:
    """The experiment's expectations as the checker's declarations (kernel regex, rule, allowed count, reason): a hazard's count, a bound
    for `STEP_PORT` and `STEP_WORK` (declared `<=`, refused otherwise: `bound_failures`), a permit for `STEP_XU`, `STEP_GLOBAL`, `BSSY`
    and a `CALL` opcode. The declarations of a rule are ordered by their allowance, largest first: the checker takes the first
    that names a kernel, which is the largest of the declarations that do."""
    entries = []
    for e in expected:
        op = e["op"]
        if op in HAZARD_RULES:
            if op in BOUND_HAZARDS and e["cmp"] != "<=":
                continue
            rule, allow = HAZARD_RULES[op], e["count"]
        elif op in PRESENCE_RULES or "CALL" in op:
            rule, allow = PRESENCE_RULES.get(op, "convergence_call"), PRESENCE_ALLOWANCE
        else:
            continue
        entries.append({"kernel": e["kernel"], "rule": rule, "allow": allow, "reason": reasons.get(op, UNSTATED)})
    return sorted(entries, key=lambda d: -d["allow"])


def bound_failures(expected: list[dict], kernels: list) -> list[str]:
    """A declared `STEP_PORT` or `STEP_WORK` that is no `<=` bound, or that bounds no kernel with tensor steps."""
    failures = []
    for e in (e for e in expected if e["op"] in BOUND_HAZARDS):
        if e["cmp"] != "<=":
            failures.append(f"{e['kernel']!r}: {e['op']} is a bound, declared \"<=\" (not {e['cmp']!r})")
        elif not any(k.regions and re.search(e["kernel"], k.demangled) for k in kernels):
            failures.append(f"no kernel with tensor steps matches {e['kernel']!r} ({e['op']} declared)")
    return failures


#: what a refusal of each rule says, with the hazard's name the experiment declares (`{n}` the findings counted, `{declared}` the count
#: declared, `{where}` the addresses of the first findings)
REFUSAL_TEXT = {
    "loop_convergence": "{n} convergence instructions inside its loops, {declared} declared (KERNEL_STANDARDS 29.13: the source "
                        "patterns of 29.9, or LOOP_CONVERGENCE)",
    "load_in_trip": "{n} global loads read in the trip that issued them (KERNEL_STANDARDS 32: a load a stage ahead of its use), "
                    "{declared} declared (LOAD_IN_TRIP)",
    "store_reload": "{n} shared loads of an address the same trip stored to (KERNEL_STANDARDS 33: a value no other warp reads "
                    "stays in a register), {declared} declared (STORE_RELOAD)",
    "store_unfenced": "{n} barrier arrivals after a shared store with no fence between, in a kernel whose steps read shared "
                      "memory (KERNEL_STANDARDS 12: the writer's fence), {declared} declared (STORE_UNFENCED)",
    "clock_uniform": "{n} sums of a cycle-counter reading in uniform registers, {declared} declared (CLOCK_UNIFORM): a lap's sum "
                     "is each thread's own (docs/internals/LEARNINGS.md, lap-in-uniform)",
    "wait_after_issue": "{n} tensor waits directly after their group's steps, {declared} declared (WAIT_AFTER_ISSUE): work that "
                        "does not read the group's accumulators belongs between the issue and the wait: KERNEL_STANDARDS 29.16 "
                        "(`lab waits` lists the candidates)",
    "step_rolled": "{n} tensor steps in loops that hold no wait on them (loops at {where}), {declared} declared (STEP_ROLLED): "
                   "a step in a rolled loop of a run-time count was measured at 37.6 cycles against 23.2 straight, 1.62 times, a fence "
                   "a step free (docs/lab/step_rolled.md), and the toolkit says nothing: write them a straight block, or a step that may be "
                   "absent under a forward branch behind its own fence (docs/internals/LEARNINGS.md)",
    "step_port": "{n:,} bytes a step through the shared-memory port, over the {declared:,} declared (STEP_PORT; `lab steps` lists "
                 "the regions)",
    "step_work": "{n:,} instructions a step, over the {declared:,} declared (STEP_WORK; `lab steps` lists the regions)",
    "step_branches": "{n} branches inside an MMA step nobody declared (KERNEL_STANDARDS 27: a step body is one basic block; a form "
                     "that keeps them declares BSSY)",
    "step_conversion": "{n} conversion-unit instructions inside an MMA step nobody declared (KERNEL_STANDARDS 29: convert on the "
                       "FMA pipe; a form that keeps them declares STEP_XU)",
    "step_global": "{n} global accesses inside an MMA step nobody declared (KERNEL_STANDARDS 29: stage by cp.async; a form that "
                   "keeps them declares STEP_GLOBAL)",
    "shared_wavefronts_excess": "{per} shared-memory wavefronts in excess of the conflict-free count a CTA-window ({n:,} in the "
                                "launch), {declared_per} declared a CTA-window ({declared:,} in the launch) (BANK_CONFLICT, which counts "
                                "a CTA-window; KERNEL_STANDARDS 30: conflict-free by construction)",
    "convergence_call": "{n} convergence calls nobody declared (a collective under role-dependent control flow becomes a call "
                        "into a convergence subroutine)",
}
_RULE_HAZARDS = {rule: hazard for hazard, rule in HAZARD_RULES.items()}


@functools.cache
def guidance() -> dict[str, str]:
    """ROLA'S GUIDANCE, a line a rule of the checker (Blake, 2026-11-20): each entry of the `checker` section of `docs/rules.json`,
    keyed by the checker rule's name, carries `guidance`, the atom or practice, the standards anchor and the lab page with the
    evidence; a refusal prints it after the checker's own fix (`note`). The checker refuses a note for a name that is no rule of its pin."""
    entries = json.loads((ROOT / "docs" / "rules.json").read_text())["checker"]
    notes = {rule: entry["guidance"] for rule, entry in entries.items()}
    checked.checker().declare.load_notes(json.dumps({"notes": notes}), checked.checker().declare.rule_names(notice_ledger()))
    return notes


def render_refusal(r, experiment: str = "the build") -> list[str]:
    """A checker refusal as the lab's failure lines: the hazard's name the experiment declares, the kernel (the experiment's name for a
    notice that names none), the count found and declared, and the lab's own words for it, then the checker's fix and rola's note."""
    place = experiment if r.kernel == "the build" else r.kernel
    note = getattr(r, "note", None)
    tail = f"\n    note: {note}" if note else ""
    ledger_rule = r.rule in _RULE_HAZARDS and _RULE_HAZARDS[r.rule] in HAZARD_STATES
    if r.rule == "notice_without_row":
        return [f"{place}: {r.count} x {r.message}:\n      {r.findings[0].what}\n    add a row to tools/notice_ledger.json with "
                "a disposition (refuse, declare, report or ignore) and the reason for it (docs/internals/tools/lab.md#ledger)" + tail]
    if r.rule.startswith("notice "):
        return [f"{place}: {f.what} ({f.count} x, {r.rule.removeprefix('notice ').replace('no_code', 'no code')}): "
                f"{r.message}" + (f" {r.fix}" if r.fix else "") + tail for f in r.findings]
    if ledger_rule:
        hazard = _RULE_HAZARDS[r.rule]
        what = "; ".join(f"{f.count} x {f.what}" for f in sorted(r.findings, key=lambda f: f.what))
        return [f"{place}: {HAZARD_STATES[hazard]}: {what}; {r.allowed} declared ({hazard}). "
                f"{r.fix} (docs/internals/LEARNINGS.md)" + tail]
    where = ", ".join(f.where for f in r.findings[:4]) + (", ..." if len(r.findings) > 4 else "")
    text = REFUSAL_TEXT[r.rule].format(n=r.count, declared=r.allowed, where=where)
    fix = f"\n    fix: {r.fix}" if r.fix else ""
    return [f"{place}: {text} [checker rule {r.rule}: at {where}]{fix}{tail}"]


def notice_names(read: list) -> dict[str, str]:
    """The functions the notices name (as the toolkit printed them, mangled), demangled."""
    functions = sorted({n.function for n in read if n.function})
    return checked.demangle(functions) if functions else {}


def read_notices(binary: Path) -> list:
    """EVERY notice of `binary`'s build, in order (the checker's reader of the lines nvcc printed beside it); none where the build
    left no notices (a packed binary run elsewhere)."""
    return checked.checker().ledgers.read_notices(notices(binary).read_text()) if notices(binary).exists() else []


def notice_report(binary: Path) -> list[str]:
    """THE REPORT: the notices of `report` rows, one line a row with its count, printed with the pack and blocking nothing."""
    read = read_notices(binary)
    return checked.checker().declare.check_notices(read, notice_ledger(), [], notice_names(read))[1] if read else []


def held_rules(experiment: str) -> set[str]:
    """The checker's rules an experiment is held to: the steps' and the calls' in every experiment, `loop_convergence` too (in the
    budgeted kernels only, for an experiment in `_PRE_TIMELINE`), and past `_PRE_TIMELINE` the rest -- `wait_after_issue` past `_PRE_WAIT`."""
    names = {"step_port", "step_work", "step_branches", "step_conversion", "step_global", "convergence_call", "loop_convergence"}
    if experiment not in _PRE_TIMELINE:
        names |= {"load_in_trip", "store_reload", "store_unfenced", "clock_uniform", "step_rolled"}
        if experiment not in _PRE_WAIT:
            names.add("wait_after_issue")
    return names


def checker_failures(binary: Path, expected: list[dict], budgeted: set[str]) -> list[str]:
    """What the checker refuses of the build: the rules the experiment is held to (`held_rules`) over every kernel and every notice
    of the build against the ledger -- under the declarations its expectations translate to."""
    c = checked.checker()
    experiment = binary.name.split(".")[0]
    names = held_rules(experiment)
    held = experiment not in _PRE_TIMELINE
    kernels = checked.kernels_of(binary, demangled=True)
    path = experiments().get(experiment)
    reasons = declared_reasons(path.read_text()) if path else {}
    declarations = load_declarations(expected, reasons)
    refusals = checked.check(kernels, names, declarations, guidance())
    refusals = [r for r in refusals if held or r.rule != "loop_convergence" or r.kernel in budgeted]
    read = read_notices(binary)
    refusals += c.declare.check_notices(read, notice_ledger(), declarations, notice_names(read), guidance())[0]
    return bound_failures(expected, kernels) + [line for r in refusals for line in render_refusal(r, experiment)]


def kernels(binary: Path) -> dict[str, dict]:
    """Per kernel (demangled): its opcode counts, registers and local bytes."""
    ops: dict[str, Counter] = {}
    current = None
    with sass.sass_lines(binary) as lines:
        for line in lines:
            if m := sass.FUNCTION.match(line):
                current = m.group(1)
                ops[current] = Counter()
            elif current and (m := sass.INSTRUCTION.match(line)):
                ops[current][m.group(3)] += 1
    usage = subprocess.run([dev_config.cuda_bin("cuobjdump"), "-res-usage", str(binary)], capture_output=True,
                           text=True, check=True).stdout
    regs: dict[str, dict] = {}
    name = None
    for line in usage.splitlines():
        if m := re.search(r"Function (\S+):", line):
            name = m.group(1)
        elif name and (m := re.search(r"REG:(\d+).*LOCAL:(\d+)", line)):
            regs[name] = {"regs": int(m.group(1)), "local": int(m.group(2))}
    names = checked.demangle(list(ops))
    return {names[k]: {"ops": dict(v), **regs.get(k, {})} for k, v in ops.items()}


def count(ops: dict[str, int], prefixes: str) -> int:
    """The instructions whose opcode starts with one of `prefixes` (`|`-separated alternatives: ptxas may spell one
    operation two ways, an integer add as `IADD3` or as `IMAD.IADD` on the FMA pipe)."""
    return sum(n for op, n in ops.items() if any(op.startswith(p) for p in prefixes.split("|")))


#: KERNEL_STANDARDS section 29's debts: the experiments recorded before a lab had to budget its MMA kernels and check
#: its outputs; each is held to both when it is next changed, and the list only shrinks
_PRE_29 = {"collective_branch", "facts_block", "mma_k8", "row_span", "tensor_tax", "vote_stash"}


#: THE LEARNINGS' DEBTS (docs/internals/LEARNINGS.md): the experiments written before an experiment's header had to
#: state its TIMELINE and every kernel's loops were held to the checker's hazards (`HAZARD_RULES`); each is held to them when it is next
#: changed, and the list only shrinks
_PRE_TIMELINE = {"bulk_drain", "byte_expand", "cluster_sum", "collective_branch", "facts_block", "facts_two",
                 "issue_latency", "reg_trade", "row_ring", "row_span", "state_move", "tensor_tax",
                 "vote_stash", "wgmma_rate", "wgmma_regs"}
#: KERNEL_STANDARDS 29.16's DEBTS: the experiments, past the learnings' debts, whose kernels wait on a commit group
#: straight after its steps; each is held to `WAIT_AFTER_ISSUE` when it is next changed, and the list only shrinks
_PRE_WAIT = {"clock_time", "fold_stream", "intra_chain", "intra_window", "pool_fill", "readout_role", "stamp_cost",
             "state_image", "step_queue", "wgmma_width"}
#: a warp collective after the `?` of a statement's conditional: only the lanes the condition takes reach it
_SELECTED = re.compile(r"\?[^;{}]*\b__(?:shfl\w*|reduce_\w+|ballot|any|all)_sync\s*\(")


def collective_selected(path: Path) -> list[int]:
    """The lines of an experiment's source where a warp collective is an operand of a conditional: a full-mask
    collective that only some lanes reach hangs its warp (docs/internals/LEARNINGS.md, shuffle-under-lane)."""
    text = re.sub(r"//[^\n]*", "", path.read_text())
    return [text.count("\n", 0, m.start()) + 1 for m in _SELECTED.finditer(text)]


def timeline(path: Path) -> bool:
    """Whether the experiment's header comment states its TIMELINE: each phase's stream, and which load is a stage
    ahead of which use (KERNEL_STANDARDS section 32, written before the kernel)."""
    head = []
    for line in path.read_text().splitlines():
        if not line.startswith("//"):
            break
        head.append(line)
    return any("TIMELINE:" in line for line in head)


def disassembled(binary: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Per kernel (mangled): its instructions as (address, opcode, line)."""
    code: dict[str, list[tuple[int, str, str]]] = {}
    current = None
    with sass.sass_lines(binary) as lines:
        for line in lines:
            if m := sass.FUNCTION.match(line):
                current = m.group(1)
                code[current] = []
            elif current and (m := sass.INSTRUCTION.match(line)):
                code[current].append((int(m.group(1), 16), m.group(3), line))
    return code


#: the warpgroup steps among the opcode table's MMAs (`tools/sass.py`): the asynchronous ones, float and integer
WGMMA = tuple(op for op in sass.OPCODE_CLASSES["mma"] if op.endswith("GMMA"))

#: an instruction's operands
OPERANDS = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?[A-Z0-9_.$]+\s+(.*?)\s*;")
#: the opcodes whose first operand is read, not written
READS_FIRST = ("ST", "BRA", "RED", "ATOM", "BSSY", "BSYNC", *WGMMA, "ISETP", "FSETP", "CALL", "EXIT", "NOP", "BAR")


#: WORK OF SUBSTANCE (WAIT_AFTER_ISSUE, KERNEL_STANDARDS 29.16): what a warp can usefully do while its tensor steps
#: run -- memory loads and stores (generic, global, shared, local; the matrix forms; atomics and reductions), other
#: asynchronous issues (`LDGSTS`, the bulk copies `UBLKCP`/`UBLKRED`, the tensor-memory-access `UTMA*`) and barrier
#: arrivals (`SYNCS.ARRIVE`, `BAR.ARV`). NOT work: the commit (it is no instruction -- the `gsb0` flag on a group's
#: last `HGMMA`), `WARPGROUP.ARRIVE`, `LDC`, register moves, address and predicate arithmetic, or the wait's setup.
SUBSTANCE = re.compile(r"^(?:LDG(?!DEPBAR)|LDS|LDL|LD\b|STG|STS|STL|ST\b|ATOM|RED|UBLK|UTMA|SYNCS\.ARRIVE|BAR\.ARV)")
#: the loads and the stores among them, for the report's classes (the rest are asynchronous issues and arrivals)
LOADS = re.compile(r"^(?:LDG(?!DEPBAR|STS)|LDS|LDL|LD\b)")
STORES = re.compile(r"^(?:STG|STS|STL|ST\b|ATOM|RED)")
#: where a straight-line scan stops: a branch, a call or exit, a barrier, a tensor issue or another wait
STOPS = ("BRA", "BSYNC", "BREAK", "JMP", "EXIT", "RET", "CALL", "BAR", "SYNCS", "WARPGROUP", "HGMMA", "IGMMA")
#: a tensor wait, and the count of groups it leaves in flight (`wgmma.wait_group N`)
TENSOR_WAIT = re.compile(r"^WARPGROUP\.DEPBAR\.LE\b.*?,\s*(0x[0-9a-f]+)")


def _operands(line: str) -> str:
    return m.group(1) if (m := OPERANDS.match(line)) else ""


def tensor_waits(body: list[tuple[int, str, str]]) -> list[tuple[int, int]]:
    """A kernel's tensor waits as (index in `body`, groups left in flight)."""
    return [(i, int(m.group(1), 16)) for i, (_, op, line) in enumerate(body)
            if op.startswith("WARPGROUP.DEPBAR") and (m := TENSOR_WAIT.match(f"{op} {_operands(line)}"))]


#: a step's shape, accumulator type and first register (`HGMMA.64x16x16.F32.BF16 R24, ...`: the opcode's `x`s are
#: past the opcode grammar `sass.INSTRUCTION` reads, so the step's operands are read from its line here)
STEP_SHAPE = re.compile(r"\b[HI]GMMA\.64x(\d+)x\d+\.(F16|F32|S32)\S*\s+R(\d+)")


def accumulators(line: str) -> set[int]:
    """The registers a step writes: its first operand and the run its shape fills (m64 nN: N/2 words a thread, N/4
    for a half-precision accumulator); none for a step whose line this does not read (`wait_candidates` then
    reports no candidates)."""
    m = STEP_SHAPE.search(line)
    if not m:
        return set()
    first = int(m.group(3))
    return set(range(first, first + int(m.group(1)) // (4 if m.group(2) == "F16" else 2)))


def touched(op: str, operands: str) -> set[int]:
    """The registers an instruction names, each widened to the run its operand width reaches (`.64`, `.WIDE`: two;
    `.128`: four) -- more than it touches, never less, so a candidate is never one that touches an accumulator."""
    width = 4 if ".128" in op else 2 if (".64" in op or "WIDE" in op) else 1
    return {int(r) + j for r, pair in re.findall(r"\bR(\d+)(\.64)?", operands) for j in range(max(width, 2 if pair else 1))}


def wait_candidates(body: list[tuple[int, str, str]], i: int) -> tuple[Counter, str]:
    """The instructions after the tensor wait at `body[i]`, in straight-line code up to the next stop (`STOPS`), that
    touch none of the registers the waited steps write -- nor read one a toucher wrote -- counted by class (`loads`,
    `stores`, `async` issues, `other` arithmetic); and what ended the scan. The steps are the `HGMMA`s since the
    previous wait in address order. Loads after a toucher's store are not counted: memory is not traced."""
    start = max((j for j in range(i) if body[j][1].startswith("WARPGROUP.DEPBAR")), default=-1)
    tainted: set[int] = set()
    for _, op, line in body[start + 1:i]:
        if op.startswith(WGMMA):
            if not (found := accumulators(line)):
                return Counter(), "an unread step"
            tainted |= found
    classes, stored = Counter(), False
    for _, op, line in body[i + 1:]:
        if op.startswith(STOPS):
            return classes, op.split(".")[0]
        operands = _operands(line)
        regs = touched(op, operands)
        written = set() if op.startswith(READS_FIRST) else touched(op, operands.split(",")[0])
        if regs & tainted:
            tainted |= written
            stored |= bool(STORES.match(op))
            continue
        if LOADS.match(op) and stored:
            continue
        classes["loads" if LOADS.match(op) else "stores" if STORES.match(op) else
                "async" if SUBSTANCE.match(op) else "other"] += 1
    return classes, "end"


#: the wait report's caps: the lines a kernel's distinct waits take, and the groups of kernels with the same lines
REPORT_LINES, REPORT_GROUPS = 6, 16


def waits_report(binary: Path) -> list[str]:
    """THE ADVISORY: per kernel (grouped by identical lines, a count each; at most `REPORT_LINES` lines a kernel and `REPORT_GROUPS` groups of kernels), each tensor
    wait's candidates to precede it -- what follows the wait in straight-line code, reads none of the registers the
    waited steps write, and could be issued before the wait (`wait_candidates`). Never a refusal."""
    code = disassembled(binary)
    names = checked.demangle(list(code))
    groups: dict[tuple, list[str]] = {}
    for mangled, body in code.items():
        rows = Counter()
        for i, left in tensor_waits(body):
            classes, stop = wait_candidates(body, i)
            what = ", ".join(f"{n} {c}" for c, n in classes.items()) or "none"
            rows[f"wait({left}): {sum(classes.values())} candidates ({what}) before {stop}"] += 1
        if rows:
            groups.setdefault(tuple(sorted(rows.items())), []).append(names[mangled])
    out = []
    for rows, kernels_ in list(groups.items())[:REPORT_GROUPS]:
        out.append(f"  {kernels_[0]}" + (f"  (and {len(kernels_) - 1} kernels with the same lines)" if kernels_[1:] else ""))
        shown = sorted(rows, key=lambda r: -r[1])
        out += [f"    {line}" + (f"  x{n}" if n > 1 else "") for line, n in shown[:REPORT_LINES]]
        if len(shown) > REPORT_LINES:
            out.append(f"    ... {len(shown) - REPORT_LINES} more distinct waits")
    if len(groups) > REPORT_GROUPS:
        out.append(f"  ... {len(groups) - REPORT_GROUPS} more groups of kernels")
    return out


def steps_report(binary: Path) -> list[str]:
    """THE ADVISORY: per kernel with tensor steps (kernels with the same lines grouped, a count each; at most
    `REPORT_GROUPS` groups), two lines: the bytes a step puts through the shared-memory port (its own operands and the
    shared loads and stores of its region; wavefronts are those bytes / 128, rounded up) and the instructions a thread
    executes a step, by class -- the figures of the checker's `step_port` and `step_work` rules, read over each of the
    kernel's regions (a kernel with several prints their range and the figures of the region with the most). Bounded by
    `STEP_PORT` and `STEP_WORK` where declared; never a refusal where not."""
    steps = checked.checker().steps
    groups: dict[tuple, list[str]] = {}
    for kernel in checked.kernels_of(binary, demangled=True):
        rows = [steps.figures(r) for r in kernel.regions]
        if not rows:
            continue
        many = f" over {len(rows)} regions; the most is" if len(rows) > 1 else ""
        if any(f.unread for f in rows):
            port = "port: a step this does not read"
        else:
            lo, hi = min(f.port for f in rows), max(f.port for f in rows)
            top = max(rows, key=lambda f: f.port)
            span = f"{lo:,}-{hi:,}" if lo != hi else f"{hi:,}"
            waves = f"{-(-lo // steps.WAVEFRONT)}-{-(-hi // steps.WAVEFRONT)}" if lo != hi else f"{-(-hi // steps.WAVEFRONT)}"
            port = (f"port: {span} B a step ({waves} wavefronts){many} operands {top.operands:,} + shared {top.shared:,}"
                    f"{f' (x {steps.WARPS} warps)' if top.shared else ''}")
        lo, hi = min(f.instructions for f in rows), max(f.instructions for f in rows)
        top = max(rows, key=lambda f: f.instructions)
        by = ", ".join(f"{c} {n}" for c, n in sorted(top.work.items(), key=lambda cn: -cn[1]))
        work = f"work: {f'{lo}-{hi}' if lo != hi else hi} instructions a step{many} {by}"
        groups.setdefault((port, work), []).append(kernel.demangled)
    out = []
    for lines, kernels_ in list(groups.items())[:REPORT_GROUPS]:
        out.append(f"  {kernels_[0]}" + (f"  (and {len(kernels_) - 1} kernels with the same lines)" if kernels_[1:] else ""))
        out += [f"    {line}" for line in lines]
    if len(groups) > REPORT_GROUPS:
        out.append(f"  ... {len(groups) - REPORT_GROUPS} more groups of kernels")
    return out


def check(binary: Path, table: dict[str, dict]) -> list[str]:
    """Every expectation the experiment declares, every hazard the checker's rules find in its kernels past what it declares, every
    notice of its build allowed by the ledger (local memory is the toolkit's own warning there), and the project's own checks of its
    source: the failures, empty when it may run."""
    expected = expectations(binary)
    failures = []
    for e in (e for e in expected if e["op"] not in HAZARDS):
        hits = [k for k in table if re.search(e["kernel"], k)]
        if not hits:
            failures.append(f"no kernel matches {e['kernel']!r}")
        for k in hits:
            n = count(table[k]["ops"], e["op"])
            ok = {">=": n >= e["count"], "==": n == e["count"], "<=": n <= e["count"]}[e["cmp"]]
            if not ok:
                failures.append(f"{k}: {e['op']} {n}, expected {e['cmp']} {e['count']}")
    experiment = binary.name.split(".")[0]
    budgeted = {k for b in budgets(binary) for k in table if re.search(b["kernel"], k)}
    failures += checker_failures(binary, expected, budgeted)
    if experiment not in _PRE_TIMELINE:
        if not timeline(experiments()[experiment]):
            failures.append(f"{experiment}: its header states no TIMELINE -- each phase's stream and which load is a "
                            "stage ahead of which use, written before the kernel (docs/internals/LEARNINGS.md)")
        for at in collective_selected(experiments()[experiment]):
            failures.append(f"{experiment}: line {at}: a warp collective is an operand of a conditional -- every "
                            "lane takes it, and the select follows (docs/internals/LEARNINGS.md)")
    if experiment not in _PRE_29:
        stated = budgets(binary) + budgets(binary, "exempt")
        for k, info in table.items():
            if count(info["ops"], "|".join(WGMMA)) and any(re.search(b["kernel"], k) for b in budgets(binary)):
                failures.append(f"{k}: a budgeted wgmma kernel -- its per-MMA divisor is read from counters the "
                                "hopper model does not name yet (owed with that model): state `Bench::exempt`")
            if count(info["ops"], "|".join(sass.OPCODE_CLASSES["mma"])) and not any(re.search(b["kernel"], k) for b in stated):
                failures.append(f"{k}: an MMA kernel with no instruction budget (KERNEL_STANDARDS 29: `Bench::budget` "
                                "-- `HMMA` and `IMMA` kernels -- or `Bench::exempt` with its reason)")
    return failures


def budgets(binary: Path, kind: str = "budget") -> list[dict]:
    """The instruction budgets the experiment states (`Bench::budget`: kernel regex, opcode class, max per MMA), or
    with `kind="mma"` the MMA each budgeted kernel issues (`Bench::mma`: kernel regex, hardware kind, K)."""
    lines = subprocess.run([str(binary), "--expect"], capture_output=True, text=True, check=True).stdout
    return [e for e in map(json.loads, lines.splitlines()) if e["lab"] == kind]


#: THE CARDS a lab's counts are priced on (`common/hardware.json`): the local card's product rates, sm_80's defaults
CARDS = (("sm_86 (RTX 3080 Ti)", "sm_86", "rtx3080ti"), ("sm_80 (A100)", "sm_80", None))


def _plain(name: str) -> str:
    """A kernel's name as both demanglers can agree on it: no return type, namespaces, casts or parameters."""
    name = re.sub(r"^void\s+", "", name)
    name = name[:name.index(">(") + 1] if ">(" in name else name.split("(")[0]
    return re.sub(r"\s+", "", re.sub(r"\((int|bool|unsigned int|long)\)", "", re.sub(r"\b\w+::", "", name)))


def counters_of(binary: Path, extra: list[str]) -> dict[str, dict]:
    """Per kernel (`_plain` name), its first launch's standard counters (docs/internals/tools/pipes.md): `pipes.short`'s
    names and `opcodes` (executed instructions by opcode)."""
    out = subprocess.run([dev_config.ncu(), "--metrics",
                          ",".join(["sass__inst_executed_per_opcode", *pipes.metric_names()]),
                          "--print-metric-instances", "details", "--csv", str(binary), "--reps", "1", *extra],
                         capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    seen: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(lines[start:]):
        seen.setdefault(_plain(row["Kernel Name"]), {}).setdefault(row["Metric Name"], row["Metric Value"])
    found = {}
    for k, m in seen.items():
        #: a metric this card or this profiler does not collect reads "n/a": `pipes.short` holds it 0, and the
        #: launch's `uncollected` names it, so a record and a report tell that zero from a measured one
        values = {metric: float(v.replace(",", "") or 0) for metric, v in m.items()
                  if metric != "sass__inst_executed_per_opcode" and v != "n/a"}
        c = pipes.short(values)
        c["uncollected"] = sorted(metric for metric, v in m.items() if v == "n/a")
        c["opcodes"] = {op: float(n) for op, n in
                        re.findall(r"([A-Z][A-Z0-9_]*)\s*[:=]\s*([0-9.]+)", m["sass__inst_executed_per_opcode"])}
        found[k] = c
    return found


def executed_classes(raw: dict[str, dict], units: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Per kernel, each opcode class's executed instructions per tensor-pipe instruction, and its shared-memory
    wavefronts and bank conflicts per tensor-pipe instruction (`_wavefronts`, `_conflicts`); a kernel with no MMAs that
    declares its units (`Bench::units`, by `_plain` name in `units`), per unit."""
    found = {}
    for k, c in raw.items():
        n = per_unit(c, k, units)
        per: dict[str, float] = {}
        for op, v in c["opcodes"].items():
            per[sass.opcode_class(op)] = per.get(sass.opcode_class(op), 0.0) + v / n
        per["_wavefronts"] = c["wavefronts"] / n
        per["_conflicts"] = c["conflicts"] / n
        found[k] = per
    return found


def per_unit(c: dict, kernel: str, units: dict[str, float] | None) -> float:
    """What a kernel's counts (summed over its SMs) are divided by: its tensor-pipe instructions, else its declared
    units, else its SMs (a launch an SM)."""
    return c["hmma"] + c["imma"] or (units or {}).get(kernel, 0.0) or pipes.sms(c)


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.2f}"


def pipes_report(raw: dict[str, dict], table: dict[str, dict], units: dict[str, float],
                 mmas: dict[str, tuple[str, int]]) -> None:
    """Every kernel's standard counters, printed (KERNEL_STANDARDS §29.14): per card the pipes' cycles a unit, the
    binding one named; on this card each pipe's busy share of the elapsed cycles, L2 sectors a unit and the three
    largest stall reasons."""
    sys.path.insert(0, str(ROOT))
    from measure import hardware

    local = hardware.product_of(smi("name")[0])
    arch = sass.device_arch()
    here = hardware.resolve(product=local) if local else hardware.resolve(arch=arch)
    kinds = {_plain(k): pipes.mma_kind(v["ops"]) for k, v in table.items()}
    for k, c in raw.items():
        n = per_unit(c, k, units)
        unit = "MMA" if c["hmma"] + c["imma"] else "unit" if k in units else "launch"
        print(f"  pipes, cycles a {unit} an SM: {k[:110]}")
        if c.get("uncollected"):
            print(f"    NOT COLLECTED on this card, read as 0 below: {', '.join(c['uncollected'])}")
        for label, card, product in CARDS:
            p = pipes.pipe_cycles(c, n, hardware.resolve(arch=card, product=product), arch, mmas.get(k), kinds.get(k))
            bound = max(p, key=p.get)
            print(f"    {label:<20} " + "  ".join(f"{name} {_num(v)}" for name, v in p.items())
                  + f"  -> {bound} binds")
        stalls = sorted(c["stalls"].items(), key=lambda kv: -kv[1])[:3]
        if "schedulers" not in here:
            #: a card whose architecture declares no rates (`hardware.json`): its measured counters alone, none priced
            print(f"    busy here, % of elapsed ({arch} declares no rates: tensor, shared and global are not priced): "
                  f"issue {c['busy_issue']:.0f}  alu {c['busy_alu']:.0f}  fma {c['busy_fma']:.0f} "
                  f"(heavy {c['busy_fmaheavy']:.0f})"
                  f"  | L2 sectors read {_num(c['l2_read'] / n)} write {_num(c['l2_write'] / n)}"
                  f"  | stalled: " + ", ".join(f"{r} {v:.0f}%" for r, v in stalls))
            continue
        b = pipes.busy_here(c, here, arch, mmas.get(k), kinds.get(k))
        print(f"    busy here, % of elapsed: issue {b['issue']:.0f}  tensor {b['tensor']:.0f} (priced)  "
              f"alu {b['alu']:.0f}  fma {b['fma']:.0f} (heavy {b['fmaheavy']:.0f})  shared {b['shared']:.0f} (priced)"
              f"  global {b['global']:.0f} (priced)"
              f"  | L2 sectors read {_num(c['l2_read'] / n)} write {_num(c['l2_write'] / n)}"
              f"  | stalled: " + ", ".join(f"{r} {v:.0f}%" for r, v in stalls))


def over_budget(binary: Path, table: dict[str, dict], raw: dict[str, dict], units: dict[str, float]) -> list[str]:
    """KERNEL_STANDARDS sections 29-30: each budgeted kernel's executed instructions per MMA by class, and its
    shared-memory wavefronts per MMA (the class `wavefronts`), against its stated budget, printed; those over it."""
    stated = budgets(binary)
    if not stated:
        return []
    measured = executed_classes(raw, units)
    over = []
    for k in table:
        mine = [b for b in stated if re.search(b["kernel"], k)]
        if not mine:
            continue
        per = measured.get(_plain(k), {})
        unit = "unit" if _plain(k) in units else "MMA"
        print(f"  budget, instructions a {unit}: {k[:110]}")
        for b in mine:
            got = per.get("_wavefronts" if b["class"] == "wavefronts" else b["class"], 0.0)
            #: a slack for once-a-kernel setup and rounding: 3% and 0.02 an MMA
            flag = "OVER" if got > b["max"] * 1.03 + 0.02 else "ok"
            print(f"    {b['class']:<16} {got:6.2f}  budget {b['max']:6.2f}  {flag}")
            if flag == "OVER":
                over.append(f"{k[:80]}: {b['class']} {got:.2f} > {b['max']:.2f}")
        print(f"    {'total':<16} {sum(v for c, v in per.items() if not c.startswith('_')):6.2f}"
              f"   shared-memory wavefronts {per.get('_wavefronts', 0.0):.2f}"
              f" ({per.get('_conflicts', 0.0):.2f} of them bank conflicts)")
    return over


def declared(binary: Path, table: dict[str, dict]) -> tuple[dict[str, float], dict[str, tuple[str, int]]]:
    """The experiment's declarations by `_plain` kernel name: each kernel's units (`Bench::units`) and the MMA it
    issues (`Bench::mma`: hardware kind, K)."""
    units = {_plain(k): float(u["n"]) for k in table for u in budgets(binary, "units") if re.search(u["kernel"], k)}
    mmas = {_plain(k): (m["kind"], int(m["k"])) for k in table for m in budgets(binary, "mma")
            if re.search(m["kernel"], k)}
    return units, mmas


def quiet() -> int:
    """The lowest GPU utilization over three looks: another consumer time-slices every kernel and inflates cycles."""
    looks = []
    for _ in range(3):
        looks.append(int(smi("utilization.gpu")[0]))
        time.sleep(0.2)
    return min(looks)


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    #: the sources a record depends on; the records themselves are outputs
    #: MODULE.bazel holds the pin of the stamp header every experiment compiles against
    sources = ["measure/lab/lab.cuh", "measure/lab/experiments", "tools/lab.py", "MODULE.bazel"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *sources], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return {"sha": sha, "dirty": bool(dirty)}


def summarize(samples: list[float]) -> dict:
    lo, hi = min(samples), max(samples)
    return {"median": statistics.median(samples), "min": lo, "max": hi,
            "unstable": len(samples) > 1 and lo > 0 and hi / lo > SPREAD}


def inputs(name: str) -> tuple[list[str], dict[str, str]]:
    """The experiment's inputs: `<name>.inputs.py`, if it has one, writes them into the scratch space; the binary's
    extra arguments and each file's digest."""
    script = EXPERIMENTS / f"{name}.inputs.py"
    if not script.exists():
        return [], {}
    directory = dev_config.scratch("lab") / "inputs" / name
    directory.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(f"{name}_inputs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.write(directory)
    digests = {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()[:16] for f in files}
    return ["--inputs", str(directory)], digests


def page(name: str, instance: str) -> int:
    """Regenerate an experiment's page's measured section from its record: `measure/lab/pages/<name>.py`'s `write`."""
    script = LAB / "pages" / f"{name}.py"
    if not script.exists():
        print(f"lab: no page generator for {name!r}: `measure/lab/pages/{name}.py` does not exist")
        return 1
    sys.path.insert(0, str(script.parent))  #: the generators' one lookup, `_record.py`, is beside them
    spec = importlib.util.spec_from_file_location(f"{name}_page", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for written in module.write(ROOT, instance):
        print(f"wrote {written.relative_to(ROOT)}")
    return 0


def execute(binary: Path, reps: int, extra: list[str], timeout: int) -> tuple[dict, list[dict]]:
    """The experiment's device line and its results, each with its samples summarized; killed at `timeout` seconds
    (KERNEL_STANDARDS: device code runs under a kill timeout)."""
    out = subprocess.run([str(binary), "--reps", str(reps), *extra], capture_output=True, text=True, check=True,
                         timeout=timeout).stdout
    device, results = {}, []
    for line in out.splitlines():
        if not line.startswith('{"lab"'):
            continue
        item = json.loads(line)
        if item["lab"] == "device":
            device = item
        elif item["lab"] == "result":
            results.append({**item, **summarize(item["samples"])})
        elif item["lab"] in ("check", "unchecked"):
            device.setdefault("_checks", []).append(item)
        elif item["lab"] == "trace":
            device.setdefault("_traces", []).append(item)
    return device, results


def unverified(name: str, device: dict, results: list[dict]) -> list[str]:
    """KERNEL_STANDARDS section 29: the variants whose outputs failed their check, and (past the debts) the recorded
    variants with neither a check nor a stated reason for none -- a result belongs to the variant whose label it starts
    with."""
    checks = device.get("_checks", [])
    bad = [f"{c['variant']}: {c['detail']}" for c in checks if c["lab"] == "check" and not c["ok"]]
    if name in _PRE_29:
        return bad
    covered = [c["variant"] for c in checks]
    bad += [f"{r['name']}: no output check" for r in results if not any(r["name"].startswith(v) for v in covered)]
    return bad


def report(name: str, arch: str, device: dict, results: list[dict], reps: int) -> None:
    width = max(len(r["name"]) for r in results)
    print(f"{name} on {device.get('name')} ({arch}), {reps} repetitions -- median [min, max]")
    for r in results:
        flag = "  UNSTABLE" if r["unstable"] else ""
        print(f"  {r['name']:<{width}}  {r['median']:10.3f} [{r['min']:.3f}, {r['max']:.3f}] {r['unit']}{flag}")


def keep(name: str, arch: str, device: dict, results: list[dict], table: dict, reps: int, utilization: int,
         record: bool, digests: dict[str, str], counters: dict[str, dict], refused: dict | None = None,
         built: dict | None = None, into: Path | None = None, blocks: int = 0) -> Path:
    """The run's record: into `measure/lab/results` under `--record`, else the scratch space (`into`: that directory
    instead). `refused`: the written exception of a host that grants no counters, in their place. `built`: a pack's
    toolkit and commit, where the binary was built on another machine (`pack`)."""
    driver, = smi("driver_version")
    built = built or {"toolkit": toolkit_line(), "git": git_state()}
    body = {"experiment": name, "question": question(experiments()[name]), "arch": arch, "device": device.get("name"),
            "checks": device.get("_checks", []),
            "driver": driver, "toolkit": built["toolkit"], "git": built["git"], "reps": reps,
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "utilization_before": utilization, "inputs": digests, "results": results,
            "kernels": {k: {"regs": v.get("regs"), "ops": dict(Counter(v["ops"]).most_common(12))}
                        for k, v in table.items()},
            "counters": {k: {m: v for m, v in c.items() if m != "opcodes"} for k, c in counters.items()}}
    if refused:
        body["counters_exception"] = refused
    if blocks:
        body["blocks"] = blocks
    target = (into / name if into else RESULTS / name if record
              else dev_config.scratch("lab") / "results" / name) / (f"{arch}.blocks{blocks}.json" if blocks else f"{arch}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=1) + "\n")
    #: THE TRACES (a traced row's timelines, `lab trace`), beside the record: a directory, a warpline record a traced
    #: row (`row<i>.json`, in the order the rows ran, the record's `variant` its name)
    traces = target.with_name(target.name.replace(".json", ".trace"))
    shutil.rmtree(traces, ignore_errors=True)
    if device.get("_traces"):
        traces.mkdir()
        for i, item in enumerate(device["_traces"]):
            (traces / f"row{i}.json").write_text(json.dumps(item["record"], separators=(",", ":")) + "\n")
    return target


def toolkit_line() -> str:
    return subprocess.run([dev_config.cuda_bin("nvcc"), "--version"], capture_output=True,
                          text=True).stdout.strip().splitlines()[-1]


def pack(name: str, arch: str, into: Path) -> int:
    """AN EXPERIMENT BUILT HERE FOR A CARD ELSEWHERE: its binary for `arch`, its SASS checked as `run` checks it, and
    beside it what a record needs of the build -- the kernels' registers and opcodes, the toolkit, the commit and the
    binary's digest -- so a box with a driver and no toolkit runs it (`carried`)."""
    binary = build(experiments()[name], arch)
    table = kernels(binary)
    failures = check(binary, table)
    if failures:
        print(f"lab: {name} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1
    if budgets(binary):
        print(f"lab: {name} has budgeted kernels, whose gate reads executed counts: it runs where the toolkit is")
        return 1
    report = notice_report(binary)
    if report:
        print("lab: toolkit notices the ledger reports (not blocking; tools/notice_ledger.json):", *report, sep="\n")
    report = waits_report(binary)
    if report:
        print("lab: tensor waits -- candidates to precede each (advisory; `lab waits` prints it alone):", *report, sep="\n")
    report = steps_report(binary)
    if report:
        print("lab: tensor steps -- the port bytes and instructions of a step (advisory unless a bound is declared; "
              "`lab steps` prints it alone):", *report, sep="\n")
    into.mkdir(parents=True, exist_ok=True)
    carried_binary = into / binary.name
    carried_binary.write_bytes(binary.read_bytes())
    carried_binary.chmod(0o755)
    body = {"experiment": name, "arch": arch, "toolkit": toolkit_line(), "git": git_state(),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "kernels": {k: {"regs": v.get("regs"), "ops": dict(v["ops"])} for k, v in table.items()}}
    (into / f"{binary.name}.pack.json").write_text(json.dumps(body, indent=1) + "\n")
    print(f"lab: packed {name} for {arch} into {into}")
    return 0


def measure(name: str, arch: str, binary: Path, table: dict, opts, *, built: dict | None = None,
            into: Path | None = None) -> int:
    """THE DEVICE'S HALF OF A RUN, the one sequence `run` and `carried` share: the inputs, a quiet GPU, the timed
    repetitions under the kill timeout, the checks, then the counters -- the profiler's, gated against the budgets, or,
    where the caller DECLARES a host that grants none (`--no-counters HOST`, KERNEL_STANDARDS section 16 addendum 4),
    the written exception naming that host. A profiler that fails undeclared fails the run."""
    extra, digests = inputs(name)
    #: `--blocks N`: the experiment's grid held to N CTAs (a lighter load on the card), its record its own
    if opts.blocks:
        extra = [*extra, "--blocks", str(opts.blocks)]
    utilization = quiet()
    if utilization > QUIET_PERCENT and not opts.force:
        print(f"lab: the GPU is {utilization}% busy before the run; another consumer would inflate every cycle "
              "(--force to measure anyway)")
        return 1

    device, results = execute(binary, opts.reps, extra, opts.timeout)
    report(name, arch, device, results, opts.reps)
    for c in device.get("_checks", []):
        print(f"  {'checked' if c['lab'] == 'check' else 'unchecked'} {c['variant']}: "
              f"{'ok' if c.get('ok') else 'FAILED' if c['lab'] == 'check' else c['reason']}"
              + (f" ({c['detail']})" if c["lab"] == "check" else ""))
    wrong = unverified(name, device, results)
    if wrong:
        print(f"lab: {name}'s outputs are not verified, not recorded:", *wrong, sep="\n  ")
        return 1
    #: warpline's `check` on each traced row's record, where warpline's library is: a record with findings is not recorded
    if device.get("_traces") and dev_config.warpline_python() is None:
        print(f"lab: {name}'s traced rows are written unchecked: warpline's library is not here (the run is outside "
              "`bazel run //tools:lab`); `lab trace` checks them where it is")
    elif device.get("_traces"):
        import lab_trace

        unsound = lab_trace.unsound(device["_traces"])
        if unsound:
            print(f"lab: {name}'s traced rows are not sound records, not recorded:", *unsound, sep="\n  ")
            return 1

    raw, refused = {}, None
    if opts.no_counters:
        if built is None and budgets(binary):
            print(f"lab: {name} has budgeted kernels, whose gate reads executed counts: not recorded on a host "
                  "that grants no counters (KERNEL_STANDARDS section 16 addendum 4)")
            return 1
        refused = {"host": opts.no_counters, "instead": "the kernels' own clocks and the SASS's static counts"}
        print(f"lab: no counters on {opts.no_counters}: recorded with the exception, no pipes report")
    else:
        raw = counters_of(binary, extra)
        units, mmas = declared(binary, table)
        pipes_report(raw, table, units, mmas)
        over = over_budget(binary, table, raw, units)
        if over and not opts.force:
            print(f"lab: {name} OVER ITS INSTRUCTION BUDGET, not recorded (--force to record anyway):", *over,
                  sep="\n  ")
            return 1
    target = keep(name, arch, device, results, table, opts.reps, utilization, opts.record, digests, raw, refused,
                  built, into, opts.blocks)
    print(f"lab: recorded {target.relative_to(ROOT) if opts.record else target}")
    return 0


def carried(name: str, source: Path, into: Path, opts) -> int:
    """A PACKED EXPERIMENT RUN ON THIS CARD, a box with no toolkit to build it: the pack's binary by its digest, on the
    pack's architecture, through `measure` -- its counters read where the box holds the pinned profiler, the host
    declared where it grants none; its build facts are the pack's."""
    arch = sass.device_arch()
    binary = source / f"{name}.{arch}"
    if not binary.exists():
        print(f"lab: no pack of {name} for this card's {arch} in {source}")
        return 1
    built = json.loads((source / f"{binary.name}.pack.json").read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest() != built["sha256"]:
        print(f"lab: {binary} is not the binary its pack describes")
        return 1
    return measure(name, arch, binary, built["kernels"], opts, built=built, into=into)


def ledger_dumps(name: str, source: Path, into: Path, patterns: list[str], skip: int = 0) -> int:
    """A CARRIED LEDGER'S DUMPS, made on a box with no toolkit: the pack's binary by its digest, then each kernel whose
    demangled name a pattern matches profiled once (its first launch, ncu's SourceCounters) into
    `ledger_<its template numbers>.csv`, and `ledger.json` naming the binary's digest and each kernel's file -- read back
    where the toolkit is by `ledger --packed --dump`. `skip` launches of each kernel pass first (an experiment's checked
    launch is its first: its timed ones follow)."""
    arch = sass.device_arch()
    binary = source / f"{name}.{arch}"
    built = json.loads((source / f"{binary.name}.pack.json").read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest() != built["sha256"]:
        print(f"lab: {binary} is not the binary its pack describes")
        return 1
    chosen = [k for k in built["kernels"] if any(re.search(p, k) for p in patterns)]
    if not chosen:
        print(f"lab: no kernel of {name}'s pack matches {patterns}")
        return 1

    into.mkdir(parents=True, exist_ok=True)
    extra, _ = inputs(name)
    files = {}
    for k in chosen:
        #: a kernel's template numbers: the long ones a single-parameter experiment has, else every `(int)` argument
        head = k.split("(")[0] if "<" not in k else k[:k.index(">") + 1]
        ids = re.findall(r"\d{4,}", head) or re.findall(r"\(int\)(\d+)", head)
        if not ids:
            print(f"lab: {k[:80]} has no template number to name its dump by")
            return 1
        core = k[:k.index(">") + 1].split("::")[-1] if "<" in k else k.split("(")[0].split("::")[-1]
        out = subprocess.run([dev_config.ncu(), "--kernel-name-base", "demangled", "--kernel-name",
                              f"regex:{re.escape(core)}", "--launch-skip", str(skip), "--launch-count", "1",
                              "--section", "SourceCounters", "--page", "source", "--print-source", "sass", "--csv",
                              str(binary), "--reps", "1", *extra], capture_output=True, text=True, timeout=900)
        dump = into / f"ledger_{'_'.join(ids)}.csv"
        dump.write_text(out.stdout)
        if out.returncode != 0 or not region_ledger.read_source_counters(dump):
            print(f"lab: {k[:80]}: no SourceCounters (exit {out.returncode}): {out.stderr[-400:]}")
            return 1
        files[k] = dump.name
        print(f"lab: {dump.name}: {k[:90]}")
    (into / "ledger.json").write_text(json.dumps({"experiment": name, "arch": arch, "sha256": built["sha256"],
                                                   "git": built.get("git"), "launch_skip": skip, "dumps": files},
                                                  indent=1) + "\n")
    return 0


def run(name: str, opts) -> int:
    arch = sass.device_arch()
    binary = build(experiments()[name], arch)
    table = kernels(binary)
    failures = check(binary, table)
    if failures:
        print(f"lab: {name} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1
    report = notice_report(binary)
    if report:
        print("lab: toolkit notices the ledger reports (not blocking; tools/notice_ledger.json):", *report, sep="\n")
    return measure(name, arch, binary, table, opts)


def ncu_counters(binary: Path, metrics: list[str], extra: list[str]) -> dict[str, dict[str, float]]:
    """Per kernel (ncu's name), each metric's median over the kernel's launches in one repetition of the experiment."""
    out = subprocess.run([dev_config.ncu(), "--metrics", ",".join(metrics), "--csv", str(binary),
                          "--reps", "1", *extra], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    seen: dict[str, dict[str, list[float]]] = {}
    for row in csv.DictReader(lines[start:]):
        raw = row["Metric Value"].replace(",", "")
        if raw == "n/a":  # a metric this card or this ncu does not collect: left out, so its absence shows
            continue
        seen.setdefault(row["Kernel Name"], {}).setdefault(row["Metric Name"], []).append(float(raw))
    return {k: {m: statistics.median(v) for m, v in ms.items()} for k, ms in seen.items()}


def counters(name: str, metrics: list[str], record: bool) -> int:
    arch = sass.device_arch()
    binary = build(experiments()[name], arch)
    failures = check(binary, kernels(binary))
    if failures:
        print(f"lab: {name} REFUSED -- its SASS is not what it measures:", *failures, sep="\n  ")
        return 1
    extra, digests = inputs(name)
    found = ncu_counters(binary, metrics, extra)
    width = max(len(k) for k in found)
    for k, ms in found.items():
        print(f"  {k:<{width}}  " + "  ".join(f"{m} {v:,.0f}" for m, v in ms.items()))
    body = {"experiment": name, "question": question(experiments()[name]), "arch": arch, "git": git_state(),
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "metrics": metrics,
            "inputs": digests, "kernels": found}
    target = (RESULTS / name if record else dev_config.scratch("lab") / "results" / name) / f"{arch}.counters.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, indent=1) + "\n")
    print(f"lab: recorded {target.relative_to(ROOT) if record else target}")
    return 0


#: a phase heading: a `//:` comment that opens a declaration block (KERNEL_STANDARDS section 29.13)
PHASE = re.compile(r"^\s*//:\s*(.*)$")


def phases(source: Path) -> list[tuple[int, str]]:
    """(first line, name) of each phase of `source`: a `//:` block opens one, named by its first words."""
    out = []
    lines = source.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if (m := PHASE.match(line)) and not PHASE.match(lines[i - 2] if i > 1 else ""):
            out.append((i, " ".join(m.group(1).split()[:6])))
    return out


def ledger(name: str, pattern: str | None, dumps: list[Path] | None = None, packed: Path | None = None,
           arch: str | None = None, per: float | None = None, top: int = 16) -> int:
    """KERNEL_STANDARDS section 29.13: each kernel's executed instructions by phase (its source's `//:` headings), per
    declared unit or per launch -- ncu's SourceCounters of the kernel's first launch, each SASS offset mapped to its
    innermost line in the experiment's source (the lab builds carry `-lineinfo`); a carried ledger's, the launch its
    manifest's `launch_skip` names (0 the first, the checked launch) -- with each phase's shared-memory
    wavefronts, those over the conflict-free count (bank conflicts) and its stall samples. CARRIED: `dumps` are the
    SourceCounters CSVs a box without the toolkit made of `packed` (`lab pack`'s binary, built here), one a kernel;
    `per` divides every count (a launch's CTA-windows)."""
    arch = arch or sass.device_arch()
    path = experiments()[name]
    manifest = None
    if dumps is not None:
        #: a carried ledger maps the dumps through the very binary they profiled: its digest the dumps' manifest's
        if packed is None:
            print("lab: --dump needs --packed, the binary the dumps profiled")
            return 1
        found = {d.parent / "ledger.json" for d in dumps}
        if len(found) != 1 or not next(iter(found)).exists():
            print(f"lab: the dumps' manifest (ledger.json beside them, `lab ledger-dumps`) is not one file: {sorted(found)}")
            return 1
        manifest = json.loads(next(iter(found)).read_text())
        if "launch_skip" not in manifest:
            print("lab: the dumps' manifest names no launch (`launch_skip`): `lab ledger-dumps` writes it")
            return 1
        print(f"  the dumps' launch: each kernel's after its first {manifest['launch_skip']} "
              f"({'the checked launch' if not manifest['launch_skip'] else 'a timed launch'})")
        if hashlib.sha256(packed.read_bytes()).hexdigest() != manifest["sha256"]:
            print(f"lab: {packed} is not the binary the dumps profiled ({manifest['sha256'][:12]})")
            return 1
    binary = packed if packed else build(path, arch)
    table = kernels(binary)
    extra, _ = inputs(name)
    units = {k: (float(u["n"]), u["unit"]) for k in table for u in budgets(binary, "units")
             if re.search(u["kernel"], k)}
    heads = phases(path)
    declarations = load_declarations(expectations(binary), declared_reasons(path.read_text()))
    refused: list[str] = []

    images = [c for c in sass.cubins(binary) if arch in c.name]
    if len(images) != 1:
        print(f"lab: {len(images)} images for {arch} in {binary.name}")
        return 1
    mangled = dict(zip(checked.demangle(list(kernels_mangled(binary))).values(), kernels_mangled(binary), strict=True))
    disassembly = sass.disassemble(images[0], "--print-line-info-inline", "-gi")
    for k in table:
        if pattern and not re.search(pattern, k):
            continue
        core = k.split("(")[0].split("::")[-1] if "<" not in k else k[:k.index(">") + 1].split("::")[-1]
        if manifest is not None:
            #: a carried kernel's dump is the one its manifest names; a kernel the pattern chose without one is refused
            named = manifest["dumps"].get(k)
            if named is None or not any(d.name == named for d in dumps):
                print(f"lab: {k[:90]}: no dump of it among {[d.name for d in dumps]}")
                return 1
            csv_path = next(d for d in dumps if d.name == named)
        else:
            out = subprocess.run([dev_config.ncu(), "--kernel-name-base", "demangled", "--kernel-name",
                                  f"regex:{re.escape(core)}", "--launch-count", "1", "--section", "SourceCounters",
                                  "--page", "source", "--print-source", "sass", "--csv", str(binary), "--reps", "1",
                                  *extra], capture_output=True, text=True, check=True).stdout
            csv_path = dev_config.scratch("lab") / f"{name}.ledger.csv"
            csv_path.write_text(out)
        rows = region_ledger.read_source_counters(csv_path)
        detail = region_ledger.read_source_detail(csv_path)

        if not rows:
            print(f"  {k}: no launch profiled")
            continue
        base = min(at for at, *_ in rows)
        frames = sass.frames(own_section(disassembly, mangled[k]))
        div, unit = (per, "CTA-window") if per else units.get(k, (1.0, "launch"))
        #: a phase's executed instructions, shared wavefronts, those a conflict-free access would not take, samples
        by: dict[str, list[float]] = {}
        #: and each source line's shared wavefronts by the kind of instruction that issued them
        sites: dict[tuple[int, str], float] = {}
        reasons: dict[str, dict[str, int]] = {}
        for at, executed, samples, why in rows:
            lines = [ln for f, ln in frames.get(at - base, []) if f == path.name]
            head = next((h for first, h in reversed(heads) if lines and first <= lines[0]), "(outside the source)")
            op, waves, ideal = detail.get(at, ("?", 0, 0))
            acc = by.setdefault(head, [0.0, 0.0, 0.0, 0.0])
            acc[0] += executed / div
            acc[1] += waves / div
            acc[2] += max(waves - ideal, 0) / div
            acc[3] += samples
            for reason, n in why.items():
                reasons.setdefault(head, {})[reason] = reasons.setdefault(head, {}).get(reason, 0) + n
            if waves:
                kind = ("wgmma operands" if op.startswith("HGMMA") else "stores" if op.startswith(("STS", "ATOMS", "REDS"))
                        else "loads" if op.startswith(("LDS", "LDSM")) else op)
                key = (lines[0] if lines else 0, kind)
                sites[key] = sites.get(key, 0.0) + waves / div
        refused += profile_failures(csv_path, k, declarations,
                                    {at - base: next((ln for f, ln in frames.get(at - base, []) if f == path.name), 0)
                                     for at, *_ in rows}, path.read_text().splitlines(), div, unit)
        total = sum(v[0] for v in by.values())
        waves_all = sum(v[1] for v in by.values())
        samples_all = max(sum(v[3] for v in by.values()), 1.0)
        print(f"  {k[:110]}: {total:,.0f} instructions and {waves_all:,.0f} shared wavefronts a {unit}")
        print(f"    {'instructions':>12}  {'wavefronts':>10}  {'conflicts':>9}  {'samples %':>9}  phase")
        for head, (n, w, c, smp) in sorted(by.items(), key=lambda x: -x[1][1]):
            print(f"    {n:12,.1f}  {w:10,.1f}  {c:9,.1f}  {100 * smp / samples_all:8.1f}%  {head}")
        print("    the phases holding the most stall samples, each with its three largest reasons:")
        for head, (_, _, _, smp) in sorted(by.items(), key=lambda x: -x[1][3])[:8]:
            top3 = sorted(reasons.get(head, {}).items(), key=lambda x: -x[1])[:3]
            print(f"    {100 * smp / samples_all:5.1f}%  {head[:60]:60}  "
                  + ", ".join(f"{r} {100 * n / max(smp, 1):.0f}%" for r, n in top3))
        source = path.read_text().splitlines()
        print(f"    the source lines that issue its shared wavefronts, a {unit}:")
        for (line, kind), w in sorted(sites.items(), key=lambda x: -x[1])[:top]:
            text = source[line - 1].strip()[:80] if 0 < line <= len(source) else "(outside the source)"
            print(f"    {w:10,.1f}  {kind:15}  line {line:5}  {text}")
    if refused:
        print("lab: REFUSED, the profile holds bank conflicts nobody declared (docs/internals/tools/lab.md#ledger):", *refused, sep="\n  ")
        return 1
    return 0


#: the share of the window by which the least-waiting role must wait less than the next for the row to name a path
PATH_MARGIN = 0.05


def is_wait(site: str) -> bool:
    return " waits on " in site or "rendezvous" in site or "fence" in site


def timelines(name: str, pattern: str | None, record: Path | None, skip: int, out: Path | None) -> int:
    """A RECORD'S TRACED ROWS read by `lab_trace`: its traces beside it, its windows' cycles from it."""
    path = record or (RESULTS / name / f"{sass.device_arch()}.json")
    traces = path.with_name(path.name.replace(".json", ".trace"))
    if not traces.is_dir():
        print(f"lab: no traces beside {path} (a record's traced rows write {traces.name})")
        return 1
    windows = {r["name"]: r["median"] for r in json.loads(path.read_text())["results"]
               if r.get("unit") == "cycles a window a CTA"}
    import lab_trace

    return lab_trace.report(lab_trace.load(traces), pattern, skip, windows, out)


def critical_path(name: str, pattern: str | None, record: Path | None) -> int:
    """THE WINDOW'S PATH (KERNEL_STANDARDS 22): for each clocked row of a record -- a row in cycles a window a CTA whose
    sites, `<row>: a <role> warp ...` in cycles a window a warp, its roles' warps clocked -- each role's waits (its
    sites that `wait on` something, and its rendezvous and fences; a warpgroup's first warp's left out), its busy
    cycles (the window less those waits) and its phases -- a phase that holds a wait marked so; the role whose waits
    are least is the window's path, and only its phases can shorten the window. A row whose two least-waiting roles
    are within `PATH_MARGIN` of the window of each other names no path."""
    arch = sass.device_arch() if record is None else None
    path = record or (RESULTS / name / f"{arch}.json")
    rows = {r["name"]: r for r in json.loads(path.read_text())["results"]}
    windows = {n for n, r in rows.items() if r.get("unit") == "cycles a window a CTA"}
    sites: dict[str, list[tuple[str, str, float]]] = {}
    for n, r in rows.items():
        if r.get("unit") != "cycles a window a warp" or ", its warpgroup's first warp" in n:
            continue
        parent = max((w for w in windows if n.startswith(w + ": ")), key=len, default=None)
        site = n[len(parent) + 2:] if parent else ""
        m = re.match(r"an? (\w+) warp", site)
        if parent and m:
            sites.setdefault(parent, []).append((m.group(1), site, r["median"]))

    shown = 0
    for row, found in sites.items():
        if pattern and not re.search(pattern, row):
            continue
        window = rows[row]["median"]
        roles: dict[str, list[float]] = {}
        for role, site, v in found:
            acc = roles.setdefault(role, [0.0, 0.0])
            acc[0 if is_wait(site) else 1] += v
        if len(roles) < 2:
            continue
        shown += 1
        ranked = sorted(roles.items(), key=lambda kv: kv[1][0])
        least = ranked[0] if ranked[1][1][0] - ranked[0][1][0] > PATH_MARGIN * window else None

        print(f"{row[:150]}: {window:,.0f} cycles a window")
        for role, (waits, phases) in sorted(roles.items(), key=lambda kv: kv[1][0]):
            mark = "  <- THE PATH" if least and role == least[0] else ""
            print(f"    a {role} warp: waits {waits:7,.0f}, busy {window - waits:7,.0f} ({100 * (window - waits) / window:3.0f}%), "
                  f"its clocked phases {phases:7,.0f}{mark}")
        if least is None:
            print(f"    no path: its two least-waiting roles wait within {PATH_MARGIN:.0%} of the window of each other")
            continue
        print(f"    the path's phases (a {least[0]} warp; '(a wait in it)' marks a phase that holds one):")
        for role, site, v in sorted((x for x in found if x[0] == least[0] and not is_wait(x[1])), key=lambda x: -x[2]):
            print(f"      {v:7,.0f}  {site[:100]}{'  (a wait in it)' if ' wait on ' in site else ''}")
    if not shown:
        print(f"lab: no clocked row of {path} matches {pattern!r}")
        return 1
    return 0


def static_kind(opcode: str) -> str:
    """An opcode's budget class (`sass.opcode_class`), a global access by what it does: the accesses that hold the
    warp that issues them are told apart."""
    kind = sass.opcode_class(opcode)
    if kind != "global":
        return kind
    return ("global copy" if opcode.startswith(("LDGSTS", "UBLK")) else "global load" if opcode.startswith("LDG")
            else "global store" if opcode.startswith("STG") else "global add")


def static_ledger(name: str, pattern: str | None, arch: str | None) -> int:
    """A kernel's instructions AS COMPILED, with no device: each innermost natural loop's body (the checker's model of the kernel) by the role that holds it (the
    call in the kernel's body the instruction was inlined from), its phases (the source's `//:` headings) and its
    kinds. A loop's cost a unit is this count times its trips, which the source states; a card that grants no
    profiler counters has no executed ledger, and this one stands in its place."""
    arch = arch or sass.device_arch()
    path = experiments()[name]
    binary = build(path, arch)
    heads = phases(path)
    source = path.read_text().splitlines()
    images = [c for c in sass.cubins(binary) if arch in c.name]
    if len(images) != 1:
        print(f"lab: {len(images)} images for {arch} in {binary.name}")
        return 1
    mangled = dict(zip(checked.demangle(list(kernels_mangled(binary))).values(), kernels_mangled(binary), strict=True))
    disassembly = sass.disassemble(images[0], "--print-line-info-inline", "-gi")
    model = {k.name: k for k in checked.kernels_of(binary, demangled=True)}
    #: a kernel the lab would refuse is counted all the same, its refusals stated first
    for failure in check(binary, kernels(binary)):
        print(f"  REFUSED AS IT STANDS: {failure}")

    for k, symbol in mangled.items():
        if pattern and not re.search(pattern, k):
            continue
        text = own_section(disassembly, symbol)
        frames = sass.frames(text)
        kernel = model[symbol]
        at_address = {i.address: i for i in kernel.instructions}
        by: dict[tuple, Counter] = {}
        for line in text.splitlines():
            if not (m := sass.INSTRUCTION.match(line)):
                continue
            at = int(m.group(1), 16)
            #: an instruction inside the model's span that the model lacks is two disassemblies that disagree, never
            #: "outside every loop"; an instruction past the model's last address (the listing's trailing padding: a
            #: self-branch, NOPs) is counted, in no loop, so under "outside every loop"
            if at not in at_address and at < max(at_address):
                print(f"lab: {k[:110]}: the instruction at {at:#06x} is in the listing and not in the checker's model of the kernel")
                return 1
            #: the innermost natural loop around the instruction, as its header's address and its last tail's
            around = kernel.loop_of(at_address[at]) if at in at_address else None
            loop = (around.address, max(kernel.block(t).end for t in around.tails)) if around else None
            own = [ln for f, ln in frames.get(at, []) if f == path.name]
            #: the role: the outermost frame that calls one (`NAME_role`) names it
            called = next((c for ln in reversed(own) if (c := re.search(r"\b(\w+_role)\b", source[ln - 1]))), None)
            role = called.group(1) if called else "(the kernel's body)"
            head = next((h for first, h in reversed(heads) if own and first <= own[0]), "(outside the source)")
            by.setdefault((role, loop), Counter())[(head, static_kind(m.group(3)))] += 1

        print(f"  {k[:110]}")
        for (role, loop), counts in sorted(by.items(), key=lambda item: (item[0][0], item[0][1] or (-1, -1))):
            where = f"loop {loop[0]:#06x}-{loop[1]:#06x}" if loop else "outside every loop"
            kinds = Counter()
            for (_, kind), n in counts.items():
                kinds[kind] += n
            print(f"    {role}, {where}: {sum(counts.values())} instructions -- "
                  + ", ".join(f"{n} {kind}" for kind, n in kinds.most_common()))
            phase = Counter()
            for (head, _), n in counts.items():
                phase[head] += n
            for head, n in phase.most_common():
                print(f"      {n:6d}  {head}")
    return 0


def own_section(disassembly: str, mangled: str) -> str:
    """A kernel's own code section of an `nvdisasm` listing: offsets restart in every function, so the frame map of the
    whole listing would merge them."""
    lines = disassembly.splitlines()
    start = next(i for i, line in enumerate(lines) if f".section\t.text.{mangled}," in line)
    end = next((i for i in range(start + 1, len(lines)) if "\t.section\t" in lines[i]), len(lines))
    return "\n".join(lines[start:end])


def kernels_mangled(binary: Path) -> list[str]:
    """The binary's kernels by their mangled names, as the disassemblers spell them."""
    with sass.sass_lines(binary) as lines:
        return [m.group(1) for line in lines if (m := sass.FUNCTION.match(line))]


def show_sass(name: str, pattern: str | None, full: bool, arch: str | None) -> int:
    path = experiments()[name]
    binary = build(path, arch or sass.device_arch())
    table = kernels(binary)
    #: a loop of steps' longest run without one: the checker's `step_gap` rule
    step_gap = checked.checker().rules.registry()["step_gap"]
    gaps = {k.demangled: step_gap.check(k) for k in checked.kernels_of(binary, demangled=True)}
    for k, info in table.items():
        if pattern and not re.search(pattern, k):
            continue
        top = ", ".join(f"{op} {n}" for op, n in Counter(info["ops"]).most_common(10))
        gap = "".join(f", {f.what}" for f in gaps.get(k, []))
        print(f"{k}\n  {info.get('regs')} registers{gap}; {top}")
    if full:
        print(subprocess.run([dev_config.cuda_bin("cuobjdump"), "-sass", str(binary)], capture_output=True,
                             text=True, check=True).stdout)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("sass")
    s.add_argument("experiment")
    s.add_argument("kernel", nargs="?")
    s.add_argument("--full", action="store_true")
    s.add_argument("--arch", help="compile for this architecture (sm_80, ...) in place of the device's: SASS only")
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--record", action="store_true", help="write the record into measure/lab/results")
    k = sub.add_parser("pack", help="build and check an experiment here for a card elsewhere")
    k.add_argument("experiment")
    k.add_argument("--arch", required=True)
    k.add_argument("--into", required=True, type=Path)
    w = sub.add_parser("waits", help="each tensor wait's candidates to precede it (KERNEL_STANDARDS 29.16), advisory")
    w.add_argument("experiment")
    w.add_argument("--arch", required=True)
    ts = sub.add_parser("steps", help="each tensor step's shared-memory port bytes and instructions, advisory beside "
                                      "the STEP_PORT and STEP_WORK bounds")
    ts.add_argument("experiment")
    ts.add_argument("--arch", required=True)
    c = sub.add_parser("carried", help="run a packed experiment on this card, a box with no toolkit to build it")
    c.add_argument("experiment")
    c.add_argument("--from", dest="source", required=True, type=Path)
    c.add_argument("--into", required=True, type=Path)
    c.set_defaults(record=False)
    for timed in (r, c):
        timed.add_argument("--reps", type=int, default=7)
        timed.add_argument("--force", action="store_true", help="measure on a busy GPU")
        timed.add_argument("--timeout", type=int, default=900, help="seconds before the experiment is killed")
        timed.add_argument("--blocks", type=int, default=0,
                           help="the grid held to N CTAs, a lighter load: its record <arch>.blocks<N>.json")
        timed.add_argument("--no-counters", metavar="HOST",
                           help="this host grants no profiler counters: name it, and the record carries the "
                                "exception in their place (a box that holds the pinned profiler reads them)")
    g = sub.add_parser("ledger", help="a kernel's executed instructions by phase (KERNEL_STANDARDS 29.13)")
    g.add_argument("experiment")
    g.add_argument("--kernel", help="a regex over the demangled kernel names")
    g.add_argument("--static", action="store_true",
                   help="the instructions as compiled, by role, loop and phase: no counters, and with --arch no device")
    g.add_argument("--arch", help="compile (--static) or read (--packed) for this architecture in place of the device's")
    g.add_argument("--dump", type=Path, action="append",
                   help="a SourceCounters CSV a box made of --packed, `ledger_<kernel core>.csv`; repeat for several")
    g.add_argument("--packed", type=Path, help="the binary `lab pack` built that the dumps profiled")
    g.add_argument("--per", type=float, help="divide every count by this: the profiled launch's CTA-windows")
    g.add_argument("--top", type=int, default=16, help="the source lines shown, by their shared wavefronts")
    c2 = sub.add_parser("path", help="each clocked row's roles' waits and busy cycles: the role the window waits on")
    c2.add_argument("experiment")
    c2.add_argument("--row", help="a regex over the rows' names")
    c2.add_argument("--record", type=Path, help="the record to read (default: this card's in measure/lab/results)")
    t = sub.add_parser("trace", help="a traced row's timelines: each role's time by state, who released its waits, "
                                     "and the window's path walked back through those releases")
    t.add_argument("experiment")
    t.add_argument("--row", help="a regex over the traced rows' names")
    t.add_argument("--record", type=Path, help="the record whose traces to read (default: this card's)")
    t.add_argument("--skip", type=int, default=3, help="the first windows left out of the steady span")
    t.add_argument("--html", type=Path, help="write three windows of each row's first CTA here, a lane a warp")
    pg = sub.add_parser("page", help="regenerate an experiment's page's measured section from its record")
    pg.add_argument("experiment")
    pg.add_argument("--instance", required=True, help="the rented box's label, which words the page's card line")
    m = sub.add_parser("ledger-dumps", help="on a box without the toolkit: profile a pack's kernels for a carried ledger")
    m.add_argument("experiment")
    m.add_argument("--from", dest="source", required=True, type=Path)
    m.add_argument("--into", required=True, type=Path)
    m.add_argument("--kernel", action="append", required=True, help="a regex over the demangled kernel names; repeat")
    m.add_argument("--skip", type=int, default=0, help="each kernel's launches passed before the one profiled")
    n = sub.add_parser("ncu")
    n.add_argument("experiment")
    n.add_argument("--metric", action="append", required=True, help="an ncu metric; repeat for several")
    n.add_argument("--record", action="store_true", help="write the counters into measure/lab/results")
    args = ap.parse_args()

    if args.cmd == "list":
        for name, path in experiments().items():
            print(f"{name}: {question(path)}")
        return 0
    if args.experiment not in experiments():
        print(f"lab: no experiment {args.experiment!r}; `list` names them")
        return 1
    if args.cmd == "pack":
        return pack(args.experiment, args.arch, args.into)
    if args.cmd == "waits":
        print(*waits_report(build(experiments()[args.experiment], args.arch)), sep="\n")
        return 0
    if args.cmd == "steps":
        print(*steps_report(build(experiments()[args.experiment], args.arch)), sep="\n")
        return 0
    if args.cmd == "carried":
        return carried(args.experiment, args.source, args.into, args)
    if args.cmd == "sass":
        return show_sass(args.experiment, args.kernel, args.full, args.arch)
    if args.cmd == "ncu":
        return counters(args.experiment, args.metric, args.record)
    if args.cmd == "path":
        return critical_path(args.experiment, args.row, args.record)
    if args.cmd == "trace":
        return timelines(args.experiment, args.row, args.record, args.skip, args.html)
    if args.cmd == "page":
        return page(args.experiment, args.instance)
    if args.cmd == "ledger-dumps":
        return ledger_dumps(args.experiment, args.source, args.into, args.kernel, args.skip)
    if args.cmd == "ledger":
        if args.static:
            return static_ledger(args.experiment, args.kernel, args.arch)
        return ledger(args.experiment, args.kernel, args.dump, args.packed, args.arch, args.per, args.top)
    return run(args.experiment, args)


if __name__ == "__main__":
    sys.exit(main())
