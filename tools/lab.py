# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB: small CUDA experiments whose measurements become kernel rules (docs/internals/tools/lab.md).

An experiment is one `measure/lab/experiments/<name>.cu` built on `measure/lab/lab.cuh`. This driver compiles it with
the pinned toolkit for the running card, reads its SASS and refuses to measure unless every expectation the experiment
declares holds (an instruction the experiment means to time is really there -- ptxas deletes work whose result is dead)
and no kernel touches local memory; then it checks the GPU is quiet, runs the repetitions, flags a result whose samples
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
import hashlib
import importlib.util
import json
import pipes
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import build_flags
import dev_config
import lab_trace
import region_ledger
import sass
import sass_gate

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
                           "-lineinfo", f"-I{LAB}", f"-I{KERNELS}", f"-I{dev_config.warpline_include()}", "--expt-relaxed-constexpr", "-o", str(out),
                           str(path)], capture_output=True, text=True, check=False)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    done.check_returncode()
    #: the toolkit's notices of this build, beside its binary: `serialized` reads them
    notices(out).write_text(done.stderr)
    return out


def notices(binary: Path) -> Path:
    return binary.with_name(binary.name + ".ptxas")


def serialized(binary: Path) -> dict[str, int]:
    """Per kernel (demangled): the toolkit's notices that it SERIALIZED the kernel's asynchronous MMAs for want of
    registers (ptxas C7512) -- each then waits out the one before, and the kernel does not measure the steps its
    source states. Nothing where the build left no notices (a packed binary run elsewhere)."""
    if not notices(binary).exists():
        return {}
    found = Counter(re.findall(r"\(C7512\).*?function '([^']+)'", notices(binary).read_text()))
    names = demangle(list(found))
    return {names[k]: n for k, n in found.items()}


def demangle(names: list[str]) -> dict[str, str]:
    out = subprocess.run([dev_config.cuda_bin("cu++filt")], input="\n".join(names), capture_output=True, text=True,
                         check=True).stdout.splitlines()
    return dict(zip(names, out, strict=True))


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
    names = demangle(list(ops))
    return {names[k]: {"ops": dict(v), **regs.get(k, {})} for k, v in ops.items()}


def count(ops: dict[str, int], prefixes: str) -> int:
    """The instructions whose opcode starts with one of `prefixes` (`|`-separated alternatives: ptxas may spell one
    operation two ways, an integer add as `IADD3` or as `IMAD.IADD` on the FMA pipe)."""
    return sum(n for op, n in ops.items() if any(op.startswith(p) for p in prefixes.split("|")))


def signatures(binary: Path) -> dict[str, dict]:
    """Per kernel (demangled): the SASS gate's signature (`tools/sass_gate.py`) -- local memory, and the calls into a
    convergence subroutine a converged warp executes (not ptxas's loop-exit `CALL.REL.NOINC`, nor a guarded fallback)."""
    found: dict[str, dict] = {}
    for cubin in sass.cubins(binary):
        sass_gate.gate(cubin, None, None)
        found |= sass_gate.STATS
    names = demangle(list(found))
    return {names[k]: v for k, v in found.items()}


#: KERNEL_STANDARDS section 29's debts: the experiments recorded before a lab had to budget its MMA kernels and check
#: its outputs; each is held to both when it is next changed, and the list only shrinks
_PRE_29 = {"collective_branch", "facts_block", "mma_k8", "row_span", "tensor_tax", "vote_stash"}


#: THE LEARNINGS' DEBTS (docs/internals/LEARNINGS.md): the experiments written before an experiment's header had to
#: state its TIMELINE and every kernel's loops were held to the hazards below; each is held to them when it is next
#: changed, and the list only shrinks
_PRE_TIMELINE = {"bulk_drain", "byte_expand", "cluster_sum", "collective_branch", "facts_block", "facts_two",
                 "issue_latency", "reg_trade", "row_ring", "row_span", "state_move", "tensor_tax",
                 "vote_stash", "wgmma_rate", "wgmma_regs"}
#: the hazards an experiment may declare a count of (`Bench::expect(kernel, NAME, "<=", n)`), beside its opcodes
HAZARDS = ("LOOP_CONVERGENCE", "LOAD_IN_TRIP", "STORE_RELOAD", "STORE_UNFENCED", "STEPS_SERIALIZED", "CLOCK_UNIFORM")


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


#: a branch's target in `cuobjdump -sass` text (function-relative)
BRANCH = re.compile(r"\bBRA(?:\.[A-Z.]+)?\s+(?:!?U?P[T\d]+,\s*)?(?:~?URZ,\s*)?(0x[0-9a-f]+)")


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


def loops_of(body: list[tuple[int, str, str]]) -> list[tuple[int, int]]:
    """A kernel's loops: the spans a backward branch closes."""
    return [(int(t.group(1), 16), at) for at, op, line in body
            if op.startswith("BRA") and (t := BRANCH.search(line)) and int(t.group(1), 16) < at]


def loop_convergence(binary: Path) -> dict[str, int]:
    """Per kernel (demangled): the convergence instructions (`BSSY`, `BSYNC`, `BRA.DIV`) inside its loops -- the spans a
    backward branch closes -- which KERNEL_STANDARDS section 29.13 has removed by the source patterns of 29.9 or
    declared (`LOOP_CONVERGENCE`)."""
    found = {}
    for name, body in disassembled(binary).items():
        loops = loops_of(body)
        found[name] = sum(1 for at, op, _ in body if op.startswith(("BSSY", "BSYNC", "BRA.DIV"))
                          and any(lo <= at <= hi for lo, hi in loops))
    names = demangle(list(found))
    return {names[k]: v for k, v in found.items()}


#: the warpgroup steps among the opcode table's MMAs (`tools/sass.py`): the asynchronous ones, float and integer
WGMMA = tuple(op for op in sass.OPCODE_CLASSES["mma"] if op.endswith("GMMA"))

#: an instruction's operands, and the registers and the one bracketed address among them
OPERANDS = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?[A-Z0-9_.$]+\s+(.*?)\s*;")
REGISTER = re.compile(r"\bR(\d+)\b")
ADDRESS = re.compile(r"\[[^\]]*\]")
#: the opcodes whose first operand is read, not written
READS_FIRST = ("ST", "BRA", "RED", "ATOM", "BSSY", "BSYNC", *WGMMA, "ISETP", "FSETP", "CALL", "EXIT", "NOP", "BAR")


def unfenced(body: list[tuple[int, str, str]]) -> int:
    """STORE_UNFENCED: in a kernel that holds a wgmma, the barrier arrivals that follow a shared store with no
    `fence.proxy.async` between them -- a step reads shared memory through the tensor pipe's own view, and a
    writer's ordinary stores reach it only past the writer's fence (KERNEL_STANDARDS section 12)."""
    if not any(op.startswith(WGMMA) for _, op, _ in body):
        return 0
    found, stored = 0, False
    for _, op, _ in body:
        if op.startswith("STS"):
            stored = True
        elif op.startswith("FENCE.VIEW.ASYNC"):
            stored = False
        elif op.startswith("SYNCS.ARRIVE"):
            found += stored
            stored = False
    return found


def loop_hazards(binary: Path) -> dict[str, dict[str, int]]:
    """Per kernel (demangled), inside its innermost loops: LOAD_IN_TRIP, the global loads whose value an instruction
    of the same trip reads -- that trip waits out its own load, where a load a stage ahead is hidden (KERNEL_STANDARDS
    section 32); and STORE_RELOAD, the shared loads of an address the same trip stored to -- a value kept in shared
    memory that no other warp reads belongs in a register (section 33). And over the whole kernel, STORE_UNFENCED
    (`unfenced`)."""
    found = {}
    for name, body in disassembled(binary).items():
        loops = loops_of(body)
        inner = [(lo, hi) for lo, hi in loops if not any((a, b) != (lo, hi) and lo <= a and b <= hi for a, b in loops)]
        in_trip = reloads = 0
        for lo, hi in inner:
            trip = [(op, m.group(1)) for at, op, line in body if lo <= at <= hi and (m := OPERANDS.match(line))]
            for i, (op, operands) in enumerate(trip):
                registers = REGISTER.findall(operands)
                if op.startswith("LDG") and not op.startswith("LDGSTS") and registers:
                    value = registers[0]
                    for later, text in trip[i + 1:]:
                        named = REGISTER.findall(text)
                        written = named[:1] if named and not later.startswith(READS_FIRST) else []
                        if value in (named[1:] if written else named):
                            in_trip += 1
                            break
                        if value in written:
                            break
                if op.startswith("STS") and (address := ADDRESS.search(operands)):
                    base = REGISTER.findall(address.group(0))
                    for later, text in trip[i + 1:]:
                        named = REGISTER.findall(text)
                        if later.startswith("LDS") and address.group(0) in text:
                            reloads += 1
                            break
                        if named and not later.startswith(READS_FIRST) and named[0] in base:
                            break
        found[name] = {"LOAD_IN_TRIP": in_trip, "STORE_RELOAD": reloads, "STORE_UNFENCED": unfenced(body)}
    names = demangle(list(found))
    return {names[k]: v for k, v in found.items()}


#: a uniform register, and the reading of the cycle counter into one
UNIFORM = re.compile(r"\bUR(\d+)\b")
CLOCK_UNIFORM = re.compile(r"\bS2UR\s+UR(\d+),\s*SR_CLOCKLO\b")


def clock_summed_uniform(body: list[tuple[int, str, str]]) -> int:
    """The uniform sums (`UIADD3`) that read a uniform register holding a cycle-counter reading: a phase clock
    accumulated in a warp's uniform registers is not a thread's count (docs/internals/LEARNINGS.md,
    lap-in-uniform). A straight-line scan: a register stops holding a reading where it is next written."""
    found, holding = 0, set()
    for _, op, line in body:
        if m := CLOCK_UNIFORM.search(line):
            holding.add(m.group(1))
            continue
        if not (ops := OPERANDS.match(line)):
            continue
        written, *read = [o.strip() for o in ops.group(1).split(",")]
        if op.startswith("UIADD3") and any(r in holding for o in read for r in UNIFORM.findall(o)):
            found += 1
        if (w := UNIFORM.fullmatch(written)) and op.startswith(("U", "R2UR", "S2UR")):
            holding.discard(w.group(1))
    return found


def clocks_summed_uniform(binary: Path) -> dict[str, int]:
    """Per kernel (demangled): `clock_summed_uniform`."""
    found = {name: clock_summed_uniform(body) for name, body in disassembled(binary).items()}
    names = demangle(list(found))
    return {names[k]: v for k, v in found.items()}


def check(binary: Path, table: dict[str, dict]) -> list[str]:
    """Every expectation the experiment declares, no local memory and no undeclared convergence call anywhere: the
    failures, empty when it may run."""
    lines = subprocess.run([str(binary), "--expect"], capture_output=True, text=True, check=True).stdout
    expected = [e for e in map(json.loads, lines.splitlines()) if e["lab"] == "expect"]
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
    declared_calls = {k for e in expected for k in table if re.search(e["kernel"], k) and "CALL" in e["op"]}
    declared_branches = {k for e in expected for k in table if re.search(e["kernel"], k) and e["op"] == "BSSY"}

    declared_loops = {k: e["count"] for e in expected if e["op"] == "LOOP_CONVERGENCE"
                      for k in table if re.search(e["kernel"], k)}
    budgeted = {k for b in budgets(binary) for k in table if re.search(b["kernel"], k)}
    #: an experiment past the learnings' debts is held to every hazard in every kernel, budgeted or not
    experiment = binary.name.split(".")[0]
    held = experiment not in _PRE_TIMELINE
    for k, n in loop_convergence(binary).items():
        if (k in budgeted or held) and n > declared_loops.get(k, 0):
            failures.append(f"{k}: {n} convergence instructions inside its loops, {declared_loops.get(k, 0)} "
                            "declared (KERNEL_STANDARDS 29.13: the source patterns of 29.9, or LOOP_CONVERGENCE)")
    if held:
        if not timeline(experiments()[experiment]):
            failures.append(f"{experiment}: its header states no TIMELINE -- each phase's stream and which load is a "
                            "stage ahead of which use, written before the kernel (docs/internals/LEARNINGS.md)")
        for k, n in clocks_summed_uniform(binary).items():
            declared = max((e["count"] for e in expected if e["op"] == "CLOCK_UNIFORM" and re.search(e["kernel"], k)),
                           default=0)
            if n > declared:
                failures.append(f"{k}: {n} sums of a cycle-counter reading in uniform registers, {declared} declared "
                                "(CLOCK_UNIFORM): a lap's sum is each thread's own (docs/internals/LEARNINGS.md, "
                                "lap-in-uniform)")
        for at in collective_selected(experiments()[experiment]):
            failures.append(f"{experiment}: line {at}: a warp collective is an operand of a conditional -- every "
                            "lane takes it, and the select follows (docs/internals/LEARNINGS.md)")
        why = {"LOAD_IN_TRIP": "global loads read in the trip that issued them (KERNEL_STANDARDS 32: a load a stage "
                               "ahead of its use)",
               "STORE_RELOAD": "shared loads of an address the same trip stored to (KERNEL_STANDARDS 33: a value no "
                               "other warp reads stays in a register)",
               "STORE_UNFENCED": "barrier arrivals after a shared store with no fence between, in a kernel whose "
                                 "steps read shared memory (KERNEL_STANDARDS 12: the writer's fence)"}
        for k, hazards in loop_hazards(binary).items():
            for kind, n in hazards.items():
                declared = max((e["count"] for e in expected if e["op"] == kind and re.search(e["kernel"], k)), default=0)
                if n > declared:
                    failures.append(f"{k}: {n} {why[kind]}, {declared} declared ({kind})")
        for k, n in serialized(binary).items():
            declared = max((e["count"] for e in expected if e["op"] == "STEPS_SERIALIZED" and re.search(e["kernel"], k)),
                           default=0)
            if n > declared:
                failures.append(f"{k}: the toolkit serialized its asynchronous MMAs for want of registers (ptxas "
                                f"C7512), {declared} declared (STEPS_SERIALIZED): a step's operand registers are "
                                "written before its commit group and the role holds the count they need "
                                "(docs/internals/LEARNINGS.md)")

    if binary.name.split(".")[0] not in _PRE_29:
        stated = budgets(binary) + budgets(binary, "exempt")
        for k, info in table.items():
            if count(info["ops"], "|".join(WGMMA)) and any(re.search(b["kernel"], k) for b in budgets(binary)):
                failures.append(f"{k}: a budgeted wgmma kernel -- its per-MMA divisor is read from counters the "
                                "hopper model does not name yet (owed with that model): state `Bench::exempt`")
            if count(info["ops"], "|".join(sass.OPCODE_CLASSES["mma"])) and not any(re.search(b["kernel"], k) for b in stated):
                failures.append(f"{k}: an MMA kernel with no instruction budget (KERNEL_STANDARDS 29: `Bench::budget` "
                                "-- `HMMA` and `IMMA` kernels -- or `Bench::exempt` with its reason)")
    declared = {op: {k for e in expected for k in table if re.search(e["kernel"], k) and e["op"] == op}
                for op in ("STEP_XU", "STEP_GLOBAL")}

    for k, sig in signatures(binary).items():
        if sig["ldl"] or sig["stack"]:
            failures.append(f"{k}: local memory ({sig['ldl']} LDL/STL, a {sig['stack']}-byte stack frame)")
        if sig["mma_loop_branches"] and k not in declared_branches:
            failures.append(f"{k}: {sig['mma_loop_branches']} branches inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 27: a step body is one basic block; a form that keeps them declares BSSY)")
        if sig["step_xu"] and k not in declared["STEP_XU"]:
            failures.append(f"{k}: {sig['step_xu']} conversion-unit instructions inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 29: convert on the FMA pipe; a form that keeps them declares STEP_XU)")
        if sig["step_global"] and k not in declared["STEP_GLOBAL"]:
            failures.append(f"{k}: {sig['step_global']} global accesses inside an MMA step nobody declared "
                            "(KERNEL_STANDARDS 29: stage by cp.async; a form that keeps them declares STEP_GLOBAL)")
        if sig["calls"] and k not in declared_calls:
            failures.append(f"{k}: {sig['calls']} convergence calls nobody declared (a collective under "
                            "role-dependent control flow becomes a call into a convergence subroutine)")
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
    #: THE TRACES (a traced row's timelines, `lab trace`), beside the record: one line a row, the commit's
    traces = target.with_name(target.name.replace(".json", ".trace.json"))
    if device.get("_traces"):
        traces.write_text(json.dumps({"experiment": name, "git": built["git"], "rows": device["_traces"]}) + "\n")
    elif traces.exists():
        traces.unlink()
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
        ids = re.findall(r"\d{4,}", k.split("(")[0] if "<" not in k else k[:k.index(">") + 1])
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

    images = [c for c in sass.cubins(binary) if arch in c.name]
    if len(images) != 1:
        print(f"lab: {len(images)} images for {arch} in {binary.name}")
        return 1
    mangled = dict(zip(demangle(list(kernels_mangled(binary))).values(), kernels_mangled(binary), strict=True))
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
    return 0


#: the share of the window by which the least-waiting role must wait less than the next for the row to name a path
PATH_MARGIN = 0.05


def is_wait(site: str) -> bool:
    return " waits on " in site or "rendezvous" in site or "fence" in site


def timelines(name: str, pattern: str | None, record: Path | None, skip: int, out: Path | None) -> int:
    """A RECORD'S TRACED ROWS read by `lab_trace`: its traces beside it, its windows' cycles from it."""
    path = record or (RESULTS / name / f"{sass.device_arch()}.json")
    traces = path.with_name(path.name.replace(".json", ".trace.json"))
    if not traces.exists():
        print(f"lab: no traces beside {path} (a record's traced rows write {traces.name})")
        return 1
    windows = {r["name"]: r["median"] for r in json.loads(path.read_text())["results"]
               if r.get("unit") == "cycles a window a CTA"}
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
    """A kernel's instructions AS COMPILED, with no device: each innermost loop's body by the role that holds it (the
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
    mangled = dict(zip(demangle(list(kernels_mangled(binary))).values(), kernels_mangled(binary), strict=True))
    disassembly = sass.disassemble(images[0], "--print-line-info-inline", "-gi")
    bodies = disassembled(binary)
    #: a kernel the lab would refuse is counted all the same, its refusals stated first
    for failure in check(binary, kernels(binary)):
        print(f"  REFUSED AS IT STANDS: {failure}")

    for k, symbol in mangled.items():
        if pattern and not re.search(pattern, k):
            continue
        text = own_section(disassembly, symbol)
        frames = sass.frames(text)
        loops = loops_of(bodies[symbol])
        by: dict[tuple, Counter] = {}
        for line in text.splitlines():
            if not (m := sass.INSTRUCTION.match(line)):
                continue
            at = int(m.group(1), 16)
            inside = [loop for loop in loops if loop[0] <= at <= loop[1]]
            loop = min(inside, key=lambda span: span[1] - span[0]) if inside else None
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
    sigs = signatures(binary)
    for k, info in table.items():
        if pattern and not re.search(pattern, k):
            continue
        top = ", ".join(f"{op} {n}" for op, n in Counter(info["ops"]).most_common(10))
        sig = sigs.get(k, {})
        gap = f", MMA gap {sig['mma_gap']} (around the loop {sig['mma_gap_around']})" if sig.get("mma_gap_around") else ""
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
