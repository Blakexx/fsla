#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name]
[--max-per-hmma N]`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass):
  local memory  -- the CHECKER's `stack_memory` (warpline's, `tools/checked.py`): a nonzero stack frame (`cuobjdump -res-usage`),
                  the LDL/STL instructions named beside it
  outlined collectives -- the checker's `convergence_call`: a `CALL.REL.NOINC` into a convergence subroutine that a CONVERGED
                  warp executes. ptxas guards the ones it emits: a `BRA.DIV` into a fallback block, or a `BRA.CONV` past the
                  call, so a converged warp runs the inline collective; a guarded call is cold code, counted (`guarded`) and
                  not red. A call to an arithmetic slow path is not one, nor a long region ptxas reconverges once on entry
  The step signatures are the checker's too (its model reads `mma.sync`'s `HMMA`/`IMMA`/`DMMA` and the warpgroup's `HGMMA`
  family alike as steps; docs/internals/tools/sass_gate.md#checker):
  predicated HMMA -- the checker's `step_predicated`: if-converted tensor work
  instructions per HMMA -- `step_density`, a ceiling the arm states (`--max-per-hmma`; issue-bound above it)
  branches in a step -- `step_branches` (KERNEL_STANDARDS section 27)
  the conversion unit in a step -- `step_conversion`: F2I/I2F/F2F/MUFU inside a loop of steps: 16 lanes a cycle an SM on sm_86 (its sm_90a re-check is owed) where
                  the FMA pipe's forms (a magic-constant FFMA, a PRMT) run at 64 (KERNEL_STANDARDS section 29)
  global memory in a step -- `step_global`: LDG/STG/RED/ATOMG or a bulk copy or reduction out inside a loop of steps: a
                  step's operands are staged ahead (`cp.async`'s LDGSTS, hopper's bulk copy in: allowed), and a
                  reduction to global memory belongs to a drain outside the steps
  The gate's own: each function's instruction count (`instr`, padding included) and the gap's two figures (`step_gap` holds the larger)
On a whole binary it also reds a PTX image (closed-world rule 2: cubins only), and on a whole extension every kernel
the source defines (`tools/kernels.py`) that no cubin carries: a gate that cannot see a kernel has not gated it. An instrument's library (`--instrument`, bazel/defs.bzl's `rola_instrument`)
carries its own kernels and owes none of the source's. Its kernels are PROBES, and a probe that keeps a step
signature on purpose (a global reduction between its HMMAs, the cost it measures) declares it in the instrument's
DECLARATIONS, by function (`sm_90a <pattern>` where one architecture's SASS keeps it alone), with that cost; an
undeclared signature is red, and so is a declaration an architecture the binary carries does not show. It prints the
counts it reads, the same way `ratify` prints registers and
spills, so a clean run is a line of zeros and a red names the signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import checked  # noqa: E402 -- path insert must precede these imports
import kernels  # noqa: E402
import sass  # noqa: E402

#: an instruction line's opcode, off the shared grammar (`tools/sass.py`)
INSTR = sass.INSTRUCTION
#: the always-false predicate: ptxas keeps an instruction under it that never runs
_NEVER = "@!PT"

#: the checker's rules the gate holds every function to, and what a red of each says
RED_RULES = {
    "stack_memory": "stack frame ({n} B; {ldl} LDL/STL)",
    "convergence_call": "convergence subroutines a converged warp executes ({n})",
    "step_predicated": "predicated HMMA ({n})",
    "step_branches": "branches inside an MMA step ({n})",
    "step_conversion": "the conversion unit inside an MMA step ({n})",
    "step_global": "global memory inside an MMA step ({n})",
    "step_density": "static instructions per HMMA {per:.1f} > {n}",
}
#: the step signatures an instrument's declarations name (`step_xu`, ...), and the checker's rule each is
KEPT_RULES = {"step_branches": "step_branches", "step_xu": "step_conversion", "step_global": "step_global"}
#: what a declaration of the instrument's permits: the probe keeps the signature whatever its count, so the allowance is no bound
_PERMIT = 10**9


def cubins_of(path: Path, member: str | None) -> list[Path]:
    return [image for image in sass.cubins(path) if member is None or member in image.name]


#: the last `gate` call's signature per function, for `--json`
STATS: dict[str, dict] = {}


def convergence_calls(kernel) -> tuple[int, int]:
    """``(hot, guarded)``: the calls into a convergence subroutine a converged warp executes (the checker's rule
    `convergence_call`), and those it never does (in a block the model finds cold: a `BRA.DIV` fallback, past a
    `BRA.CONV`, dead code). ``kernel`` is the checker's model of the function."""
    calls = [c for c in kernel.calls if c.subroutine]
    return sum(c.hot for c in calls), sum(not c.hot for c in calls)


def gap_split(kernel) -> tuple[int, int]:
    """``(inside, around)``: the figure the checker's `step_gap` takes the larger of, per loop of steps -- the longest run without a
    step between two steps of a trip, and the run around the back edge. The rule holds the one figure; the report keeps the two."""
    inside = around = 0
    for region in (r for r in kernel.regions if r.kind == "loop"):
        steps = {t.instruction.address for t in region.tensor if not t.null}
        at = [k for k, i in enumerate(region.instructions) if i.address in steps]
        inside = max(inside, *(b - a - 1 for a, b in zip(at, at[1:])), 0)
        around = max(around, len(region.instructions) - 1 - (at[-1] - at[0]) if at else 0)
    return inside, around


def instructions(cubin: Path) -> dict[str, int]:
    """Each function's instruction count off the listing, padding to the next 128-byte boundary included (the checker's model drops
    it); the instructions under `@!PT`, which never run, are not counted."""
    counts: dict[str, int] = {}
    cur = None
    for line in sass.disassemble(cubin).splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            counts[cur] = 0
            continue
        m = INSTR.match(line)
        if m and cur is not None and (m.group(2) or "").strip() != _NEVER:
            counts[cur] += 1
    return counts


def gate(cubin: Path, kernel: str | None, max_per_hmma: int | None) -> tuple[bool, str]:
    global ARCH
    here = ARCH = sass.machine(cubin)
    c = checked.checker()
    registry = c.rules.registry()
    counts = instructions(cubin)
    modelled = {k.name: k for k in checked.kernels_of(cubin)}
    held = [modelled[name] for name in counts if not kernel or re.search(kernel, name)]
    #: an instrument's declarations as the checker's: each function the key's pattern names, on the architectures the key
    #: says, is permitted what the key declares; the arm's ceiling is the density rule's bound on every function
    entries, declared = [], {}
    for k in held:
        for key, lints in INSTRUMENT.items():
            if _declares(key, k.name, here):
                for lint, cost in lints.items():
                    entries.append({"kernel": "^" + re.escape(k.demangled) + "$", "rule": KEPT_RULES[lint], "allow": _PERMIT,
                                    "reason": cost})
                    declared.setdefault(k.name, []).append((key, lint))
    if max_per_hmma is not None:
        entries.append({"kernel": "", "rule": "step_density", "allow": max_per_hmma, "reason": "the arm's stated ceiling"})
    refused: dict[str, dict[str, int]] = {}
    for r in checked.check(held, RED_RULES, checked.declarations(entries), {}):
        refused.setdefault(r.kernel, {})[r.rule] = r.count

    ok = True
    report = []
    STATS.clear()
    for k in held:
        found = {name: registry[name].count(k) for name in ("step_predicated", "step_branches", "step_conversion",
                                                           "step_global", "step_density")}
        hmma = sum(not s.null and not s.instruction.dead for s in k.steps)
        per = found["step_density"] if hmma else float("nan")
        ldl = sum(a.space == "local" for a in k.accesses)
        calls, guarded = convergence_calls(k)
        gap, around = gap_split(k)
        said = {"stack_memory": k.stack_bytes, "convergence_call": calls, "step_density": max_per_hmma}
        reds = [RED_RULES[rule].format(n=said.get(rule, refused[k.demangled][rule]), ldl=ldl, per=per)
                for rule in RED_RULES if rule in refused.get(k.demangled, {})]
        ok &= not reds
        exhibited = {"step_branches": found["step_branches"], "step_xu": found["step_conversion"],
                     "step_global": found["step_global"]}
        n = counts[k.name]
        STATS[k.name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                         "stack": k.stack_bytes, "calls": calls, "guarded": guarded, "pred_hmma": found["step_predicated"],
                         "mma_loop_branches": found["step_branches"], "mma_gap": gap, "mma_gap_around": around,
                         "step_xu": found["step_conversion"], "step_global": found["step_global"], "registers": k.registers,
                         "reds": reds,
                         "kept": sorted([key, lint] for key, lint in declared.get(k.name, []) if exhibited[lint])}
        report.append(f"{'RED ' if reds else 'ok  '} {k.name[:70]}: regs {k.registers} instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {k.stack_bytes} calls {calls} guarded {guarded} predHMMA {found['step_predicated']} "
                      f"mma-loop-branches {found['step_branches']} mma-gap {gap} around {around} "
                      f"step-xu {found['step_conversion']} step-global {found['step_global']}"
                      + (" :: " + "; ".join(reds) if reds else ""))
        if LOOPS:
            report.extend(f"       {rule}: {f.what}" for rule in ("step_branches", "step_conversion", "step_global")
                          for f in registry[rule].check(k))
    return ok, "\n".join(report)


#: `--loops`: each function's report is followed by the checker's findings of its step rules
LOOPS = False
#: `--instrument`: an instrument's declarations, `{name pattern: {step lint: the cost the probe measures}}` (the module
#: docstring); empty for the extension, whose functions keep no step signature
INSTRUMENT: dict[str, dict[str, str]] = {}
#: the architecture of the cubin `gate` read last, which a cached verdict carries with it
ARCH = ""


def _declared(key: str) -> tuple[str | None, str]:
    """A declaration's architecture and pattern: `sm_90a <pattern>` keeps its signature on that architecture's SASS
    alone (ptxas's output is an architecture's), a bare pattern on every one."""
    arch, _, pattern = key.partition(" ")
    return (arch, pattern) if pattern and re.fullmatch(r"sm_\d+a?", arch) else (None, key)


def _declares(key: str, function: str, arch: str | None) -> bool:
    on, pattern = _declared(key)
    return (on is None or on == arch) and re.search(pattern, function) is not None


def missing_kernels(seen: set[str]) -> list[str]:
    """The source's kernels (`tools/kernels.py`) whose name no gated function carries."""
    names = set(kernels.demangled_names(sorted(seen)))
    return [f"{k.where()}: `{k.name}` ({k.family}) has no SASS in this binary" for k in kernels.kernels()
            if k.name not in names]


#: THE VERDICT CACHE: a cubin's verdict, report and statistics under the key of everything that decides them -- the
#: cubin's bytes, the gate's own source closure (every module of `tools/` it has loaded, and the checker's), the
#: disassemblers' identity and the arguments -- so an unchanged cubin's verdict is the reading of identical bytes under
#: an identical gate, replayed. `--no-cache` reads every cubin.
#: -- see docs/internals/tools/sass_gate.md#cache
_CLOSURE: str | None = None


def _closure() -> str:
    """The gate's source and toolchain identity, once a process: every loaded module under `tools/`, the checker's own
    source, `nvdisasm` and `cuobjdump` by their version lines."""
    global _CLOSURE
    if _CLOSURE is None:
        tools = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for path in sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                            if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(tools)}):
            h.update(str(path.relative_to(tools)).encode() + b"\0" + path.read_bytes())
        package = checked.checker().root / "warpline"
        for path in sorted(package.rglob("*.py")):
            h.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
        for tool in ("nvdisasm", "cuobjdump"):
            h.update(subprocess.run([sass.tool(tool), "--version"], capture_output=True, text=True,
                                    check=True).stdout.encode())
        _CLOSURE = h.hexdigest()
    return _CLOSURE


def cached_gate(cubin: Path, kernel: str | None, max_per_hmma: int | None, use: bool) -> tuple[bool, str]:
    """`gate`, replayed from the cache when the key matches; a miss reads the cubin and records the verdict."""
    if not use:
        return gate(cubin, kernel, max_per_hmma)
    key = hashlib.sha256(json.dumps([_closure(), hashlib.sha256(cubin.read_bytes()).hexdigest(), kernel,
                                     max_per_hmma, LOOPS, sorted(INSTRUMENT.items())]).encode()).hexdigest()
    entry = sass.scratch("sass_gate_verdicts") / f"{key}.json"
    global ARCH
    if entry.exists():
        doc = json.loads(entry.read_text())
        STATS.clear()
        STATS.update(doc["stats"])
        ARCH = doc["arch"]
        return doc["ok"], doc["report"]
    ok, report = gate(cubin, kernel, max_per_hmma)
    tmp = entry.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"ok": ok, "report": report, "stats": STATS, "arch": ARCH}, sort_keys=True))
    tmp.replace(entry)
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel", default=None, help="narrow to the functions matching this regex (a measurement)")
    ap.add_argument("--member", default=None, help="narrow to the cubins whose name holds this (a measurement)")
    ap.add_argument("--instrument", type=Path, default=None, metavar="DECLARATIONS",
                    help="an instrument's library: no kernel of the source is owed, and its probes keep the step "
                         "signatures DECLARATIONS names (a JSON file, docs/internals/tools/sass_gate.md#instrument)")
    ap.add_argument("--max-per-hmma", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write every function's signature here, by cubin")
    ap.add_argument("--loops", action="store_true",
                    help="after each function, what the checker's step rules (branches, conversion unit, global memory) found in it")
    ap.add_argument("--no-cache", action="store_true",
                    help="read every cubin, the verdict cache neither replayed nor written (docs/internals/tools/"
                         "sass_gate.md#cache)")
    ap.add_argument("--cuobjdump", default=None, help="the graph's cuobjdump (with --nvdisasm: the graph's toolkit, "
                    "scratch of this run's own, no verdict cache)")
    ap.add_argument("--nvdisasm", default=None, help="the graph's nvdisasm")
    a = ap.parse_args()

    if (a.cuobjdump is None) != (a.nvdisasm is None):
        raise SystemExit("--cuobjdump and --nvdisasm name the graph's toolkit together")
    if a.cuobjdump:
        sass.use_graph(cuobjdump=a.cuobjdump, nvdisasm=a.nvdisasm)
        a.no_cache = True
    global LOOPS, INSTRUMENT
    LOOPS = a.loops
    INSTRUMENT = json.loads(a.instrument.read_text())["kernels"] if a.instrument else {}
    allok = True
    doc = {}
    seen: set[str] = set()
    guarded = 0
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = cached_gate(cub, a.kernel, a.max_per_hmma, not a.no_cache)
        guarded += sum(st["guarded"] for st in STATS.values())
        if rep:
            print(f"== {cub.name}\n{rep}")
        doc[cub.name] = {"ok": ok, "arch": ARCH, "functions": dict(STATS)}
        seen |= set(STATS)
        allok &= ok
    if not seen:
        print(f"RED: no function matched (--kernel {a.kernel!r}, --member {a.member!r}): a gate with no target does "
              "not pass")
        allok = False
    #: CLOSED-WORLD RULE 2 on the binary itself: no PTX image a driver could just-in-time compile for an architecture
    #: nobody measured
    if a.binary.suffix != ".cubin":
        listed = subprocess.run([sass.tool("cuobjdump"), "--list-ptx", str(a.binary)], capture_output=True, text=True,
                                check=True).stdout
        for line in (line for line in listed.splitlines() if line.startswith("PTX file")):
            print(f"RED  rule 2: {line.strip()} -- a binary carries cubins alone (docs/build.md#what-the-build-checks)")
            allok = False
    if a.binary.suffix != ".cubin" and a.kernel is None and a.member is None and not a.instrument:
        for line in missing_kernels(seen):
            print(f"RED  {line}")
            allok = False
    #: a declaration an architecture does not show keeps nothing there: red, so the declarations only say what is so
    #: -- a plain one on every architecture the binary carries, an `sm_XX` one on its own (an iteration build of one
    #: card reads none of another's)
    if a.instrument and a.kernel is None and a.member is None:
        used = {(cubin["arch"], *k) for cubin in doc.values() for st in cubin["functions"].values() for k in st["kept"]}
        carried = sorted({cubin["arch"] for cubin in doc.values()})
        for key, lints in sorted(INSTRUMENT.items()):
            on = _declared(key)[0]
            for arch in [on] if on else carried:
                for lint in sorted(lints):
                    if arch in carried and (arch, key, lint) not in used:
                        hint = "" if on else " (`sm_XX <pattern>` declares a signature one architecture keeps)"
                        print(f"RED  {a.instrument}: `{key}` keeps {lint}, which no function of {arch}'s SASS shows"
                              + hint)
                        allok = False
    if a.json:
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    print(f"guarded convergence fallbacks (cold code, not red): {guarded}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
