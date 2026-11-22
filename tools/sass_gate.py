"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name] [--declare FILE]
[--notices LOG]...`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass): the CHECKER's facts
(warpline's, `tools/checked.py`) of the rules below, and the build's ptxas notices read against the project's ledger
(`tools/notice_ledger.json`), compared with the declarations the build states beside its kernels (`--declare`, the file
bazel/defs.bzl writes from them): undeclared is zero, a declaration's `allow` the count its kernels may keep, and what the
comparison refuses is red, in the checker's own lines. The lab judges through the same function (`checked.compare`).
  local memory  -- `stack_memory`: a nonzero stack frame (`cuobjdump -res-usage`), the LDL/STL instructions named beside it
  outlined collectives -- `convergence_call`: a `CALL.REL.NOINC` into a convergence subroutine that a CONVERGED
                  warp executes. ptxas guards the ones it emits: a `BRA.DIV` into a fallback block, or a `BRA.CONV` past the
                  call, so a converged warp runs the inline collective; a guarded call is cold code, counted (`guarded`) and
                  not red. A call to an arithmetic slow path is not one, nor a long region ptxas reconverges once on entry
  predicated HMMA -- `step_predicated`: if-converted tensor work
  instructions per HMMA -- `step_density`, a bound the arm declares (issue-bound above it)
  branches in a step -- `step_branches` (KERNEL_STANDARDS section 27)
  the conversion unit in a step -- `step_conversion`: F2I/I2F/F2F/MUFU inside a loop of steps: the unit runs below the rate of
                  the FMA pipe's forms (a magic-constant FFMA, a PRMT) (KERNEL_STANDARDS section 29)
  global memory in a step -- `step_global`: LDG/STG/RED/ATOMG or a bulk copy or reduction out inside a loop of steps: a
                  step's operands are staged ahead (`cp.async`'s LDGSTS, hopper's bulk copy in: allowed), and a
                  reduction to global memory belongs to a drain outside the steps
  the figures, shown and never red until a declaration bounds them: `function_size` (a function's instructions, padding
                  excluded) and `step_gap` (the longest run without a step in a loop of steps)
  the build's notices -- each counted under its row's rule at the function it names (`register_spill`, `mma_serialized`, ...)
On a whole binary it also reds a PTX image (closed-world rule 2), and on a whole extension every kernel
the source defines (`tools/kernels.py`) that no cubin carries and every arm the declarations name (`arms`, each arm's exact kernel pattern)
that the binary does not carry once: a gate that cannot see a kernel has not gated it. A library of PROBES (`fsla_instrument`) owes none of the source's kernels; a
probe that keeps a signature on purpose (a global reduction between its HMMAs, the cost it measures) declares it, with that cost; a
declaration that allows a count its architecture's SASS never shows is red, so the declarations only say what is so. It prints the
counts it reads, the same way `ratify` prints registers and spills, so a clean run is a line of zeros and a red names the signature.
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

#: THE GATE APPLIES EVERY RULE OF THE CHECKER'S REGISTRY to every function (`rules()`): there is no subset. A rule a kernel
#: predates is declared beside the kernel with its count (`bazel/cuda.bzl`'s `declaration`), never left out of the gate
#: the model rules' counts the report and the JSON carry, by the statistic's name
COUNTED = {"pred_hmma": "step_predicated", "mma_loop_branches": "step_branches", "step_xu": "step_conversion", "step_global": "step_global",
           "instr": "function_size", "mma_gap": "step_gap"}


def rules() -> tuple[str, ...]:
    """Every model rule of the checker's registry, by name: the rules the gate holds every function to."""
    return tuple(sorted(checked.checker().rules.registry()))


def cubins_of(path: Path, member: str | None) -> list[Path]:
    return [image for image in sass.cubins(path) if member is None or member in image.name]


#: the last `gate` call's signature per function, for `--json`; and the functions it read (demangled name, the count of each rule), which
#: the aggregate checks of `main` read
STATS: dict[str, dict] = {}
PLACES: list[tuple[str, dict[str, int]]] = []


def convergence_calls(kernel) -> tuple[int, int]:
    """``(hot, guarded)``: the calls into a convergence subroutine a converged warp executes (the checker's rule
    `convergence_call`), and those it never does (in a block the model finds cold: a `BRA.DIV` fallback, past a
    `BRA.CONV`, dead code). ``kernel`` is the checker's model of the function."""
    calls = [c for c in kernel.calls if c.subroutine]
    return sum(c.hot for c in calls), sum(not c.hot for c in calls)


def read_notices(paths: list[Path]) -> list:
    """Every notice of the build's messages (the units' ptxas reports), in order."""
    return checked.checker().ledgers.read_notices("".join(path.read_text() + "\n" for path in paths))


def gate(cubin: Path, kernel: str | None, declarations: list[dict], notices: list = ()) -> tuple[bool, str]:
    """One cubin's verdict: the checker's facts of its functions under every rule of its registry (`rules()`), and of the `notices` that name them, compared with
    `declarations` (this architecture's entries: `kernel` a regex over the demangled name, `rule`, `allow`, `reason`)."""
    global ARCH
    ARCH = sass.machine(cubin)
    c = checked.checker()
    registry = c.rules.registry()
    modelled = checked.kernels_of(cubin)
    held = [k for k in modelled if not kernel or re.search(kernel, k.name)]
    mine = {k.name for k in held}
    comparison, facts = checked.compare(held, rules(), checked.declarations(declarations), {}, notices=[n for n in notices if n.function in mine])
    refused: dict[str, list] = {}
    for r in comparison.refusals:
        refused.setdefault(r.kernel, []).append(r)
    place = {p.name: p for p in facts.places if p.kernel}

    ok = True
    report = []
    STATS.clear()
    PLACES.clear()
    for k in held:
        count = {rule: found.count for rule, found in place[k.name].rules.items()}
        hmma = sum(not s.null and not s.instruction.dead for s in k.steps)
        per = count["step_density"] if hmma else float("nan")
        ldl = sum(a.space == "local" for a in k.accesses)
        calls, guarded = convergence_calls(k)
        reds = [f"{r.rule}: {r.count} found, {r.allowed} declared ({r.reason or 'undeclared'})" for r in refused.get(k.demangled, [])]
        ok &= not reds
        STATS[k.name] = {"hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl, "stack": k.stack_bytes, "calls": calls,
                         "guarded": guarded, "registers": k.registers, "reds": reds,
                         **{stat: count[rule] for stat, rule in COUNTED.items()}}
        PLACES.append((k.demangled, count))
        s = STATS[k.name]
        report.append(f"{'RED ' if reds else 'ok  '} {k.name[:70]}: regs {k.registers} instr {s['instr']} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {k.stack_bytes} calls {calls} guarded {guarded} predHMMA {s['pred_hmma']} "
                      f"mma-loop-branches {s['mma_loop_branches']} mma-gap {s['mma_gap']} "
                      f"step-xu {s['step_xu']} step-global {s['step_global']}"
                      + (" :: " + "; ".join(reds) if reds else ""))
        if reds:
            report.append(f"       kernel: {k.demangled}")
        report.extend("       " + line for r in refused.get(k.demangled, []) for line in r.lines()[1:])
        if LOOPS:
            report.extend(f"       {rule}: {f.what}" for rule in ("step_branches", "step_conversion", "step_global")
                          for f in registry[rule].check(k))
    return ok, "\n".join(report)


#: `--loops`: each function's report is followed by the checker's findings of its step rules
LOOPS = False
#: the architecture of the cubin `gate` read last, which a cached verdict carries with it
ARCH = ""


def missing_kernels(seen: set[str]) -> list[str]:
    """The source's kernels (`tools/kernels.py`) whose name no gated function carries."""
    names = set(kernels.demangled_names(sorted(seen)))
    return [f"{k.where()}: `{k.name}` ({k.family}) has no SASS in this binary" for k in kernels.kernels()
            if k.name not in names]


def unshown(spec: dict, places: list[tuple[str, dict[str, int]]], arch: str, source: str) -> list[str]:
    """The reds of an architecture's declarations over what its SASS shows (`places`: each function's demangled name and rule counts): a
    declaration that allows a count (not a bound, not a tightening) where no function it names has one, and an arm whose exact kernel
    pattern the binary does not carry once. The declarations say only what is so."""
    bound = {name for name, rule in checked.checker().rules.registry().items() if rule.bound}
    reds = []
    for d in spec.get("declarations", []):
        if d["allow"] == 0 or d["rule"] in bound:
            continue
        named = [counts for name, counts in places if re.search(d["kernel"], name)]
        if not any(counts.get(d["rule"], 0) for counts in named):
            reds.append(f"RED  {source}: `{d['kernel']}` declares {d['rule']} (allow {d['allow']}), which no function of {arch}'s SASS shows"
                        + ("" if named else " (it names no function there)"))
    for arm in spec.get("arms", []):
        found = sum(re.search(arm["kernel"], name) is not None for name, _ in places)
        if found != 1:
            reds.append(f"RED  {source}: arm {arm['family']}[{arm['arm']}] (`{arm['kernel']}`) matches {found} functions of {arch}'s SASS, not one")
    return reds


#: THE VERDICT CACHE: a cubin's verdict, report and statistics under the key of everything that decides them -- the
#: cubin's bytes, the gate's own source closure (every module of `tools/` it has loaded, and the checker's), the
#: disassemblers' identity, the ledger, the declarations and the notices, and the arguments -- so an unchanged cubin's verdict is the reading of identical bytes under an
#: identical gate, replayed. `--no-cache` reads every cubin.
#: -- see docs/internals/tools/sass_gate.md#cache
_CLOSURE: str | None = None


def _closure() -> str:
    """The gate's source and toolchain identity, once a process: every loaded module under `tools/`, the project's ledger, the checker's own
    source, `nvdisasm` and `cuobjdump` by their version lines."""
    global _CLOSURE
    if _CLOSURE is None:
        tools = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for path in sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                            if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(tools)}):
            h.update(str(path.relative_to(tools)).encode() + b"\0" + path.read_bytes())
        h.update(checked.PROJECT_LEDGER.read_bytes())
        package = checked.checker().root / "warpline"
        for path in sorted(package.rglob("*.py")):
            h.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
        for tool in ("nvdisasm", "cuobjdump"):
            h.update(subprocess.run([sass.tool(tool), "--version"], capture_output=True, text=True,
                                    check=True).stdout.encode())
        _CLOSURE = h.hexdigest()
    return _CLOSURE


def cached_gate(cubin: Path, kernel: str | None, declarations: list[dict], notices: list, use: bool) -> tuple[bool, str]:
    """`gate`, replayed from the cache when the key matches; a miss reads the cubin and records the verdict."""
    if not use:
        return gate(cubin, kernel, declarations, notices)
    key = hashlib.sha256(json.dumps([_closure(), hashlib.sha256(cubin.read_bytes()).hexdigest(), kernel, LOOPS, declarations,
                                     [n.raw for n in notices]]).encode()).hexdigest()
    entry = sass.scratch("sass_gate_verdicts") / f"{key}.json"
    global ARCH
    if entry.exists():
        doc = json.loads(entry.read_text())
        STATS.clear()
        STATS.update(doc["stats"])
        PLACES[:] = [(name, counts) for name, counts in doc["places"]]
        ARCH = doc["arch"]
        return doc["ok"], doc["report"]
    ok, report = gate(cubin, kernel, declarations, notices)
    tmp = entry.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"ok": ok, "report": report, "stats": STATS, "places": PLACES, "arch": ARCH}, sort_keys=True))
    tmp.replace(entry)
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel", default=None, help="narrow to the functions matching this regex (a measurement)")
    ap.add_argument("--member", default=None, help="narrow to the cubins whose name holds this (a measurement)")
    ap.add_argument("--declare", type=Path, default=None, metavar="DECLARATIONS",
                    help="the build's declarations, by architecture (the file bazel/defs.bzl writes from those beside the kernels: "
                         "`{\"architectures\": {\"sm_90a\": {\"declarations\": [...], \"arms\": [...]}}}`, docs/internals/tools/sass_gate.md#declarations)")
    ap.add_argument("--notices", type=Path, action="append", default=[], metavar="LOG",
                    help="a unit's ptxas report: its notices are read against the project's ledger (repeat for each unit)")
    ap.add_argument("--probes", action="store_true",
                    help="a library of probes: no kernel of the source is owed (docs/internals/tools/sass_gate.md#probes)")
    ap.add_argument("--loops", action="store_true",
                    help="after each function, what the checker's step rules (branches, conversion unit, global memory) found in it")
    ap.add_argument("--no-cache", action="store_true",
                    help="read every cubin, the verdict cache neither replayed nor written (docs/internals/tools/"
                         "sass_gate.md#cache)")
    ap.add_argument("--cuobjdump", default=None, help="the graph's cuobjdump (with --nvdisasm and --cuxxfilt: the graph's toolkit, "
                    "scratch of this run's own, no verdict cache)")
    ap.add_argument("--nvdisasm", default=None, help="the graph's nvdisasm")
    ap.add_argument("--cuxxfilt", default=None, help="the graph's cu++filt")
    ap.add_argument("--json", type=Path, default=None, help="also write every function's signature here, by cubin")
    a = ap.parse_args()

    if not (a.cuobjdump is None) == (a.nvdisasm is None) == (a.cuxxfilt is None):
        raise SystemExit("--cuobjdump, --nvdisasm and --cuxxfilt name the graph's toolkit together")
    if a.cuobjdump:
        sass.use_graph(cuobjdump=a.cuobjdump, nvdisasm=a.nvdisasm, cuxxfilt=a.cuxxfilt)
        a.no_cache = True
    global LOOPS
    LOOPS = a.loops
    declared = json.loads(a.declare.read_text())["architectures"] if a.declare else {}
    notices = read_notices(a.notices)
    allok = True
    doc = {}
    seen: set[str] = set()
    places: dict[str, list] = {}
    guarded = 0
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        arch = sass.machine(cub)
        ok, rep = cached_gate(cub, a.kernel, declared.get(arch, {}).get("declarations", []), notices, not a.no_cache)
        guarded += sum(st["guarded"] for st in STATS.values())
        if rep:
            print(f"== {cub.name}\n{rep}")
        doc[cub.name] = {"ok": ok, "arch": ARCH, "functions": dict(STATS)}
        seen |= set(STATS)
        places.setdefault(ARCH, []).extend(PLACES)
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
    if a.binary.suffix != ".cubin" and a.kernel is None and a.member is None and not a.probes:
        for line in missing_kernels(seen):
            print(f"RED  {line}")
            allok = False
    if a.kernel is None and a.member is None:
        #: what no cubin's functions claim: the notices of the build as a whole (a front end's warning, a function no cubin carries)
        #: are judged once, under every architecture's declarations
        unclaimed = [n for n in notices if n.function not in seen]
        entries = [d for arch in sorted(declared) for d in declared[arch].get("declarations", [])]
        for r in checked.check([], (), checked.declarations(entries), {}, notices=unclaimed):
            print("RED  " + "\n".join(r.lines()))
            allok = False
        for arch, found in sorted(places.items()):
            for line in unshown(declared.get(arch, {}), found, arch, str(a.declare)):
                print(line)
                allok = False
    if a.json:
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    print(f"guarded convergence fallbacks (cold code, not red): {guarded}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
