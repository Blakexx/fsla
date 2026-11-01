#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name]
[--max-per-hmma N]`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass):
  local memory  -- LDL/STL instructions, or a nonzero stack frame (`cuobjdump -res-usage`)
  outlined collectives -- a `CALL.REL.NOINC` into a convergence subroutine (a short block, `_SUBROUTINE`
                  instructions at most, that syncs the warp, runs one collective and returns) that a CONVERGED
                  warp executes. ptxas guards the ones it emits: a `BRA.DIV` into a fallback block, or a
                  `BRA.CONV` past the call, so a converged warp runs the inline collective; a guarded call is cold
                  code, counted (`guarded`) and not red. A call to an arithmetic slow path is not one, nor a long
                  region ptxas reconverges once on entry
  predicated HMMA -- `@P HMMA`: if-converted tensor work
  instructions per HMMA -- the static ratio, a ceiling the arm states (issue-bound above it)
  the conversion unit in a step -- F2I/I2F/F2F/MUFU inside an MMA loop: 16 lanes a cycle an SM where the FMA pipe's
                  forms (a magic-constant FFMA, a PRMT) run at 64 (KERNEL_STANDARDS section 29)
  global memory in a step -- LDG/STG/RED/ATOMG inside an MMA loop: a step's operands are staged ahead by `cp.async`
                  (LDGSTS, allowed), and a reduction to global memory belongs to a drain outside the steps
  the step cover -- each innermost MMA loop simulated for one warp from ptxas's control words (`pipe_sim`), its pipe
                  share alone times the warps a scheduler its registers admit, under `STEP_LINE` (KERNEL_STANDARDS
                  section 32); a debt kernel held at its floor (`--write-floors` records them)
On a whole extension it also reds every kernel the source defines (`tools/kernels.py`) that no cubin carries: a gate
that cannot see a kernel has not gated it. It prints the counts it reads, the same way `ratify` prints registers and
spills, so a clean run is a line of zeros and a red names the signature.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_config  # noqa: E402 -- path insert must precede these imports
import kernels  # noqa: E402
import sass  # noqa: E402
import sass_control  # noqa: E402

#: an instruction line's opcode, off the shared grammar (`tools/sass.py`)
INSTR = sass.INSTRUCTION
#: the same line whole: address, predicate, opcode, operands
WHOLE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_.$]+)\s*([^;]*);")
_COLLECTIVE = re.compile(r"^(SHFL|VOTE|MATCH|REDUX|BAR)")

#: the always-false predicate: ptxas keeps a branch under it that it never takes
_NEVER = "@!PT"
#: an unconditional transfer: nothing falls through it
_TERMINATOR = re.compile(r"^(BRA|EXIT|RET|JMP|BRX|JMX)$|^(EXIT|RET)\.")
#: a convergence subroutine's length at most: its operand moves, `WARPSYNC`, the collective (`RET` not counted)
_SUBROUTINE = 8


def cubins_of(path: Path, member: str | None) -> list[Path]:
    return [image for image in sass.cubins(path) if member is None or member in image.name]


_RES = re.compile(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+)")


def resources(cubin: Path) -> dict[str, tuple[int, int]]:
    """`{function: (registers, stack frame bytes)}` off `cuobjdump -res-usage`."""
    text = subprocess.run([dev_config.cuda_bin("cuobjdump"), "-res-usage", str(cubin)], capture_output=True,
                          text=True, check=True).stdout
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in _RES.finditer(text)}


#: the last `gate` call's signature per function, for `--json`
STATS: dict[str, dict] = {}


def convergence_calls(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """``(hot, guarded)``: the calls into a convergence subroutine a converged warp executes, and those it never does
    (in a block a `BRA.DIV` enters, past a `BRA.CONV` in the call's block, or in a dead block: one only a branch under
    the always-false predicate `@!PT` enters, right after an unconditional transfer). ``lines`` are ``(address,
    opcode, operands, predicate)``, the function's own; ``labels`` ``{name: line index}``. A target is an address
    (`cuobjdump`) or a label (`nvdisasm`: `` `($__internal_1_$__cuda_sm70_shflsync_bfly_p) ``)."""
    at = {addr: k for k, (addr, _, _, _) in enumerate(lines)}

    def resolve(operands: str) -> int | None:
        address = re.match(r"(?:~URZ,\s*)?(0x[0-9a-f]+)", operands)
        label = re.search(r"\(([^)]+)\)", operands)
        return at.get(int(address.group(1), 16)) if address else labels.get(label.group(1)) if label else None

    def subroutine(start: int) -> bool:
        block = []
        for _, op, _, _ in lines[start:start + _SUBROUTINE + 1]:
            if op.startswith("RET"):
                return "WARPSYNC" in block and any(_COLLECTIVE.match(o) for o in block)
            block.append(op)
        return False

    def targets(prefix: str, live: bool) -> set[int]:
        found = (resolve(o) for _, op, o, p in lines if op.startswith(prefix) and (p != _NEVER) == live)
        return {k for k in found if k is not None}

    taken = targets("BRA", True)
    starts = {0, *labels.values()} | taken | targets("BRA", False)
    fallback = targets("BRA.DIV", True)
    dead = {k for k in starts - taken
            if k > 0 and lines[k - 1][3] == "" and _TERMINATOR.match(lines[k - 1][1])}
    hot = guarded = 0
    block, conv = 0, False
    for k, (_, opcode, operands, predicate) in enumerate(lines):
        if k in starts:
            block, conv = k, False
        if predicate == _NEVER:
            continue
        conv = conv or opcode.startswith("BRA.CONV")
        target = resolve(operands) if opcode.startswith("CALL.REL") else None
        if target is not None and subroutine(target):
            if block in fallback or block in dead or conv:
                guarded += 1
            else:
                hot += 1
    return hot, guarded


_MMA = re.compile(r"^(HMMA|IMMA)")
#: KERNEL_STANDARDS 27's debts: the kernels built before the rule whose steps still branch, each with what replaces it
_STEP_BRANCH_DEBT = {
    "intra_kernel": "the intra's port onto the fp16 family rebuilds its step",
    "readout_part": "the readout component's rebuild replaces the skeleton's",
    "calib_kernel": "a calibration probe, not a production step",
}


def _resolver(lines: list[tuple[int, str, str, str]], labels: dict[str, int]):
    at = {addr: k for k, (addr, _, _, _) in enumerate(lines)}

    def resolve(operands: str) -> int | None:
        address = re.match(r"(0x[0-9a-f]+)", operands)
        label = re.search(r"\(([^)]+)\)", operands)
        return at.get(int(address.group(1), 16)) if address else labels.get(label.group(1)) if label else None

    return resolve


def mma_loops(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> list[tuple[int, int]]:
    """The innermost loops that issue MMAs, as (first line, backward branch): a loop a backward branch, its body the
    lines from its target to it, innermost when no other loop that issues MMAs lies inside. A loop inside it that
    issues none -- a compare-and-swap spin (sm_80/86's shared fp32 add), a one-trip guard -- is part of its body, read
    once. Targets resolve as `convergence_calls`' do."""
    resolve = _resolver(lines, labels)
    back = []
    for k, (_, opcode, operands, predicate) in enumerate(lines):
        target = resolve(operands) if opcode == "BRA" and predicate != _NEVER else None
        if target is not None and target <= k:
            back.append((target, k))
    mma = [(t, k) for t, k in back if any(_MMA.match(x[1]) for x in lines[t:k + 1])]
    return [(target, k) for target, k in mma
            if not any(target <= t and kk < k for t, kk in mma if (t, kk) != (target, k))]


def mma_gap(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """KERNEL_STANDARDS 27's interleave as ptxas emitted it: the longest run of instructions without an MMA in an
    innermost MMA loop's body, and the run around the loop (its tail into its head -- a rolled step loop's own gap, an
    unrolled one's fill and drain). In-order issue makes a run the time a warp leaves the tensor pipe to the
    scheduler's other warps."""
    inside = around = 0
    for target, k in mma_loops(lines, labels):
        at = [j for j in range(target, k + 1) if _MMA.match(lines[j][1])]
        inside = max(inside, *(b - a - 1 for a, b in zip(at, at[1:])), 0)
        around = max(around, k - target - (at[-1] - at[0]))
    return inside, around


#: KERNEL_STANDARDS section 29's debts: the kernels built before the step lints whose steps still do it
_STEP_XU_DEBT = {
    "intra_kernel": "its epilogue's F2I (160 in its steps); the intra's port onto the fp16 family converts on the FMA pipe",
}
_STEP_GLOBAL_DEBT = {
    "carry_kernel": "the fold's steps past a list's 16th entry read records from global memory; the readout-first "
                    "rebuild stages every step's records",
    "produce_kernel": "126 global accesses inside its MMA loop: its operands are not staged by cp.async",
}


def step_classes(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> tuple[int, int]:
    """KERNEL_STANDARDS section 29: inside the innermost MMA loops (`mma_loops`), the instructions on the conversion
    unit and the global memory accesses other than `cp.async` (LDGSTS)."""
    xu = memory = 0
    for target, k in mma_loops(lines, labels):
        for j in range(target, k + 1):
            _, op, _, pred = lines[j]
            if pred == _NEVER:
                continue
            kind = sass.opcode_class(op)
            xu += kind == "conversion unit"
            memory += kind == "global" and not op.startswith("LDGSTS")
    return xu, memory


def mma_loop_branches(lines: list[tuple[int, str, str, str]], labels: dict[str, int]) -> int:
    """KERNEL_STANDARDS 27: the branches inside a step -- a convergence barrier (BSSY) or a forward conditional BRA in an
    innermost MMA loop (`mma_loops`; the BRA.DIV / BRA.CONV guards of a collective's cold fallback not counted). Each
    one ends ptxas's scheduling block inside the step."""
    resolve = _resolver(lines, labels)
    found = set()
    for target, k in mma_loops(lines, labels):
        for j in range(target, k):
            _, op, ops, pred = lines[j]
            if pred == _NEVER:
                continue
            forward = resolve(ops) if op == "BRA" and pred else None
            if op == "BSSY" or (forward is not None and forward > j):
                found.add(j)
    return len(found)


#: THE STEP LINE (KERNEL_STANDARDS section 32): the share of a step loop's time its tensor pipe can be kept busy, at
#: least 1 / section 28's line
STEP_LINE = 100 / 1.15
#: the kernels whose step loops are under the line, each with what replaces them; each function held at its floor
#: (`STEP_FLOORS`, by architecture and mangled name: a ratchet), and the list only shrinks
_STEP_COVER_DEBT = {
    "carry_kernel": "the read side's software pipeline (the readout's next tile loaded under its MMAs, the image's "
                    "batches stored while read) and the fold chain's",
    "produce_kernel": "its operands staged by cp.async a step ahead (with section 29's global-memory debt)",
    "intra_kernel": "the intra's port onto the fp16 family rebuilds its step",
    "readout_part": "the readout component's rebuild replaces the skeleton's",
    "calib_kernel": "a calibration probe, not a production step",
}
STEP_FLOORS = Path(__file__).resolve().parent / "budgets" / "step_floors.json"
_PARTITION_REGISTERS = 16384


def step_cover(cubin: Path, name: str, lines: list[tuple[int, str, str, str]], labels: dict[str, int],
               registers: int) -> dict | None:
    """KERNEL_STANDARDS section 32 on the SASS: each innermost MMA loop's period for one warp alone, simulated from
    ptxas's control words with the tensor pipe as a resource (`pipe_sim`, the model `burst_gate` reads), its pipe
    time's share of it (the occupancy), and the COVER: that share times the warps a scheduler the registers admit,
    at most 100 -- the most the loop's warps together can keep the pipe busy, since a warp sharing its scheduler
    never runs faster than alone. The worst loop's reading, or None without an MMA loop."""
    loops = mma_loops(lines, labels)
    if not loops:
        return None
    import burst_gate

    arch = sass_control.machine(cubin)
    mach = sass_control.MACHINES[arch]
    encoded = sass_control.encoded(cubin, f"^{re.escape(name)}$")
    grain = -(-registers // 8) * 8
    warps = max(1, _PARTITION_REGISTERS // (grain * 32)) if registers else 1
    worst = None
    for target, k in loops:
        body = [(a, *encoded[a]) for a, _, _, _ in lines[target:k + 1] if a in encoded]
        c = burst_gate.coverage(body, mach, arch)
        c["cover"] = round(min(100.0, c["occupancy"] * warps), 1)
        c["warps"] = warps
        if worst is None or c["cover"] < worst["cover"]:
            worst = c
    worst["loops"] = len(loops)
    worst["key"] = f"{arch} {name}"
    return worst


def gate(cubin: Path, kernel: str | None, max_per_hmma: float | None) -> tuple[bool, str]:
    funcs: dict[str, list[str]] = {}
    whole: dict[str, list[tuple[int, str, str]]] = {}
    labels: dict[str, dict[str, int]] = {}
    cur = None
    for line in sass.disassemble(cubin).splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            whole[cur] = []
            labels[cur] = {}
            continue
        m = re.match(r"^(\$?[\w.$]+):\s*$", line)
        if m and cur is not None:
            labels[cur][m.group(1)] = len(whole[cur])
            continue
        m = INSTR.match(line)
        if m and cur is not None:
            predicate = (m.group(2) or "").strip()
            if predicate != _NEVER:
                funcs[cur].append(predicate + " " + m.group(3))
            w = WHOLE.match(line)
            whole[cur].append((int(w.group(1), 16), w.group(3), w.group(4).strip(), predicate) if w else
                              (int(m.group(1), 16), m.group(3), "", predicate))

    ok = True
    report = []
    STATS.clear()
    frame = resources(cubin)
    floors = json.loads(STEP_FLOORS.read_text()) if STEP_FLOORS.exists() else {}
    for name, ins in funcs.items():
        if kernel and not re.search(kernel, name):
            continue
        n = len(ins)
        registers, stack = frame.get(name, (0, 0))
        ldl = sum(1 for i in ins if re.search(r"\b(LDL|STL)\b", i))
        calls, guarded = convergence_calls(whole[name], labels[name])
        branches = mma_loop_branches(whole[name], labels[name])
        gap, around = mma_gap(whole[name], labels[name])
        xu, memory = step_classes(whole[name], labels[name])
        phmma = sum(1 for i in ins if re.match(r"@!?P\d+ HMMA", i))
        hmma = sum(1 for i in ins if " HMMA" in i or i.startswith("HMMA"))
        per = (n / hmma) if hmma else float("nan")
        reds = []
        if ldl:
            reds.append(f"local memory ({ldl} LDL/STL)")
        if stack:
            reds.append(f"stack frame ({stack} B)")
        if calls:
            reds.append(f"convergence subroutines a converged warp executes ({calls})")
        if phmma:
            reds.append(f"predicated HMMA ({phmma})")
        if branches and not any(debt in name for debt in _STEP_BRANCH_DEBT):
            reds.append(f"branches inside an MMA step ({branches})")
        if xu and not any(debt in name for debt in _STEP_XU_DEBT):
            reds.append(f"the conversion unit inside an MMA step ({xu})")
        if memory and not any(debt in name for debt in _STEP_GLOBAL_DEBT):
            reds.append(f"global memory inside an MMA step ({memory})")
        if max_per_hmma is not None and hmma and per > max_per_hmma:
            reds.append(f"static instructions per HMMA {per:.1f} > {max_per_hmma}")
        cover = step_cover(cubin, name, whole[name], labels[name], registers)
        debt = next((d for d in _STEP_COVER_DEBT if d in name), None)
        if cover is not None and cover["cover"] < STEP_LINE:
            if debt is None:
                reds.append(f"a step loop's pipe cover {cover['cover']}% < {STEP_LINE:.1f}%")
            elif cover["key"] not in floors:
                reds.append(f"a step loop's pipe cover {cover['cover']}% with no floor (`--write-floors`)")
            elif cover["cover"] < floors[cover["key"]]:
                reds.append(f"a step loop's pipe cover {cover['cover']}% below its floor {floors[cover['key']]}%")
        ok &= not reds
        STATS[name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                       "stack": stack, "calls": calls, "guarded": guarded, "pred_hmma": phmma,
                       "mma_loop_branches": branches, "mma_gap": gap, "mma_gap_around": around,
                       "step_xu": xu, "step_global": memory, "registers": registers,
                       "step_cover": None if cover is None else {k: cover[k] for k in (
                           "cover", "occupancy", "warps", "period", "pipe", "exposed", "worst", "loops", "key")},
                       "reds": reds}
        short = name[:70]
        report.append(f"{'RED ' if reds else 'ok  '} {short}: instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {stack} calls {calls} guarded {guarded} predHMMA {phmma} "
                      f"mma-loop-branches {branches} mma-gap {gap} around {around} step-xu {xu} step-global {memory}"
                      + ("" if cover is None else f" step-cover {cover['cover']}% (alone {cover['occupancy']}% x "
                                                  f"{cover['warps']} warps, exposed {cover['exposed']})")
                      + (" :: " + "; ".join(reds) if reds else ""))
    return ok, "\n".join(report)


def missing_kernels(seen: set[str]) -> list[str]:
    """The source's kernels (`tools/kernels.py`) whose name no gated function carries."""
    names = set(kernels.demangled_names(sorted(seen)))
    return [f"{k.where()}: `{k.name}` ({k.family}) has no SASS in this binary" for k in kernels.kernels()
            if k.name not in names]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel", default=None, help="narrow to the functions matching this regex (a measurement)")
    ap.add_argument("--member", default=None, help="narrow to the cubins whose name holds this (a measurement)")
    ap.add_argument("--max-per-hmma", type=float, default=None)
    ap.add_argument("--json", type=Path, default=None, help="also write every function's signature here, by cubin")
    ap.add_argument("--write-floors", action="store_true",
                    help="record each step-cover debt's worst reading as its floor (a ratchet: a later build may not "
                         "read lower)")
    a = ap.parse_args()
    allok = True
    doc = {}
    seen: set[str] = set()
    guarded = 0
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = gate(cub, a.kernel, a.max_per_hmma)
        guarded += sum(st["guarded"] for st in STATS.values())
        if rep:
            print(f"== {cub.name}\n{rep}")
        doc[cub.name] = {"ok": ok, "functions": dict(STATS)}
        seen |= set(STATS)
        allok &= ok
    if not seen:
        print(f"RED: no function matched (--kernel {a.kernel!r}, --member {a.member!r}): a gate with no target does "
              "not pass")
        allok = False
    if a.binary.suffix != ".cubin" and a.kernel is None and a.member is None:
        for line in missing_kernels(seen):
            print(f"RED  {line}")
            allok = False
    if a.json:
        a.json.write_text(json.dumps(doc, sort_keys=True) + "\n")
    if a.write_floors:
        floors = json.loads(STEP_FLOORS.read_text()) if STEP_FLOORS.exists() else {}
        for cubin in doc.values():
            for name, st in cubin["functions"].items():
                if any(d in name for d in _STEP_COVER_DEBT) and st["step_cover"] is not None:
                    floors[st["step_cover"]["key"]] = st["step_cover"]["cover"]
        STEP_FLOORS.write_text(json.dumps(dict(sorted(floors.items())), indent=1) + "\n")
        print(f"step floors written to {STEP_FLOORS}")

    print(f"guarded convergence fallbacks (cold code, not red): {guarded}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
