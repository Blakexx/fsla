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


def stacks(cubin: Path) -> dict[str, int]:
    """`{function: stack frame bytes}` off `cuobjdump -res-usage`."""
    text = subprocess.run([dev_config.cuda_bin("cuobjdump"), "-res-usage", str(cubin)], capture_output=True,
                          text=True, check=True).stdout
    return {m.group(1): int(m.group(3)) for m in _RES.finditer(text)}


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
    frame = stacks(cubin)
    for name, ins in funcs.items():
        if kernel and not re.search(kernel, name):
            continue
        n = len(ins)
        stack = frame.get(name, 0)
        ldl = sum(1 for i in ins if re.search(r"\b(LDL|STL)\b", i))
        calls, guarded = convergence_calls(whole[name], labels[name])
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
        if max_per_hmma is not None and hmma and per > max_per_hmma:
            reds.append(f"static instructions per HMMA {per:.1f} > {max_per_hmma}")
        ok &= not reds
        STATS[name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                       "stack": stack, "calls": calls, "guarded": guarded, "pred_hmma": phmma, "reds": reds}
        short = name[:70]
        report.append(f"{'RED ' if reds else 'ok  '} {short}: instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {stack} calls {calls} guarded {guarded} predHMMA {phmma}"
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

    print(f"guarded convergence fallbacks (cold code, not red): {guarded}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
