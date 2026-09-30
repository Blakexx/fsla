#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name]
[--max-per-hmma N]`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass):
  local memory  -- LDL/STL instructions, or a nonzero stack frame (`cuobjdump -res-usage`)
  outlined collectives -- a `CALL.REL.NOINC` into a convergence subroutine: a short block (`_SUBROUTINE`
                  instructions at most) that syncs the warp, runs one collective and returns -- paid on every
                  collective it wraps. A call to an arithmetic slow path is not one, nor a long region ptxas
                  reconverges once on entry
  predicated HMMA -- `@P HMMA`: if-converted tensor work
  instructions per HMMA -- the static ratio, a ceiling the arm states (issue-bound above it)
On a whole extension it also reds every kernel the source defines (`tools/kernels.py`) that no cubin carries: a gate
that cannot see a kernel has not gated it. THE CONVERGENCE BACKLOG (`tools/sass_backlog.json`) records, per function,
the convergence subroutines the tree carried when the check first read them (it read none before 2026-09-29: it
matched a call's operands against its opcode alone): a function above its recorded count, or not recorded, is red;
one below it is red too until the record shrinks (`--write-backlog`), so the backlog only pays down. It prints the counts it reads, the same way `ratify` prints registers and
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

BACKLOG = Path(__file__).resolve().parent / "sass_backlog.json"


def backlog() -> dict[str, int]:
    """`{function: convergence subroutines owed}`."""
    return json.loads(BACKLOG.read_text())["convergence"] if BACKLOG.exists() else {}


def convergence_calls(lines: list[tuple[int, str, str]], labels: dict[str, int]) -> int:
    """The calls whose target block, up to its `RET`, holds a `WARPSYNC` and a collective: `(address, opcode,
    operands)` a line, the function's own, and its labels (`{name: line index}`). A target is an address
    (`cuobjdump`) or a label (`nvdisasm`: `` `($__internal_1_$__cuda_sm70_shflsync_bfly_p) ``)."""
    at = {addr: k for k, (addr, _, _) in enumerate(lines)}
    count = 0
    for _, opcode, operands in lines:
        if not opcode.startswith("CALL.REL"):
            continue
        address = re.match(r"(0x[0-9a-f]+)", operands)
        label = re.search(r"\(([^)]+)\)", operands)
        start = (at.get(int(address.group(1), 16)) if address else labels.get(label.group(1)) if label else None)
        if start is None:
            continue
        block = []
        for _, op, _ in lines[start:start + _SUBROUTINE + 1]:
            if op.startswith("RET"):
                count += "WARPSYNC" in block and any(_COLLECTIVE.match(o) for o in block)
                break
            block.append(op)
    return count


def gate(cubin: Path, kernel: str | None, max_per_hmma: float | None,
         owed: dict[str, int] | None = None) -> tuple[bool, str]:
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
            funcs[cur].append((m.group(2) or "").strip() + " " + m.group(3))
            w = WHOLE.match(line)
            whole[cur].append((int(w.group(1), 16), w.group(3), w.group(4).strip()) if w else
                              (int(m.group(1), 16), m.group(3), ""))

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
        calls = convergence_calls(whole[name], labels[name])
        phmma = sum(1 for i in ins if re.match(r"@!?P\d+ HMMA", i))
        hmma = sum(1 for i in ins if " HMMA" in i or i.startswith("HMMA"))
        per = (n / hmma) if hmma else float("nan")
        reds = []
        if ldl:
            reds.append(f"local memory ({ldl} LDL/STL)")
        if stack:
            reds.append(f"stack frame ({stack} B)")
        debt = (owed or {}).get(name, 0)
        if calls > debt:
            reds.append(f"convergence subroutines ({calls} calls into WARPSYNC + a collective, {debt} owed)")
        if phmma:
            reds.append(f"predicated HMMA ({phmma})")
        if max_per_hmma is not None and hmma and per > max_per_hmma:
            reds.append(f"static instructions per HMMA {per:.1f} > {max_per_hmma}")
        ok &= not reds
        STATS[name] = {"instr": n, "hmma": hmma, "per_hmma": None if not hmma else round(per, 2), "ldl": ldl,
                       "stack": stack, "calls": calls, "pred_hmma": phmma, "reds": reds}
        short = name[:70]
        report.append(f"{'RED ' if reds else 'ok  '} {short}: instr {n} hmma {hmma} per-hmma {per:.1f} "
                      f"ldl {ldl} stack {stack} calls {calls} predHMMA {phmma}"
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
    ap.add_argument("--write-backlog", action="store_true",
                    help="record this binary's convergence subroutines as the backlog (after reading every red)")
    a = ap.parse_args()
    allok = True
    doc = {}
    seen: set[str] = set()
    owed = {} if a.write_backlog else backlog()
    #: a function's convergence subroutines, its most over the cubins read (one per arch)
    found: dict[str, int] = {}

    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = gate(cub, a.kernel, a.max_per_hmma, owed=owed)
        for n, st in STATS.items():
            found[n] = max(found.get(n, 0), st["calls"])
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

    if a.write_backlog:
        debt = {n: c for n, c in sorted(found.items()) if c}
        BACKLOG.write_text(json.dumps({"about": _BACKLOG_ABOUT, "convergence": debt}, indent=1) + "\n")
        print(f"wrote {BACKLOG.name}: {len(debt)} function(s), {sum(debt.values())} convergence subroutine(s)")
        return 0
    for name in sorted(n for n in found if found[n] < owed.get(n, 0)):
        print(f"RED  {name[:70]}: convergence backlog paid down ({found[name]} of {owed[name]} owed) -- shrink "
              f"{BACKLOG.name} (`--write-backlog`)")
        allok = False
    print(f"convergence backlog: {sum(owed.values())} subroutine(s) owed across {len(owed)} function(s)")
    return 0 if allok else 1


_BACKLOG_ABOUT = ("The convergence subroutines each function carried when tools/sass_gate.py first read them "
                  "(2026-09-29; the check read none before). The gate reds a function above its count or below it; "
                  "pay one down and shrink its entry.")


if __name__ == "__main__":
    sys.exit(main())
