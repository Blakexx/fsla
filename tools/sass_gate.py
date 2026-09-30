#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE: the ptxas signatures of KERNEL_STANDARDS §20, read off a cubin or a built extension, red on any
of them. Runs on every build (`tools/sass_gate.py <cubin|.so> [--kernel regex] [--member name]
[--max-per-hmma N]`); exit 1 on a red.

What it reads, per function of every cubin (every kernel and every outlined device function; `--kernel` and
`--member` narrow a MEASUREMENT, and a narrowing that matches nothing is red, never a silent pass):
  local memory  -- LDL/STL instructions, or a nonzero stack frame (`cuobjdump -res-usage`)
  outlined collectives -- `CALL.REL.NOINC $__internal_*` (a convergence subroutine)
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


def gate(cubin: Path, kernel: str | None, max_per_hmma: float | None) -> tuple[bool, str]:
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in sass.disassemble(cubin).splitlines():
        m = re.match(r"\s*\.text\.(\S+):", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = INSTR.match(line)
        if m and cur is not None:
            funcs[cur].append((m.group(2) or "").strip() + " " + m.group(3))

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
        calls = sum(1 for i in ins if "CALL.REL.NOINC $__internal" in i or "$__cuda_sm" in i)
        phmma = sum(1 for i in ins if re.match(r"@!?P\d+ HMMA", i))
        hmma = sum(1 for i in ins if " HMMA" in i or i.startswith("HMMA"))
        per = (n / hmma) if hmma else float("nan")
        reds = []
        if ldl:
            reds.append(f"local memory ({ldl} LDL/STL)")
        if stack:
            reds.append(f"stack frame ({stack} B)")
        if calls:
            reds.append(f"convergence subroutines ({calls} CALL $__internal)")
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
    a = ap.parse_args()
    allok = True
    doc = {}
    seen: set[str] = set()
    for cub in cubins_of(a.binary, a.member if a.binary.suffix != ".cubin" else None):
        ok, rep = gate(cub, a.kernel, a.max_per_hmma)
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
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
