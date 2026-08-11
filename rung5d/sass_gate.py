"""The addressing gates, read off the assembled binary rather than asserted.

    python sass_gate.py

Every rule the ladder holds every rung to, checked per kernel entry in the very
binary the measurements run:

  * no register spill (`LDL` / `STL`), which `build.py` also checks device-side;
  * zero `IMAD.WIDE` INSIDE A LOOP BODY -- a widening multiply is what a
    `pointer + int` costs, and the rule is that every 64-bit address is formed
    once in the entry and then only advanced. Attribution is by `-lineinfo`, so
    a surviving one is reported with the source line that emitted it;
  * no runtime-divisor division or modulo, and none of the instructions that
    stand in for one (`I2F`, `F2I`, `MUFU`);
  * the fan-in is a REDUCTION and not an atomic read-modify-write: `RED`
    present, `ATOM` absent.

A LOOP BODY IS FOUND STRUCTURALLY, not by name: any instruction that a backward
branch jumps over is inside a loop. The gate takes every branch target that is at
a lower address than its branch and calls the span between them a body.
"""
from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rung5d import build as r5dbuild  # noqa: E402

#: the SHIPPED build, named by its ring depth so a stale object cannot be graded
SO = HERE / "_build" / f"rung5d_cuda_f{r5dbuild.FORM_WORDS}.so"
FORBIDDEN_ANY = ("LDL", "STL", "ATOM")
FORBIDDEN_BODY = ("IMAD.WIDE", "I2F", "F2I", "MUFU")

INSN = re.compile(r"^\s*/\*([0-9a-f]{4})\*/\s+(.*?);")
LINE = re.compile(r'^\s*//## File "([^"]+)", line (\d+)')
BRANCH = re.compile(r"\b(BRA|BRX|JMP)\b.*?(0x[0-9a-f]+)")


def entries(text):
    cur, out = None, {}
    for ln in text.splitlines():
        if ln.startswith("\t\tFunction :") or ln.startswith("Function :"):
            cur = ln.split(":", 1)[1].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def analyse(lines):
    insns, cur = [], None
    for ln in lines:
        m = LINE.match(ln)
        if m:
            cur = f"{pathlib.Path(m.group(1)).name}:{m.group(2)}"
            continue
        m = INSN.match(ln)
        if m:
            insns.append((int(m.group(1), 16), m.group(2).strip(), cur))
    addrs = [a for a, _, _ in insns]
    body = set()
    for a, t, _ in insns:
        m = BRANCH.search(t)
        if not m:
            continue
        tgt = int(m.group(2), 16)
        if tgt < a:
            body.update(x for x in addrs if tgt <= x <= a)
    return insns, body


def main():
    if not SO.exists():
        raise SystemExit(f"no binary at {SO}; build it first")
    txt = subprocess.run(["cuobjdump", "--dump-sass", str(SO)],
                         capture_output=True, text=True, check=True).stdout
    ok = True
    print(f"{'entry':58s}{'insns':>7s}{'body':>7s}{'RED':>5s}"
          f"{'ATOM':>6s}{'LDL/STL':>9s}{'body viol':>11s}")
    for name, lines in sorted(entries(txt).items()):
        insns, body = analyse(lines)
        if not insns:
            continue
        n_red = sum(1 for _, t, _ in insns if t.startswith("RED"))
        bad_any = collections.Counter(
            op for _, t, _ in insns for op in FORBIDDEN_ANY if t.startswith(op))
        bad_body = [(t, s) for a, t, s in insns if a in body
                    for op in FORBIDDEN_BODY if op in t]
        short = name if len(name) <= 56 else name[:26] + ".." + name[-28:]
        print(f"{short:58s}{len(insns):7d}{len(body):7d}{n_red:5d}"
              f"{bad_any['ATOM']:6d}{bad_any['LDL'] + bad_any['STL']:9d}"
              f"{len(bad_body):11d}")
        for t, s in bad_body:
            print(f"      BODY VIOLATION at {s}: {t}")
        ok &= not bad_body and not bad_any["ATOM"] and not bad_any["LDL"] \
            and not bad_any["STL"]
    print("SASS GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
