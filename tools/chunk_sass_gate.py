"""THE CHUNK CONSUMER'S SASS OPCODE GATE (ported from the ladder at P69).

Stricter than the ratchet manifests: the chunk kernels ship with ZERO spill
(`LDL`/`STL`), `RED` never `ATOM`, and no `IMAD.WIDE`/`I2F`/`F2I`/`MUFU`
inside a loop body.  Loop bodies are detected as backward-branch spans.  The
gate DUMPS THE BUILT EXTENSION (`rola/_C.*.so`), so it verdicts on the
artifact that ships -- and it verdicts the sm_86 cubins it finds there; the
sm_80 bodies get the same gate on the box that ratifies them.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN_ANY = ("LDL", "STL", "ATOM")
FORBIDDEN_BODY = ("IMAD.WIDE", "I2F", "F2I", "MUFU")

INSN = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?);")
BRANCH = re.compile(r"\b(BRA|BRX|JMP)\b.*?(0x[0-9a-f]+)")


def so_path() -> pathlib.Path:
    cand = sorted((HERE / "rola").glob("_C.*.so"))
    if not cand:
        raise SystemExit("no built extension under rola/")
    return max(cand, key=lambda p: p.stat().st_mtime)


def entries(text):
    cur, out = None, {}
    for ln in text.splitlines():
        if "Function :" in ln:
            cur = ln.split(":", 1)[1].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def analyse(lines):
    insns = []
    for ln in lines:
        m = INSN.match(ln)
        if m:
            insns.append((int(m.group(1), 16), m.group(2).strip()))
    addrs = [a for a, _ in insns]
    body = set()
    for a, t in insns:
        m = BRANCH.search(t)
        if not m:
            continue
        tgt = int(m.group(2), 16)
        if tgt < a:
            body.update(x for x in addrs if tgt <= x <= a)
    return insns, body


def main() -> bool:
    so = so_path()
    text = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True,
                          text=True, check=True).stdout
    ok, seen = True, 0
    print(f"{'entry':<58s}{'insns':>7s}{'body':>7s}{'ANY':>5s}{'body viol':>10s}")
    for name, lines in entries(text).items():
        if "chunk" not in name:
            continue
        seen += 1
        insns, body = analyse(lines)
        bad_any = [(a, t) for a, t in insns
                   for op in FORBIDDEN_ANY
                   if re.search(rf"\b{op}\b", t) and "RED" not in t.split()[0]]
        bad_body = [(a, t) for a, t in insns if a in body
                    for op in FORBIDDEN_BODY if op in t]
        print(f"{name[:56]:<58s}{len(insns):>7d}{len(body):>7d}"
              f"{len(bad_any):>5d}{len(bad_body):>10d}")
        for a, t in (bad_any + bad_body)[:6]:
            print(f"      VIOLATION at {a:#x}: {t[:80]}")
        ok &= not bad_any and not bad_body
    if seen == 0:
        print("NO CHUNK ENTRIES FOUND -- the gate matched nothing; that is a FAIL")
        ok = False
    print("CHUNK SASS GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
