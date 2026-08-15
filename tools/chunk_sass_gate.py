"""THE CHUNK CONSUMER'S SASS OPCODE GATE (ported from the ladder repo).

Stricter than the ratchet manifests: the chunk kernels ship with ZERO spill
(`LDL`/`STL`), `RED` never `ATOM`, and no `IMAD.WIDE`/`I2F`/`F2I`/`MUFU`
inside a loop body.  Loop bodies are detected as backward-branch spans.  The
gate DUMPS THE BUILT EXTENSION (`rola/_C.*.so`), so it verdicts on the
artifact that ships -- and it verdicts the sm_86 cubins it finds there; the
sm_80 bodies get the same gate on the box that ratifies them.

A violation is judged against its INNERMOST enclosing loop, and one loop is
exempt: the WORK LOOP of the persistent launch variant, which is the only
kernel here that has one -- the variant is named in the mangled symbol, so
the exemption cannot leak into a kernel whose outermost loop is its hot one.
Its body is a whole block -- a CTA prologue plus a full token sweep -- so an
address computation there runs exactly as often per block as the same
computation ran in the CTA prologue of a kernel that owned one block, which
is where every instruction it reports came from and where nothing gated it.
Widening the gate to that loop would not be a stricter check of the same
property; it would be a different property, and one no launch shape can
satisfy. The count is REPORTED per entry so a growth in it is visible, and
every OTHER loop -- the super-chunk loop and everything inside it -- is held
at zero exactly as before (docs/internals/chunk/chunk_kernel.md#work-loop).
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


#: The persistent launch variant, off the mangled name:
#: `compacted_kernel<C, D, B, BC, STATE, PERSIST>` with `PERSIST = true`. It is
#: the only entry with a work loop, and therefore the only one the exemption
#: below may touch.
PERSISTENT = re.compile(r"compacted_kernelILi\d+ELi\d+ELi\d+ELi\d+ELb\dELb1E")


def analyse(lines, persistent=False):
    """`(insns, body, work)` -- every instruction, the ones inside SOME loop, and
    the ones whose INNERMOST loop is the work loop (empty unless `persistent`)."""
    insns = []
    for ln in lines:
        m = INSN.match(ln)
        if m:
            insns.append((int(m.group(1), 16), m.group(2).strip()))
    addrs = [a for a, _ in insns]
    spans = []
    for a, t in insns:
        m = BRANCH.search(t)
        if not m:
            continue
        tgt = int(m.group(2), 16)
        if tgt < a:
            spans.append((tgt, a))
    body = {x for lo, hi in spans for x in addrs if lo <= x <= hi}
    #: THE WORK LOOP is the widest span, and an instruction belongs to it only if
    #: NO narrower span also holds it -- innermost attribution, so nothing inside
    #: the super-chunk loop or below can reach the exemption.
    work: set[int] = set()
    if persistent and spans:
        lo, hi = max(spans, key=lambda s: s[1] - s[0])
        inner = [(s, e) for s, e in spans if (s, e) != (lo, hi)]
        work = {x for x in addrs if lo <= x <= hi
                and not any(s <= x <= e for s, e in inner)}
    return insns, body, work


def main() -> bool:
    so = so_path()
    text = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True,
                          text=True, check=True).stdout
    ok, seen = True, 0
    print(f"{'entry':<58s}{'insns':>7s}{'body':>7s}{'ANY':>5s}{'body viol':>10s}{'work':>6s}")
    for name, lines in entries(text).items():
        if "chunk" not in name:
            continue
        seen += 1
        insns, body, work = analyse(lines, PERSISTENT.search(name) is not None)
        bad_any = [(a, t) for a, t in insns
                   for op in FORBIDDEN_ANY
                   if re.search(rf"\b{op}\b", t) and "RED" not in t.split()[0]]
        bad_body = [(a, t) for a, t in insns if a in body and a not in work
                    for op in FORBIDDEN_BODY if op in t]
        in_work = [(a, t) for a, t in insns if a in work
                   for op in FORBIDDEN_BODY if op in t]
        print(f"{name[:56]:<58s}{len(insns):>7d}{len(body):>7d}"
              f"{len(bad_any):>5d}{len(bad_body):>10d}{len(in_work):>6d}")
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
