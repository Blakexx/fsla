"""PER-INSTANTIATION SASS BODY HASHES -- the codegen-equivalence instrument.

**THE STANDING RULING THIS MECHANIZES.** When a change claims to be a restructure and
not a codegen change, the thing to compare is the ASSEMBLED BODY of each kernel, not
its symbol name and not the register count. Names are the weaker check twice over: a
mangling can move while the code is identical (a translation-unit restructure does
exactly that to anonymous-namespace kernels), and the code can move while every name
is identical. Register counts are weaker still -- two different instruction schedules
routinely allocate the same number of registers.

So: disassemble both binaries, index every entry point by `(arch, mangled name)`, and
hash the instruction text. Two builds are codegen-equivalent iff the maps are equal.

**WHAT IS AND IS NOT IN THE HASH.** Every line of the disassembly that carries
codegen, whitespace-normalised: the instruction lines (offset marker, opcode, operands
and the encoding word printed beside them) and the standalone continuation lines that
carry the second encoding word. The encoding words are IN the hash and must be -- two
instructions can print identically and encode differently (a predicate, a scheduling
control field, a reuse flag). Not in the hash: `Function :` headers, `.headerflags`,
`//## File` lineinfo lines and anything else without an offset marker or an encoding
comment; the padding cuobjdump inserts to align the encoding column is normalised away,
because column alignment is a property of the LISTING and moves with the longest
instruction in it.

**THE NAMES ARE PATH-DEPENDENT, SO BOTH BUILDS MUST BE MADE AT THE SAME PATH.**
`decode.cu` and `entmax.cu` put their kernels in an ANONYMOUS namespace, so the mangled
name embeds a discriminator derived from the source path (`_GLOBAL__N__<hash>_11_
consumer_cu_<hash>`). A baseline built in a worktree at a different directory therefore
mismatches on every non-consumer entry while being byte-identical code. Build the two
trees at the SAME path -- check one out over the other -- rather than side by side.

**A WRITTEN MAP CARRIES THE HASHER'S VERSION, AND A MISMATCH IS REFUSED.** What is
hashed is a property of THIS FILE, not of the binary: widening the capture once moved
every digest in an unchanged binary. Two maps written by different versions therefore
disagree on all 1,228 entries while describing identical code -- a false verdict with
the same shape as the path trap above, and one the P06 review hit. `HASH_VERSION` is
written into every map and `--compare` refuses a pair that does not agree on it, and
refuses an unstamped map outright rather than guessing which version wrote it. BUMP IT
whenever `bodies()` changes what it feeds the digest.

Two uses, both binary verdicts:

* a TU restructure (P02's shard) -- pre-shard vs post-shard `.so` must be identical
  over all 1,216 ratified entries;
* the `--split-compile` trial -- the flag is adopted only on an empty diff.

Usage:

    python tools/sass_bodies.py build_a/_C.so --json a.json
    python tools/sass_bodies.py --compare a.json b.json

Both `.so`s must have been BUILT AT THE SAME PATH -- see the path-dependence note
above.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ratify  # noqa: E402

#: THE HASHER'S OWN VERSION, written into every map. Bump it in the same commit that
#: changes what `bodies()` hashes; a map is only comparable with another map written by
#: the same version. 1 = instruction text only (the pre-2026-08-05 capture, which
#: dropped the encoding words); 2 = every offset-marked and encoding line,
#: whitespace-normalised.
HASH_VERSION = 2

_ARCH = re.compile(r"^arch = sm_(\d+)\s*$")
_FUNC = re.compile(r"^\s*Function : (\S+)")
#: An INSTRUCTION line: it carries an offset marker. The whole line is codegen --
#: opcode, operands and the encoding word printed beside them.
_INST = re.compile(r"/\*[0-9a-f]{4,}\*/")
#: The CONTINUATION line: an encoding word alone, the second half of the instruction's
#: encoding. It has no offset marker, so it needs its own pattern or it is not hashed.
_ENC = re.compile(r"^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")


def bodies(so: Path) -> dict[str, str]:
    """`{"sm_XX:mangled" -> sha256 of the instruction text}` for every entry point."""
    proc = subprocess.Popen([ratify.cuda_bin("cuobjdump"), "-sass", str(so)],
                            stdout=subprocess.PIPE, text=True)
    out: dict[str, str] = {}
    arch, key, h = None, None, None

    def flush():
        if key is not None:
            #: LAST WRITER WINS IS WRONG HERE, so it is refused: a name appearing in
            #: two cubins of one arch means the kernel is duplicated across objects,
            #: which is `tools/fatbin_entries.py`'s failure, and silently hashing one
            #: of the two copies would hide it behind a green comparison.
            if key in out and out[key] != h.hexdigest():
                raise SystemExit(f"{key} is disassembled twice in {so.name} with "
                                 f"DIFFERENT bodies; run tools/fatbin_entries.py")
            out[key] = h.hexdigest()

    for line in proc.stdout:
        m = _ARCH.match(line)
        if m:
            flush()
            arch, key, h = f"sm_{m.group(1)}", None, None
            continue
        m = _FUNC.match(line)
        if m:
            flush()
            key, h = f"{arch}:{m.group(1)}", hashlib.sha256()
            continue
        if h is None:
            continue
        if _INST.search(line) or _ENC.match(line):
            h.update(" ".join(line.split()).encode())
            h.update(b"\n")
    flush()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"cuobjdump -sass FAILED (rc={rc})")
    return out


def write_map(path: Path, bodies_by_key: dict[str, str], so: Path) -> None:
    """A map is a STAMPED RECORD: what hashed it, and what it hashed."""
    path.write_text(json.dumps(
        {"hash_version": HASH_VERSION, "so": str(so), "entries": len(bodies_by_key),
         "bodies": bodies_by_key}, indent=1, sort_keys=True) + "\n")


def read_map(path: str) -> dict[str, str]:
    """The bodies out of a stamped map, or a REFUSAL naming what is wrong.

    An unstamped map is refused rather than read as version 1: a map that predates the
    stamp cannot say which capture wrote it, and the whole point of the stamp is that
    guessing produces a confident wrong answer.
    """
    blob = json.loads(Path(path).read_text())
    if not isinstance(blob, dict) or "hash_version" not in blob:
        raise SystemExit(
            f"{path} carries no `hash_version`, so nothing knows what hashed it. "
            f"Maps written by different captures disagree on EVERY entry while "
            f"describing identical code. Re-hash the binary with this tool "
            f"(version {HASH_VERSION}).")
    if blob["hash_version"] != HASH_VERSION:
        raise SystemExit(
            f"{path} was written by hash version {blob['hash_version']}; this tool is "
            f"version {HASH_VERSION}. The two hash DIFFERENT text and their digests "
            f"are not comparable -- re-hash both binaries with one version.")
    return blob["bodies"]


def compare(a: dict[str, str], b: dict[str, str], label_a: str, label_b: str) -> bool:
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    moved = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    for label, rows in ((f"only in {label_a}", only_a), (f"only in {label_b}", only_b),
                        ("SASS BODY DIFFERS", moved)):
        if rows:
            print(f"\n{len(rows)} {label}:")
            for name in rows[:12]:
                print(f"    {name}")
            if len(rows) > 12:
                print(f"    ... and {len(rows) - 12} more")
    ok = not (only_a or only_b or moved)
    print(f"\n{len(set(a) & set(b))} entries compared, {len(moved)} with a different "
          f"body, {len(only_a)} absent from {label_b}, {len(only_b)} absent from "
          f"{label_a}")
    print(f"{'PASS' if ok else 'FAIL'}  SASS body equivalence")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", type=Path, nargs="?", help="a built extension to hash")
    ap.add_argument("--json", metavar="PATH", help="write the hash map")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two written hash maps; nonzero exit on any difference")
    a = ap.parse_args()
    if a.compare:
        maps = [read_map(p) for p in a.compare]
        return 0 if compare(maps[0], maps[1], *a.compare) else 1
    if a.so is None:
        raise SystemExit("pass a .so to hash, or --compare two written maps")
    got = bodies(a.so)
    print(f"{len(got)} entry points hashed in {a.so.name}")
    if a.json:
        write_map(Path(a.json), got, a.so)
        print(f"written to {a.json} (hash version {HASH_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
