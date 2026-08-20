"""THE SHARD GATE, READ OFF THE BINARY.

**WHAT THIS CATCHES THAT NOTHING ELSE DOES.** A translation-unit split has two
failure modes that leave every other gate green:

1. the per-arm launcher's DEFINITION becomes visible to the dispatch translation
   unit, which then implicitly instantiates every kernel right back into itself;
2. a query path (residency, footprint) keeps taking kernel ADDRESSES in the
   dispatch translation unit, duplicating the arms it queries.

In both cases the build links, the ratification manifest is byte-identical, the
geometry census still reports the whole arm list -- and the compile time simply does
not fall, for a reason no gate in the design can see. `nm` cannot see it either: a
`__global__` template's host-side stub symbol is WEAK, so the linker keeps exactly
one however many objects emitted it, and `nm` reports the full set either way.

**THE FATBIN CAN SEE IT.** Device code is not linked that way: each object registers
its OWN cubin, and `cuobjdump -symbols` lists them one fatbin section at a time, each
tagged with the architecture and with the source file it was compiled from. So the
question "is this actually sharded?" has an exact answer in the shipped `.so`:

* exactly one `STO_ENTRY` per instantiation per architecture -- one per built chunk
  arm per state variant, for each of sm_80 and sm_86, no more (duplication) and no
  fewer (a dropped arm);
* every entry in the object built from the shard `tools/gen_shards.py` assigns it,
  which is a stronger statement than any count: a SWAP (one arm added, one dropped)
  keeps the count unchanged and is caught here;
* every mangled name the ratification manifest records is present in the binary,
  which is what makes the manifest a description of the shipped codegen rather than
  of a probe nobody ships.

**AND THE HOST SIDE, WHICH THE LINKER WILL NOT ENFORCE FOR US.** The dispatch reaches
each arm through a host wrapper it can declare but not define, so a shard member that
went missing leaves an UNRESOLVED reference. On a static link that would be a link
error and the design would need nothing more. This is a Python extension module: `ld
-shared` leaves undefined symbols for the loader to resolve, by design and correctly
(the CPython and torch symbols in here are resolved that way), so the missing arm
would surface as an `ImportError` at first import instead of as a build failure --
and, for the occupancy wrapper, which has no device code of its own, the fatbin count
above would not see it at all. `--no-undefined` is not available as a fix, for exactly
the reason the loader-resolved symbols exist. So the closure is checked here: NO
undefined facts-launcher symbol may remain.
Verified by removal -- dropping one arm from its shard leaves exactly one undefined
symbol, naming its tuple.

Usage:

    python tools/fatbin_entries.py --check rola/_C*.so     # the gate (setup.py runs it)
    python tools/fatbin_entries.py rola/_C*.so             # the listing
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_shards  # noqa: E402
import ratify  # noqa: E402

_ARCH = re.compile(r"^arch = sm_(\d+)\s*$")
_IDENT = re.compile(r"^identifier = (.*)$")
#: `cuobjdump -symbols` prints one line per symbol; a device ENTRY POINT (a
#: `__global__` function) is the one marked `STO_ENTRY`, which is what distinguishes
#: it from the device functions and objects sharing the cubin.
_ENTRY = re.compile(r"^STT_FUNC\s+\S+\s+STO_ENTRY\s+(\S+)\s*$")


def device_entries(so: Path) -> list[tuple[str, str, str]]:
    """Every device entry point in `so`, as (arch, source-file, mangled name).

    One tuple per (cubin, entry), NOT a set: the duplication this gate exists to
    catch is exactly a name appearing in two cubins of the same architecture.
    """
    proc = subprocess.run([ratify.cuda_bin("cuobjdump"), "-symbols", str(so)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"cuobjdump -symbols FAILED (rc={proc.returncode}); the shard "
                         f"gate cannot pass a binary it cannot read")
    out, arch, ident = [], None, "?"
    for line in proc.stdout.splitlines():
        m = _ARCH.match(line)
        if m:
            arch = f"sm_{m.group(1)}"
            continue
        m = _IDENT.match(line)
        if m:
            ident = m.group(1)
            continue
        m = _ENTRY.match(line)
        if m and arch is not None:
            out.append((arch, ident, m.group(1)))
    if not out:
        raise SystemExit(f"{so} carries no device entry points at all; that is not a "
                         f"built RoLA extension")
    return out


#: The two host entry points a shard is the only possible definer of.
_SHARD_SYMBOLS = ("launch_chunk_facts_arm", "launch_chunk_atom_bitmap_arm",
                  "launch_chunk_block_bitmap_arm")


def unresolved_arms(so: Path) -> list[str]:
    """Per-arm host wrappers that nothing in this binary defines."""
    proc = subprocess.run(["nm", "--undefined-only", str(so)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"nm FAILED on {so}; the shard gate cannot pass a binary it "
                         f"cannot read")
    return sorted(line.split()[-1] for line in proc.stdout.splitlines()
                  if any(name in line for name in _SHARD_SYMBOLS))


def expected_shards() -> dict[str, str]:
    """`{canonical arm -> shard stem}` for the whole arm list, from the generator.

    Keyed on the generator's canonical spelling rather than on a mangled name, so
    this side never has to reproduce a mangling: `tools/ratify.py:parse_shape` is the
    one decoder, and :func:`shape_key` composes it with the same canonical form.
    """
    return {gen_shards.chunk_canonical(arm): name
            for name, arms in gen_shards.chunk_shards().items() for arm in arms}


def shape_key(mangled: str) -> str | None:
    """A facts instantiation's mangled name, as the generator's canonical tuple."""
    shape = ratify.parse_shape(mangled)
    if shape is None:
        return None
    kind = {"chunk_facts_tables": "FACTS", "block_bitmap_kernel": "BLOCK",
            "atom_bitmap_kernel": "ATOM"}[shape["kernel"]]
    if kind == "FACTS":
        row = ("FACTS", shape["D"], shape["B"], shape["BC"], shape["mask"])
    elif kind == "BLOCK":
        row = ("BLOCK", shape["D"], shape["B"], shape["BC"])
    else:
        row = ("ATOM", shape["D"], shape["B"])
    return gen_shards.chunk_canonical(row)


def check(so: Path, arches=None) -> bool:
    entries = device_entries(so)
    want = expected_shards()
    ok = True

    #: 1. THE COUNT AND THE OWNERSHIP, per arch.
    seen: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for arch, ident, name in entries:
        key = shape_key(name)
        if key is not None:
            seen[arch][key].append(Path(ident).stem)
    if arches is None:
        arches = sorted(seen)
    for arch in arches:
        got = seen.get(arch, {})
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        dup = sorted(k for k, v in got.items() if len(v) > 1)
        wrong = sorted(f"{k}: in {v[0]}, the arm list says {want[k]}"
                       for k, v in sorted(got.items()) if k in want and v[0] != want[k])
        if missing or extra or dup or wrong:
            ok = False
            print(f"\nFAIL  {arch}: {len(got)} chunk instantiations in the fatbin, "
                  f"{len(want)} in the arm list.")
            for label, rows in (("absent from the binary", missing),
                                ("in the binary, not in the arm list", extra),
                                ("EMITTED BY MORE THAN ONE OBJECT (nothing is sharded)", dup),
                                ("in the wrong shard", wrong)):
                if rows:
                    print(f"      {len(rows)} {label}:")
                    for row in rows[:6]:
                        print(f"        {row}")
        else:
            print(f"fatbin OK  {arch}: {len(got)} chunk entries, one per "
                  f"instantiation, each in the shard the arm list assigns it")

    #: 2. THE DISPATCH CLOSURE, host side. An arm the dispatch can select and no shard
    #: defines is an unresolved reference that this kind of link does not refuse.
    unresolved = unresolved_arms(so)
    if unresolved:
        ok = False
        print(f"\nFAIL  {len(unresolved)} per-arm entry point(s) are referenced by the "
              f"dispatch and defined by no shard. This binary would raise ImportError "
              f"on first import; the shard partition and the dispatch have drifted "
              f"apart:")
        for name in unresolved[:6]:
            print(f"        {name}")
    else:
        print("link closure OK  every arm the dispatch can select is defined by a shard")

    #: 3. THE MANIFEST DESCRIBES THIS BINARY. Every ratified name is a name the
    #: shipped `.so` actually carries -- which was NOT true while the register/spill
    #: numbers were measured on a probe translation unit that shipped in nothing.
    present = {f"{arch}:{name}" for arch, _, name in entries}
    manifest = ratify.load_manifest([a.removeprefix("sm_") for a in arches])
    absent = sorted(set(manifest["entries"]) - present)
    if absent:
        ok = False
        print(f"\nFAIL  {len(absent)} ratified instantiation(s) name no entry point in "
              f"{so.name}. The manifest certifies a binary that is not this one:")
        for name in absent[:6]:
            print(f"        {name}")
    else:
        print(f"manifest OK  all {len(manifest['entries'])} ratified names are entry "
              f"points in {so.name}")
    print(f"\n{'PASS' if ok else 'FAIL'}  fatbin instantiation gate")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", type=Path, help="the built extension")
    ap.add_argument("--check", action="store_true",
                    help="gate: nonzero exit unless the binary carries exactly the "
                         "arm list, one entry per instantiation per arch, in the right "
                         "shard")
    ap.add_argument("--json", metavar="PATH", help="dump the raw (arch, object, entry) list")
    a = ap.parse_args()
    if a.json:
        Path(a.json).write_text(json.dumps(device_entries(a.so), indent=1) + "\n")
    if a.check:
        return 0 if check(a.so) else 1
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for arch, ident, _ in device_entries(a.so):
        counts[(arch, Path(ident).name)] += 1
    for (arch, ident), n in sorted(counts.items()):
        print(f"{arch}  {n:4d}  {ident}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
