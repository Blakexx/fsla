"""THE §0.2 RENAME GATE -- proof that the namespace rename moved KEYS and nothing else.

The manifest is keyed on MANGLED C++ symbol names. Renaming `fla_rola::v3` to `rola`
therefore invalidates every one of its 1024 keys at once, and the ordinary spill gate
cannot tell that apart from the instantiation set moving: it would report 1024 entries
missing and 1024 added and it would be right to. Hand-porting the keys is the exact
failure the gate exists to prevent -- it would assert spills nobody measured.

So the rename is proved instead of asserted. This script takes the PRE-rename manifest
and the POST-rename one, applies a LITERAL, declared transform to the old keys, and
requires the result to equal the new file EXACTLY -- keys and values. Any value
difference means codegen moved and the rename is not what it claimed to be.

**WHY THE COMPARISON IS DONE ON DEMANGLED NAMES.** Itanium mangling compresses repeated
components with back-references (`S_`, `S0_`, `S1_`, ...). Dropping one namespace level
renumbers every one of them, so a substitution written against the mangled string would
have to reimplement the compression scheme to be correct -- and a transform that is
itself a reimplementation proves nothing. `c++filt` is asked to expand both sides
instead, and the rename is expressed in the source language it was actually written in:

    fla_rola::v3::            ->  rola::
    fla_rola::p3::            ->  rola::paging::
    v3_consumer_forward_kernel->  consumer_forward_kernel
    V3ConsumerParams          ->  ConsumerParams

Usage:

    python tools/verify_rename_transform.py OLD.json NEW.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: THE DECLARED TRANSFORM. Longest first; applied to the DEMANGLED spelling.
TRANSFORM: list[tuple[str, str]] = [
    ("fla_rola::v3::mma::", "rola::mma::"),
    ("fla_rola::v3::detail::", "rola::detail::"),
    ("fla_rola::v3::", "rola::"),
    ("fla_rola::p3::", "rola::paging::"),
    ("v3_consumer_forward_kernel", "consumer_forward_kernel"),
    ("V3ConsumerParams", "ConsumerParams"),
    ("V3ScanParams", "ScanParams"),
]


def demangle(names: list[str]) -> list[str]:
    """One `c++filt` invocation for the whole list -- 2048 subprocesses would be a
    minute of nothing, and the batch form is exact."""
    proc = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise SystemExit(f"c++filt failed (rc={proc.returncode}): {proc.stderr}")
    out = proc.stdout.split("\n")
    while out and out[-1] == "":
        out.pop()
    if len(out) != len(names):
        raise SystemExit(f"c++filt returned {len(out)} lines for {len(names)} names")
    return out


def demangled_entries(manifest: dict) -> dict[str, dict]:
    """`{"sm_80:<mangled>": {...}}` -> `{"sm_80:<demangled>": {...}}`.

    A collision would mean two distinct mangled names demangle alike, which would make
    the comparison weaker than the manifest's own keying; it is checked rather than
    assumed.
    """
    keys = list(manifest["entries"])
    archs = [k.split(":", 1)[0] for k in keys]
    plain = demangle([k.split(":", 1)[1] for k in keys])
    out: dict[str, dict] = {}
    for arch, name, key in zip(archs, plain, keys):
        new_key = f"{arch}:{name}"
        if new_key in out:
            raise SystemExit(f"demangled key collision: {new_key}")
        out[new_key] = manifest["entries"][key]
    return out


def apply_transform(key: str) -> str:
    for old, new in TRANSFORM:
        key = key.replace(old, new)
    return key


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    old = json.loads(Path(argv[1]).read_text())
    new = json.loads(Path(argv[2]).read_text())

    # ---- the toolchain pin must be BYTE-IDENTICAL ---------------------------
    if old["toolchain"] != new["toolchain"]:
        print("FAIL  the toolchain pin moved. The rename cannot be separated from a\n"
              f"      re-ratification under a different assembler.\n"
              f"        old {old['toolchain']}\n        new {new['toolchain']}")
        return 1
    print(f"toolchain byte-identical  {old['toolchain']['ptxas']}")

    old_d = demangled_entries(old)
    new_d = demangled_entries(new)
    transformed = {apply_transform(k): v for k, v in old_d.items()}
    if len(transformed) != len(old_d):
        print("FAIL  the transform is not injective: it merged distinct old keys")
        return 1

    print(f"{len(old_d)} old entries, {len(new_d)} new entries")

    ok = True
    missing = sorted(set(transformed) - set(new_d))
    added = sorted(set(new_d) - set(transformed))
    if missing or added:
        ok = False
        print(f"\nFAIL  KEY SET. {len(missing)} transformed old key(s) absent from the "
              f"new manifest, {len(added)} new key(s) the transform does not produce. "
              f"The rename is not the only thing that changed.")
        for name in missing[:5]:
            print(f"    only-old  {name}")
        for name in added[:5]:
            print(f"    only-new  {name}")
    else:
        print("key sets identical under the declared transform")

    moved = [(k, transformed[k], new_d[k]) for k in sorted(set(transformed) & set(new_d))
             if transformed[k] != new_d[k]]
    if moved:
        ok = False
        print(f"\nFAIL  VALUE SET. {len(moved)} entr(ies) changed. A rename does not "
              f"move registers or spills, so CODEGEN MOVED -- stop and find out why "
              f"before proceeding:")
        for name, was, now in moved[:20]:
            print(f"    {name}\n        was {was}\n        now {now}")
        if len(moved) > 20:
            print(f"    ... and {len(moved) - 20} more")
    else:
        print("every value identical: registers, spill stores, spill loads")

    print(f"\n{'PASS' if ok else 'FAIL'}  §0.2 rename transform "
          f"(transformed old manifest == new manifest, keys AND values)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
