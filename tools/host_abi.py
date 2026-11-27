# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HOST TOOLCHAIN AND THE LIBRARY'S ABI, read through the graph (`//tests:host_abi`): the one C++ toolchain is the
pinned GCC with its own sysroot (MODULE.bazel's `gcc_toolchain`), and the extension it links loads wherever torch does.

The probe (`tests/unit/fixtures/host_toolchain_probe.cpp`, compiled by that toolchain) prints the compiler's version
and the C and C++ libraries its headers are; the pinned binutils' `readelf --version-info` lists every symbol version `_C.abi3.so`
needs, by the library it needs it from. The check holds the compiler to the pin's release, the C++ ABI to torch's
(the new `std::string`), and each needed version to its floor: glibc to torch's own (`manylinux_2_28`) and libstdc++
to the runtime Ubuntu 22.04 carries (GCC 12's, `GLIBCXX_3.4.30`, `CXXABI_1.3.13`) -- above a floor the library would
fail to load on the desk or in the devbox image. Docs: docs/build-graph.md#environment.

    host_abi.py --probe BIN --readelf BIN --library FILES... --compiler 12.5.0 --floor GLIBC_2.28 --floor GLIBCXX_3.4.30 ...
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

#: a needed version in `readelf --version-info`'s `.gnu.version_r` section: `Name: GLIBCXX_3.4.21  Flags: none ...`
_NEEDED = re.compile(r"Name: ([A-Z]+)_([0-9][0-9.]*)\b")


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def needed(readelf: str, so: str) -> dict[str, tuple[int, ...]]:
    """The highest version of each symbol family (`GLIBC`, `GLIBCXX`, `CXXABI`, `GCC`) the library needs."""
    listing = subprocess.run([readelf, "--wide", "--version-info", so], capture_output=True, text=True, check=True).stdout
    section = listing.split("Version needs section", 1)[-1]
    highest: dict[str, tuple[int, ...]] = {}
    for family, version in _NEEDED.findall(section):
        highest[family] = max(highest.get(family, ()), _version(version))
    return highest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="the probe the toolchain compiled")
    ap.add_argument("--readelf", required=True, help="the pinned binutils' readelf")
    ap.add_argument("--library", nargs="+", required=True, help="`//:library`'s files, the linked extension among them")
    ap.add_argument("--compiler", required=True, help="the pinned GCC release")
    ap.add_argument("--floor", action="append", default=[], help="FAMILY_VERSION: the highest version the library may need")
    a = ap.parse_args()
    facts = dict(line.split(" ", 1) for line in subprocess.run([a.probe], capture_output=True, text=True,
                                                                check=True).stdout.splitlines())
    print("the probe:", ", ".join(f"{k} {v}" for k, v in facts.items()))
    red = []
    if facts["compiler"].split(" ", 1)[0] != a.compiler:
        red.append(f"the compiler is {facts['compiler']!r}, not the pinned GCC {a.compiler}")
    if facts["cxx11_abi"] != "1":
        red.append("the C++ library's ABI is the old std::string: torch's extensions take the new one")
    so = [f for f in a.library if f.endswith("/_C.abi3.so")]
    if len(so) != 1:
        raise SystemExit(f"host_abi: the library's files name {len(so)} `_C.abi3.so`, not one: {a.library}")
    highest = needed(a.readelf, so[0])
    print("the library needs:", ", ".join(f"{k}_{'.'.join(map(str, v))}" for k, v in sorted(highest.items())))
    for floor in a.floor:
        family, version = floor.split("_", 1)
        if family not in highest:
            red.append(f"the library needs no {family} version: the floor {floor} reads nothing (a wrong library?)")
        elif highest[family] > _version(version):
            red.append(f"the library needs {family}_{'.'.join(map(str, highest[family]))}, above the floor {floor}")
    for line in red:
        print("RED ", line, file=sys.stderr)
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
