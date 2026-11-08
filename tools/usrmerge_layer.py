# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A PACKAGE LAYER FOR A MERGED-/USR BASE (the root package's `devbox_debs`): on Ubuntu 22.04 `/bin`, `/lib`, `/lib64`
and `/sbin` are links into `/usr`, and a layer that carries those paths as directories replaces the links, so `/bin/sh`
and the loader's own path go. Every entry under one of them moves under `usr/`, as dpkg writes it through the link, and
the entries for the four directories themselves go. Members keep their metadata, in order; a later one at a path wins.
Docs: docs/setup.md#the-graphs-image.

    usrmerge_layer.py --in flat.tar --out usrmerged.tar
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

#: the top-level directories a merged-/usr system links into `usr/`
MERGED = ("bin", "lib", "lib64", "sbin")


def merged_name(name: str) -> str | None:
    """Where an entry lands on a merged-/usr system: under `usr/` when it is under a merged directory, None for one of
    those directories itself, else where it was."""
    path = name.removeprefix("./")
    head, _, rest = path.partition("/")
    if head in MERGED:
        return f"./usr/{head}/{rest}" if rest else None
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="source", type=Path, required=True, help="the flattened package layer")
    ap.add_argument("--out", type=Path, required=True, help="the layer for a merged-/usr base")
    a = ap.parse_args()
    with tarfile.open(a.source) as src:
        members = []
        for member in src.getmembers():
            name = merged_name(member.name)
            if name is not None:
                members.append((name, member))
        last = {name: i for i, (name, _) in enumerate(members)}
        with tarfile.open(a.out, "w", format=tarfile.PAX_FORMAT) as out:
            for i, (name, member) in enumerate(members):
                if last[name] != i:
                    continue
                payload = src.extractfile(member) if member.isfile() else None
                member.name = name
                #: a hard link names its target by archive path, which moved as its target did
                if member.islnk():
                    member.linkname = merged_name(member.linkname) or member.linkname
                out.addfile(member, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
