# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LAB'S FACTS, read in the graph (bazel/lab.bzl's `lab_experiment`, docs/internals/tools/lab.md#graph): what one
experiment's binary says of itself to the graph's own disassemblers, written once as the files the lab reads -- the cubin,
`cuobjdump`'s listing and resource report, `nvdisasm`'s block graph and its listing with the frame chain, the demangled
names, the toolkit's version line. No lab command runs a toolkit tool: it reads these.

    lab_facts.py --binary B --ptxas LOG --cuobjdump T --nvdisasm T --cuxxfilt T --nvcc T
                 --cubin O --sass O --usage O --graph O --lines O --names O --toolkit O
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import sass  # noqa: E402 -- path insert must precede this import

#: a mangled symbol, as the toolkit's notices name a function by it
_MANGLED = re.compile(r"_Z[0-9A-Za-z_]+")


def run(*argv: str, cwd: str | None = None, text: str | None = None) -> str:
    return subprocess.run(argv, capture_output=True, input=text, text=True, check=True, cwd=cwd).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--ptxas", type=Path, required=True, help="the build's diagnostics: the functions they name are demangled too")
    for tool in ("cuobjdump", "nvdisasm", "cuxxfilt", "nvcc"):
        ap.add_argument(f"--{tool}", required=True, help=f"the graph's {tool}")
    for out in ("cubin", "sass", "usage", "graph", "lines", "names", "toolkit"):
        ap.add_argument(f"--{out}", type=Path, required=True)
    a = ap.parse_args()
    #: by absolute path: the extraction runs in a directory of its own
    a.cuobjdump, a.nvdisasm, a.cuxxfilt, a.nvcc = (str(Path(t).absolute()) for t in (a.cuobjdump, a.nvdisasm, a.cuxxfilt, a.nvcc))
    listing = run(a.cuobjdump, "-sass", str(a.binary))
    a.sass.write_text(listing)
    a.usage.write_text(run(a.cuobjdump, "-res-usage", str(a.binary)))
    with tempfile.TemporaryDirectory(prefix="rola_lab_cubins_") as scratch:
        run(a.cuobjdump, "-xelf", "all", str(a.binary.absolute()), cwd=scratch)
        found = sorted(Path(scratch).glob("*.cubin"))
        if len(found) != 1:
            raise SystemExit(f"lab_facts: {len(found)} ELF images in {a.binary.name}; an experiment carries one cubin")
        shutil.copyfile(found[0], a.cubin)
    a.graph.write_text(run(a.nvdisasm, "-bbcfg", "-poff", str(a.cubin)))
    a.lines.write_text(run(a.nvdisasm, "--print-line-info-inline", "-gi", str(a.cubin)))
    #: every function the listing or the notices name, demangled by the toolkit's own `cu++filt`
    functions = [m.group(1) for line in listing.splitlines() if (m := sass.FUNCTION.match(line))]
    named = list(dict.fromkeys([*functions, *_MANGLED.findall(a.ptxas.read_text())]))
    pretty = run(a.cuxxfilt, text="\n".join(named)).splitlines()
    a.names.write_text("".join(f"{m}\t{p}\n" for m, p in zip(named, pretty, strict=True)))
    a.toolkit.write_text(run(a.nvcc, "--version"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
