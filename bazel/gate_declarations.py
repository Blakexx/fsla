#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SASS GATE'S DECLARATIONS, one file a built architecture: what fsla states for warpline's check of a library (bazel/defs.bzl's
`_sass_gate`, which runs `warpline_check_actions` once an architecture over the file written here). warpline's checker computes every
finding and makes the one judgment; what is fsla's is what the declarations say:

- the declarations beside the units (`--stated`, `{"architectures": {arch: [entry, ...]}}`, the library rule's from `cuda_unit(declarations
  = ...)`), then the arm families' (`--file`, `bazel/arm_emit.py`'s `declarations`, each entry with its architecture), in that order: the
  first declaration of a rule whose pattern names a kernel is the one that applies;
- each arm the build carries as a PRESENCE, its exact kernel pattern naming one function of its architecture's SASS (`exactly: 1`);
- every kernel the source defines (`tools/kernels.py`) as a presence, its name naming a function (`at_least: 1`): a gate that cannot see a
  kernel has not gated it. A library of probes (`--probes`, an instrument's) owes none of them;
- fsla's guidance a rule as the notes, printed after a refusal's fix (the `checker` section of `docs/rules.json`, `--rules`);
- `strict`: a declaration that allows a count no function it names has is refused, so the declarations say only what is so.

    gate_declarations.py --stated FILE [--file F]... --rules docs/rules.json [--probes] --out sm_90a=OUT [--out ...]

Docs: docs/internals/tools/sass_gate.md#declarations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools"))
import kernels  # noqa: E402 -- path insert must precede this import

_KEYS = ("kernel", "rule", "allow", "reason")


def source_pattern(name: str) -> str:
    """The pattern of a source kernel's name over a demangled function name: the name the toolkit's demangler writes before the
    parameter list, after the return type and the namespaces, its template arguments cut (`tools/kernels.py`'s `function_name`)."""
    return rf"^(?:\S+ )?(?:[^\s:<>()]+::)*{re.escape(name)}(?:<.*>)?(?:\(|$)"


def declarations(stated: dict, files: list[dict], rules: dict, probes: bool, source: list) -> dict[str, dict]:
    """Each architecture's declarations file (`stated`'s architectures: the build's), from the stated entries, the arm families'
    files, the `checker` section of `docs/rules.json` and the source's kernels (none for `probes`)."""
    out = {arch: [{k: d[k] for k in _KEYS} for d in entries] for arch, entries in stated["architectures"].items()}
    arms: dict[str, list[dict]] = {arch: [] for arch in out}
    for data in files:
        for d in data["declarations"]:
            if d["arch"] in out:
                out[d["arch"]].append({k: d[k] for k in _KEYS})
        for arm in data["arms"]:
            if arm["arch"] in out:
                arms[arm["arch"]].append(arm)
    notes = {rule: entry["guidance"] for rule, entry in rules["checker"].items()}
    owed = [] if probes else [{"kernel": source_pattern(k.name), "check": "present", "at_least": 1,
                               "reason": f"{k.where()}: `{k.name}` ({k.family}), a kernel the source defines: a gate that cannot see "
                                         "a kernel has not gated it"} for k in source]
    return {arch: {"declarations": [*entries,
                                    *({"kernel": arm["kernel"], "check": "present", "exactly": 1,
                                       "reason": f"arm {arm['family']}[{arm['arm']}], which the build carries: its exact kernel "
                                                 "pattern names one function"} for arm in arms[arch]),
                                    *owed],
                   "notes": notes, "strict": True}
            for arch, entries in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stated", type=Path, required=True, help="the declarations beside the units, by architecture")
    ap.add_argument("--file", action="append", default=[], type=Path, help="an arm family's declarations (bazel/arm_emit.py)")
    ap.add_argument("--rules", type=Path, required=True, help="docs/rules.json: its `checker` section's guidance are the notes")
    ap.add_argument("--probes", action="store_true", help="a library of probes: no kernel of the source is owed")
    ap.add_argument("--out", action="append", default=[], required=True, metavar="ARCH=PATH",
                    help="where an architecture's file is written; one for each architecture of --stated")
    a = ap.parse_args()
    found = declarations(json.loads(a.stated.read_text()), [json.loads(p.read_text()) for p in a.file],
                         json.loads(a.rules.read_text()), a.probes, [] if a.probes else kernels.kernels())
    outs = dict(o.split("=", 1) for o in a.out)
    if set(outs) != set(found):
        raise SystemExit(f"gate_declarations: --out names {sorted(outs)}, the build's architectures are {sorted(found)}")
    for arch, doc in found.items():
        Path(outs[arch]).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
