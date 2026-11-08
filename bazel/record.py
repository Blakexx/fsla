#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRAPH LIBRARY'S BUILD RECORD, AFTER RULES 1 AND 3: the graph's ptxas is the toolchain record's byte for byte
(rule 1, the one tie between the record and `MODULE.bazel`'s toolkit pin), every architecture is ratified (rule 3),
and `rola_cu13/_build_config.py` is written for one build's parameters (`tools/build_record.py`, the writer `setup.py`
uses). Run by the graph's library rule (`bazel/defs.bzl`'s `rola_extension`). Docs: docs/build-graph.md#compile.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_record  # noqa: E402
import gen_shards  # noqa: E402


def _list(value: str) -> str:
    return value.replace(",", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--toolchain", type=Path, required=True, help="the toolchain record (tools/toolchains/<name>.json)")
    ap.add_argument("--ptxas", required=True, help="the graph's ptxas")
    ap.add_argument("--archs", required=True)
    ap.add_argument("--carry-arms", default="")
    ap.add_argument("--decode-arms", default="")
    ap.add_argument("--carry-parts", default="all")
    ap.add_argument("--torch", required=True, help="the environment's torch version")
    ap.add_argument("--cxx11-abi", required=True, choices=("True", "False"))
    a = ap.parse_args()

    record = json.loads(a.toolchain.read_text())
    ptxas = " ".join(subprocess.run([a.ptxas, "--version"], capture_output=True, text=True, check=True).stdout.split())
    if ptxas != record["ptxas"]:
        raise SystemExit(f"CLOSED-WORLD RULE 1: the graph's ptxas is not the toolchain record's.\n  record: "
                         f"{record['ptxas']}\n  graph : {ptxas}\nThe manifests were ratified under the record's "
                         f"assembler; MODULE.bazel's toolkit pin must name that release.")
    archs = [x for x in a.archs.split(",") if x]
    manifests = ROOT / "tools" / "manifests" / a.toolchain.stem
    ratified = sorted(p.stem.removeprefix("sm_") for p in manifests.glob("sm_*.json"))
    manifest = build_record.ratified_manifest(archs, manifests)
    carry = gen_shards.family_arms(gen_shards.CARRY, _list(a.carry_arms), "--//:carry_arms")
    decode = gen_shards.family_arms(gen_shards.DECODE, _list(a.decode_arms), "--//:decode_arms")
    parts, trivial = gen_shards.carry_parts_spec(a.carry_parts, "--//:carry_parts")
    a.out.write_text(build_record.render(build_record.config(
        version=build_record.version(ROOT / "rola" / "__init__.py"), archs=archs,
        iteration=gen_shards.is_iteration(carry, decode, parts, trivial, archs, ratified), toolchain=a.toolchain.stem,
        ptxas=ptxas, torch=a.torch, cxx11_abi=a.cxx11_abi == "True", manifest=manifest,
        cutlass_pin=json.loads((ROOT / "csrc" / "third_party" / "cutlass" / "PIN.json").read_text()), sccache=None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
