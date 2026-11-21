#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRAPH LIBRARY'S BUILD GATES AND RECORD: rule 1 (`tools/toolchains.py`'s `rule1`: the graph's ptxas, cuobjdump and
nvdisasm are the toolchain record's byte for byte -- the ties between that record and `MODULE.bazel`'s two mentions of
the toolkit's manifest -- and the environment's torch is the toolchain's CUDA major), rule 3 (every architecture is a
built one) and rule 5 (the vendored tree is its pin's), then `rola_cu13/_build_config.py` for one build's parameters
(`tools/build_record.py`'s writer). Run by `bazel/defs.bzl`'s `rola_extension`.
Docs: docs/build-graph.md#gates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / "bazel"), str(ROOT / "tools"), str(ROOT)]
import arms  # noqa: E402
import build_record  # noqa: E402
import toolchains  # noqa: E402

from measure import hardware  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--toolchain", required=True, help="the toolchain's name (its record tools/toolchains/<name>.json)")
    ap.add_argument("--ptxas", required=True, help="the graph's ptxas")
    ap.add_argument("--cuobjdump", required=True, help="the graph's cuobjdump")
    ap.add_argument("--nvdisasm", required=True, help="the graph's nvdisasm")
    ap.add_argument("--archs", required=True)
    ap.add_argument("--carry-table", type=Path, required=True, help="the carry family's generated table")
    ap.add_argument("--decode-table", type=Path, required=True, help="the decode family's generated table")
    ap.add_argument("--carry-arms", default="")
    ap.add_argument("--decode-arms", default="")
    ap.add_argument("--torch", required=True, help="the environment's torch version")
    ap.add_argument("--torch-cuda", required=True, help="the CUDA release the environment's torch was built for")
    ap.add_argument("--cxx11-abi", required=True, choices=("True", "False"))
    a = ap.parse_args()

    toolchain = toolchains.named(a.toolchain)
    archs = [x for x in a.archs.split(",") if x]
    built = hardware.built()
    build_record.rule3(archs, built)

    #: the binary gates' disassemblers are the record's too: MODULE.bazel names the toolkit's manifest twice
    ptxas = toolchains.rule1(toolchain, toolchains.version_text(a.ptxas), a.torch_cuda,
                             {"cuobjdump": toolchains.version_text(a.cuobjdump),
                              "nvdisasm": toolchains.version_text(a.nvdisasm)})
    print(f"rule 1 OK  toolchain {toolchain.name}, torch CUDA {a.torch_cuda}, ptxas: {ptxas}")
    carry_table, decode_table = arms.Table(a.carry_table), arms.Table(a.decode_table)
    carry = arms.family_arms(carry_table, a.carry_arms, "--//:carry_arms")
    decode = arms.family_arms(decode_table, a.decode_arms, "--//:decode_arms")
    cutlass = build_record.vendored_pin(ROOT / "csrc" / "third_party" / "cutlass")
    a.out.write_text(build_record.render(build_record.config(
        version=build_record.version(ROOT / "rola" / "__init__.py"), archs=archs,
        iteration=arms.is_iteration(carry, carry_table, decode, decode_table, archs, built),
        toolchain=toolchain.name,
        ptxas=ptxas, torch=a.torch, cxx11_abi=a.cxx11_abi == "True",
        cutlass_pin=cutlass)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
