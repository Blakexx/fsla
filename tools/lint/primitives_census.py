#!/usr/bin/env python3
"""SASS-SHAPE CENSUS for arch-selected primitives.

**WHAT THIS CATCHES.** `csrc/rola/src/common/ops.cuh` names a primitive by its
CONTRACT (KERNEL_STANDARDS section 8: "feature selection lives where the
primitive is defined ... never as `#if` at call sites") and picks a NATIVE
instruction per architecture behind an `arch::Caps` flag. The failure mode this
gate exists to catch is the `mul_bf16x2`-before-K35 shape: a primitive quietly
falls back to a WIDER-TYPE EMULATION (unpack to fp32, compute, repack) on some
built arch instead of the one native instruction the ISA actually offers
there -- and nothing before this gate notices, because the emulation is
functionally correct and only costs SASS shape. `mul_bf16x2` passed the fp64
oracle for months carrying exactly this defect; it was found by the K4
instruction ledger, not by an audit. This file is the audit, made permanent:
an emulation FAILS THE DAY IT IS WRITTEN rather than the day someone happens to
ledger that kernel again.

**HOW.** One probe translation unit (`primitives_census_probe.cu`) `#include`s `ops.cuh` and calls each primitive from
its own `__global__` kernel; the graph's CUDA rule compiles it for the build's architectures (tools/lint/BUILD.bazel's
`primitives_census_probe`), and this test reads the object's SASS with the graph's `cuobjdump -sass` and counts opcodes by
FAMILY (the mnemonic up to its first `.` modifier -- `HFMA2.BF16_V2` and `HFMA2.MMA` are both family `HFMA2`). Each cell
asserts an exact count for one or more families; each table row's comment states the instruction the primitive must lower to
and what its emulation would show instead, and this file asserts the shape that comment gives.

**BUILT ARCHES ONLY.** The probe is compiled for the build's architectures (`BUILT_ARCHS`), so this gate and the real
build can never disagree about which arches "built" means; each is read from the SASS's own `arch = sm_XX` line.

**WIRING.** A graph test, a compile-only gate: no GPU, no `flock`. `bazel test //tools/lint:primitives_census`.

Usage (no GPU required):

    bazel test //tools/lint:primitives_census --test_output=all     # every cell is printed
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

#: THE CENSUS TABLE.  Each row: (primitive, function symbol, family -> exact
#: count).  A family absent from a row's dict is NOT asserted (e.g. `mul_bf16x2`
#: does not assert `PRMT`, because `hi_bf16x2`'s presence there is a different
#: primitive's contract) -- only families that WOULD appear under emulation, or
#: that the native route is defined by, are asserted, per the card's own
#: worked example ("exactly 1 HFMA2 and 0 FMUL").
CENSUS: dict[str, dict] = {
    "mul_bf16x2": {
        "symbol": "census_mul_bf16x2",
        #: FMA route (sm_80/86/87/89, K35 R-C): one `fma.rn.bf16x2` with a
        #: `-0.0` addend lowers to one HFMA2; the WIDER-TYPE EMULATION this
        #: gate exists to catch would show FMUL (fp32 multiply) plus PRMT/pack
        #: traffic instead.
        "families": {"HFMA2": 1, "FMUL": 0, "F2FP": 0},
    },
    "pack_bf16x2": {
        "symbol": "census_pack_bf16x2",
        #: `__floats2bfloat162_rn`: one packed fp32->bf16x2 convert (F2FP
        #: family on every tabulated arch -- the operand order differs
        #: between sm_80's PACK_AB and sm_90's F32.PACK_AB, both same family).
        "families": {"F2FP": 1, "FMUL": 0, "FADD": 0},
    },
    "hi_bf16x2": {
        "symbol": "census_hi_bf16x2",
        #: byte-select PRMT, unconditioned by any Caps flag -- available on
        #: every built arch. An emulation route here would show FMUL/F2FP
        #: traffic where one PRMT suffices.
        "families": {"PRMT": 1, "FMUL": 0, "F2FP": 0},
    },
}

_OPCODE_RE = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9_]*)")


def opcode_families(sass_text: str) -> dict[str, int]:
    """Count SASS opcodes by FAMILY: the mnemonic up to (excluding) its first
    `.` modifier -- the capturing group already stops there (`[A-Z][A-Z0-9_]*`
    contains no `.`), so `HFMA2.BF16_V2` and `HFMA2.MMA` both count as `HFMA2`.
    """
    counts: dict[str, int] = {}
    for line in sass_text.splitlines():
        m = _OPCODE_RE.search(line)
        if not m:
            continue
        family = m.group(1)
        if family in ("NOP", "BRA", "EXIT"):
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def disassemble(obj: Path, cuobjdump: Path) -> dict[str, dict[str, dict[str, int]]]:
    """The object's SASS by architecture and function, each as its opcode families: a count never leaks across probe
    kernels sharing one object."""
    sass = subprocess.run([str(cuobjdump), "-sass", str(obj)], capture_output=True, text=True, check=True).stdout
    per_function: dict[str, dict[str, list[str]]] = {}
    arch = function = None
    for line in sass.splitlines():
        m = re.match(r"\s*arch = (sm_\w+)", line)
        if m:
            arch, function = m.group(1), None
            continue
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            function = m.group(1)
            per_function.setdefault(arch, {})[function] = []
        elif function is not None:
            per_function[arch][function].append(line)
    return {arch: {name: opcode_families("\n".join(lines)) for name, lines in functions.items()}
            for arch, functions in per_function.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", type=Path, required=True, help="the probe's object, the graph's compile of it")
    ap.add_argument("--cuobjdump", type=Path, required=True, help="the graph's cuobjdump")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every cell's measured opcode families, not just failures")
    args = ap.parse_args()

    ok = True
    by_arch = disassemble(args.object, args.cuobjdump)
    if not by_arch:
        raise SystemExit(f"primitives_census: {args.object} holds no SASS")
    for arch, by_function in sorted(by_arch.items()):
        print(f"=== {arch} ===")
        for prim, row in CENSUS.items():
            symbol = row["symbol"]
            measured = by_function.get(symbol)
            if measured is None:
                ok = False
                print(f"  FAIL  {prim}: symbol {symbol} not found in disassembly "
                      f"(compiled but produced no code? check the probe TU)")
                continue
            mismatches = []
            for family, expected in row["families"].items():
                got = measured.get(family, 0)
                if got != expected:
                    mismatches.append((family, expected, got))
            if mismatches:
                ok = False
                print(f"  FAIL  {prim} ({arch}): SASS shape moved")
                for family, expected, got in mismatches:
                    print(f"          {family}: expected {expected}, measured {got}")
                print(f"          full measured opcode counts: {measured}")
            elif args.verbose:
                wanted = {k: measured.get(k, 0) for k in row["families"]}
                print(f"  pass  {prim} ({arch}): {wanted}")

    print(f"\n{'PASS' if ok else 'FAIL'}  primitives census "
          f"({len(CENSUS)} primitive(s) x {len(by_arch)} built arch(es))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
