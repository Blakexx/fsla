#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTER LIFE-RANGE REPORT (KERNEL_STANDARDS §22): peak LIVE registers per source region
of a kernel, beside ptxas's allocation. A budget argument cites the peak live at the program
point it concerns, never the allocation. The registers held at each instruction are warpline's
(`flow.pressure` over the checker's model with `nvdisasm --print-life-ranges`'s marks); what is this
tool's is the attribution of each instruction to its source region (docs/internals/tools/life_ranges.md).

    bazel run //tools:life_ranges -- --cubin <cubin> --source <kernel file>
    bazel run //tools:life_ranges -- --so <extension .so> --member <member> --kernel <regex> \
        --source <kernel file> [--arch sm_90a] [--top N]
    bazel run //tools:life_ranges -- --so <extension .so> --member <member> --kernel <regex> --source <kernel file> \
        --source <component files>... --budget tools/budgets/<kernel>.json   # the register budget

The cubin must carry line info (`-lineinfo`); an extension built without it reports the peak
and the histogram but no regions. Regions are the innermost source function (struct methods as
`Struct::method`, as `tools/region_ledger.py` names them).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
import checked  # noqa: E402
from region_ledger import fn_of, function_map, owner_of  # noqa: E402

#: a vector register, the registers a budget's `registers` line counts
_VECTOR = re.compile(r"R\d+")


def component_peaks(frames: dict[int, list[tuple[str, int]]], rows: list[tuple[int, str, int]], sources: list[Path],
                    components: dict[str, str]) -> dict[str, int]:
    """THE REGISTER BUDGET's reading (KERNEL_STANDARDS section 32): each budget component's peak live registers, an
    instruction attributed as `tools/region_ledger.py` attributes it (`owner_of`: the innermost inline frame in
    `sources` whose function is a component's, walking outward no further than the kernel's own body, else that body,
    its `glue`)."""
    fmaps = {p.name: function_map(p.read_text().splitlines()) for p in sources}
    by_function = {fn: name for name, fn in components.items()}
    glue = next((name for name, fn in components.items() if fn.endswith("_kernel")), None)
    peaks: dict[str, int] = collections.defaultdict(int)
    glue_fn = components.get(glue) if glue is not None else None
    for offset, _ins, live in rows:
        c = owner_of(frames.get(offset, []), fmaps, set(by_function), glue_fn)
        owner = by_function[c] if c is not None else glue
        if owner is not None:
            peaks[owner] = max(peaks[owner], live)
    return dict(peaks)


def cubin_of(args) -> Path:
    if args.cubin:
        return Path(args.cubin)
    return checked.cubin(Path(args.so), args.member, args.arch or checked.device_arch())


def line_map(frames: dict[int, list[tuple[str, int]]], source_name: str) -> dict[int, int | None]:
    """SASS offset -> innermost line in `source_name` of the function's inline frames."""
    return {offset: next((line for name, line in chain if name == source_name), None) for offset, chain in frames.items()}


def live_counts(cubin: Path, kernel: str | None) -> list[tuple[int, str, int]]:
    """(offset, instruction, live registers) per SASS instruction of the function `kernel` names: the vector registers warpline's
    flow holds across it (`flow.pressure`, over the life-range marks), the outlined subroutines' bodies left out."""
    k = checked.kernel_of(cubin, kernel)
    held = checked.checker().flow.pressure(k)
    return [(ins.address, f"{ins.opcode} {ins.operands}".strip(), sum(1 for r in held[ins.address] if _VECTOR.fullmatch(r)))
            for ins in k.instructions if ins.address in held]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubin", type=Path, default=None)
    ap.add_argument("--so", type=Path, default=None)
    ap.add_argument("--arch", default=None, help="with --so: the architecture (default: this machine's GPU)")
    ap.add_argument("--member", required=False, default=None, help="with --so: the cubin member's name")
    ap.add_argument("--kernel", default=None,
                    help="the function to read, a regex over the cubin's (needed where it holds several)")
    ap.add_argument("--source", type=Path, required=True, action="append",
                    help="the kernel's source (first) and, with --budget, its components' sources")
    ap.add_argument("--budget", type=Path, default=None,
                    help="a budget (tools/budgets/<kernel>.json): red where a component's peak live registers pass its "
                         "`registers` line, or it has none")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--json", type=Path, default=None, help="also write the peak and every region's live counts here")
    a = ap.parse_args()
    if (a.cubin is None) == (a.so is None) or (a.so is not None and a.member is None):
        ap.error("give --cubin, or --so with --member")
    cubin = cubin_of(a)
    rows = live_counts(cubin, a.kernel)
    if not rows:
        print("no life-range rows: is this a cubin of the kernel?")
        return 1

    source = a.source[0]
    frames = checked.frames(checked.inline_listing(cubin), a.kernel, cubin.name)
    lmap = line_map(frames, source.name)
    fmap = function_map(source.read_text().splitlines())
    peak = max(r[2] for r in rows)
    hist = collections.Counter((r[2] // 20) * 20 for r in rows)
    print(f"instructions {len(rows)}  peak live registers {peak}")
    print("live histogram (instructions per 20):",
          " ".join(f"{k}-{k + 19}:{v}" for k, v in sorted(hist.items())))
    per = collections.defaultdict(list)
    for off, _ins, live in rows:
        ln = lmap.get(off)
        per[fn_of(fmap, ln) if ln else "?"].append(live)
    if a.json:
        a.json.write_text(json.dumps({"instructions": len(rows), "peak_live": peak, "line_info": any(lmap.values()),
                                      "by_region": {fn: {"peak": max(v), "mean": round(sum(v) / len(v), 1), "instr": len(v)}
                                                    for fn, v in sorted(per.items())}}, sort_keys=True) + "\n")
    if not any(lmap.values()):
        print("no line info in this cubin: build with -lineinfo for regions")
        return 1 if a.budget else 0
    red = False
    if a.budget:
        budget = json.loads(a.budget.read_text())
        lines = budget.get("registers", {})
        peaks = component_peaks(frames, rows, a.source, budget["components"])
        print(f"\n{'component':14s} {'peak live':>9s} {'line':>5s}")
        for name in budget["components"]:
            line, got = lines.get(name), peaks.get(name, 0)
            over = line is None or got > line
            red |= over
            print(f"{'RED ' if over else 'ok  '}{name:10s} {got:9d} {'-' if line is None else line:>5}")
        if red:
            print(f"RED: a component past its `registers` line in {a.budget}, or with none")
    print(f"\n{'region':26s} {'peak':>5s} {'mean':>6s} {'instr':>6s}")
    for fn, v in sorted(per.items(), key=lambda x: -max(x[1]))[: a.top]:
        print(f"{fn:26s} {max(v):5d} {sum(v) / len(v):6.1f} {len(v):6d}")
    print("\npeak program points:")
    for off, ins, live in sorted(rows, key=lambda r: -r[2])[:6]:
        ln = lmap.get(off)
        print(f"  {off:#07x} live {live}  {fn_of(fmap, ln) if ln else '?'}:{ln}  {ins[:60]}")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
