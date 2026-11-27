#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REALIZED SUPPORT: what a layer cell's producer routes -- per level and side, the nonzero digits a token in the
rows the kernels read (the bundle, after quantization), as the mean, the median, the 99th percentile and the level's
fraction, and whether the side is declared sparse. A layer reading's regime is this, not the cell's name: a carry cell
DECLARES its support, a layer cell's weights realize one.

    bazel run //tools:realized_support -- [layer-alt-entmax1.5-256x256-T65536-h4] [--json support.json]   # one layer cell, or every one

Docs: docs/internals/tools/realized_support.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def support(routes) -> list[dict]:
    """Each level's and side's nonzero digits a token, over every batch row, token and head."""
    import torch

    from fsla.routing.types import activation_walks

    walks = activation_walks(routes.topology)
    out = []
    for level in range(routes.topology.D):
        for side, flag in (("read", 0), ("write", 1)):
            rows = routes.rows(level, side)
            count = (rows != 0).sum(dim=-1).flatten().double()
            q = torch.quantile(count[:: max(1, count.numel() // (1 << 24))], torch.tensor([0.5, 0.99], device=count.device,
                                                                                         dtype=torch.float64))
            width = rows.shape[-1]
            out.append({"level": level, "side": side, "sparse": bool(walks[flag][level]), "width": width,
                        "mean": round(float(count.mean()), 2), "median": float(q[0]), "p99": float(q[1]),
                        "fraction": round(float(count.mean()) / width, 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", nargs="?", default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    from fsla_devtools.cells import build as build_cell
    from fsla_devtools.cells import central
    from fsla_devtools.locks.gpu import gpu_lock

    from measure.cells import by_name
    from measure.cells.layer import build

    chosen = [by_name(a.cell)] if a.cell else [c for c in map(build_cell, central().cells.values()) if c.kind == "layer"]
    rows = []
    with gpu_lock(mode="shared"):
        for cell in chosen:
            for r in support(build(cell)["routes"]):
                rows.append({"cell": cell.name, **r})
    for r in rows:
        print(f"{r['cell']} level {r['level']} {r['side']:5s} {'sparse' if r['sparse'] else 'dense '} "
              f"{r['mean']:7.2f} a token (median {r['median']:.0f}, p99 {r['p99']:.0f}) of {r['width']}, "
              f"{100 * r['fraction']:.1f}%")
    if a.json:
        a.json.write_text(json.dumps({"support": rows}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
