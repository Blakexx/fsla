#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PHASE LEDGER RUN: one carry launch on a cell with the kernel's phase clock bound, printed
as cycles a warp a window per phase, averaged over the CTAs and warps; and per warp, so an
owner imbalance shows. KERNEL_STANDARDS §19's per-phase instrument for the composed kernel.

    python tools/phase_ledger.py flagship-alt-k4 [--launches 3]

The ledger is `rola.ops.carry.LEDGER_PHASES` columns by `LEDGER_WARPS` warps by one row a CTA, a CTA a box of
the carve (docs/internals/carry/carry_kernel.md#phase-ledger). Measured work: takes the GPU lock exclusive.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ctas(cell: str) -> int:
    """The launch's CTAs: the carve's boxes (one batch-head: a registry cell draws one)."""
    from measure.cells import by_name
    from measure.subjects import cell_carve

    spec = by_name(cell)
    carve = cell_carve(spec)
    return (spec.dv // carve.channels) * math.prod(w // e for w, e in zip(carve.widths, carve.extents))


def cta_windows(cell: str) -> int:
    """The launch's CTA-windows, CTAs times windows: the one divisor that turns a launch total into a unit."""
    from measure.cells import by_name
    from rola.ops import carry as c

    return ctas(cell) * ((by_name(cell).tokens + c.WINDOW - 1) // c.WINDOW)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="launches run before the ledger is bound (a fresh binary's first launch is not a reading)")
    ap.add_argument("--per-warp", default="readout,fold",
                    help="the phases whose per-warp rows are printed: an imbalance is seen, not inferred")
    ap.add_argument("--json", type=Path, default=None, help="also write the reading here: every phase, every warp")
    ap.add_argument("--forced", default=None,
                    help="the plan form each window takes where it fits (`rola.ops.carry.forced_word`)")
    a = ap.parse_args()

    import torch
    from rola_devtools.locks.gpu import gpu_lock

    from measure.cells import by_name
    from measure.subjects import carry_operands
    from rola.ops import carry as c

    spec = by_name(a.cell)

    def launch(*ops):
        return c.carry(*ops, forced=a.forced)

    operands = carry_operands(spec)
    phases = c.LEDGER_PHASES
    warps = c.LEDGER_WARPS
    n_ctas = ctas(a.cell)
    windows = (spec.tokens + c.WINDOW - 1) // c.WINDOW
    with gpu_lock(mode="exclusive"):
        ledger = torch.zeros((n_ctas, warps, len(phases)), dtype=torch.int64, device="cuda")
        for _ in range(a.warmup):
            launch(*operands)
        torch.cuda.synchronize()
        c.ledger_bind(ledger)
        for _ in range(a.launches):
            launch(*operands)
        torch.cuda.synchronize()
        c.ledger_bind(None)
    per = ledger.double().mean(dim=(0, 1)).cpu() / a.launches / windows
    print(f"{a.cell}: cycles a warp a window (mean over CTAs, warps): "
          + ", ".join(f"{n} {x:.0f}" for n, x in zip(phases, per.tolist())) + f"  total {per.sum():.0f}")
    by_warp = ledger.double().mean(dim=0).cpu() / a.launches / windows
    for ph in a.per_warp.split(","):
        i = phases.index(ph)
        print(f"  {ph:8s} a warp: " + " ".join(f"{by_warp[w][i]:6.0f}" for w in range(warps)))
    if a.json:
        a.json.write_text(json.dumps({"cell": a.cell, "launches": a.launches, "warmup": a.warmup,
                                      "cta_windows": n_ctas * windows,
                                      "per_phase": {n: round(x, 1) for n, x in zip(phases, per.tolist())},
                                      "total": round(float(per.sum()), 1),
                                      "per_warp": {n: [round(by_warp[w][i].item(), 1) for w in range(warps)]
                                                   for i, n in enumerate(phases)}}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
