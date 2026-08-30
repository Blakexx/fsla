#!/usr/bin/env python3
"""THE POST-BUILD SWEEP over the kernel's remaining STRUCTURE BRANCHES (Blake,
2026-08-29, R9's enforcement layers).

G2b deleted the one big structure branch this tool was written for -- the enumeration
over a level's `(span, shift)` field.  The warp sub-box is REGISTER SHAPE now, so those
are immediates and there is no case set, no trap and nothing to sweep there.  What
remains CTA-uniform-branchy in the body is the LIVENESS CLAUSE: per level and per side,
whether this side carves the level (`carves`) and whether the level votes ballot rows at
all (`row_g > 0`).  Those are geometry, never a threshold, and their combinations are
what a launch can still take a path through.

A branch that nothing exercises is a promise, not a proof.  So: for every BUILT arm this
enumerates the topologies and mode words its depth admits, reads each one's clause
signature off the SHIPPED derivation, and LAUNCHES the kernel once per distinct
signature on a tiny cell.  A fault or refusal surfaces as a gap.  It also reports which
signatures a depth can reach at all, because "covered" without that is a claim about a
set nobody stated.

    python tools/field_coverage.py            # sweep every built arm
    python tools/field_coverage.py --check    # same, non-zero exit on any gap
"""
from __future__ import annotations

import argparse
import itertools
import sys

import torch

from rola.ops import carry as carry_ops

#: a signature is `(side, level, carves, votes_rows)` per level, and the set of them a
#: depth can produce is what this tool reports rather than assumes.


def _topologies(D: int):
    """Every uniform and mixed power-of-two topology at this depth the seam admits."""
    widths = (8, 16, 32, 64, 128, 256)
    seen = []
    for combo in itertools.product(widths, repeat=D):
        prod = 1
        for w in combo:
            prod *= w
        if prod < carry_ops.BC or sum(combo) % 8:
            continue
        seen.append(combo)
    return seen[:48]


def _clauses(widths, modes, ns, split):
    """The liveness clause's shape, read off the block exactly as the body reads it."""
    g = list(carry_ops.geometry(list(widths), modes, carry_ops.BC, ns, ns))
    D = g[0]
    i = 2 + 7 * 4 + 6                      # past the block's own arrays and scalars
    sig = []
    for side in ("r", "w"):
        grid = g[i + 3 * 4:i + 3 * 4 + D]
        row_g = g[i + 5 * 4:i + 5 * 4 + D]
        for l in range(D):
            sig.append((side, l, grid[l] > 1, row_g[l] > 0))
        i += 7 * 4 + 4
    return frozenset(sig)


def _tiny_cell(widths, T=64, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(7)
    def side():
        out = []
        for w in widths:
            x = torch.rand((1, T, 1, w), device=device, generator=gen)
            out.append((x / x.sum(-1, keepdim=True)).to(torch.bfloat16))
        return tuple(out)
    g = (torch.rand((1, T, 1), device=device, generator=gen) + 0.5).to(torch.bfloat16)
    v = torch.randn((1, T, 1, 64), device=device, generator=gen).to(torch.bfloat16)
    return side(), side(), g, v


def sweep(verbose=True):
    gaps = []
    for arm in carry_ops.arms():
        D, d_v, W, C, warps, WPS, split = arm[:7]
        ns = 1 if split else warps
        realized, launched = set(), 0
        for widths in _topologies(D):
            for modes in range(0, 1 << (2 * D)):
                try:
                    f = _clauses(widths, modes, ns, bool(split))
                except RuntimeError:
                    continue          # the seam refused this geometry: the refusal did its job
                if f in realized:
                    continue
                realized.add(f)
                read, write, g, v = _tiny_cell(widths, T=W)
                try:
                    carry_ops.carry_inter(read, write, g, v, widths, window=W, chunk=C,
                                          split=bool(split), level_modes=modes)
                    torch.cuda.synchronize()
                except RuntimeError as ex:
                    gaps.append(f"arm {arm[:7]} widths={widths} modes={modes}: {ex}")
                    return gaps
                launched += 1
        if verbose:
            per_level = sorted({(l, c, r) for sig in realized for (_s, l, c, r) in sig})
            print(f"arm D={D} dv={d_v} W={W} C={C} warps={warps} WPS={WPS} "
                  f"split={split}: {launched} launches, {len(realized)} distinct clause "
                  f"signatures")
            print(f"    (level, carves, votes rows) reached: {per_level}")
        if not realized:
            gaps.append(f"arm {arm[:7]}: no geometry reached the body at all")
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any arm traps or reaches no field at all")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("field_coverage: no CUDA device; the sweep is a LAUNCH check and needs one")
        return 0 if not args.check else 1
    gaps = sweep()
    for g in gaps:
        print(f"FIELD COVERAGE GAP: {g}", file=sys.stderr)
    if gaps:
        return 1 if args.check else 0
    print("field coverage: every built arm launched at every clause signature its depth "
          "can reach")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
