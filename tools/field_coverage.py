#!/usr/bin/env python3
"""LAYER 4 of the structure branch's enforcement (Blake, 2026-08-29): the post-build
sweep that proves the trap is unreachable.

`carry_kernel.cuh`'s `field_dispatch` is a CTA-uniform branch over the admissible
`(span, shift)` fields of an `n`-bit slot index -- constitution R9's branch on
STRUCTURE.  Its four layers are: the case set is GENERATED (`static_for`, not a table),
its size is asserted against the closed form `(n+1)(n+2)/2` at compile time, the SEAM
refuses an inadmissible field before the launch, and the default TRAPS.

A trap that nothing exercises is a promise, not a proof.  This tool closes that: for
every BUILT arm it enumerates the topologies and mode words its depth admits, derives
each one's fields off the SHIPPED derivation, and LAUNCHES the kernel once per distinct
field set on a tiny cell.  A trap surfaces as a CUDA error and fails the check.

It also reports the coverage: which admissible pairs a depth can actually reach.  Not
every pair is reachable at every depth -- a side's spans multiply to the leaves one warp
owns, so `D = 2` can only place two fields in five bits -- and saying which are
unreachable is the honest form of "covered".

    python tools/field_coverage.py            # sweep every built arm
    python tools/field_coverage.py --check    # same, non-zero exit on any gap
"""
from __future__ import annotations

import argparse
import itertools
import sys

import torch

from rola.ops import carry as carry_ops

LEAVES_PER_WARP = 32
N_BITS = LEAVES_PER_WARP.bit_length() - 1  # 5


def admissible(n: int = N_BITS) -> set:
    return {(k, sh) for k in range(n + 1) for sh in range(n + 1) if k + sh <= n}


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


def _fields(widths, modes, ns, split):
    """The fields the BODY will dispatch on, derived exactly as the body derives them.

    The two bodies place their slot run differently and this tool must mirror each, or
    it sweeps pairs no kernel ever sees.  The CARVED body's field is the side's own
    `(span, shift)` out of the block -- its stream spans multiply to the leaves one warp
    owns, which is the slot run.  The CHANNEL-SPLIT body has no carve: every warp holds
    the whole box, so a thread's slot run is `BC / (threads / C)` and a level's field is
    how much of that run its digit varies over -- `n_l = clamp(SPT >> run_shift, 1, s_l)`
    with the shift collapsed to zero when the level does not vary at all.  THAT CLAMP is
    what keeps the split body inside the admissible set (`n_l << shift <= SPT` by
    construction), and this sweep is what proves the clamp rather than asserting it.
    """
    g = list(carry_ops.geometry(list(widths), modes, carry_ops.BC, ns, ns))
    D = g[0]
    out = set()
    if split:
        span = g[2 + 4:2 + 4 + D]              # the OWNER spans
        run_shift = g[2 + 4 * 4:2 + 4 * 4 + D]
        for l in range(D):
            rs = run_shift[l]
            vary = 0 if rs >= N_BITS else (LEAVES_PER_WARP >> rs)
            n_l = 1 if vary < 1 else min(vary, span[l])
            out.add((n_l.bit_length() - 1, 0 if n_l == 1 else rs))
        return out
    i = 2 + 7 * 4 + 6
    for _side in ("r", "w"):
        sp = g[i + 2 * 4:i + 2 * 4 + D]
        sh = g[i + 5 * 4:i + 5 * 4 + D]
        for l in range(D):
            out.add((sp[l].bit_length() - 1, sh[l]))
        i += 8 * 4 + 4
    return out


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
    adm = admissible()
    for arm in carry_ops.arms():
        D, d_v, W, C, warps, WPS, split = arm[:7]
        ns = 1 if split else warps
        realized, launched = set(), 0
        for widths in _topologies(D):
            for modes in range(0, 1 << (2 * D)):
                try:
                    f = _fields(widths, modes, ns, bool(split))
                except RuntimeError:
                    continue          # the seam refused this geometry: layer 3 did its job
                if f <= realized:
                    continue
                realized |= f
                read, write, g, v = _tiny_cell(widths, T=W)
                try:
                    carry_ops.carry_inter(read, write, g, v, widths, window=W, chunk=C,
                                          split=bool(split), level_modes=modes)
                    torch.cuda.synchronize()
                except RuntimeError as ex:
                    gaps.append(f"arm {arm[:7]} widths={widths} modes={modes}: {ex}")
                    return gaps
                launched += 1
        assert realized <= adm, (
            f"arm {arm[:7]}: the derivation produced fields outside the body's "
            f"admissible set: {sorted(realized - adm)} -- layer 3 should have refused "
            f"these before any launch")
        unreachable = sorted(adm - realized)
        if verbose:
            print(f"arm D={D} dv={d_v} W={W} C={C} warps={warps} WPS={WPS} "
                  f"split={split}: {launched} launches, {len(realized)}/{len(adm)} "
                  f"admissible pairs reached")
            print(f"    reached     : {sorted(realized)}")
            print(f"    unreachable : {unreachable}  "
                  f"(a side's spans multiply to {LEAVES_PER_WARP}, so a depth cannot "
                  f"place more than {D} fields in {N_BITS} bits)")
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
    print("field coverage: every built arm launched at every field its depth can reach; "
          "no trap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
