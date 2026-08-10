"""THE CANONICAL ORACLE, AT BOTH ENDS OF THE SPARSITY SWEEP.

`rola/ops/naive.py::naive_rola` is the repository's executable spec. Rung 2
reached it; this rung must keep it, and must keep it on data the schedule can
actually skip -- a wrong skip is exactly the failure a spec gate is for.

ONE THING DOES NOT LINE UP, AND IT IS NAMED RATHER THAN WORKED AROUND. The
oracle has no un-normalised mode: `normalization` admits only `"global"`
(`rola/routing/types.py::NORMALIZATION_KINDS`), so it returns the RATIO

    y[t] = num[t] / (mass[t] + eps),     num[t] = sum_s R[t,s] S[t,s,:]

whereas this rung computes the NUMERATOR (normalisation is rung 5). The
numerator is recovered EXACTLY, by algebra on the oracle's own outputs:

    run 1: v            ->  y1 = num / (mass + eps)
    run 2: v + 1        ->  y2 = (num + mass) / (mass + eps)

because the recurrence is linear in the value and a value of all ones IS the
mass state's own recurrence. Then rho = y2 - y1 = mass/(mass+eps), identical in
every d_v column (checked), mass = eps*rho/(1-rho) and num = y1*(mass+eps).
`eps` is a caller argument and is set to 1 so the inversion never subtracts two
nearly equal numbers.

The comparison is against the oracle's EXACT fp64 coefficient -- the design's
own error -- and the GATE is the RATIO to the design's own bf16 quantisation
floor, not an absolute tolerance, because that floor moves with N and with the
support.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rola.ops.naive import naive_rola  # noqa: E402
from rola.routing.types import (  # noqa: E402
    IndependentRouting,
    LevelConfig,
    Topology,
    softmax,
)
from rung3 import build, reference  # noqa: E402

EPS = 1.0


def topology(widths, d_v=64):
    routing = IndependentRouting(tied=False, read_activation=softmax(),
                                 write_activation=softmax())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=d_v)


def oracle_numerator(v_bhld, pr, pw, widths):
    """`num[t] = sum_s R[t,s] S[t,s,:]`, from `naive_rola` and nothing else."""
    top = topology(widths)
    v = v_bhld.permute(0, 2, 1, 3).contiguous()
    read = tuple(x.permute(0, 2, 1, 3).contiguous()
                 for x in reference.split_levels(pr, widths))
    write = tuple(x.permute(0, 2, 1, 3).contiguous()
                  for x in reference.split_levels(pw, widths))
    B, T, H, _ = v.shape
    g_write = torch.ones(B, T, H, device=v.device, dtype=torch.float64)
    kw = dict(g_read=None, g_write=g_write, topology=top, decay=None,
              normalization="global", eps=EPS)
    y1, _ = naive_rola(v, read, write, **kw)
    y2, _ = naive_rola(v.double() + 1.0, read, write, **kw)
    rho = y2 - y1
    spread = (rho - rho[..., :1]).abs().max().item()
    mass = EPS * rho / (1.0 - rho)
    num = y1 * (mass + EPS)
    return num.permute(0, 2, 1, 3), spread


def run():
    ok = True
    dev = "cuda"
    #: (widths, BC, B, H, L, supports). The supports are the level width (dense)
    #: and two the schedule can act on.
    cases = [((64, 64), 64, 1, 1, 64, (64, 4, 1)),
             ((64, 64), 64, 1, 2, 256, (64, 4, 1)),
             ((64, 64), 32, 1, 1, 128, (64, 4)),
             ((64, 64), 128, 1, 1, 128, (64, 4)),
             ((16, 16), 64, 1, 2, 256, (16, 2)),
             ((8, 8), 64, 2, 2, 128, (8, 2))]
    print(f"{'widths':>10s}{'BC':>5s}{'blk':>5s}{'B':>3s}{'H':>3s}{'L':>6s}{'C':>4s}"
          f"{'sup':>5s}{'skip':>8s}{'rel_L2 vs naive_rola':>22s}{'design floor':>14s}"
          f"{'ratio':>8s}{'rho spread':>12s}")
    for widths, bc, B, H, L, sups in cases:
        for sup in sups:
            pr = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=11)
            pw = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=12)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            num, spread = oracle_numerator(v, pr, pw, widths)
            Rb = reference.kron_bf16(pr, widths)
            Wb = reference.kron_bf16(pw, widths)
            built = {a[0] for a in build.arms()
                     if not a[1] and a[2] == bc and (a[3], a[4]) == widths}
            for c in sorted(built):
                if L % c:
                    continue
                got = build.forward(pr, pw, v, c, False, bc, *widths)
                torch.cuda.synchronize()
                # AT THE SPARSEST SUPPORT THE ORACLE'S OWN ANSWER CAN BE ZERO --
                # no read digit coincides with a write digit anywhere in the
                # cell, and nothing routes. A ratio is then undefined, so the
                # absolute residual is gated instead of a NaN being compared.
                nrm = num.norm().item()
                dk = (got.double() - num).norm().item()
                df = (reference.split(Rb, Wb, v, c, bc) - num).norm().item()
                floor = df / nrm if nrm else df
                rel = dk / nrm if nrm else dk
                nb = (widths[0] * widths[1]) // bc
                skip = 1.0 - reference.schedule(pr, pw, widths, bc,
                                                c).double().mean().item()
                ratio = rel / floor if floor else (1.0 if rel == 0 else float("inf"))
                print(f"{str(widths):>10s}{bc:5d}{nb:5d}{B:3d}{H:3d}{L:6d}{c:4d}"
                      f"{sup:5d}{100*skip:7.1f}%{rel:22.3e}{floor:14.3e}"
                      f"{ratio:8.4f}{spread:12.2e}")
                # THE GATE: the disagreement with the oracle must BE the design's
                # own bf16 quantisation and not a hair more.
                ok &= ratio <= 1.02 and spread < 1e-9
    return ok


if __name__ == "__main__":
    good = run()
    print("CANONICAL ORACLE:", "REACHED" if good else "NOT REACHED")
    sys.exit(0 if good else 1)
