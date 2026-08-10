"""FIRST CONTACT WITH THE PROJECT'S CANONICAL ORACLE.

`rola/ops/naive.py::naive_rola` is the repository's executable spec: routing-
native, consuming already-materialised per-level simplex allocations, deriving
its own leaf addresses. With routing (rung 1) and capacity (rung 2) the kernel's
inputs are finally that function's inputs -- one level tensor per level per side,
MSB-first mixed radix, N = prod widths -- so the ladder can be gated against the
spec itself rather than against a reference written beside it.

ONE THING DOES NOT LINE UP, AND IT IS NAMED RATHER THAN WORKED AROUND. The
oracle has no un-normalised mode: `normalization` admits only `"global"`
(`rola/routing/types.py::NORMALIZATION_KINDS`), and `"raw"` was deleted from the
contract. So the oracle always returns the RATIO

    y[t] = num[t] / (mass[t] + eps),     num[t] = sum_s R[t,s] S[t,s,:]

whereas this rung computes the NUMERATOR (normalisation is rung 5). The
numerator is recovered from the oracle EXACTLY, by algebra on the oracle's own
outputs rather than by reimplementing it:

    run 1: v            ->  y1 = num / (mass + eps)
    run 2: v + 1        ->  y2 = (num + mass) / (mass + eps)

because the recurrence is linear in the value and a value of all ones IS the
mass state's own recurrence -- the ones-column trick the oracle already uses,
applied from outside. Then

    rho  = y2 - y1 = mass / (mass + eps)   (identical in every d_v column: checked)
    mass = eps * rho / (1 - rho)
    num  = y1 * (mass + eps)

`eps` is a caller-supplied argument, so it is set to 1 rather than left at
`READOUT_EPS`: with `eps` far below `mass` the inversion would subtract two
nearly equal numbers, and the point of an oracle gate is that the oracle's
answer arrives intact.

The comparison is therefore against the oracle's EXACT fp64 coefficient -- the
design's own error, which includes the bf16 rounding of the amplitude product --
not against a reference that models the kernel's rounding.
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
from rung2 import build, reference  # noqa: E402

EPS = 1.0


def topology(widths, d_v=64):
    routing = IndependentRouting(tied=False, read_activation=softmax(),
                                 write_activation=softmax())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=d_v)


def oracle_numerator(v_bhld, pr, pw, widths):
    """`num[t] = sum_s R[t,s] S[t,s,:]`, from `naive_rola` and nothing else."""
    top = topology(widths)
    # [B,H,L,.] -> the oracle's [B,T,H,.]
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
    cases = [((64, 64), 64, 1, 1, 64), ((64, 64), 64, 1, 2, 256),
             ((64, 64), 32, 1, 1, 128), ((64, 64), 128, 1, 1, 128),
             ((16, 16), 64, 1, 2, 256), ((8, 8), 64, 2, 2, 128)]
    print(f"{'widths':>10s}{'BC':>5s}{'blocks':>8s}{'B':>3s}{'H':>3s}{'L':>6s}{'C':>4s}"
          f"{'rel_L2 vs naive_rola':>22s}{'design floor':>14s}{'ratio':>8s}"
          f"{'rho spread':>12s}")
    for widths, bc, B, H, L in cases:
        pr = reference.amplitudes(B, H, L, widths, dev, seed=11)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=12)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        num, spread = oracle_numerator(v, pr, pw, widths)
        # the design's own floor at this cell: the kernel's rounding modelled
        Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
        built = {a[0] for a in build.arms()
                 if not a[1] and a[2] == bc and (a[3], a[4]) == widths}
        for c in sorted(built):
            if L % c:
                continue
            got = build.forward(pr, pw, v, c, False, bc, *widths)
            torch.cuda.synchronize()
            floor = ((reference.split(Rb, Wb, v, c, bc) - num).norm()
                     / num.norm()).item()
            rel = ((got.double() - num).norm() / num.norm()).item()
            nb = (widths[0] * widths[1]) // bc
            print(f"{str(widths):>10s}{bc:5d}{nb:8d}{B:3d}{H:3d}{L:6d}{c:4d}"
                  f"{rel:22.3e}{floor:14.3e}{rel/floor:8.4f}{spread:12.2e}")
            # THE GATE: the disagreement with the oracle must BE the design's own
            # bf16 quantisation and not a hair more. An absolute tolerance would
            # be a different, weaker claim -- and one that a smaller N would fail
            # for the honest reason that its coefficients carry fewer slots.
            ok &= rel <= 1.02 * floor and spread < 1e-9
    return ok


if __name__ == "__main__":
    good = run()
    print("CANONICAL ORACLE:", "REACHED" if good else "NOT REACHED")
    sys.exit(0 if good else 1)
