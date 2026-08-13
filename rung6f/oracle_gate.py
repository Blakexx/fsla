"""THE CANONICAL ORACLE, REACHED DIRECTLY, WITH THE COMPACTED KERNEL.

`rola/ops/naive.py::naive_rola` is the repository's executable spec, and since
rung 4 the comparison against it is the plain one:

    naive_rola(v, read, write, normalization="global", eps=eps)   vs   forward()

with the SAME `eps`. `g_read` is `None` because spec 4.2 fixes `g_read := 1`
under global normalisation, which is exactly what the kernel implements by not
having one. `forward()` here is the table pass, the chunk kernel and the divide
pass -- the whole of what a caller runs.

The comparison is against the oracle's EXACT fp64 coefficient -- the design's own
error -- and the GATE is the RATIO to the design's own bf16 quantisation floor,
not an absolute tolerance, because that floor moves with N and with the support.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rola.ops.constants import READOUT_EPS  # noqa: E402
from rola.ops.naive import naive_rola  # noqa: E402
from rola.routing.types import (  # noqa: E402
    IndependentRouting,
    LevelConfig,
    Topology,
    softmax,
)
from rung5b import reference  # noqa: E402
from rung6f import build  # noqa: E402

#: THE SECOND FLOOR, AND WHY THERE ARE TWO. The design's bf16 floor is not the
#: only lower bound on the disagreement: the kernel accumulates in fp32 through a
#: fan-in whose order is undefined, and it divides in fp32. Where the routing is
#: sparse enough that every coefficient is EXACTLY representable in bf16 the
#: design floor collapses to zero, and what is left is that fp32 precision --
#: measured, not assumed, at <= 2.0e-7 over five runs of every multi-block arm
#: (`check.py` layer 9). A ratio to a zero floor is not a gate, so the binding
#: bound is the larger of the two.
KERNEL_FP32_FLOOR = 2.0e-7


def topology(widths, d_v=64):
    routing = IndependentRouting(tied=False, read_activation=softmax(),
                                 write_activation=softmax())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=d_v)


def _oracle_args(v_bhld, pr, pw, widths):
    top = topology(widths)
    v = v_bhld.permute(0, 2, 1, 3).contiguous()
    read = tuple(x.permute(0, 2, 1, 3).contiguous()
                 for x in reference.split_levels(pr, widths))
    write = tuple(x.permute(0, 2, 1, 3).contiguous()
                  for x in reference.split_levels(pw, widths))
    B, T, H, _ = v.shape
    g_write = torch.ones(B, T, H, device=v.device, dtype=torch.float64)
    return v, read, write, dict(g_read=None, g_write=g_write, topology=top,
                                decay=None, normalization="global")


def oracle_ratio(v_bhld, pr, pw, widths, eps=READOUT_EPS):
    """`y = num/(den+eps)` from `naive_rola` and nothing else. [B,H,L,d_v]."""
    v, read, write, kw = _oracle_args(v_bhld, pr, pw, widths)
    y, _ = naive_rola(v, read, write, eps=eps, **kw)
    return y.permute(0, 2, 1, 3)


def rel(got, ref):
    n = ref.norm().item()
    d = (got.double() - ref).norm().item()
    return (d / n) if n else d


def direct():
    """The gate: `forward()` against `naive_rola`, at every support."""
    ok = True
    dev = "cuda"
    cases = [((64, 64), 64, 1, 1, 64, (64, 4, 1)),
             ((64, 64), 64, 1, 2, 256, (64, 4, 1)),
             ((64, 64), 32, 1, 1, 128, (64, 4)),
             ((64, 64), 128, 1, 1, 128, (64, 4)),
             ((64, 64), 256, 1, 1, 128, (64, 4)),
             ((64, 64), 256, 1, 2, 4096, (64, 4)),
             ((16, 16), 64, 1, 2, 256, (16, 2)),
             ((8, 8), 64, 2, 2, 128, (8, 2)),
             # G4's DEEP arms, at the flagship's matched N = 4096: three levels
             # of 16, four levels of 8 -- and the BC = 32 depth-4 arm whose
             # three-clause form and two-scalar upper chain exist to be gated.
             ((16, 16, 16), 128, 1, 1, 256, (16, 4, 1)),
             ((16, 16, 16), 128, 1, 1, 2048, (16, 4)),
             ((8, 8, 8, 8), 128, 1, 1, 256, (8, 4, 1)),
             ((8, 8, 8, 8), 128, 1, 1, 2048, (8, 4)),
             ((8, 8, 8, 8), 32, 1, 1, 256, (8, 4, 1)),
             ((8, 8, 8, 8), 32, 1, 1, 2048, (8, 4))]
    print(f"{'widths':>10s}{'BC':>5s}{'blk':>5s}{'B':>3s}{'H':>3s}{'L':>6s}{'C':>4s}"
          f"{'sup':>5s}{'skip':>8s}{'rel_L2 vs naive_rola':>22s}{'design floor':>14s}"
          f"{'r/bound':>8s}")
    for widths, bc, B, H, L, sups in cases:
        for sup in sups:
            pr = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=11)
            pw = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=12)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            y = oracle_ratio(v, pr, pw, widths)
            built = {a[0] for a in build.arms()
                     if not a[1] and a[2] == bc
                     and (a[3], a[4]) == (len(widths), widths[0])}
            for c in sorted(built):
                if L % c:
                    continue
                got = build.forward(pr, pw, v, c, False, bc, widths)
                torch.cuda.synchronize()
                # AT THE SPARSEST SUPPORT THE ORACLE'S OWN ANSWER CAN BE ZERO --
                # no read digit coincides with a write digit anywhere in the
                # cell, and nothing routes. A ratio is then undefined, so the
                # absolute residual is gated instead of a NaN being compared.
                #: THE FLOOR IS COMPACTION-AWARE, AND IT HAS TO BE. Compaction
                #: REGROUPS an exact sum, so it changes nothing in exact
                #: arithmetic -- but it moves WHERE the state is rounded to the
                #: operand format, and that is exactly what a design floor
                #: measures. Modelling the floor on the contiguous chunking would
                #: charge this kernel for a rounding it does not perform.
                md = reference.ratio(
                    reference.model_compacted(pr, pw, v, widths, c, bc), READOUT_EPS)
                r, floor = rel(got, y), rel(md, y)
                nb = 1
                for w_ in widths:
                    nb *= w_
                nb //= bc
                skip = 1.0 - reference.schedule(pr, pw, widths, bc,
                                                c).double().mean().item()
                bound = max(floor, KERNEL_FP32_FLOOR)
                ratio = r / bound
                print(f"{str(widths):>10s}{bc:5d}{nb:5d}{B:3d}{H:3d}{L:6d}{c:4d}"
                      f"{sup:5d}{100*skip:7.1f}%{r:22.3e}{floor:14.3e}{ratio:8.4f}"
                      f"{'  (fp32 floor binds)' if floor < KERNEL_FP32_FLOOR else ''}")
                # THE GATE: the disagreement with the oracle must BE the design's
                # own bf16 quantisation -- or, where that is exactly zero because
                # the coefficients are exactly representable, the kernel's own
                # measured fp32 precision -- and not a hair more.
                ok &= ratio <= 1.02
    return ok




def trend():
    """THE ACCURACY TREND AGAINST L, against the oracle itself.

    The denominator is a monotone positive sum that grows without bound over the
    sequence, so its error could be length-dependent in a way one length hides.
    The oracle's own mass is printed beside the error so the growth is visible
    rather than assumed.
    """
    ok = True
    dev = "cuda"
    widths, bc, c = (16, 16), 64, 32
    print(f"\n{'L':>7s}{'oracle den mean':>17s}{'rel_L2 vs naive_rola':>22s}"
          f"{'design floor':>14s}{'ratio':>8s}{'den rel_L2':>12s}")
    for L in (256, 512, 1024, 2048, 4096, 8192, 16384):
        pr = reference.support_amplitudes(1, 1, L, widths, widths[0], dev, seed=60)
        pw = reference.support_amplitudes(1, 1, L, widths, widths[0], dev, seed=61)
        v = torch.randn(1, 1, L, 64, device=dev).bfloat16()
        y = oracle_ratio(v, pr, pw, widths)
        got = build.forward(pr, pw, v, c, False, bc, widths)
        tbl = build.union_table(pr, pw, v, c, bc, widths)
        num, den = build.forward_raw(pr, pw, v, tbl, c, False, bc, widths)
        torch.cuda.synchronize()
        mdn = reference.model_compacted(pr, pw, v, widths, c, bc)
        exn = reference.chunked_norm(reference.kron(pr, widths),
                                     reference.kron(pw, widths), v, c)
        md = reference.ratio(mdn, READOUT_EPS)
        r, floor = rel(got, y), rel(md, y)
        print(f"{L:7d}{exn[..., -1].mean().item():17.3f}{r:22.3e}{floor:14.3e}"
              f"{r/floor:8.4f}{rel(den.unsqueeze(-1), exn[..., -1:]):12.3e}")
        ok &= r / floor <= 1.05
        del mdn, exn, y
        torch.cuda.empty_cache()
    return ok


def run():
    print("== THE DIRECT GATE: forward() vs naive_rola, same eps, no inversion ==")
    ok = direct()
    print("\n== THE TREND AGAINST L ==")
    ok &= trend()
    return ok


if __name__ == "__main__":
    good = run()
    print("CANONICAL ORACLE:", "REACHED DIRECTLY" if good else "NOT REACHED")
    sys.exit(0 if good else 1)
