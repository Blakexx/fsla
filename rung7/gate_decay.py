"""THE DECAY GATE: the mass clock against the canonical oracle, directly.

`naive_rola(decay=LeafMassDecay(...))` is the reference; the global per-head
rate enters it as uniform dials `delta_l = rate^(1/D)` (leaf rate = their
product = rate exactly, up to fp64).  The gate is ratio-to-floor with the
decay-aware floor model (`reference_decay.model_compacted_decay`), which
mirrors the kernel's dressing arithmetic.  Every case runs g=1 AND lognormal
gain: the clock must stay UNGAINED while the deposit is gained.

Dead chunks under sparse support exercise compaction-decay-neutrality against
the oracle (whose clock advances only on actual write mass) on every sparse
row here — the event-clock semantics are gated by agreement, not asserted.
"""
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rola.ops.constants import READOUT_EPS  # noqa: E402
from rola.ops.naive import naive_rola  # noqa: E402
from rola.routing.types import LeafMassDecay  # noqa: E402
from rung5b import reference  # noqa: E402
from rung7 import build  # noqa: E402
from rung7.oracle_gate import rel, topology  # noqa: E402
from rung7.reference_decay import model_compacted_decay  # noqa: E402

dev = "cuda"
KERNEL_FP32_FLOOR = 2.0e-7


def oracle_decay(v_bhld, pr, pw, g, widths, rate, eps=READOUT_EPS):
    top = topology(widths)
    D = len(widths)
    dials = tuple(
        rate.double().pow(1.0 / D).unsqueeze(1).expand(rate.shape[0], w).contiguous()
        for w in widths)
    v = v_bhld.permute(0, 2, 1, 3).contiguous()
    read = tuple(x.permute(0, 2, 1, 3).contiguous()
                 for x in reference.split_levels(pr, widths))
    write = tuple(x.permute(0, 2, 1, 3).contiguous()
                  for x in reference.split_levels(pw, widths))
    y, _ = naive_rola(v, read, write, g_read=None,
                      g_write=g.permute(0, 2, 1).double(), topology=top,
                      decay=LeafMassDecay(dials=dials), normalization="global",
                      eps=eps)
    return y.permute(0, 2, 1, 3)


def main():
    torch.manual_seed(0)
    ok = True
    B, H = 1, 2
    rates = {"mild": torch.tensor([0.30, 0.05], device=dev),
             "hot": torch.tensor([0.60, 0.45], device=dev)}
    cases = [((8, 8), 64, 32, 512, (8, 2)),
             ((64, 64), 128, 32, 1024, (64, 4)),
             ((8, 8, 8, 8), 128, 32, 512, (8, 4))]
    print(f"{'widths':>14s}{'rate':>6s}{'g':>6s}{'L':>6s}{'sup':>5s}"
          f"{'rel vs naive':>14s}{'floor':>11s}{'r/bound':>9s}")
    for widths, bc, c, L, sups in cases:
        for sup in sups:
            pr = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=21)
            pw = reference.support_amplitudes(B, H, L, widths, sup, dev, seed=22)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            for rname, rate in rates.items():
                for gmode in ("g=1", "logn"):
                    if gmode == "g=1":
                        g = torch.ones(B, H, L, device=dev).bfloat16()
                    else:
                        gl = torch.Generator(dev).manual_seed(23)
                        g = torch.empty(B, H, L, device=dev).log_normal_(
                            0.0, 1.0, generator=gl).bfloat16()
                    y = oracle_decay(v, pr, pw, g, widths, rate)
                    got = build.forward(pr, pw, g, v, c, False, bc, widths,
                                        rate=rate, mode=1)
                    md = reference.ratio(
                        model_compacted_decay(pr, pw, g, v, widths, c, bc, rate),
                        READOUT_EPS)
                    r, floor = rel(got, y), rel(md, y)
                    bound = max(floor, KERNEL_FP32_FLOOR)
                    print(f"{str(widths):>14s}{rname:>6s}{gmode:>6s}{L:6d}{sup:5d}"
                          f"{r:14.3e}{floor:11.3e}{r / bound:9.4f}")
                    #: 1.05, not 1.02: the kernel's EX2/fast-log path is ~1e-7
                    #: relative per factor against the model's exact pow.
                    ok &= r / bound <= 1.05
    print("DECAY GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
