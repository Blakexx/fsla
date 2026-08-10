"""The ablation ladder's correctness gate.

    python ladder_gate.py

Three claims, all binding, none of them observed rather than asserted:

  * DEVICE-SIDE BUILD IDENTITY per arm and per stage, including a nonzero
    `localSizeBytes` failing the gate, so a spill introduced by one stage cannot
    ride along in another stage's number;
  * AT DENSE EVERY STAGE IS THE SAME FUNCTION. The stages differ only in
    mechanisms that are the identity there -- the contiguous producer IS the
    cook's dense partition, `J[i] = i + 1` IS the triangle, and every region is
    live -- so stages 1..4 must be BIT-IDENTICAL to one another AT ONE BLOCK.
    Above one block the fan-in's order is not determined (rung 2: the reduction
    is fire-and-forget, measured spread 1.4e-7), so what is available there is
    agreement INSIDE that spread and it is reported as `rel_L2`;
  * STAGE 4 IS RUNG 5B, on every arm and every regime, to the same standard --
    which is what closes the ladder with no unattributed residual;
  * STAGE 5 IS STAGE 4, on every arm and EVERY REGIME. The window changes only
    where the cook's words come from, never which partition it produces, so
    unlike stages 1..3 this one is binding away from dense as well.
"""
from __future__ import annotations

import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5b import build as bbuild  # noqa: E402
from rung5c import build as cbuild  # noqa: E402
from rung5c import reference  # noqa: E402

REGIMES = ("dense", "lead4", "split", "d4")
STAGES = (1, 2, 3, 4, 5)


def main():
    dev = "cuda"
    B, H, L = 1, 2, 512
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    arms = sorted({a[:5] for a in cbuild.arms()})
    ok = True

    print(f"{'arm':34s}{'stage':>6s}{'regs':>6s}{'local':>7s}{'stages':>8s}")
    for c, w, bc, w0, w1 in arms:
        for abl in STAGES:
            d = cbuild.assert_build_identity(c, w, bc, w0, w1, abl)
            print(f"{f'C={c} SWZ={int(w)} BC={bc} N={w0*w1}':34s}{abl:6d}"
                  f"{d['regs_per_thread']:6d}{d['local_bytes']:7d}{d['stages']:8d}")

    print("\n-- AT DENSE THE FOUR STAGES ARE ONE FUNCTION, and stage 4 IS rung 5B")
    TOL = 1e-6

    def agree(a, b, one_block):
        if one_block:
            return all(torch.equal(x, y) for x, y in zip(a, b)), 0.0
        r = max((x.double() - y.double()).norm().item()
                / max(y.double().norm().item(), 1e-30) for x, y in zip(a, b))
        return r < TOL, r

    print(f"{'arm':34s}{'regime':>8s}{'1v4':>10s}{'2v4':>10s}{'3v4':>10s}"
          f"{'5v4':>10s}{'5B':>10s}")
    for c, w, bc, w0, w1 in arms:
        #: THE REGIMES ARE THE FLAGSHIP'S. The (8,8) control and the (16,16)
        #: small configuration have their own width, so they carry the dense
        #: draw only -- which is the row the ladder's deltas are measured on.
        rgs = REGIMES if (w0, w1) == (64, 64) else ("dense",)
        for rg in rgs:
            pr, pw = (reference.regime_amplitudes(B, H, L, rg, dev, seed=5)
                      if (w0, w1) == (64, 64) else
                      (reference.amplitudes(B, H, L, (w0, w1), dev, seed=2),
                       reference.amplitudes(B, H, L, (w0, w1), dev, seed=3)))
            tbl = cbuild.union_table(pr, pw, v, c, bc, w0, w1)
            one = w0 * w1 == bc
            ref = cbuild.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1, 4)
            b5 = bbuild.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1)
            same5, r5 = agree(ref, b5, one)
            eq, rr = {}, {}
            for abl in (1, 2, 3, 5):
                got = cbuild.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1, abl)
                eq[abl], rr[abl] = agree(ref, got, one)
            #: THE STAGES BELOW 4 ARE THE IDENTITY ONLY AT DENSE. Away from dense
            #: they compute the CONTIGUOUS partition, which is a different (and
            #: also correct) grouping, so only `dense` is a binding row.
            binding = rg == "dense"
            def mark(b, r):
                return "BIT-EQ" if one and b else (f"{r:.1e}" if b else f"FAIL {r:.1e}")
            print(f"{f'C={c} SWZ={int(w)} BC={bc} N={w0*w1}':34s}{rg:>8s}"
                  f"{mark(eq[1], rr[1]):>10s}{mark(eq[2], rr[2]):>10s}"
                  f"{mark(eq[3], rr[3]):>10s}{mark(eq[5], rr[5]):>10s}"
                  f"{mark(same5, r5):>10s}")
            #: STAGE 5 IS BINDING IN EVERY REGIME. It changes only WHERE the
            #: cook's words come from, so it must produce the same partition
            #: wherever the partition is not the identity as well.
            ok &= same5 and eq[5] and (not binding or eq[1] and eq[2] and eq[3])
    print("\nLADDER GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
