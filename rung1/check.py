"""Correctness gate for RUNG 1. No performance number is reported unless this passes.

The base's layers plus one that is specific to this rung:

0. COEFFICIENT -- the factored production against the leaf-major DEFINITION
   (explicit divide/modulo digit extraction). The kernel never computes a digit,
   so this is a genuine cross-check of the digit-major loop nest against the
   mixed-radix definition rather than a restatement of it.
1. ALGORITHM   -- fp64 chunked recurrence vs the fp64 quadratic definition on
   ROUTED coefficients. Pure maths, no kernel.
2. SINGLE CHUNK (L == C) -- state identically zero: isolates production, the
   intra path, the causal mask and the output path with no state noise.
3. FULL LENGTH -- L = 4096, the measurement cell, against BOTH the exact-product
   fp64 recurrence (the DESIGN error, which includes the routed form's own
   coefficient quantisation) and the fp64 recurrence that models every rounding
   the kernel performs (bf16 coefficient, bf16 state, bf16 output).
4. RING PHASE  -- lengths that are not multiples of the ring depth, so the
   staging wrap and the deferred output drain are exercised at every phase.
5. ARM IDENTITY -- every arm of a chunk size must be BIT-IDENTICAL to every
   other. That is what proves the layout axis changes no arithmetic.

Every arm in the binary is gated, not just the default one. The binary now holds
SIX: two axes, `C` and `SWZ`. The staging depth is DERIVED per arm and is printed
with the census below, so a quantity that stopped being an axis is still visible.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1 import build, reference  # noqa: E402

CHUNKS = (16, 32, 64)
WIDTHS = build.WIDTHS


def metrics(got, ref):
    got, ref = got.double(), ref.double()
    d = got - ref
    return dict(
        max_abs=d.abs().max().item(),
        rel_max=(d.abs().max() / ref.abs().max()).item(),
        rel_l2=(d.norm() / ref.norm()).item(),
    )


def fmt(e):
    return (f"max_abs {e['max_abs']:.4e}  max_abs/|ref|max {e['rel_max']:.3e}  "
            f"rel_L2 {e['rel_l2']:.3e}")


def run():
    ok = True
    dev = "cuda"
    arms = build.arms()
    default = [(c, False) for c in CHUNKS]

    print("== arms in the binary (two axes; S and the CTA target are DERIVED) ==")
    for c, w, s, b, t in arms:
        st = build.assert_build_identity(c, w)
        print(f"  C={c:3d} SWZ={int(w)}  S={s} (derived)  ctas={t}  "
              f"smem={b:7,d} B  regs={st['regs_per_thread']:3d}  "
              f"local={st['local_bytes']}  caps={st['caps_digest']}")

    # ---- 0. the factored coefficient IS the mixed-radix definition ----------
    p = reference.amplitudes(1, 1, 8, WIDTHS, dev, seed=7)
    lv = reference.split_levels(p.double(), WIDTHS)
    st_, n_slots = reference.strides(WIDTHS)
    brute = torch.empty(1, 1, 8, n_slots, dtype=torch.float64, device=dev)
    for n in range(n_slots):
        acc = torch.ones(1, 1, 8, dtype=torch.float64, device=dev)
        for l, w in enumerate(WIDTHS):
            acc = acc * lv[l][..., (n // st_[l]) % w]
        brute[..., n] = acc
    e = metrics(reference.kron(p, WIDTHS), brute)
    print(f"\n[0 coefficient] widths={WIDTHS} strides={st_}  vectorised vs "
          f"slot-by-slot mixed radix   {fmt(e)}")
    ok &= e["rel_l2"] == 0.0

    # ---- 1. algorithm -------------------------------------------------------
    pr = reference.amplitudes(2, 3, 512, WIDTHS, dev, seed=0)
    pw = reference.amplitudes(2, 3, 512, WIDTHS, dev, seed=1)
    v = torch.randn(2, 3, 512, 64, device=dev).bfloat16()
    R, W = reference.kron(pr, WIDTHS), reference.kron(pw, WIDTHS)
    quad = reference.quadratic(R, W, v)
    for c in CHUNKS:
        e = metrics(reference.chunked(R, W, v, c), quad)
        print(f"[1 algorithm ] chunk={c:3d}  fp64 chunked vs fp64 quadratic   {fmt(e)}")
        ok &= e["rel_l2"] < 1e-13

    # ---- 2. one chunk, state identically zero -------------------------------
    for c, w in default:
        pr = reference.amplitudes(4, 4, c, WIDTHS, dev, seed=10 + c)
        pw = reference.amplitudes(4, 4, c, WIDTHS, dev, seed=20 + c)
        vv = torch.randn(4, 4, c, 64, device=dev).bfloat16()
        got = build.forward(pr, pw, vv, c, w)
        torch.cuda.synchronize()
        Rb, Wb = reference.kron_bf16(pr, WIDTHS), reference.kron_bf16(pw, WIDTHS)
        e = metrics(got, reference.chunked_bf16_state(Rb, Wb, vv, c).bfloat16())
        print(f"[2 one chunk ] chunk={c:3d}  L=C, state==0, intra only        {fmt(e)}")
        ok &= e["rel_l2"] < 1e-4

    # ---- 3. the measurement cell, EVERY arm ---------------------------------
    L = 4096
    pr = reference.amplitudes(2, 8, L, WIDTHS, dev, seed=2)
    pw = reference.amplitudes(2, 8, L, WIDTHS, dev, seed=3)
    v = torch.randn(2, 8, L, 64, device=dev).bfloat16()
    Rx, Wx = reference.kron(pr, WIDTHS), reference.kron(pw, WIDTHS)
    Rb, Wb = reference.kron_bf16(pr, WIDTHS), reference.kron_bf16(pw, WIDTHS)
    ex = {c: reference.chunked(Rx, Wx, v, c) for c in CHUNKS}
    bs = {c: reference.chunked_bf16_state(Rb, Wb, v, c) for c in CHUNKS}
    first = {}
    for c, w, s, _, _ in arms:
        got = build.forward(pr, pw, v, c, w)
        torch.cuda.synchronize()
        e1, e2 = metrics(got, ex[c]), metrics(got, bs[c].bfloat16())
        d = metrics(bs[c], ex[c])
        tag = f"C={c:3d} SWZ={int(w)} S={s}"
        print(f"[3 L={L}   ] {tag}  vs exact fp64 {fmt(e1)}")
        print(f"                             vs bf16 coeff+state fp64, bf16 out "
              f"{fmt(e2)}   (design floor {d['rel_l2']:.3e})")
        ok &= e2["rel_l2"] < 1e-3 and e1["rel_l2"] < 5e-3
        if c not in first:
            first[c] = (got.clone(), tag)
        else:
            same = torch.equal(got, first[c][0])
            print(f"                             bit-identical to {first[c][1]}: "
                  f"{'YES' if same else 'NO'}")
            ok &= same

    # ---- 4. ring phase ------------------------------------------------------
    for Lx in (64, 128, 192, 320, 1024):
        pr = reference.amplitudes(2, 2, Lx, WIDTHS, dev, seed=30)
        pw = reference.amplitudes(2, 2, Lx, WIDTHS, dev, seed=31)
        vv = torch.randn(2, 2, Lx, 64, device=dev).bfloat16()
        Rb2, Wb2 = reference.kron_bf16(pr, WIDTHS), reference.kron_bf16(pw, WIDTHS)
        worst = 0.0
        for c, w, _, _, _ in arms:
            if Lx % c:
                continue
            got = build.forward(pr, pw, vv, c, w)
            torch.cuda.synchronize()
            worst = max(worst, metrics(
                got, reference.chunked_bf16_state(Rb2, Wb2, vv, c).bfloat16())["rel_l2"])
        print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm  {worst:.3e}")
        ok &= worst < 1e-3
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
