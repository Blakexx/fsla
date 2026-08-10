"""Correctness gate for RUNG 2. No performance number is reported unless this passes.

Rung 1's layers, each extended by what capacity adds, plus two that are new:

0. BLOCK PLAN  -- the blocks tile the slot space exactly once, and each block's
   slots are the mixed-radix rectangle the law describes, built with an EXPLICIT
   divide and modulo (the leaf-major form the kernel's loop nest never executes).
1. ALGORITHM   -- fp64 chunked vs fp64 quadratic on routed coefficients, AND the
   sum over blocks of the exact partials against the exact whole: the statement
   that the split is exact algebra rather than an approximation.
2. SINGLE CHUNK (L == C) -- state identically zero.
3. FULL LENGTH -- against the exact-product fp64 recurrence (the DESIGN error)
   and against the recurrence that models every rounding the kernel performs,
   PER BLOCK: bf16 coefficient, bf16 state, bf16 Gram, partials summed.
4. RING PHASE  -- lengths that are not multiples of the ring depth.
5. ARM IDENTITY / DETERMINISM -- arms of one (chunk, config) are compared to each
   other. With ONE block the sum has one contributor and the comparison is
   BIT-IDENTITY. With more, the fan-in's order is not defined -- that is the
   contract of a fire-and-forget reduction and the price of dropping the
   scratch-and-reduce -- so what is gated is the MEASURED spread, run to run and
   arm to arm, against the fp32 summation error it may not exceed.
6. THE CONTROL -- the one-block configuration against RUNG 1 itself, rounded to
   rung 1's output format. Bit-identity there is the statement that the capacity
   machinery changed no arithmetic.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung1 import build as r1build  # noqa: E402
from rung2 import build, reference  # noqa: E402

#: (widths, cells) per configuration. The fp64 references materialise [.., N] so
#: the gate's cell is chosen by N, not by the measurement's cell.
CONFIGS = {
    (8, 8): dict(name="CTRL  N=64   1 block", full=(4, 4, 1024), one=(4, 4, None)),
    (16, 16): dict(name="SMALL N=256  4 blocks", full=(2, 2, 1024), one=(4, 4, None)),
    (64, 64): dict(name="FLAG  N=4096 up to 128 blocks", full=(1, 1, 4096), one=(1, 2, None)),
}


def metrics(got, ref):
    got, ref = got.double(), ref.double()
    d = got - ref
    return dict(max_abs=d.abs().max().item(),
                rel_max=(d.abs().max() / ref.abs().max()).item(),
                rel_l2=(d.norm() / ref.norm()).item())


def fmt(e):
    return (f"max_abs {e['max_abs']:.4e}  max_abs/|ref|max {e['rel_max']:.3e}  "
            f"rel_L2 {e['rel_l2']:.3e}")


def run():
    ok = True
    dev = "cuda"
    arms = build.arms()

    print("== arms in the binary (C, SWZ are axes; BC is on trial; everything "
          "else is DERIVED) ==")
    for c, w, bc, w0, w1, nb, ls, ts, s, b, t in arms:
        st = build.assert_build_identity(c, w, bc, w0, w1)
        print(f"  C={c:3d} SWZ={int(w)} BC={bc:4d} widths=({w0:2d},{w1:2d})  "
              f"N={w0*w1:5d} blocks={nb:3d} rect={ls}x{ts:<3d}  S={s}  ctas={t}  "
              f"smem={b:7,d} B  regs={st['regs_per_thread']:3d}  "
              f"local={st['local_bytes']}")

    # ---- 0. the block plan tiles the slot space, and its slots are the law's --
    for widths in CONFIGS:
        for bc in sorted({a[2] for a in arms if (a[3], a[4]) == widths}):
            tab = reference.block_slots(widths, bc, dev)
            flat = tab.reshape(-1)
            perm = torch.equal(torch.sort(flat).values,
                               torch.arange(widths[0] * widths[1], device=dev))
            contig = bool((tab[:, 1:] - tab[:, :-1] == 1).all())
            plan, ls, ts = reference.block_plan(widths, bc)
            print(f"[0 block plan] widths={widths} BC={bc:4d}  {len(plan):3d} blocks "
                  f"rect={ls}x{ts:<3d}  tiles N exactly once: {'YES' if perm else 'NO'}  "
                  f"each block contiguous in slot index: {'YES' if contig else 'NO'}")
            ok &= perm and contig

    # ---- 0b. the factored coefficient IS the mixed-radix definition ----------
    for widths in CONFIGS:
        p = reference.amplitudes(1, 1, 8, widths, dev, seed=7)
        lv = reference.split_levels(p.double(), widths)
        st_, n_slots = reference.strides(widths)
        brute = torch.empty(1, 1, 8, n_slots, dtype=torch.float64, device=dev)
        for n in range(n_slots):
            acc = torch.ones(1, 1, 8, dtype=torch.float64, device=dev)
            for l, w in enumerate(widths):
                acc = acc * lv[l][..., (n // st_[l]) % w]
            brute[..., n] = acc
        e = metrics(reference.kron(p, widths), brute)
        print(f"[0 coefficient] widths={widths} strides={st_}  vectorised vs "
              f"slot-by-slot mixed radix   {fmt(e)}")
        ok &= e["rel_l2"] == 0.0

    # ---- 1. algorithm, and the exactness of the split -----------------------
    for widths in CONFIGS:
        B, H, L = 1, 1, 256
        pr = reference.amplitudes(B, H, L, widths, dev, seed=0)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=1)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        R, W = reference.kron(pr, widths), reference.kron(pw, widths)
        quad = reference.quadratic(R, W, v)
        for c in (16, 64):
            e = metrics(reference.chunked(R, W, v, c), quad)
            print(f"[1 algorithm ] widths={widths} chunk={c:3d}  fp64 chunked vs "
                  f"fp64 quadratic   {fmt(e)}")
            ok &= e["rel_l2"] < 1e-13
        for bc in sorted({a[2] for a in arms if (a[3], a[4]) == widths}):
            n = widths[0] * widths[1]
            parts = sum(reference.chunked(R[..., s:s + bc], W[..., s:s + bc], v, 32)
                        for s in range(0, n, bc))
            e = metrics(parts, reference.chunked(R, W, v, 32))
            print(f"[1 split     ] widths={widths} BC={bc:4d}  sum of exact block "
                  f"partials vs the exact whole   {fmt(e)}")
            ok &= e["rel_l2"] < 1e-13

    # ---- 2/3/4. the kernel, every arm ---------------------------------------
    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        if not sel:
            continue
        print(f"\n---- {cfg['name']} ----")

        # 2. one chunk, state identically zero
        Bo, Ho, _ = cfg["one"]
        for c, w, bc, w0, w1, *_ in sel:
            if w:
                continue
            pr = reference.amplitudes(Bo, Ho, c, widths, dev, seed=10 + c)
            pw = reference.amplitudes(Bo, Ho, c, widths, dev, seed=20 + c)
            vv = torch.randn(Bo, Ho, c, 64, device=dev).bfloat16()
            got = build.forward(pr, pw, vv, c, w, bc, w0, w1)
            torch.cuda.synchronize()
            Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
            e = metrics(got, reference.split(Rb, Wb, vv, c, bc))
            print(f"[2 one chunk ] C={c:3d} BC={bc:4d}  L=C, state==0    {fmt(e)}")
            ok &= e["rel_l2"] < 1e-4

        # 3. full length, every arm of the configuration
        B, H, L = cfg["full"]
        pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=3)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        Rx, Wx = reference.kron(pr, widths), reference.kron(pw, widths)
        Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
        ex, md = {}, {}
        first = {}
        for c, w, bc, w0, w1, nb, *_ in sel:
            if c not in ex:
                ex[c] = reference.chunked(Rx, Wx, v, c)
            if (c, bc) not in md:
                md[(c, bc)] = reference.split(Rb, Wb, v, c, bc)
            got = build.forward(pr, pw, v, c, w, bc, w0, w1)
            torch.cuda.synchronize()
            e1, e2 = metrics(got, ex[c]), metrics(got, md[(c, bc)])
            d = metrics(md[(c, bc)], ex[c])
            print(f"[3 L={L:5d}  ] C={c:3d} SWZ={int(w)} BC={bc:4d} blocks={nb:3d}  "
                  f"vs exact fp64 {fmt(e1)}")
            print(f"                              vs the modelled rounding "
                  f"{fmt(e2)}   (design floor {d['rel_l2']:.3e})")
            ok &= e2["rel_l2"] < 1e-3 and e1["rel_l2"] < 5e-3
            key = (c, bc)
            if key not in first:
                first[key] = (got.clone(), f"SWZ={int(w)}", nb)
            else:
                ref0, tag, blocks = first[key]
                if blocks == 1:
                    same = torch.equal(got, ref0)
                    print(f"                              bit-identical to {tag}: "
                          f"{'YES' if same else 'NO'}")
                    ok &= same
                else:
                    e = metrics(got, ref0)
                    print(f"                              vs {tag} (fan-in order is "
                          f"not defined): rel_L2 {e['rel_l2']:.3e}")
                    ok &= e["rel_l2"] < 1e-6

        # 3b. the fan-in's own non-determinism, measured rather than assumed
        for c, w, bc, w0, w1, nb, *_ in sel:
            if w or nb == 1:
                continue
            runs = []
            for _ in range(5):
                runs.append(build.forward(pr, pw, v, c, w, bc, w0, w1).clone())
            torch.cuda.synchronize()
            bits = all(torch.equal(runs[0], r) for r in runs[1:])
            spread = max(metrics(r, runs[0])["rel_l2"] for r in runs[1:])
            print(f"[3 fan-in    ] C={c:3d} BC={bc:4d} blocks={nb:3d}  5 runs "
                  f"bit-identical: {'YES' if bits else 'NO'}   worst rel_L2 spread "
                  f"{spread:.3e}")
            ok &= spread < 1e-6

        # 4. ring phase
        for Lx in (64, 128, 192, 320):
            pr = reference.amplitudes(2, 2, Lx, widths, dev, seed=30)
            pw = reference.amplitudes(2, 2, Lx, widths, dev, seed=31)
            vv = torch.randn(2, 2, Lx, 64, device=dev).bfloat16()
            Rb2 = reference.kron_bf16(pr, widths)
            Wb2 = reference.kron_bf16(pw, widths)
            worst = 0.0
            for c, w, bc, w0, w1, *_ in sel:
                if Lx % c:
                    continue
                got = build.forward(pr, pw, vv, c, w, bc, w0, w1)
                torch.cuda.synchronize()
                worst = max(worst, metrics(
                    got, reference.split(Rb2, Wb2, vv, c, bc))["rel_l2"])
            print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm  {worst:.3e}")
            ok &= worst < 1e-3

    # ---- 6. the one-block control against RUNG 1 ----------------------------
    print("\n---- the control: one block, against rung 1 itself ----")
    B, H, L = 2, 8, 1024
    pr = reference.amplitudes(B, H, L, (8, 8), dev, seed=2)
    pw = reference.amplitudes(B, H, L, (8, 8), dev, seed=3)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    for c, w, bc, w0, w1, nb, *_ in arms:
        if (w0, w1) != (8, 8) or nb != 1:
            continue
        r1build.assert_build_identity(c, w)
        a = r1build.forward(pr, pw, v, c, w)
        b = build.forward(pr, pw, v, c, w, bc, w0, w1)
        torch.cuda.synchronize()
        same = torch.equal(a, b.bfloat16())
        e = metrics(b, a)
        print(f"[6 control   ] C={c:3d} SWZ={int(w)}  rung2 rounded to rung 1's "
              f"output format: {'BIT-IDENTICAL' if same else 'DIFFERS'}   "
              f"(unrounded rel_L2 vs rung 1 {e['rel_l2']:.3e})")
        ok &= same
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
