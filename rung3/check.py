"""Correctness gate for RUNG 3. No performance number is reported unless this passes.

Rung 2's layers, run at BOTH ends of the sparsity sweep, plus the three that a
schedule adds:

0s. THE MEMBERSHIP TEST -- the kernel's prefix test against the definition (ANY
    nonzero coefficient over the block's enumerated slots). The prefix test may
    only ever be a SUPERSET of the definition: it may call a dead group live and
    never the reverse. That is the rung's safety property and it is checked on
    sparse data, where the two can actually differ.
6.  THE DENSE RUN IS RUNG 2's. With dense amplitudes every test returns true, so
    the kernel must compute exactly what rung 2 computed. At ONE block that is
    BIT-IDENTITY; with more, the fan-in's order is undefined by contract, so what
    is gated is agreement inside the measured fp32 summation spread.
7.  SPARSE CORRECTNESS -- the same fp64 layers on data the schedule can actually
    skip, which is the only place a wrong skip could show up at all.

A skipped group contributes the additive identity to every term, so THERE IS NO
NEW REFERENCE: rung 2's `split` is the model of this kernel's arithmetic at every
sparsity, and that it stays so is the claim under test.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung2 import build as r2build  # noqa: E402
from rung3 import build, reference  # noqa: E402

#: (widths, cells) per configuration, and the supports swept at each. The fp64
#: references materialise [.., N] so the gate's cell is chosen by N.
CONFIGS = {
    (8, 8): dict(name="CTRL  N=64   1 block", full=(4, 4, 1024), one=(4, 4, None),
                 supports=(8, 2)),
    (16, 16): dict(name="SMALL N=256  4 blocks", full=(2, 2, 1024), one=(4, 4, None),
                   supports=(16, 2)),
    (64, 64): dict(name="FLAG  N=4096 up to 128 blocks", full=(1, 1, 4096),
                   one=(1, 2, None), supports=(64, 4, 1)),
}


def metrics(got, ref):
    """A relative error against a reference that MAY BE IDENTICALLY ZERO.

    At the sparsest support a whole cell can route to nothing -- no read digit
    coincides with a write digit anywhere in the chunk -- and the answer really
    is the zero tensor. A ratio is then undefined, so agreement is reported as
    exact and disagreement as infinite, rather than as a NaN that a threshold
    would silently pass.
    """
    got, ref = got.double(), ref.double()
    d = got - ref
    n = ref.norm().item()
    rel = (d.norm().item() / n) if n else (0.0 if d.norm().item() == 0 else float("inf"))
    m = ref.abs().max().item()
    return dict(max_abs=d.abs().max().item(),
                rel_max=(d.abs().max().item() / m) if m else rel,
                rel_l2=rel)


def fmt(e):
    return (f"max_abs {e['max_abs']:.4e}  max_abs/|ref|max {e['rel_max']:.3e}  "
            f"rel_L2 {e['rel_l2']:.3e}")


def amps(B, H, L, widths, support, dev, seed):
    return reference.support_amplitudes(B, H, L, widths, support, dev, seed=seed)


def run():
    ok = True
    dev = "cuda"
    arms = build.arms()

    print("== arms in the binary (rung 2's set, arm for arm; everything but "
          "C and SWZ is DERIVED) ==")
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

    # ---- 0s. the prefix test is a SUPERSET of the definition ------------------
    print("\n---- the membership test: the prefix test vs slot enumeration ----")
    for widths, cfg in CONFIGS.items():
        for support in cfg["supports"]:
            B, H, L = 2, 2, 256
            pr = amps(B, H, L, widths, support, dev, 40)
            pw = amps(B, H, L, widths, support, dev, 41)
            for bc in sorted({a[2] for a in arms if (a[3], a[4]) == widths}):
                for c in sorted({a[0] for a in arms if (a[3], a[4]) == widths}):
                    pre = reference.schedule(pr, pw, widths, bc, c)
                    exa = reference.schedule_exact(pr, pw, widths, bc, c)
                    sup = bool((exa & ~pre).sum() == 0)
                    live_p = pre.double().mean().item()
                    live_e = exa.double().mean().item()
                    print(f"[0 membership] widths={widths} BC={bc:4d} C={c:3d} "
                          f"support={support:3d}  live prefix {100*live_p:6.2f}%  "
                          f"live exact {100*live_e:6.2f}%  prefix covers exact: "
                          f"{'YES' if sup else 'NO'}")
                    ok &= sup

    # ---- 1. algorithm, and the exactness of the split -----------------------
    for widths in CONFIGS:
        B, H, L = 1, 1, 256
        pr = amps(B, H, L, widths, CONFIGS[widths]["supports"][-1], dev, 0)
        pw = amps(B, H, L, widths, CONFIGS[widths]["supports"][-1], dev, 1)
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

    # ---- 2/3/4/7. the kernel, every arm, at every support -------------------
    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        if not sel:
            continue
        for support in cfg["supports"]:
            dense = support >= widths[0]
            print(f"\n---- {cfg['name']}   support={support} "
                  f"({'DENSE' if dense else 'SPARSE'}) ----")

            # 2. one chunk, state identically zero
            Bo, Ho, _ = cfg["one"]
            for c, w, bc, w0, w1, *_ in sel:
                if w:
                    continue
                pr = amps(Bo, Ho, c, widths, support, dev, 10 + c)
                pw = amps(Bo, Ho, c, widths, support, dev, 20 + c)
                vv = torch.randn(Bo, Ho, c, 64, device=dev).bfloat16()
                got = build.forward(pr, pw, vv, c, w, bc, w0, w1)
                torch.cuda.synchronize()
                Rb = reference.kron_bf16(pr, widths)
                Wb = reference.kron_bf16(pw, widths)
                e = metrics(got, reference.split(Rb, Wb, vv, c, bc))
                print(f"[2 one chunk ] C={c:3d} BC={bc:4d}  L=C, state==0    {fmt(e)}")
                ok &= e["rel_l2"] < 1e-4

            # 3. full length, every arm of the configuration
            B, H, L = cfg["full"]
            pr = amps(B, H, L, widths, support, dev, 2)
            pw = amps(B, H, L, widths, support, dev, 3)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            Rx, Wx = reference.kron(pr, widths), reference.kron(pw, widths)
            Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
            ex, md, first = {}, {}, {}
            for c, w, bc, w0, w1, nb, *_ in sel:
                if c not in ex:
                    ex[c] = reference.chunked(Rx, Wx, v, c)
                if (c, bc) not in md:
                    md[(c, bc)] = reference.split(Rb, Wb, v, c, bc)
                got = build.forward(pr, pw, v, c, w, bc, w0, w1)
                torch.cuda.synchronize()
                e1, e2 = metrics(got, ex[c]), metrics(got, md[(c, bc)])
                d = metrics(md[(c, bc)], ex[c])
                skip = 1.0 - reference.schedule(pr, pw, widths, bc,
                                                c).double().mean().item()
                print(f"[3 L={L:5d}  ] C={c:3d} SWZ={int(w)} BC={bc:4d} blocks={nb:3d} "
                      f"skipped {100*skip:5.1f}%  vs exact fp64 {fmt(e1)}")
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

            # 4. ring phase
            for Lx in (64, 128, 192, 320):
                pr2 = amps(2, 2, Lx, widths, support, dev, 30)
                pw2 = amps(2, 2, Lx, widths, support, dev, 31)
                vv = torch.randn(2, 2, Lx, 64, device=dev).bfloat16()
                Rb2 = reference.kron_bf16(pr2, widths)
                Wb2 = reference.kron_bf16(pw2, widths)
                worst = 0.0
                for c, w, bc, w0, w1, *_ in sel:
                    if Lx % c:
                        continue
                    got = build.forward(pr2, pw2, vv, c, w, bc, w0, w1)
                    torch.cuda.synchronize()
                    worst = max(worst, metrics(
                        got, reference.split(Rb2, Wb2, vv, c, bc))["rel_l2"])
                print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm  "
                      f"{worst:.3e}")
                ok &= worst < 1e-3

    # ---- 6. THE DENSE RUN IS RUNG 2's ---------------------------------------
    print("\n---- the dense run against RUNG 2 itself: the schedule skips nothing "
          "and the kernel must compute what rung 2 computed ----")
    B, H, L = 2, 4, 1024
    for widths in CONFIGS:
        pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=3)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        for c, w, bc, w0, w1, nb, *_ in arms:
            if (w0, w1) != widths:
                continue
            r2build.assert_build_identity(c, w, bc, w0, w1)
            a = r2build.forward(pr, pw, v, c, w, bc, w0, w1)
            b = build.forward(pr, pw, v, c, w, bc, w0, w1)
            torch.cuda.synchronize()
            same = torch.equal(a, b)
            e = metrics(b, a)
            if nb == 1:
                print(f"[6 vs rung2  ] C={c:3d} SWZ={int(w)} BC={bc:4d} N={w0*w1:5d} "
                      f"1 block: {'BIT-IDENTICAL' if same else 'DIFFERS'}")
                ok &= same
            else:
                print(f"[6 vs rung2  ] C={c:3d} SWZ={int(w)} BC={bc:4d} N={w0*w1:5d} "
                      f"{nb:3d} blocks: bit-identical {'YES' if same else 'NO'}   "
                      f"rel_L2 {e['rel_l2']:.3e} (fan-in order is not defined)")
                ok &= e["rel_l2"] < 1e-6
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
