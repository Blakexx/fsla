"""Correctness gate for RUNG 4. No performance number is reported unless this passes.

Rung 3's layers, run on `d_v + 1` columns, plus the four the ones column adds:

6.  THE NUMERATOR IS RUNG 3's. The mass channel is an extra `m` row of three
    products and it writes only its own accumulators, so the value channels'
    arithmetic must be untouched. At ONE block that is BIT-IDENTITY of the
    NUMERATOR against rung 3's output; with more blocks the fan-in's order is
    undefined by contract, so what is gated is agreement inside the measured
    spread. This is the strongest statement in the rung.
8.  THE ACCURACY TREND AGAINST L. The denominator is a MONOTONE POSITIVE SUM
    with no cancellation, so its error is one-directional and could in principle
    grow with the sequence; the numerator's terms are signed and partially
    cancel, so the two need not drift together. One length would hide that
    entirely, so the ratio is gated at six lengths and the trend is printed.
9.  THE FAN-IN SPREAD, RE-MEASURED WITH NORMALISATION ON. Determinism has been
    gone by construction since rung 2 (measured spread 1.4e-7); the denominator
    now accumulates through that same fire-and-forget path, so the question is
    whether the spread grows. Five runs, on the ratio and on the denominator.
10. THE DIVIDE PASS IS THE SPEC'S LINE. `divide_pass` against the same division
    performed in fp64, which is what says the pass rounds once and does not
    reciprocate.

A skipped group contributes the additive identity to the mass as well, so THERE
IS STILL NO NEW REFERENCE: rung 2's `split`, run on `with_ones(v)`, is the model
of this kernel's arithmetic at every sparsity.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rola.ops.constants import READOUT_EPS  # noqa: E402
from rung3 import build as r3build  # noqa: E402
from rung4 import build, reference  # noqa: E402

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


def kernel(pr, pw, v, c, w, bc, w0, w1, eps=READOUT_EPS):
    """(the readout's d_v+1 columns, the ratio) -- one launch, both views."""
    num, den = build.forward_raw(pr, pw, v, c, w, bc, w0, w1)
    y = build.divide_pass(num, den, eps)
    torch.cuda.synchronize()
    return torch.cat((num, den.unsqueeze(-1)), dim=-1), y, num


def run():
    ok = True
    dev = "cuda"
    arms = build.arms()
    eps = READOUT_EPS

    print("== arms in the binary (rung 3's set MINUS the two BC=256 arms, whose "
          "mass tile does not fit the register file) ==")
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

    # ---- 1. algorithm, and the exactness of the split, ON d_v+1 COLUMNS -------
    for widths in CONFIGS:
        B, H, L = 1, 1, 256
        pr = amps(B, H, L, widths, CONFIGS[widths]["supports"][-1], dev, 0)
        pw = amps(B, H, L, widths, CONFIGS[widths]["supports"][-1], dev, 1)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        v1 = reference.with_ones(v)
        R, W = reference.kron(pr, widths), reference.kron(pw, widths)
        quad = reference.quadratic(R, W, v1)
        for c in (16, 64):
            e = metrics(reference.chunked(R, W, v1, c), quad)
            print(f"[1 algorithm ] widths={widths} chunk={c:3d}  fp64 chunked vs "
                  f"fp64 quadratic, d_v+1 columns   {fmt(e)}")
            ok &= e["rel_l2"] < 1e-13
        for bc in sorted({a[2] for a in arms if (a[3], a[4]) == widths}):
            n = widths[0] * widths[1]
            parts = sum(reference.chunked(R[..., s:s + bc], W[..., s:s + bc], v1, 32)
                        for s in range(0, n, bc))
            e = metrics(parts, reference.chunked(R, W, v1, 32))
            print(f"[1 split     ] widths={widths} BC={bc:4d}  sum of exact block "
                  f"partials vs the exact whole   {fmt(e)}")
            ok &= e["rel_l2"] < 1e-13

    # ---- 2/3/4/7. the kernel, every arm, at every support --------------------
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
                got, y, _ = kernel(pr, pw, vv, c, w, bc, w0, w1, eps)
                Rb = reference.kron_bf16(pr, widths)
                Wb = reference.kron_bf16(pw, widths)
                md = reference.split_norm(Rb, Wb, vv, c, bc)
                e = metrics(got, md)
                ey = metrics(y, reference.ratio(md, eps))
                print(f"[2 one chunk ] C={c:3d} BC={bc:4d}  L=C, state==0  "
                      f"columns {fmt(e)}")
                print(f"                                      ratio   {fmt(ey)}")
                ok &= e["rel_l2"] < 1e-4 and ey["rel_l2"] < 1e-4

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
                    ex[c] = reference.chunked_norm(Rx, Wx, v, c)
                if (c, bc) not in md:
                    md[(c, bc)] = reference.split_norm(Rb, Wb, v, c, bc)
                got, y, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
                e1, e2 = metrics(got, ex[c]), metrics(got, md[(c, bc)])
                d = metrics(md[(c, bc)], ex[c])
                y1 = metrics(y, reference.ratio(ex[c], eps))
                y2 = metrics(y, reference.ratio(md[(c, bc)], eps))
                dy = metrics(reference.ratio(md[(c, bc)], eps),
                             reference.ratio(ex[c], eps))
                skip = 1.0 - reference.schedule(pr, pw, widths, bc,
                                                c).double().mean().item()
                print(f"[3 L={L:5d}  ] C={c:3d} SWZ={int(w)} BC={bc:4d} blocks={nb:3d} "
                      f"skipped {100*skip:5.1f}%  columns vs exact fp64 {fmt(e1)}")
                print(f"                              columns vs the modelled rounding "
                      f"{fmt(e2)}   (design floor {d['rel_l2']:.3e})")
                print(f"                              RATIO   vs exact fp64 "
                      f"{fmt(y1)}")
                print(f"                              RATIO   vs the modelled rounding "
                      f"{fmt(y2)}   (design floor {dy['rel_l2']:.3e})")
                ok &= e2["rel_l2"] < 1e-3 and e1["rel_l2"] < 5e-3
                ok &= y2["rel_l2"] < 5e-3 and y1["rel_l2"] < 2e-2
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
                    got, y, _ = kernel(pr2, pw2, vv, c, w, bc, w0, w1, eps)
                    mdx = reference.split_norm(Rb2, Wb2, vv, c, bc)
                    worst = max(worst, metrics(got, mdx)["rel_l2"],
                                metrics(y, reference.ratio(mdx, eps))["rel_l2"])
                print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm, columns "
                      f"and ratio  {worst:.3e}")
                ok &= worst < 1e-3

    # ---- 6. THE NUMERATOR IS RUNG 3's ---------------------------------------
    print("\n---- the NUMERATOR against RUNG 3 itself: the ones column writes only "
          "its own accumulators, so no value channel may move ----")
    B, H, L = 2, 4, 1024
    for widths in CONFIGS:
        pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=3)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        for c, w, bc, w0, w1, nb, *_ in arms:
            if (w0, w1) != widths:
                continue
            r3build.assert_build_identity(c, w, bc, w0, w1)
            a = r3build.forward(pr, pw, v, c, w, bc, w0, w1)
            _, _, num = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
            same = torch.equal(a, num)
            e = metrics(num, a)
            if nb == 1:
                print(f"[6 vs rung3  ] C={c:3d} SWZ={int(w)} BC={bc:4d} N={w0*w1:5d} "
                      f"1 block: {'BIT-IDENTICAL' if same else 'DIFFERS'}")
                ok &= same
            else:
                print(f"[6 vs rung3  ] C={c:3d} SWZ={int(w)} BC={bc:4d} N={w0*w1:5d} "
                      f"{nb:3d} blocks: bit-identical {'YES' if same else 'NO'}   "
                      f"rel_L2 {e['rel_l2']:.3e} (fan-in order is not defined)")
                ok &= e["rel_l2"] < 1e-6

    # ---- 8. THE ACCURACY TREND AGAINST L ------------------------------------
    print("\n---- the trend against L: the denominator is a monotone positive sum, "
          "so a single length would hide a length-dependent drift ----")
    print(f"{'widths':>10s}{'BC':>5s}{'C':>4s}{'L':>7s}{'den mean':>11s}"
          f"{'den rel_L2':>12s}{'num rel_L2':>12s}{'ratio rel_L2':>14s}"
          f"{'floor':>11s}{'ratio/floor':>13s}")
    for widths, bc, c in (((16, 16), 64, 32), ((64, 64), 128, 32)):
        support = widths[0]
        for L in (256, 512, 1024, 2048, 4096, 8192):
            if widths == (64, 64) and L > 4096:
                continue
            pr = amps(1, 1, L, widths, support, dev, 60)
            pw = amps(1, 1, L, widths, support, dev, 61)
            v = torch.randn(1, 1, L, 64, device=dev).bfloat16()
            got, y, _ = kernel(pr, pw, v, c, False, bc, *widths, eps)
            Rx, Wx = reference.kron(pr, widths), reference.kron(pw, widths)
            Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
            exn = reference.chunked_norm(Rx, Wx, v, c)
            mdn = reference.split_norm(Rb, Wb, v, c, bc)
            den = metrics(got[..., -1:], exn[..., -1:])["rel_l2"]
            num = metrics(got[..., :-1], exn[..., :-1])["rel_l2"]
            rat = metrics(y, reference.ratio(exn, eps))["rel_l2"]
            floor = metrics(reference.ratio(mdn, eps),
                            reference.ratio(exn, eps))["rel_l2"]
            print(f"{str(widths):>10s}{bc:5d}{c:4d}{L:7d}"
                  f"{exn[..., -1].mean().item():11.2f}{den:12.3e}{num:12.3e}"
                  f"{rat:14.3e}{floor:11.3e}{rat/floor:13.4f}")
            ok &= rat <= 1.05 * floor
            del Rx, Wx, Rb, Wb, exn, mdn
            torch.cuda.empty_cache()

    # ---- 9. THE FAN-IN SPREAD, WITH NORMALISATION ON ------------------------
    print("\n---- the fan-in spread with the denominator on it (rung 2 measured "
          "1.4e-7 without): five runs of every multi-block arm ----")
    B, H, L = 1, 2, 1024
    for widths in ((16, 16), (64, 64)):
        pr = reference.amplitudes(B, H, L, widths, dev, seed=7)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=8)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        for c, w, bc, w0, w1, nb, *_ in arms:
            if (w0, w1) != widths or w or nb == 1:
                continue
            runs = [kernel(pr, pw, v, c, w, bc, w0, w1, eps) for _ in range(5)]
            bit = all(torch.equal(runs[0][1], r[1]) for r in runs[1:])
            wy = max(metrics(r[1], runs[0][1])["rel_l2"] for r in runs[1:])
            wd = max(metrics(r[0][..., -1:], runs[0][0][..., -1:])["rel_l2"]
                     for r in runs[1:])
            wn = max(metrics(r[0][..., :-1], runs[0][0][..., :-1])["rel_l2"]
                     for r in runs[1:])
            print(f"[9 spread    ] C={c:3d} BC={bc:4d} N={w0*w1:5d} blocks={nb:3d}  "
                  f"bit-identical across runs: {'YES' if bit else 'NO'}   "
                  f"numerator {wn:.2e}  denominator {wd:.2e}  ratio {wy:.2e}")
            ok &= wy < 1e-5

    # ---- 10. THE DIVIDE PASS IS THE SPEC'S LINE -----------------------------
    print("\n---- the divide pass against the same division in fp64 ----")
    num, den = build.forward_raw(pr, pw, v, 32, False, 128, 64, 64)
    for e in (READOUT_EPS, 1.0):
        y = build.divide_pass(num, den, e)
        torch.cuda.synchronize()
        ref = num.double() / (den.double().unsqueeze(-1) + e)
        m = metrics(y, ref)
        print(f"[10 divide   ] eps={e:g}  {fmt(m)}")
        ok &= m["rel_max"] < 1e-6
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
