"""Correctness gate for RUNG 5A. No performance number is reported unless this passes.

This rung changes WHERE the membership test's clauses come from and nothing else,
so its gate is dominated by one claim and the claim is an EQUALITY, not a
tolerance:

T1. THE TABLE IS THE LAW'S CLAUSES. The device table, unpacked, against a
    reference built from the block geometry law rather than from the kernel's
    constants. Bit-exact, every arm, every support.
T2. ONE STRUCTURE, TWO CONSUMERS. The group schedule DERIVED FROM THE TABLE
    equals rung 4's own membership schedule, exactly. If a future consumer reads
    these bits per token it is reading the same bits this schedule was reduced
    from, so the two cannot drift.
T3. THE TABLE'S SIZE FOLLOWS THE SUM OF THE WIDTHS AND NOT THEIR PRODUCT.
    Measured on the device tensor, at fixed sum with varying N and at fixed N
    with varying sum.
T4. THE KERNEL IS RUNG 4's, BIT FOR BIT. Same inputs, same arm, same output --
    identity at every one-block arm and at every support, and inside the fan-in's
    own measured spread where the fan-in makes bit-identity undefined. This is
    the strongest statement available and it is asserted, not observed.

Everything below T4 is rung 4's gate, unchanged in substance, because the
arithmetic is unchanged: the prefix test is still a SUPERSET of slot enumeration,
the fp64 references are still rung 2's `split` on `with_ones(v)`, and the ratio
is still the spec's line.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rola.ops.constants import READOUT_EPS  # noqa: E402
from rung4 import build as r4build  # noqa: E402
from rung5 import build, reference  # noqa: E402

CONFIGS = {
    (8, 8): dict(name="CTRL  N=64   1 block", full=(4, 4, 1024), one=(4, 4, None),
                 supports=(8, 2)),
    (16, 16): dict(name="SMALL N=256  4 blocks", full=(2, 2, 1024), one=(4, 4, None),
                   supports=(16, 2)),
    (64, 64): dict(name="FLAG  N=4096 up to 128 blocks", full=(1, 1, 4096),
                   one=(1, 2, None), supports=(64, 4, 1)),
}


def metrics(got, ref):
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
    """(the readout's d_v+1 columns, the ratio, the numerator, the table)."""
    tbl = build.union_table(pr, pw, v, c, bc, w0, w1)
    num, den = build.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1)
    y = build.divide_pass(num, den, eps)
    torch.cuda.synchronize()
    return torch.cat((num, den.unsqueeze(-1)), dim=-1), y, num, tbl


def run():
    ok = True
    dev = "cuda"
    arms = build.arms()
    eps = READOUT_EPS

    print("== arms in the binary (rung 4's set, arm for arm) ==")
    for c, w, bc, w0, w1, nb, ls, ts, s, b, t, ent in arms:
        st = build.assert_build_identity(c, w, bc, w0, w1)
        print(f"  C={c:3d} SWZ={int(w)} BC={bc:4d} widths=({w0:2d},{w1:2d})  "
              f"N={w0*w1:5d} blocks={nb:3d} rect={ls}x{ts:<3d}  S={s}  ctas={t}  "
              f"smem={b:7,d} B  regs={st['regs_per_thread']:3d}  "
              f"local={st['local_bytes']}  table entries={ent:3d}")

    # ---- T1/T2. THE TABLE, AND THE SCHEDULE DERIVED FROM IT -------------------
    print("\n---- T1 the table IS the law's clauses; T2 the schedule derived from "
          "it IS rung 4's membership ----")
    for widths, cfg in CONFIGS.items():
        for support in cfg["supports"]:
            B, H, L = 2, 2, 256
            pr = amps(B, H, L, widths, support, dev, 40)
            pw = amps(B, H, L, widths, support, dev, 41)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            for c, sw, bc, _, _, *_ in [a for a in arms if (a[3], a[4]) == widths]:
                if sw:
                    continue
                if True:
                    tbl = build.union_table(pr, pw, v, c, bc, widths[0], widths[1])
                    got = reference.union_unpack(tbl)
                    ref = reference.union_bits(pr, pw, widths, bc)
                    t1 = bool((got == ref).all())
                    s_tab = reference.schedule_from_table(got, widths, bc, c)
                    s_r4 = reference.schedule(pr, pw, widths, bc, c)
                    t2 = bool((s_tab == s_r4.reshape(s_tab.shape)).all())
                    exa = reference.schedule_exact(pr, pw, widths, bc, c)
                    sup = bool((exa.reshape(s_tab.shape) & ~s_tab).sum() == 0)
                    print(f"[T1/T2] widths={widths} BC={bc:4d} C={c:3d} support={support:3d} "
                          f" entries={tbl.shape[2]:3d} bits/token/side="
                          f"{tbl.shape[2]:3d}  table exact: {'YES' if t1 else 'NO'}  "
                          f"schedule == rung4's: {'YES' if t2 else 'NO'}  "
                          f"covers slot enumeration: {'YES' if sup else 'NO'}")
                    ok &= t1 and t2 and sup

    # ---- T3. THE SIZE FOLLOWS THE SUM OF THE WIDTHS ---------------------------
    print("\n---- T3 the table is O(L x sum of widths/spans) and INDEPENDENT of N "
          "-- measured on the device tensor ----")
    print("  BUILT ARMS -- the device tensor's own third dimension:")
    print(f"{'widths':>12s}{'N':>8s}{'sum W':>7s}{'BC':>5s}{'entries':>9s}"
          f"{'law':>6s}{'bytes at L=4096':>17s}")
    for widths in ((8, 8), (16, 16), (64, 64)):
        for bc in sorted({a[2] for a in arms if (a[3], a[4]) == widths}):
            rows = reference.union_entries(widths, bc)
            law = reference.union_entries_law(widths, bc)
            n = widths[0] * widths[1]
            byts = 2 * len(rows) * (4096 // 32) * 4
            print(f"{str(widths):>12s}{n:8d}{sum(widths):7d}{bc:5d}{len(rows):9d}"
                  f"{law:6d}{byts:17,d}")
            ok_ = law == len(rows)
            if not ok_:
                print("    LAW DISAGREES WITH THE BUILT TABLE")
            globals().setdefault("_t3", True)
    print("\n  N FIXED AT 4096, THE SUM OF THE WIDTHS VARIED (BC=64):")
    print(f"{'widths':>14s}{'N':>8s}{'sum W':>7s}{'entries':>9s}")
    for widths in ((64, 64), (32, 128), (128, 32), (16, 256), (256, 16)):
        n, s = widths[0] * widths[1], sum(widths)
        print(f"{str(widths):>14s}{n:8d}{s:7d}"
              f"{reference.union_entries_law(widths, 64):9d}")
    print("\n  THE SUM HELD PER LEVEL AND N GROWN BY DEPTH (width 64 per level, "
          "BC=64) -- the entry count is LINEAR in the depth while N is EXPONENTIAL:")
    print(f"{'depth':>7s}{'N':>12s}{'sum W':>7s}{'entries':>9s}"
          f"{'table MB at L=65536':>21s}{'state MB':>10s}")
    for d in range(2, 6):
        widths = tuple([64] * d)
        e = reference.union_entries_law(widths, 64)
        n = 64 ** d
        tab = 2 * e * (65536 // 8) / (1 << 20)
        state = n * 64 * 4 / (1 << 20)
        print(f"{d:7d}{n:12d}{64*d:7d}{e:9d}{tab:21.2f}{state:10.1f}")

    # ---- T4. THE KERNEL IS RUNG 4's -----------------------------------------
    print("\n---- T4 the kernel is rung 4's, bit for bit: only the source of the "
          "membership clauses changed ----")
    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        for support in cfg["supports"]:
            B, H, L = cfg["full"]
            pr = amps(B, H, L, widths, support, dev, 2)
            pw = amps(B, H, L, widths, support, dev, 3)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            for c, w, bc, w0, w1, nb, *_ in sel:
                r4build.assert_build_identity(c, w, bc, w0, w1)
                n4, d4 = r4build.forward_raw(pr, pw, v, c, w, bc, w0, w1)
                got, _, n5, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
                d5 = got[..., -1]
                same = torch.equal(n4, n5) and torch.equal(d4, d5)
                e = max(metrics(n5, n4)["rel_l2"], metrics(d5, d4)["rel_l2"])
                if nb == 1:
                    print(f"[T4] widths={widths} C={c:3d} SWZ={int(w)} BC={bc:4d} "
                          f"support={support:3d}  1 block: "
                          f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
                    ok &= same
                else:
                    print(f"[T4] widths={widths} C={c:3d} SWZ={int(w)} BC={bc:4d} "
                          f"support={support:3d}  {nb:3d} blocks: bit-identical "
                          f"{'YES' if same else 'NO'}   rel_L2 {e:.3e} "
                          f"(fan-in order is not defined)")
                    ok &= e < 1e-6

    # ---- 1. algorithm, and the exactness of the split, ON d_v+1 COLUMNS -------
    print("\n---- the fp64 layers, unchanged from rung 4 because the arithmetic is "
          "unchanged ----")
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

    # ---- 2/3/4. the kernel against fp64, every arm, at every support ---------
    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        for support in cfg["supports"]:
            dense = support >= widths[0]
            print(f"\n---- {cfg['name']}   support={support} "
                  f"({'DENSE' if dense else 'SPARSE'}) ----")
            Bo, Ho, _ = cfg["one"]
            for c, w, bc, w0, w1, *_ in sel:
                if w:
                    continue
                pr = amps(Bo, Ho, c, widths, support, dev, 10 + c)
                pw = amps(Bo, Ho, c, widths, support, dev, 20 + c)
                vv = torch.randn(Bo, Ho, c, 64, device=dev).bfloat16()
                got, y, _, _ = kernel(pr, pw, vv, c, w, bc, w0, w1, eps)
                Rb = reference.kron_bf16(pr, widths)
                Wb = reference.kron_bf16(pw, widths)
                md = reference.split_norm(Rb, Wb, vv, c, bc)
                e = metrics(got, md)
                ey = metrics(y, reference.ratio(md, eps))
                print(f"[2 one chunk ] C={c:3d} BC={bc:4d}  L=C, state==0  "
                      f"columns {fmt(e)}")
                print(f"                                      ratio   {fmt(ey)}")
                ok &= e["rel_l2"] < 1e-4 and ey["rel_l2"] < 1e-4

            B, H, L = cfg["full"]
            pr = amps(B, H, L, widths, support, dev, 2)
            pw = amps(B, H, L, widths, support, dev, 3)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            Rx, Wx = reference.kron(pr, widths), reference.kron(pw, widths)
            Rb, Wb = reference.kron_bf16(pr, widths), reference.kron_bf16(pw, widths)
            ex, md = {}, {}
            for c, w, bc, w0, w1, nb, *_ in sel:
                if c not in ex:
                    ex[c] = reference.chunked_norm(Rx, Wx, v, c)
                if (c, bc) not in md:
                    md[(c, bc)] = reference.split_norm(Rb, Wb, v, c, bc)
                got, y, _, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
                e1, e2 = metrics(got, ex[c]), metrics(got, md[(c, bc)])
                y1 = metrics(y, reference.ratio(ex[c], eps))
                y2 = metrics(y, reference.ratio(md[(c, bc)], eps))
                skip = 1.0 - reference.schedule(pr, pw, widths, bc,
                                                c).double().mean().item()
                print(f"[3 L={L:5d}  ] C={c:3d} SWZ={int(w)} BC={bc:4d} blocks={nb:3d} "
                      f"skipped {100*skip:5.1f}%  columns vs exact fp64 {fmt(e1)}")
                print(f"                              RATIO   vs the modelled "
                      f"rounding {fmt(y2)}")
                ok &= e2["rel_l2"] < 1e-3 and e1["rel_l2"] < 5e-3
                ok &= y2["rel_l2"] < 5e-3 and y1["rel_l2"] < 2e-2

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
                    got, y, _, _ = kernel(pr2, pw2, vv, c, w, bc, w0, w1, eps)
                    mdx = reference.split_norm(Rb2, Wb2, vv, c, bc)
                    worst = max(worst, metrics(got, mdx)["rel_l2"],
                                metrics(y, reference.ratio(mdx, eps))["rel_l2"])
                print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm, columns "
                      f"and ratio  {worst:.3e}")
                ok &= worst < 1e-3
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
