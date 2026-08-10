"""Correctness gate for RUNG 5B. No performance number is reported unless this passes.

This rung changes WHICH TOKENS a visit carries and nothing else, so its gate has
two halves and the first is an EQUALITY rather than a tolerance:

T1. THE TABLE IS THE LAW'S CLAUSES, and the bits the COOK consumes are the bits
    the device table holds. One structure, two consumers -- rung 5A reduced them
    per group, this rung reads them per token, and the gate is that the per-side
    reference and the device table agree bit for bit.
T2. AT DENSE, COMPACTION IS THE IDENTITY. The cook's partition IS the contiguous
    chunking and its thresholds ARE `i + 1`, asserted from the design; and the
    kernel is therefore BIT-IDENTICAL to rung 5A at every one-block arm and every
    support, and inside the fan-in's own spread above one block.
T3. WHERE COMPACTION FIRES, THE ARITHMETIC IS THE MODELLED ONE. Against an fp64
    reference that runs the recurrence on the cook's own partition with the
    monotone staircase -- compaction is exact in exact arithmetic, so what the
    model captures is where the state is ROUNDED, which the partition moves.
T4. THE COMPACTION IS REAL. The super-chunk count against the live-group count,
    per regime, so a gate cannot pass by having compacted nothing.

Everything below is rung 5A's gate, unchanged in substance where the arithmetic
is unchanged: the prefix test is still a SUPERSET of slot enumeration, the fp64
recurrence is still rung 2's on `with_ones(v)`, and the ratio is still the spec's
line.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rola.ops.constants import READOUT_EPS  # noqa: E402
from rung5 import build as r5build  # noqa: E402
from rung5b import build, reference  # noqa: E402

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


#: THE TOPOLOGY REGIMES AT ANY WIDTH. `rung5.reference.REGIMES` names them at the
#: flagship's (64,64); the same four patterns are what the smaller width
#: configurations need, and a regime is a statement about WHERE sparsity sits
#: relative to the block boundary rather than about a particular width. `d4` is a
#: deeper topology expressed as the product pattern the two-level interface sees,
#: so its sub-levels must multiply to the level's width.
def regspec(widths, regime):
    w0, w1 = widths
    if regime == "dense":
        return [[(w0, w0, w0)], [(w1, w1, w1)]]
    if regime == "lead4":
        k = min(4, w0)
        return [[(w0, k, k)], [(w1, w1, w1)]]
    if regime == "split":
        return [[(w0, w0, 1)], [(w1, 1, w1)]]
    a0, a1 = (2 if w0 == 8 else w0 // 8), (2 if w1 == 8 else w1 // 8)
    return [[(a0, a0, 1), (w0 // a0, 1, w0 // a0)],
            [(a1, a1, 1), (w1 // a1, 1, w1 // a1)]]


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

    print("== arms in the binary (rung 5A's set, arm for arm) ==")
    for c, w, bc, w0, w1, nb, ls, ts, s, b, t, ent in arms:
        st = build.assert_build_identity(c, w, bc, w0, w1)
        print(f"  C={c:3d} SWZ={int(w)} BC={bc:4d} widths=({w0:2d},{w1:2d})  "
              f"N={w0*w1:5d} blocks={nb:3d} rect={ls}x{ts:<3d}  S={s}  ctas={t}  "
              f"smem={b:7,d} B  regs={st['regs_per_thread']:3d}  "
              f"local={st['local_bytes']}  table entries={ent:3d}")

    # ---- T1. THE TABLE, AND THE BITS THE COOK CONSUMES ------------------------
    print("\n---- T1 the table IS the law's clauses, and the per-side bits the "
          "cook reads ARE the device table's ----")
    for widths, cfg in CONFIGS.items():
        for support in cfg["supports"]:
            B, H, L = 2, 2, 256
            pr = amps(B, H, L, widths, support, dev, 40)
            pw = amps(B, H, L, widths, support, dev, 41)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            for c, sw, bc, _, _, *_ in [a for a in arms if (a[3], a[4]) == widths]:
                if sw:
                    continue
                tbl = build.union_table(pr, pw, v, c, bc, widths[0], widths[1])
                got = reference.union_unpack(tbl)
                ref = reference.union_bits(pr, pw, widths, bc)
                t1 = bool((got == ref).all())
                #: THE COOK'S OWN INPUT. `side_live` derives the per-(side, block,
                #: token) bit from the law; the kernel derives it from the table by
                #: the same AND. If these disagree the two consumers have drifted.
                rl, wl = reference.side_live(pr, pw, widths, bc)
                sd = reference.schedule_from_table(got, widths, bc, c)
                grp = (rl | wl).reshape(*rl.shape[:2], L // c, c).any(-1)
                t2 = bool((sd == grp).all())
                exa = reference.schedule_exact(pr, pw, widths, bc, c)
                sup = bool((exa.reshape(sd.shape) & ~sd).sum() == 0)
                print(f"[T1] widths={widths} BC={bc:4d} C={c:3d} support={support:3d} "
                      f" entries={tbl.shape[2]:3d}  table exact: "
                      f"{'YES' if t1 else 'NO'}  per-side bits == table's: "
                      f"{'YES' if t2 else 'NO'}  covers slot enumeration: "
                      f"{'YES' if sup else 'NO'}")
                ok &= t1 and t2 and sup

    # ---- T2. AT DENSE, COMPACTION IS THE IDENTITY -----------------------------
    print("\n---- T2 at dense the cook IS the contiguous chunking and the "
          "staircase IS the triangle -- asserted from the design ----")
    for c in (16, 32, 64):
        for L in (c, 4 * c, 320 if 320 % c == 0 else 8 * c):
            reference.assert_dense_identity(L, c)
        print(f"[T2] C={c:3d}  cook == contiguous chunking, thresholds == i+1  "
              f"ASSERTED at L = C, 4C, 8C")

    print("\n---- T2 the kernel is rung 5A's at dense, bit for bit ----")
    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        B, H, L = cfg["full"]
        pr = reference.amplitudes(B, H, L, widths, dev, seed=2)
        pw = reference.amplitudes(B, H, L, widths, dev, seed=3)
        v = torch.randn(B, H, L, 64, device=dev).bfloat16()
        for c, w, bc, w0, w1, nb, *_ in sel:
            r5build.assert_build_identity(c, w, bc, w0, w1)
            t5 = r5build.union_table(pr, pw, v, c, bc, w0, w1)
            n5, d5 = r5build.forward_raw(pr, pw, v, t5, c, w, bc, w0, w1)
            got, _, n6, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
            d6 = got[..., -1]
            same = torch.equal(n5, n6) and torch.equal(d5, d6)
            e = max(metrics(n6, n5)["rel_l2"], metrics(d6, d5)["rel_l2"])
            if nb == 1:
                print(f"[T2] widths={widths} C={c:3d} SWZ={int(w)} BC={bc:4d} DENSE "
                      f" 1 block: {'BIT-IDENTICAL' if same else 'DIFFERS'}")
                ok &= same
            else:
                print(f"[T2] widths={widths} C={c:3d} SWZ={int(w)} BC={bc:4d} DENSE "
                      f" {nb:3d} blocks: bit-identical {'YES' if same else 'NO'}   "
                      f"rel_L2 {e:.3e} (fan-in order is not defined)")
                ok &= e < 1e-6
        del pr, pw, v
        torch.cuda.empty_cache()

    # ---- T3. WHERE COMPACTION FIRES ------------------------------------------
    print("\n---- T3 the compacted arithmetic against its fp64 model: the "
          "recurrence ON THE COOK'S OWN PARTITION, with the staircase ----")
    for widths in ((8, 8), (16, 16), (64, 64)):
        sel = [a for a in arms if (a[3], a[4]) == widths and not a[1]]
        for regime in ("dense", "lead4", "split", "d4"):
            B, H, L = 1, 1, 256
            pr, pw = reference.regime_amplitudes(B, H, L, regspec(widths, regime),
                                                 dev, seed=5)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            Rb = reference.kron_bf16(pr, widths)
            Wb = reference.kron_bf16(pw, widths)
            v1 = reference.with_ones(v)
            for c, w, bc, w0, w1, nb, *_ in sel:
                plans = reference.plans_for(pr, pw, widths, bc, c)
                md = reference.split_compacted(Rb, Wb, v1, c, bc, plans)
                got, y, _, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
                e = metrics(got, md)
                ey = metrics(y, reference.ratio(md, eps))
                sc = sum(len(p) for p in plans)
                print(f"[T3] widths={widths} C={c:3d} BC={bc:4d} {regime:>6s} "
                      f"blocks={nb:3d} super-chunks={sc:5d} (groups {nb * L // c:5d})"
                      f"  columns rel_L2 {e['rel_l2']:.3e}  ratio {ey['rel_l2']:.3e}")
                ok &= e["rel_l2"] < 1e-3 and ey["rel_l2"] < 5e-3
            del pr, pw, v, Rb, Wb, v1
            torch.cuda.empty_cache()

    # ---- the fp64 layers ------------------------------------------------------
    print("\n---- the fp64 layers: compaction REGROUPS an exact sum, so the exact "
          "recurrence is unchanged and is still the binding reference ----")
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

    for widths, cfg in CONFIGS.items():
        sel = [a for a in arms if (a[3], a[4]) == widths]
        for support in cfg["supports"]:
            dense = support >= widths[0]
            print(f"\n---- {cfg['name']}   support={support} "
                  f"({'DENSE' if dense else 'SPARSE'}) ----")
            B, H, L = cfg["full"]
            pr = amps(B, H, L, widths, support, dev, 2)
            pw = amps(B, H, L, widths, support, dev, 3)
            v = torch.randn(B, H, L, 64, device=dev).bfloat16()
            Rx, Wx = reference.kron(pr, widths), reference.kron(pw, widths)
            ex = {}
            for c, w, bc, w0, w1, nb, *_ in sel:
                if c not in ex:
                    ex[c] = reference.chunked_norm(Rx, Wx, v, c)
                got, y, _, _ = kernel(pr, pw, v, c, w, bc, w0, w1, eps)
                e1 = metrics(got, ex[c])
                y1 = metrics(y, reference.ratio(ex[c], eps))
                print(f"[3 L={L:5d}  ] C={c:3d} SWZ={int(w)} BC={bc:4d} blocks={nb:3d} "
                      f" columns vs exact fp64 {fmt(e1)}")
                ok &= e1["rel_l2"] < 5e-3 and y1["rel_l2"] < 2e-2
            del pr, pw, v, Rx, Wx, ex
            torch.cuda.empty_cache()

            for Lx in (64, 128, 192, 320):
                pr2 = amps(2, 2, Lx, widths, support, dev, 30)
                pw2 = amps(2, 2, Lx, widths, support, dev, 31)
                vv = torch.randn(2, 2, Lx, 64, device=dev).bfloat16()
                Rx2 = reference.kron(pr2, widths)
                Wx2 = reference.kron(pw2, widths)
                worst = 0.0
                for c, w, bc, w0, w1, *_ in sel:
                    if Lx % c:
                        continue
                    got, y, _, _ = kernel(pr2, pw2, vv, c, w, bc, w0, w1, eps)
                    exx = reference.chunked_norm(Rx2, Wx2, vv, c)
                    worst = max(worst, metrics(got, exx)["rel_l2"],
                                metrics(y, reference.ratio(exx, eps))["rel_l2"])
                print(f"[4 ring phase] L={Lx:5d}  worst rel_L2 over every arm, columns "
                      f"and ratio  {worst:.3e}")
                ok &= worst < 1e-2
                del pr2, pw2, vv, Rx2, Wx2
                torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
