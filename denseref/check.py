"""Correctness gate. No performance number is reported unless this passes.

Three layers, each isolating a different thing:

1. ALGORITHM   -- the fp64 chunked recurrence vs the fp64 quadratic definition.
                  Pure maths, no kernel. Must agree to fp64 round-off.
2. SINGLE CHUNK (L == C) -- the state is identically zero, so the ONLY rounding
                  in the kernel that the fp64 reference must model is the bf16
                  Gram. Isolates the intra path, the causal mask, the operand
                  conventions and the output path with no state noise at all.
3. FULL LENGTH -- L = 4096, the measurement cell. Reported against BOTH the
                  exact fp64 chunked recurrence (the DESIGN error: what a bf16
                  chunked kernel costs you) and the fp64 recurrence with the
                  state rounded to bf16 at every readout (the IMPLEMENTATION
                  error: what THIS kernel costs you on top of the design).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from denseref import build, reference  # noqa: E402

CHUNKS = (16, 32, 64)
DEFAULT_STAGES = {16: 4, 32: 4, 64: 3}


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
    ARMS = build.arms()

    # THE CAPABILITY TABLE, as the built binary reports it. Printed before any
    # number, because every derived quantity below is downstream of it.
    print("[0 caps      ] the architecture capability table")
    for cc in build.TABULATED_CC:
        r = build.arch_row(cc)
        req = " ".join(f"{n}={int(r[n])}" for n in build.REQUIRED)
        sel = " ".join(f"{n}={int(r[n])}" for n in build.SELECTING)
        mark = "BUILT" if cc == build.TARGET_CC else ("     " if r["supported"]
                                                      else "REFUSED")
        print(f"               sm_{cc//10:<3d} {mark:7s} REQUIRED[{req}] "
              f"SELECTING[{sel}]")
        print(f"                       FACTS smem/SM={r['smem_per_sm']:,d} "
              f"smem/CTA<={r['smem_per_cta_max']:,d} regs/SM={r['regs_per_sm']:,d} "
              f"threads/SM={r['max_threads_per_sm']} CTAs/SM<={r['max_ctas_per_sm']} "
              f"mma_unit={r['mma_unit_threads']}t fp32_accum={r['tensor_f32_accum_pct']}%")

    print(f"[0 arms      ] {len(ARMS)} built (C, STAGES, SWZ, smem_B); residency "
          f"DERIVED from the table, not from a literal:")
    for c, st, w, b in ARMS:
        cells = []
        for cc in (860, 800):
            if not build.arch_row(cc)["supported"]:
                continue
            p = build.residency_plan(b, 255, cc)
            cells.append(f"sm_{cc//10}:{p['by_smem']}")
        print(f"               C={c:3d} S={st} SWZ={int(w)}  smem={b:7,d} B  "
              f"CTAs/SM by smem  {'  '.join(cells)}")
    for t in (128, 256):
        b86 = build.residency_plan(0, 255, 860, t)["reg_budget"]
        b80 = build.residency_plan(0, 255, 800, t)["reg_budget"]
        print(f"[0 reg budget] per-lane registers sustaining n CTAs of {t:3d} threads:"
              f"  sm_86 {b86}   sm_80 {b80}")
    # The production consumer kernel's shape, planned from the table alone. This
    # box owns no A100; the row does. It is the check that the derivation is the
    # one that answered P58's question, not a reimplementation of it.
    for cc in (860, 800):
        p = build.residency_plan(44544, 128, cc, 256)
        bind = " + ".join(n for n, key in (("smem", "by_smem"), ("regs", "by_regs"),
                                           ("threads", "by_threads"),
                                           ("cta cap", "by_cta_cap"))
                          if p[key] == p["ctas"])
        print(f"[0 plan 44544] sm_{cc//10} at 256 threads / 128 regs: by_smem "
              f"{p['by_smem']}  by_regs {p['by_regs']}  by_threads {p['by_threads']}"
              f"  -> {p['ctas']} CTAs/SM, bound by {bind}")

    torch.manual_seed(0)
    q = (torch.randn(2, 3, 512, 64, device=dev) * 0.5).bfloat16()
    k = (torch.randn(2, 3, 512, 64, device=dev) * 0.5).bfloat16()
    v = torch.randn(2, 3, 512, 64, device=dev).bfloat16()
    quad = reference.quadratic(q, k, v)
    for c in CHUNKS:
        e = metrics(reference.chunked(q, k, v, c), quad)
        print(f"[1 algorithm ] chunk={c:3d}  fp64 chunked vs fp64 quadratic   {fmt(e)}")
        ok &= e["rel_l2"] < 1e-13

    for c in CHUNKS:
        torch.manual_seed(1)
        qq = (torch.randn(4, 4, c, 64, device=dev) * 0.5).bfloat16()
        kk = (torch.randn(4, 4, c, 64, device=dev) * 0.5).bfloat16()
        vv = torch.randn(4, 4, c, 64, device=dev).bfloat16()
        ref2 = reference.chunked_bf16_state(qq, kk, vv, c).bfloat16()
        for cc, ss, ww, _ in ARMS:
            if cc != c:
                continue
            st = build.assert_build_identity(cc, ss, ww)
            got = build.forward(qq, kk, vv, cc, ss, ww)
            torch.cuda.synchronize()
            e = metrics(got, ref2)
            print(f"[2 one chunk ] C={cc:3d} S={ss} SWZ={int(ww)}  L=C, state==0   {fmt(e)}"
                  f"   regs={st['regs_per_thread']} dynsmem={st['dynamic_smem']}")
            ok &= e["rel_l2"] < 1e-4

    torch.manual_seed(2)
    L = 4096
    q = (torch.randn(2, 8, L, 64, device=dev) * 0.5).bfloat16()
    k = (torch.randn(2, 8, L, 64, device=dev) * 0.5).bfloat16()
    v = torch.randn(2, 8, L, 64, device=dev).bfloat16()
    for c in CHUNKS:
        ex = reference.chunked(q, k, v, c)
        bs = reference.chunked_bf16_state(q, k, v, c)
        d = metrics(bs, ex)
        bsb = bs.bfloat16()
        base = None
        for cc, ss, ww, _ in ARMS:
            if cc != c:
                continue
            got = build.forward(q, k, v, cc, ss, ww)
            torch.cuda.synchronize()
            e1, e2 = metrics(got, ex), metrics(got, bsb)
            # Every arm of one chunk size must be BIT-IDENTICAL: staging depth
            # and shared-memory layout change no arithmetic whatsoever.
            if base is None:
                base = got.clone()
                same = "(reference arm)"
            else:
                same = "BIT-IDENTICAL" if torch.equal(got, base) else "*** DIVERGES ***"
                ok &= torch.equal(got, base)
            print(f"[3 L={L}   ] C={cc:3d} S={ss} SWZ={int(ww)} vs exact fp64 {fmt(e1)}  {same}")
            print(f"                        vs bf16-state fp64, bf16 out {fmt(e2)}")
            ok &= e2["rel_l2"] < 1e-3 and e1["rel_l2"] < 5e-3
        print(f"                        (design floor: bf16-state fp64 vs exact fp64 "
              f"rel_L2 {d['rel_l2']:.3e})")

    # 4. LAYERED: a length that is not a multiple of the staging depth, so the
    #    pipeline prologue/epilogue and the deferred output store are exercised
    #    at every ring phase, plus a single-CTA and a 1-chunk sequence.
    torch.manual_seed(3)
    for Lx in (64, 128, 192, 320, 1024):
        qx = (torch.randn(1, 2, Lx, 64, device=dev) * 0.5).bfloat16()
        kx = (torch.randn(1, 2, Lx, 64, device=dev) * 0.5).bfloat16()
        vx = torch.randn(1, 2, Lx, 64, device=dev).bfloat16()
        for cc, ss, ww, _ in ARMS:
            if Lx % cc:
                continue
            got = build.forward(qx, kx, vx, cc, ss, ww)
            torch.cuda.synchronize()
            e = metrics(got, reference.chunked_bf16_state(qx, kx, vx, cc).bfloat16())
            bad = e["rel_l2"] >= 1e-3
            if bad:
                print(f"[4 ring phase] L={Lx:5d} C={cc:3d} S={ss} SWZ={int(ww)}  {fmt(e)}  FAIL")
            ok &= not bad
    print(f"[4 ring phase] L in (64,128,192,320,1024) x all arms: "
          f"{'all within 1e-3' if ok else 'FAILURES ABOVE'}")
    return ok


if __name__ == "__main__":
    good = run()
    print("CORRECTNESS:", "PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
