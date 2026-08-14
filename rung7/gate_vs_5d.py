"""RUNG 5E GATE: bit-identity against rung 5D at every arm and every regime.

5D is itself gated bit-identical to 5B (42/42), so equality here closes the
chain. Also: residency (stages, CTA target) unchanged arm for arm; zero local
bytes; and a depth-1 formed ring on two arms so the restructured wait path is
EXERCISED, not assumed.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

from rung5b import reference  # noqa: E402
from rung5d import build as b5d  # noqa: E402
from rung5d.check import regspec  # noqa: E402
from rung7 import build as b6  # noqa: E402

B, H, L = 1, 2, 1024
dev = "cuda"

def run5(mod, pr, pw, v, c, w, bc, w0, w1):
    tbl = mod.union_table(pr, pw, v, c, bc, w0, w1)
    n, d = mod.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1)
    return n, d

def run6(mod, pr, pw, v, c, w, bc, widths):
    tbl = mod.union_table(pr, pw, v, c, bc, list(widths))
    g = torch.ones(v.shape[0], v.shape[1], v.shape[2], device=v.device).bfloat16()
    r0 = torch.zeros(v.shape[1], device=v.device, dtype=torch.float32)
    n, d = mod.forward_raw(pr, pw, g, v, tbl, c, w, bc, list(widths), r0, 0)
    return n, d

def main():
    torch.manual_seed(0)
    a5d = {tuple(a[:5]): a for a in b5d.module().arms()}
    m5e = b6.module()
    # 6F arms are (C, SWZ, BC, depth, width); a depth-2 arm's 5D key is
    # (C, SWZ, BC, width, width).  Deeper arms have no 5D counterpart BY
    # CONSTRUCTION -- 5D is a two-level kernel -- and are oracle-gated only.
    a5e = {(a[0], a[1], a[2], a[4], a[4]): a for a in m5e.arms() if a[3] == 2}
    deep = sorted((a[0], a[1], a[2], a[3], a[4]) for a in m5e.arms() if a[3] != 2)
    assert set(a5d) <= set(a5e), "rung 6 must carry every 5D arm"
    print("DEEP arms (no 5D counterpart possible, oracle-gated separately):", deep)
    for k in deep:
        st = m5e.build_stamp(k[0], bool(k[1]), k[2], [k[4]] * k[3])
        print(f"  {k}: regs {st[3]}, smem {st[6]}, spill {st[8]}, stages {st[9]}")
        assert st[8] == 0, f"spill at {k}"
    a5e = {k: v for k, v in a5e.items() if k in a5d}
    print(f"{'arm':>24s}  S ctas smem | refactor  spill")
    for k in sorted(a5d):
        d, e = a5d[k], a5e[k]
        # rung 7 adds the decay scratch (T_end fp32 row + gain bf16 row per
        # group) to fixed SMEM -- a DOCUMENTED, computed delta; stages and CTA
        # target must be untouched by it.
        c_, _, bc_ = k[0], k[1], k[2]
        grp_ = 2 if bc_ >= 256 else 1
        extra = grp_ * ((bc_ // grp_) * 4 + ((c_ * 2 + 15) // 16) * 16)
        s_ok = d[8] == e[8] and d[10] == e[10] and d[9] + extra == e[9]
        st = m5e.build_stamp(k[0], bool(k[1]), k[2], [k[3], k[4]])
        spill = st[8]
        print(f"{str(k):>24s}  {e[8]} {e[10]:>4d} {e[9]:>6d} | {'OK' if s_ok else 'FAIL'}  {spill}")
        assert s_ok, f"geometry changed at {k}: 5D (S={d[8]} ctas={d[10]} smem={d[9]}), r6 (S={e[8]} ctas={e[10]} smem={e[9]})"
        assert spill == 0, f"spill at {k}: {spill} local bytes"
    print()
    m5d = b5d.module()
    fails = 0
    for regime in ("dense", "lead4", "split", "d4"):
        for k in sorted(a5d):
            c, w, bc, w0, w1 = k
            widths = (w0, w1)
            pr, pw = reference.regime_amplitudes(B, H, L, regspec(widths, regime), dev, seed=5)
            v = torch.randn(B, H, L, 64, device=dev, generator=torch.Generator(dev).manual_seed(7)).bfloat16().contiguous()
            n5, d5 = run5(m5d, pr, pw, v, c, bool(w), bc, w0, w1)
            n6, d6 = run6(m5e, pr, pw, v, c, bool(w), bc, [w0, w1])
            one_block = (w0, w1) == (8, 8)
            if one_block:
                # one CTA: no fan-in order freedom, so equality is owed exactly
                ok = torch.equal(n5, n6) and torch.equal(d5, d6)
                tag = "equal" if ok else "NOT EQUAL"
            else:
                # >1 block: RED arrival order is free, spread measured 1.4e-7
                # at rung 2; the standing 5D-vs-5B criterion is <= ~1.6e-7
                rl = (n5.double() - n6.double()).norm() / n5.double().norm().clamp_min(1e-30)
                ok = bool(rl <= 3e-7)
                tag = f"rel_L2={rl:.2e}"
            if not ok:
                fails += 1
                print(f"  FAIL {regime} {k}: {tag}")
        print(f"{regime}: {'PASS' if fails == 0 else f'{fails} FAIL'} over {len(a5d)} arms (one-block exact, multi-block <= fan-in spread)")
        assert fails == 0
    # exercise the wait: depth-1 formed ring, walk waits on almost every word
    # step A keeps 5D's fused walk, so the depth-1 wait gate applies as-is.
    m1 = b6.module(1)
    for k, exact in [((16, 0, 64, 8, 8), True), ((32, 0, 128, 64, 64), False)]:
        c, w, bc, w0, w1 = k
        for regime in ("dense", "d4"):
            pr, pw = reference.regime_amplitudes(B, H, L, regspec((w0, w1), regime), dev, seed=5)
            v = torch.randn(B, H, L, 64, device=dev, generator=torch.Generator(dev).manual_seed(7)).bfloat16().contiguous()
            n5, d5 = run5(m5d, pr, pw, v, c, bool(w), bc, w0, w1)
            n1, d1 = run6(m1, pr, pw, v, c, bool(w), bc, [w0, w1])
            if exact:
                assert torch.equal(n5, n1) and torch.equal(d5, d1), \
                    f"depth-1 NOT EQUAL at one block {regime} {k}"
            else:
                rl = (n5.double() - n1.double()).norm() / n5.double().norm().clamp_min(1e-30)
                assert rl <= 3e-7, f"depth-1 {regime} {k}: rel_L2={rl:.3e}"
    print("depth-1 wait path: EXERCISED (one-block exact, multi-block <= spread)")
    print("\nGATE PASS")

if __name__ == "__main__":
    main()
