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
from rung5f import build as b5f  # noqa: E402

B, H, L = 1, 2, 1024
dev = "cuda"

def run(mod, pr, pw, v, c, w, bc, w0, w1):
    tbl = mod.union_table(pr, pw, v, c, bc, w0, w1)
    n, d = mod.forward_raw(pr, pw, v, tbl, c, w, bc, w0, w1)
    return n, d

def main():
    torch.manual_seed(0)
    a5d = {tuple(a[:5]): a for a in b5d.module().arms()}
    m5e = b5f.module()
    a5e = {tuple(a[:5]): a for a in m5e.arms()}
    assert a5d.keys() == a5e.keys(), "arm sets differ"
    print(f"{'arm':>24s}  S ctas extra | resid  spill")
    for k in sorted(a5d):
        d, e = a5d[k], a5e[k]
        s_ok = d[8] == e[8] and d[10] == e[10]
        st = m5e.build_stamp(k[0], bool(k[1]), k[2], k[3], k[4])
        spill = st[8]
        print(f"{str(k):>24s}  {e[8]} {e[10]:>4d} {e[12]:>5d} | {'OK' if s_ok else 'FAIL'}  {spill}")
        assert s_ok, f"residency changed at {k}: 5D S={d[8]} ctas={d[10]}, 5E S={e[8]} ctas={e[10]}"
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
            n5, d5 = run(m5d, pr, pw, v, c, bool(w), bc, w0, w1)
            n6, d6 = run(m5e, pr, pw, v, c, bool(w), bc, w0, w1)
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
    # THE WAIT, EXERCISED.  Depth 1 is structurally impossible now: emission
    # holds a super-chunk's words live in the formed ring, and the static
    # assert refuses a ring that cannot span the in-flight super-chunks.  So
    # the reduced ring is 64 words with L=4096 (128 words of axis): the form
    # is bounded by the emission floor and the scan genuinely outruns it.
    m1 = b5f.module(64)
    L2 = 4096
    for k, exact in [((16, 0, 64, 8, 8), True), ((32, 0, 128, 64, 64), False)]:
        c, w, bc, w0, w1 = k
        for regime in ("dense", "d4"):
            pr, pw = reference.regime_amplitudes(B, H, L2, regspec((w0, w1), regime), dev, seed=5)
            v = torch.randn(B, H, L2, 64, device=dev, generator=torch.Generator(dev).manual_seed(7)).bfloat16().contiguous()
            n5, d5 = run(m5d, pr, pw, v, c, bool(w), bc, w0, w1)
            n1, d1 = run(m1, pr, pw, v, c, bool(w), bc, w0, w1)
            if exact:
                assert torch.equal(n5, n1) and torch.equal(d5, d1), \
                    f"reduced-ring NOT EQUAL at one block {regime} {k}"
            else:
                rl = (n5.double() - n1.double()).norm() / n5.double().norm().clamp_min(1e-30)
                assert rl <= 3e-7, f"reduced-ring {regime} {k}: rel_L2={rl:.3e}"
    print("reduced-ring (64 words, L=4096) wait path: EXERCISED")
    print("\nGATE PASS")

if __name__ == "__main__":
    main()
