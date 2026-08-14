"""TRANSITIONAL GATE (P69 stage B->C): the adopted chunk consumer against the
ladder's rung-6G module, arm for arm, same inputs, one process.

Criterion is the ladder's own: ONE-BLOCK arms owe exact equality; multi-block
arms owe rel_L2 <= 3e-7 (fan-in arrival order is free between launches).
DEV-ONLY: requires the ladder repo checkout; retired when the ladder gates
are re-pointed at this tree (stage D).
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

#: BIND the standalone's package FIRST -- the ladder repo carries its own
#: `rola/`, and path order decides which one `import rola` means.
from rola.ops._ext import extension  # noqa: E402

sys.path.insert(0, "/home/blake/rola-wt-ladder")
from rung5b import reference  # noqa: E402  isort:skip
from rung6g import build as lad  # noqa: E402  isort:skip

def main():
    m = extension()
    lad.module()  # force the ladder build before timing-sensitive comparisons
    torch.manual_seed(0)
    B, H, L = 1, 2, 1024
    fails = 0
    ladder_arms = {tuple(a[:5]) for a in lad.arms()}
    for arm in m.chunk_arms():
        c, bc, d, wid = arm[0], arm[1], arm[2], arm[3]
        widths = (wid,) * d
        #: the standalone retired the swizzle axis; match the ladder's
        #: SWZ=false rows, and skip arms born here (the flat D=1 baseline).
        if (c, 0, bc, d, wid) not in ladder_arms:
            continue
        for regime_sup in (wid, 2):
            pr = reference.support_amplitudes(B, H, L, widths, regime_sup, "cuda", seed=5)
            pw = reference.support_amplitudes(B, H, L, widths, regime_sup, "cuda", seed=6)
            g = torch.ones(B, H, L, device="cuda").bfloat16()
            v = torch.randn(B, H, L, 64, device="cuda",
                            generator=torch.Generator("cuda").manual_seed(7)).bfloat16()
            tl = lad.union_table(pr, pw, v, c, bc, list(widths))
            nl, dl = lad.forward_raw(pr, pw, g, v, tl, c, False, bc, list(widths))
            ts = m.chunk_union_table(pr, pw, v, c, bc, list(widths))
            ns, ds = m.chunk_forward_raw(pr, pw, g, v, ts, c, bc, list(widths))
            assert torch.equal(tl, ts), f"TABLE differs at {arm}"
            one_block = (wid ** d) <= bc
            if one_block:
                ok = torch.equal(nl, ns) and torch.equal(dl, ds)
                tag = "equal" if ok else "NOT EQUAL"
            else:
                rl = (nl.double() - ns.double()).norm() / nl.double().norm().clamp_min(1e-30)
                ok = bool(rl <= 3e-7)
                tag = f"rel={rl:.2e}"
            if not ok:
                fails += 1
                print(f"  FAIL {tuple(arm[:5])} sup={regime_sup}: {tag}")
    n = len(m.chunk_arms())
    print(f"LADDER PARITY: {'PASS' if fails == 0 else f'{fails} FAIL'} over {n} arms x 2 regimes")
    return fails == 0

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
