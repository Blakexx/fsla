"""COMMIT-1 PRICE: the factored Gram against the panel Gram (rung 6 at
GROUPS=1 == 5D), same arms, same instrument."""
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung5b import reference  # noqa: E402
from rung5d.check import regspec  # noqa: E402
from rung6 import build as b6  # noqa: E402
from rung6b import build as b6b  # noqa: E402

FLAG = (64, 64)


def sw(fn):
    best = None
    for _ in range(3):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        mins = []
        for _ in range(10):
            ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(15)]
            for a, b in ev:
                a.record()
                fn()
                b.record()
            torch.cuda.synchronize()
            mins.append(min(a.elapsed_time(b) for a, b in ev))
        m = statistics.median(mins)
        best = m if best is None else min(best, m)
    return best


def main():
    m6, mb = b6.module(), b6b.module()
    torch.manual_seed(0)
    for name, (B, H, L) in (("A", (2, 8, 4096)), ("B", (20, 8, 1024))):
        v = torch.randn(B, H, L, 64, device="cuda").bfloat16()
        print(f"cell {name}: B={B} H={H} L={L}")
        for regime in ("dense", "lead4", "d4"):
            if regime == "dense":
                pr = reference.amplitudes(B, H, L, FLAG, "cuda", seed=2)
                pw = reference.amplitudes(B, H, L, FLAG, "cuda", seed=3)
            else:
                pr, pw = reference.regime_amplitudes(B, H, L, regspec(FLAG, regime), "cuda", seed=5)
            for c, bc in ((32, 128), (16, 128), (64, 128)):
                if L % c:
                    continue
                tbl = mb.union_table(pr, pw, v, c, bc, *FLAG)

                def f0():
                    return m6.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

                def f1():
                    return mb.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

                sw(f0)
                a1, b1 = sw(f0), sw(f1)
                b2, a2 = sw(f1), sw(f0)
                a, b = min(a1, a2), min(b1, b2)
                print(f"  {regime:6s} C={c} BC={bc}: panel {a:8.4f}  factored {b:8.4f}  {100*(b-a)/a:+6.1f}%")
        print(f"  peak_alloc {torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB")


if __name__ == "__main__":
    main()
