"""THE FUSION'S VERDICT: one BC=256 block (8 warps, 1 CTA, Gram exchange)
against the same work as two BC=128 blocks (4 warps, 2-3 CTAs co-resident).
Same inputs, same instrument as every stop gate: A-B-B-A, block-min-median,
discarded warm-up, VRAM guard."""
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung5b import reference  # noqa: E402
from rung5d.check import regspec  # noqa: E402
from rung6 import build  # noqa: E402

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
    m = build.module()
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
            t128 = m.union_table(pr, pw, v, 32, 128, *FLAG)
            t256 = m.union_table(pr, pw, v, 32, 256, *FLAG)
            def f1():
                return m.forward_raw(pr, pw, v, t128, 32, False, 128, *FLAG)

            def f2():
                return m.forward_raw(pr, pw, v, t256, 32, False, 256, *FLAG)
            sw(f1)  # discarded warm-up
            a1, b1 = sw(f1), sw(f2)
            b2, a2 = sw(f2), sw(f1)
            a, b = min(a1, a2), min(b1, b2)
            print(f"  {regime:6s} C=32: BC=128 {a:8.4f} ms   BC=256-fused {b:8.4f} ms   {100*(b-a)/a:+6.1f}%")
        used = torch.cuda.max_memory_allocated() / (1 << 30)
        print(f"  peak_alloc {used:.3f} GiB")

if __name__ == "__main__":
    main()
