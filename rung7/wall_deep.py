"""THE DEEP-ARM PRICE: depth 3 and 4 against the flagship, MATCHED N = 4096.

Same kernel, same tokens, same values -- what changes is the TOPOLOGY the
capacity is factored into: (64,64) vs (16,16,16) vs (8,8,8,8).  At BC = 128
all three run the same 32 blocks, so the comparison is the schedule's and the
produce chain's, not the grid's.  The amplitude tensors differ in width by
construction (128 vs 48 vs 32 columns) -- the producer upstream is NOT priced
here, the kernel is.

Support is PER LEVEL (support_amplitudes), and the MATCHING MATTERS: equal
per-level support gives sup^D active leaves, so a fixed sup compares DIFFERENT
densities across depths (256 leaves at depth 4 vs 16 at depth 2 for sup = 4)
-- that sweep is reported, but as a density artifact, not a depth price.  The
depth price is read from the MATCHED-LEAVES sweep, where sup is chosen per
topology so the active leaf count agrees.  The skip column is the fp64
schedule's whole-chunk accounting; token-level compaction inside a live chunk
is what the wall itself shows.
"""
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from rung5b import reference  # noqa: E402
from rung7 import build  # noqa: E402

ARMS = [((64, 64), 32, 128), ((16, 16, 16), 32, 128),
        ((8, 8, 8, 8), 32, 128), ((8, 8, 8, 8), 32, 32)]


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
    mod = build.module()
    torch.manual_seed(0)
    B, H, L = 2, 8, 4096
    v = torch.randn(B, H, L, 64, device="cuda").bfloat16()
    print(f"cell A: B={B} H={H} L={L}, matched N = 4096, kernel only (table prebuilt)")

    def row(label, widths, c, bc, s, base):
        pr = reference.support_amplitudes(B, H, L, widths, s, "cuda", seed=11)
        pw = reference.support_amplitudes(B, H, L, widths, s, "cuda", seed=12)
        skip = 1.0 - reference.schedule(pr, pw, widths, bc, c).double().mean().item()
        tbl = mod.union_table(pr, pw, v, c, bc, list(widths))
        g = torch.ones(B, H, L, device="cuda").bfloat16()

        r0 = torch.zeros(H, device="cuda", dtype=torch.float32)

        def f():
            return mod.forward_raw(pr, pw, g, v, tbl, c, False, bc, list(widths),
                                   r0, 0)

        sw(f)
        t = min(sw(f), sw(f))
        tag = f"D{len(widths)} BC={bc}"
        print(f"{label:>10s}{str(widths) + ' ' + tag:>22s}{100*skip:7.1f}%"
              f"{t:10.4f}{100*(t-base)/base if base else 0.0:+12.1f}%")
        return t

    hdr = f"{'match':>10s}{'arm':>22s}{'skip':>8s}{'wall ms':>10s}{'vs flagship':>13s}"
    print("\n== per-level support (a DENSITY sweep, not a depth price) ==")
    print(hdr)
    for sup in ("dense", 4, 1):
        base = None
        for widths, c, bc in ARMS:
            s = widths[0] if sup == "dense" else sup
            t = row(str(sup), widths, c, bc, s, base)
            base = base or t

    print("\n== matched active leaves (THE depth price) ==")
    print(hdr)
    for leaves, sups in ((16, {2: 4, 4: 2}), (64, {2: 8, 3: 4}),
                         (256, {2: 16, 4: 4})):
        base = None
        for widths, c, bc in ARMS:
            s = sups.get(len(widths))
            if s is None or bc != 128:
                continue
            t = row(f"{leaves} lv", widths, c, bc, s, base)
            base = base or t
    print(f"\npeak_alloc {torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB")


if __name__ == "__main__":
    main()
