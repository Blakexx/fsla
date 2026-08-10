"""Wall clock for the ladder: rung 4 (`72d1a5c`) and rung 5A, ARM FOR ARM, ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's and
p59's tables.

    python bench.py headline [A|B|both]     the cost of the representation
    python bench.py sweep    [A|B|both]     that the saving still fires

WHAT IS TIMED AGAINST RUNG 4 IS `forward_raw`, THE KERNEL, on a table BUILT ONCE
OUTSIDE THE TIMED REGION -- because that is what the table is for. Rung 4 paid
its membership test inside every one of the `N/BC` blocks that share it; this
rung pays it once. The table pass is then timed on ITS OWN LINE and, because it
is per (b,h) rather than per block, it is also reported amortised.

Both numbers are needed and neither is the whole answer:
  * KERNEL vs KERNEL is what the chunk loop's schedule costs;
  * KERNEL + TABLE vs KERNEL is what a caller pays end to end.

MEASUREMENT HYGIENE, binding:
  * A DISCARDED WARM-UP runs before the first timed arm of every cell: the first
    arm timed in a cold process bears the clock ramp from 210 MHz and reads ~14%
    slow (p59d).
  * peak VRAM per cell; a cell above 10 GiB of the 12 GiB card is DISCARDED.
  * clocks cannot be pinned on this box, so the two kernels are INTERLEAVED arm
    by arm and the drift is reported.
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung4 import build as r4build  # noqa: E402
from rung5 import build as r5build  # noqa: E402
from rung5 import reference  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
ORDER = False
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)
SWEEP_ARMS = [(16, 128), (32, 64), (32, 32), (64, 32)]
REGIMES = ("dense", "lead4", "d4")


def clocks():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,"
         "clocks_throttle_reasons.active", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()


def stopwatch(fn):
    best = None
    for _ in range(ROUNDS):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        mins = []
        for _ in range(BLOCKS):
            ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(REPS)]
            for a, b in ev:
                a.record()
                fn()
                b.record()
            torch.cuda.synchronize()
            mins.append(min(a.elapsed_time(b) for a, b in ev))
        m = statistics.median(mins)
        best = m if best is None else min(best, m)
    return best


def vram(tag):
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    print(f"    VRAM {tag}: peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} "
          f"GiB  device_used={used:.3f}/{total/(1<<30):.3f} GiB  "
          f"{'VALID' if used < VRAM_DISCARD_GIB else 'DISCARDED (>10 GiB)'}", flush=True)
    return used < VRAM_DISCARD_GIB


def headline(name, B, H, L):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== CELL {name}: B={B} H={H} L={L}   {B*H} (b,h) on {sms} SMs", flush=True)
    arms = [a for a in r5build.arms() if L % a[0] == 0]
    for c, w, bc, w0, w1, *_ in arms:
        r4build.assert_build_identity(c, w, bc, w0, w1)
        r5build.assert_build_identity(c, w, bc, w0, w1)
    dense = {wds: (reference.amplitudes(B, H, L, wds, dev, seed=2),
                   reference.amplitudes(B, H, L, wds, dev, seed=3))
             for wds in ((8, 8), (16, 16), (64, 64))}
    pr, pw = dense[FLAG]
    tbl0 = r5build.union_table(pr, pw, v, 32, 64, 64, 64)
    stopwatch(lambda: r5build.forward_raw(pr, pw, v, tbl0, 32, False, 64, 64, 64))
    print(f"    clocks before: {clocks()}", flush=True)

    t4, t5, tt = {}, {}, {}
    for c, w, bc, w0, w1, *_ in arms:
        p, q = dense[(w0, w1)]
        k = (c, w, bc, w0, w1)
        tbl = r5build.union_table(p, q, v, c, bc, w0, w1)
        #: THE PAIR IS TIMED IN BOTH ORDERS. Clocks cannot be pinned on this box
        #: and they RISE through a session, so whichever kernel is timed second
        #: gets the faster clock. Timing the pair once each way and taking the
        #: better of each cancels the direction of that bias rather than hiding
        #: it; the two orders' raw deltas are reported by `--order`.
        a1 = stopwatch(lambda: r4build.forward_raw(p, q, v, c, w, bc, w0, w1))
        b1 = stopwatch(lambda: r5build.forward_raw(p, q, v, tbl, c, w, bc, w0, w1))
        b2 = stopwatch(lambda: r5build.forward_raw(p, q, v, tbl, c, w, bc, w0, w1))
        a2 = stopwatch(lambda: r4build.forward_raw(p, q, v, c, w, bc, w0, w1))
        t4[k], t5[k] = min(a1, a2), min(b1, b2)
        if ORDER:
            print(f"    [order] {k}  r4 first {a1:.4f}/{b1:.4f} ({100*(b1-a1)/a1:+.1f}%)"
                  f"   r5 first {a2:.4f}/{b2:.4f} ({100*(b2-a2)/a2:+.1f}%)", flush=True)
        tt[k] = stopwatch(lambda: r5build.union_table(p, q, v, c, bc, w0, w1))
    print(f"    clocks after: {clocks()}", flush=True)
    if not vram("headline"):
        return

    print("\n-- THE SCHEDULE'S REPRESENTATION: dense data, the chunk kernel only")
    print(f"{'arm':32s}{'rung4 ms':>10s}{'rung5 ms':>10s}{'kernel':>9s}"
          f"{'table ms':>10s}{'+table':>9s}")
    for k in t5:
        c, w, bc, w0, w1 = k
        a, b, t = t4[k], t5[k], tt[k]
        print(f"{f'C={c} SWZ={int(w)} BC={bc} N={w0*w1}':32s}{a:10.4f}{b:10.4f}"
              f"{100*(b-a)/a:+8.1f}%{t:10.4f}{100*(b+t-a)/a:+8.1f}%")
    for tag, wds in (("flagship N=4096", FLAG), ("control  N=64", (8, 8)),
                     ("small    N=256", (16, 16))):
        a = min(t4[k] for k in t4 if k[3:] == wds)
        b = min(t5[k] for k in t5 if k[3:] == wds)
        ka = min(t5, key=lambda k: t5[k] if k[3:] == wds else 1e9)
        print(f"{'FLOOR ' + tag:32s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%"
              f"{tt[ka]:10.4f}{100*(b+tt[ka]-a)/a:+8.1f}%")
    torch.cuda.empty_cache()


def sweep(name, B, H, L):
    """The saving must survive, and BC=32's penalty must not."""
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    print(f"\n=== SWEEP, CELL {name}: B={B} H={H} L={L}", flush=True)
    amps = {r: reference.regime_amplitudes(B, H, L, r, dev, seed=5) for r in REGIMES}
    pr, pw = amps["dense"]
    tbl0 = r5build.union_table(pr, pw, v, 32, 64, 64, 64)
    stopwatch(lambda: r5build.forward_raw(pr, pw, v, tbl0, 32, False, 64, 64, 64))
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"{'arm':16s}{'regime':>8s}{'skip vis':>10s}{'rung4 ms':>10s}"
          f"{'rung5 ms':>10s}{'kernel':>9s}{'table ms':>10s}", flush=True)
    for c, bc in SWEEP_ARMS:
        if L % c:
            continue
        for rg in REGIMES:
            p, q = amps[rg]
            skip = 1.0 - reference.schedule(p, q, FLAG, bc, c).double().mean().item()
            tbl = r5build.union_table(p, q, v, c, bc, *FLAG)
            a = stopwatch(lambda: r4build.forward_raw(p, q, v, c, False, bc, *FLAG))
            b = stopwatch(lambda: r5build.forward_raw(p, q, v, tbl, c, False, bc, *FLAG))
            t = stopwatch(lambda: r5build.union_table(p, q, v, c, bc, *FLAG))
            print(f"{f'C={c} BC={bc}':16s}{rg:>8s}{100*skip:9.1f}%"
                  f"{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%{t:10.4f}", flush=True)
    print(f"    clocks after: {clocks()}", flush=True)
    vram("sweep")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headline"
    ORDER = mode == "order"
    mode = "headline" if mode == "order" else mode
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    fn = headline if mode == "headline" else sweep
    for nm in (CELLS if which == "both" else [which]):
        fn(nm, *CELLS[nm])
