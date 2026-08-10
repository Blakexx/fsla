"""Wall clock for the ladder: rung 3 (`f66ab95`) and rung 4, ARM FOR ARM, ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's and
p59's tables.

    python bench.py headline [A|B|both]     the cost of the ones column
    python bench.py sweep    [A|B|both]     that it still rides the schedule

WHAT IS TIMED AGAINST RUNG 3 IS `forward_raw`, THE KERNEL. Rung 3 has no
denominator and therefore no division, so a kernel-to-kernel delta must not carry
one. `divide_pass` is timed on its own line, as rung 2 timed the accumulator's
zeroing: it is a real cost of the feature and it is not a cost of the kernel.

MEASUREMENT HYGIENE, binding:
  * A DISCARDED WARM-UP runs before the first timed arm of every cell: the first
    arm timed in a cold process bears the clock ramp from 210 MHz and reads ~14%
    slow (p59d).
  * peak VRAM per cell; a cell above 10 GiB of the 12 GiB card is DISCARDED.
  * clocks cannot be pinned on this box, so the two kernels are INTERLEAVED arm
    by arm and the drift is reported.
  * determinism is gone by construction since rung 2 (multi-block reductions);
    no row here is a bit-reproducible quantity.
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung3 import build as r3build  # noqa: E402
from rung4 import build as r4build  # noqa: E402
from rung4 import reference  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)
SWEEP_ARMS = [(16, 64), (32, 64), (16, 128), (32, 128)]
REGIMES = ("dense", "lead1")


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
    arms = [a for a in r4build.arms() if L % a[0] == 0]
    for c, w, bc, w0, w1, *_ in arms:
        r3build.assert_build_identity(c, w, bc, w0, w1)
        r4build.assert_build_identity(c, w, bc, w0, w1)
    dense = {wds: (reference.amplitudes(B, H, L, wds, dev, seed=2),
                   reference.amplitudes(B, H, L, wds, dev, seed=3))
             for wds in ((8, 8), (16, 16), (64, 64))}
    pr, pw = dense[FLAG]
    stopwatch(lambda: r4build.forward_raw(pr, pw, v, 32, False, 64, 64, 64))  # DISCARDED
    print(f"    clocks before: {clocks()}", flush=True)

    t3, t4 = {}, {}
    for c, w, bc, w0, w1, *_ in arms:
        p, q = dense[(w0, w1)]
        k = (c, w, bc, w0, w1)
        t3[k] = stopwatch(lambda: r3build.forward(p, q, v, c, w, bc, w0, w1))
        t4[k] = stopwatch(lambda: r4build.forward_raw(p, q, v, c, w, bc, w0, w1))
    #: THE DIVIDE PASS, ON ITS OWN LINE. It is one elementwise pass over the
    #: numerator and it cannot be in the kernel (see the source header), so it is
    #: reported as its own cost and never folded into a kernel delta.
    num, den = r4build.forward_raw(pr, pw, v, 32, False, 128, 64, 64)
    tdiv = stopwatch(lambda: r4build.divide_pass(num, den))
    print(f"    clocks after: {clocks()}", flush=True)
    if not vram("headline"):
        return

    print("\n-- THE COST OF THE ONES COLUMN: dense data, the kernel only")
    print(f"{'arm':32s}{'rung3 ms':>10s}{'rung4 ms':>10s}{'delta':>9s}")
    for k in t4:
        c, w, bc, w0, w1 = k
        a, b = t3[k], t4[k]
        print(f"{f'C={c} SWZ={int(w)} BC={bc} N={w0*w1}':32s}{a:10.4f}{b:10.4f}"
              f"{100*(b-a)/a:+8.1f}%")
    for tag, wds in (("flagship N=4096", FLAG), ("control  N=64", (8, 8)),
                     ("small    N=256", (16, 16))):
        a = min(t3[k] for k in t3 if k[3:] == wds)
        b = min(t4[k] for k in t4 if k[3:] == wds)
        print(f"{'FLOOR ' + tag:32s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%")
    fl = min(t4[k] for k in t4 if k[3:] == FLAG)
    print(f"\n{'DIVIDE PASS (own line)':32s}{'':10s}{tdiv:10.4f}"
          f"{100*tdiv/fl:+8.1f}% of the flagship floor arm")
    torch.cuda.empty_cache()


def sweep(name, B, H, L):
    """The mass channel is skipped with its group: the saving must survive."""
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    print(f"\n=== SWEEP, CELL {name}: B={B} H={H} L={L}", flush=True)
    amps = {r: reference.regime_amplitudes(B, H, L, r, dev, seed=5) for r in REGIMES}
    pr, pw = amps["dense"]
    stopwatch(lambda: r4build.forward_raw(pr, pw, v, 32, False, 64, 64, 64))  # DISCARDED
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"{'arm':16s}{'regime':>8s}{'skip vis':>10s}{'rung3 ms':>10s}"
          f"{'rung4 ms':>10s}{'delta':>9s}", flush=True)
    for c, bc in SWEEP_ARMS:
        if L % c:
            continue
        for rg in REGIMES:
            p, q = amps[rg]
            skip = 1.0 - reference.schedule(p, q, FLAG, bc, c).double().mean().item()
            a = stopwatch(lambda: r3build.forward(p, q, v, c, False, bc, *FLAG))
            b = stopwatch(lambda: r4build.forward_raw(p, q, v, c, False, bc, *FLAG))
            print(f"{f'C={c} BC={bc}':16s}{rg:>8s}{100*skip:9.1f}%"
                  f"{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%", flush=True)
    print(f"    clocks after: {clocks()}", flush=True)
    vram("sweep")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headline"
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    fn = headline if mode == "headline" else sweep
    for nm in (CELLS if which == "both" else [which]):
        fn(nm, *CELLS[nm])
