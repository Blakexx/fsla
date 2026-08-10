"""Wall clock for the ABLATION LADDER: rung 5A and the four stages, ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's and
p59's tables.

    python bench.py ladder [A|B|both]     each mechanism's dense wall
    python bench.py stopgate [A|B|both]   the stop gate, 5A against stage 4
    python bench.py sweep [A|B|both]      what compaction buys, by regime

WHAT IS TIMED IS `forward_raw`, THE CHUNK KERNEL, on a table BUILT ONCE OUTSIDE
THE TIMED REGION, which is what the table is for.

MEASUREMENT HYGIENE, binding:
  * A DISCARDED WARM-UP before the first timed arm of every cell: the first arm
    timed in a cold process bears the clock ramp and reads ~14% slow (p59d).
  * peak VRAM per cell; a cell above 10 GiB of the 12 GiB card is DISCARDED.
  * clocks cannot be pinned on this box, so every PAIR is timed in BOTH ORDERS
    and the better of each is taken, which cancels the direction of the bias
    rather than hiding it.
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5 import build as r4build  # noqa: E402
from rung5c import build as acbuild  # noqa: E402
from rung5c import reference  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)
LADDER_ARMS = [(16, 128), (32, 128), (64, 128), (32, 64), (64, 64), (32, 32)]
SWEEP_ARMS = [(16, 128), (32, 128), (64, 128), (32, 64), (32, 32), (64, 32)]
REGIMES = ("dense", "lead4", "split", "d4")
STAGE_LABEL = {1: "+1 gather", 2: "+2 staircase", 3: "+3 gates", 4: "+4 cook(5B)"}


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


def _setup(B, H, L, name, mode):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== {mode}, CELL {name}: B={B} H={H} L={L}   {B*H} (b,h) on {sms} SMs",
          flush=True)
    return dev, v


def ladder(name, B, H, L):
    """EACH MECHANISM'S DENSE WALL. Stage `s` against stage `s-1` is exactly the
    mechanism stage `s` admits, because at dense every stage computes the same
    function (`ladder_gate.py`)."""
    dev, v = _setup(B, H, L, name, "LADDER")
    pr, pw = (reference.amplitudes(B, H, L, FLAG, dev, seed=2),
              reference.amplitudes(B, H, L, FLAG, dev, seed=3))
    for c, bc in LADDER_ARMS:
        r4build.assert_build_identity(c, False, bc, *FLAG)
        for s in (1, 2, 3, 4):
            acbuild.assert_build_identity(c, False, bc, *FLAG, s)
    tbl0 = acbuild.union_table(pr, pw, v, 32, 128, *FLAG)
    stopwatch(lambda: acbuild.forward_raw(pr, pw, v, tbl0, 32, False, 128, *FLAG, 4))
    print(f"    clocks before: {clocks()}", flush=True)

    print(f"\n{'arm':16s}{'stage':14s}{'ms':>10s}{'vs 5A':>9s}{'mechanism':>11s}")
    for c, bc in LADDER_ARMS:
        if L % c:
            continue
        tbl = acbuild.union_table(pr, pw, v, c, bc, *FLAG)
        base = _both(lambda: r4build.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG),
                     lambda: acbuild.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG, 1))
        t = {0: base[0], 1: base[1]}
        for s in (2, 3, 4):
            a, b = _both(
                lambda: acbuild.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG, s - 1),
                lambda: acbuild.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG, s))
            t[s - 1] = min(t[s - 1], a)
            t[s] = b
        print(f"{f'C={c} BC={bc}':16s}{'5A baseline':14s}{t[0]:10.4f}"
              f"{'':>9s}{'':>11s}")
        for s in (1, 2, 3, 4):
            print(f"{'':16s}{STAGE_LABEL[s]:14s}{t[s]:10.4f}"
                  f"{100*(t[s]-t[0])/t[0]:+8.1f}%{100*(t[s]-t[s-1])/t[0]:+10.1f}%")
    print(f"    clocks after: {clocks()}", flush=True)
    vram("ladder")
    torch.cuda.empty_cache()


def _both(fa, fb):
    """A PAIR TIMED IN BOTH ORDERS, better of each. Clocks rise through a session,
    so whichever kernel runs second gets the faster one."""
    a1 = stopwatch(fa)
    b1 = stopwatch(fb)
    b2 = stopwatch(fb)
    a2 = stopwatch(fa)
    return min(a1, a2), min(b1, b2)


def stopgate(name, B, H, L):
    """THE STOP GATE, on section P59i's arms and cells, unreinterpreted."""
    dev, v = _setup(B, H, L, name, "STOP GATE")
    arms = sorted({a[:5] for a in acbuild.arms() if L % a[0] == 0})
    dense = {}
    for _, _, _, w0, w1 in arms:
        if (w0, w1) not in dense:
            dense[(w0, w1)] = (reference.amplitudes(B, H, L, (w0, w1), dev, seed=2),
                               reference.amplitudes(B, H, L, (w0, w1), dev, seed=3))
    pr, pw = dense[FLAG]
    tbl0 = acbuild.union_table(pr, pw, v, 32, 128, *FLAG)
    stopwatch(lambda: acbuild.forward_raw(pr, pw, v, tbl0, 32, False, 128, *FLAG, 4))
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"\n{'arm':30s}{'rung5A':>10s}{'stage4':>10s}{'delta':>9s}{'table':>9s}")
    t4, t5, tt = {}, {}, {}
    for c, w, bc, w0, w1 in arms:
        p, q = dense[(w0, w1)]
        tbl = acbuild.union_table(p, q, v, c, bc, w0, w1)
        a, b = _both(lambda: r4build.forward_raw(p, q, v, tbl, c, w, bc, w0, w1),
                     lambda: acbuild.forward_raw(p, q, v, tbl, c, w, bc, w0, w1, 4))
        k = (c, w, bc, w0, w1)
        t4[k], t5[k] = a, b
        tt[k] = stopwatch(lambda: acbuild.union_table(p, q, v, c, bc, w0, w1))
        print(f"{f'C={c} BC={bc} N={w0*w1}':30s}{a:10.4f}{b:10.4f}"
              f"{100*(b-a)/a:+8.1f}%{tt[k]:9.4f}")
    for tag, wds in (("flagship N=4096", FLAG), ("control  N=64", (8, 8)),
                     ("small    N=256", (16, 16))):
        ks = [k for k in t4 if k[3:] == wds]
        if not ks:
            continue
        a, b = min(t4[k] for k in ks), min(t5[k] for k in ks)
        print(f"{'FLOOR ' + tag:30s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%")
    print(f"    clocks after: {clocks()}", flush=True)
    vram("stopgate")
    torch.cuda.empty_cache()


def sweep(name, B, H, L):
    """WHAT COMPACTION BUYS, by regime, stage 4 against rung 5A."""
    dev, v = _setup(B, H, L, name, "SWEEP")
    amps = {r: reference.regime_amplitudes(B, H, L, r, dev, seed=5) for r in REGIMES}
    pr, pw = amps["dense"]
    tbl0 = acbuild.union_table(pr, pw, v, 32, 128, *FLAG)
    stopwatch(lambda: acbuild.forward_raw(pr, pw, v, tbl0, 32, False, 128, *FLAG, 4))
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"{'arm':16s}{'regime':>8s}{'grp/sc':>9s}{'rung5A ms':>11s}"
          f"{'stage4 ms':>11s}{'kernel':>9s}", flush=True)
    for c, bc in SWEEP_ARMS:
        if L % c:
            continue
        for rg in REGIMES:
            p_, q_ = amps[rg]
            plans = reference.plans_for(p_, q_, FLAG, bc, c)
            sc = sum(len(x) for x in plans)
            grp = int(reference.schedule(p_, q_, FLAG, bc, c)[0, 0].sum().item())
            tbl = acbuild.union_table(p_, q_, v, c, bc, *FLAG)
            a, b = _both(
                lambda: r4build.forward_raw(p_, q_, v, tbl, c, False, bc, *FLAG),
                lambda: acbuild.forward_raw(p_, q_, v, tbl, c, False, bc, *FLAG, 4))
            print(f"{f'C={c} BC={bc}':16s}{rg:>8s}{grp/max(sc,1):9.2f}"
                  f"{a:11.4f}{b:11.4f}{100*(b-a)/a:+8.1f}%", flush=True)
    print(f"    clocks after: {clocks()}", flush=True)
    vram("sweep")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    fn = {"ladder": ladder, "stopgate": stopgate, "sweep": sweep}[mode]
    for nm in (CELLS if which == "both" else [which]):
        fn(nm, *CELLS[nm])
