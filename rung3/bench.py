"""Wall clock for the ladder: rung 2 (`47fd58d`) and rung 3, ARM FOR ARM, ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's and
p59's tables.

    python bench.py headline [A|B|both]     the dense cost of the machinery
    python bench.py sweep    [A|B|both]     what the machinery buys, by topology
    python bench.py analysis                the schedule's arithmetic, no GPU

THE MEASUREMENT THIS RUNG EXISTS FOR IS THE DENSE ONE. With dense amplitudes the
schedule skips nothing and every membership test returns true, so rung 3 does
exactly rung 2's work: the delta is the COST OF THE MACHINERY, paid in full and
earning nothing. That number decides whether one uniform path is viable.

THE SWEEP IS OVER TOPOLOGY, NOT ONLY OVER DENSITY, because whether a block can
be skipped at all is decided by WHERE each side is sparse relative to the block
boundary -- sparsity above it confines a side to few blocks, sparsity below it
spreads that side over every block at one slot each. `reference.REGIMES` names
the configurations and says how a deeper topology is expressed at the kernel's
two-level interface.

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
from rung2 import build as r2build  # noqa: E402
from rung3 import build as r3build  # noqa: E402
from rung3 import reference  # noqa: E402

sys.path.insert(0, "/home/blake/rola-scratch/workflows/perf/p59f_evidence")
from table import sided  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)
#: the arms the sweep is run on: the flagship's padded arms at the block widths
#: rung 2 found interior. Every arm is in the headline; the sweep is narrowed so
#: that four topologies fit one session.
SWEEP_ARMS = [(16, 64), (32, 64), (16, 128), (32, 128), (64, 128)]
REGIMES = ("dense", "lead4", "lead1", "split", "d4")


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
    r3arms = [a for a in r3build.arms() if L % a[0] == 0]
    for c, w, bc, w0, w1, *_ in r3arms:
        r2build.assert_build_identity(c, w, bc, w0, w1)
        r3build.assert_build_identity(c, w, bc, w0, w1)
    dense = {wds: (reference.amplitudes(B, H, L, wds, dev, seed=2),
                   reference.amplitudes(B, H, L, wds, dev, seed=3))
             for wds in ((8, 8), (16, 16), (64, 64))}
    pr, pw = dense[FLAG]
    stopwatch(lambda: r3build.forward(pr, pw, v, 32, False, 64, 64, 64))   # DISCARDED
    print(f"    clocks before: {clocks()}", flush=True)

    t2, t3 = {}, {}
    for c, w, bc, w0, w1, *_ in r3arms:
        p, q = dense[(w0, w1)]
        k = (c, w, bc, w0, w1)
        t2[k] = stopwatch(lambda: r2build.forward(p, q, v, c, w, bc, w0, w1))
        t3[k] = stopwatch(lambda: r3build.forward(p, q, v, c, w, bc, w0, w1))
    print(f"    clocks after: {clocks()}", flush=True)
    if not vram("headline"):
        return

    print("\n-- THE DENSE COST OF THE MACHINERY: every test true, nothing skipped")
    print(f"{'arm':32s}{'rung2 ms':>10s}{'rung3 ms':>10s}{'delta':>9s}")
    for k in t3:
        c, w, bc, w0, w1 = k
        a, b = t2[k], t3[k]
        print(f"{f'C={c} SWZ={int(w)} BC={bc} N={w0*w1}':32s}{a:10.4f}{b:10.4f}"
              f"{100*(b-a)/a:+8.1f}%")
    for tag, wds in (("flagship N=4096", FLAG), ("control  N=64", (8, 8)),
                     ("small    N=256", (16, 16))):
        a = min(t2[k] for k in t2 if k[3:] == wds)
        b = min(t3[k] for k in t3 if k[3:] == wds)
        print(f"{'FLOOR ' + tag:32s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%")
    torch.cuda.empty_cache()


def sweep(name, B, H, L):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    print(f"\n=== SWEEP, CELL {name}: B={B} H={H} L={L}", flush=True)
    amps = {r: reference.regime_amplitudes(B, H, L, r, dev, seed=5)
            for r in REGIMES}
    pr, pw = amps["dense"]
    stopwatch(lambda: r3build.forward(pr, pw, v, 32, False, 64, 64, 64))   # DISCARDED
    print(f"    clocks before: {clocks()}", flush=True)
    print(f"{'arm':16s}{'regime':>8s}{'skip vis':>10s}{'skip mma':>10s}"
          f"{'per-side':>10s}{'rung2 ms':>10s}{'rung3 ms':>10s}{'delta':>9s}",
          flush=True)
    for c, bc in SWEEP_ARMS:
        if L % c:
            continue
        for rg in REGIMES:
            p, q = amps[rg]
            rt, wt = reference.schedule_sides(p, q, FLAG, bc)
            s = sided(c, bc, rt, wt)
            a = stopwatch(lambda: r2build.forward(p, q, v, c, False, bc, *FLAG))
            b = stopwatch(lambda: r3build.forward(p, q, v, c, False, bc, *FLAG))
            print(f"{f'C={c} BC={bc}':16s}{rg:>8s}{100*s['skip_union']:9.1f}%"
                  f"{100*s['mma_skipped_union']:9.1f}%"
                  f"{100*s['mma_skipped_side']:9.1f}%"
                  f"{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%", flush=True)
            del rt, wt
    print(f"    clocks after: {clocks()}", flush=True)
    vram("sweep")
    torch.cuda.empty_cache()


def analysis():
    """The schedule's arithmetic on the flagship, per side and per term. No GPU
    timing, so it can range over topologies and chunk sizes freely."""
    dev = "cuda"
    B, H, L = 2, 8, 4096
    print("== WHAT THE SCHEDULE CAN SEE, by topology and by side ==")
    print("   skip = fraction of (block, group) VISITS a schedule may drop;")
    print("   mma  = fraction of the dense mma that dropping them removes.")
    print(f"\n{'regime':>8s}{'C':>4s}{'BC':>5s}{'per-token union':>16s}"
          f"{'skip read':>11s}{'skip write':>11s}{'skip union':>12s}"
          f"{'mma union':>11s}{'mma per-side':>14s}{'V bytes side':>14s}")
    for rg in REGIMES:
        p, q = reference.regime_amplitudes(B, H, L, rg, dev, seed=5)
        for bc in (64, 128):
            rt, wt = reference.schedule_sides(p, q, FLAG, bc)
            tokdead = 1.0 - (rt | wt).double().mean().item()
            for c in (16, 32, 64):
                s = sided(c, bc, rt, wt)
                print(f"{rg:>8s}{c:4d}{bc:5d}{100*(1-tokdead):15.1f}%"
                      f"{100*s['skip_read']:10.1f}%{100*s['skip_write']:10.1f}%"
                      f"{100*s['skip_union']:11.1f}%"
                      f"{100*s['mma_skipped_union']:10.1f}%"
                      f"{100*s['mma_skipped_side']:13.1f}%"
                      f"{100*s['bytes_skipped_side']:13.1f}%", flush=True)
            del rt, wt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headline"
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    if mode == "analysis":
        analysis()
    else:
        fn = headline if mode == "headline" else sweep
        for nm in (CELLS if which == "both" else [which]):
            fn(nm, *CELLS[nm])
