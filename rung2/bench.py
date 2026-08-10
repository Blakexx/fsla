"""Wall clock for the ladder: rung 1 (`adde2f1`) and rung 2, ARM FOR ARM, ONE session.

Stopwatch: P51's protocol unchanged -- block-min-median, 12 blocks x 20 reps,
cuda events, best of 3 rounds -- so every row is commensurable with p57's and
p59's tables.

THE COMPARISON A CAPACITY RUNG NEEDS. Rung 2's flagship holds 64x rung 1's state
and does 64x the mma, so its wall against rung 1's is not a delta, it is a
different problem. Two comparisons are therefore reported and they answer
different questions:

  * THE CONTROL, N = 64 with ONE block, against rung 1 arm for arm. Same work,
    same mma count, so this IS a floor comparison: what the capacity machinery
    and the fan-in cost when the capacity is not used.
  * NANOSECONDS PER mma, and per USEFUL mma, for every arm. That is the figure
    that survives a change of problem size, and it is what makes the BC arms
    (which do DIFFERENT amounts of mma for the same answer) comparable at all.

MEASUREMENT HYGIENE, binding:
  * A DISCARDED WARM-UP runs before the first timed arm of every cell: the first
    arm timed in a cold process bears the clock ramp from 210 MHz and reads ~14%
    slow (p59d).
  * peak VRAM per cell; a cell above 10 GiB of the 12 GiB card is DISCARDED.
  * clocks cannot be pinned on this box, so arms are INTERLEAVED and the drift
    is reported.

    python bench.py [cell]        cell in {A, B, both}
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung1 import build as r1build  # noqa: E402
from rung2 import build as r2build  # noqa: E402
from rung2 import reference  # noqa: E402

sys.path.insert(0, "/home/blake/rola-scratch/workflows/perf/p59e_evidence")
from table import tiles, useful_fraction  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}


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


def mma_count(B, H, L, C, BC, N):
    return B * H * (N // BC) * (L // C) * tiles(C, BC)["total"]


def cell(name, B, H, L):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    amps = {}
    for w in ((8, 8), (16, 16), (64, 64)):
        amps[w] = (reference.amplitudes(B, H, L, w, dev, seed=2),
                   reference.amplitudes(B, H, L, w, dev, seed=3))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== CELL {name}: B={B} H={H} L={L}   {B*H} (b,h) on {sms} SMs")

    r1arms = r1build.arms()
    r2arms = [a for a in r2build.arms() if L % a[0] == 0]
    for c, w, *_ in r1arms:
        r1build.assert_build_identity(c, w)
    for c, w, bc, w0, w1, *_ in r2arms:
        r2build.assert_build_identity(c, w, bc, w0, w1)

    # DISCARDED: the cold-process clock ramp is an instrument property and is
    # charged to nobody.
    pr, pw = amps[(64, 64)]
    stopwatch(lambda: r2build.forward(pr, pw, v, 32, False, 64, 64, 64))
    print(f"    clocks before: {clocks()}")

    r1t, r2t = {}, {}
    for c, w, bc, w0, w1, nb, *_ in r2arms:
        if (w0, w1) == (8, 8):          # interleave the control with its rung-1 arm
            for c1, w1_, *_ in r1arms:
                if (c1, w1_) != (c, w):
                    continue
                q, k = amps[(8, 8)]
                r1t[(c, w)] = stopwatch(
                    lambda c=c1, w=w1_: r1build.forward(q, k, v, c, w))
        p, q = amps[(w0, w1)]
        r2t[(c, w, bc, w0, w1)] = stopwatch(
            lambda: r2build.forward(p, q, v, c, w, bc, w0, w1))

    print(f"    clocks after:  {clocks()}")
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    print(f"    VRAM: peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB  "
          f"device_used={used:.3f}/{total/(1<<30):.3f} GiB  "
          f"{'VALID' if used < VRAM_DISCARD_GIB else 'DISCARDED (>10 GiB)'}")
    if used >= VRAM_DISCARD_GIB:
        return

    # THE ACCUMULATOR'S ZEROING IS PART OF THE CALL AND IS MEASURED, NOT ASSUMED.
    # A fan-in ADDS, so the identity has to be in the buffer before the first
    # contribution; `forward` therefore allocates a zeroed fp32 accumulator where
    # rung 1 allocated an uninitialised bf16 output. At the flagship that is a
    # rounding error against the kernel, and at the control it is most of the
    # wall -- so it is reported on its own line rather than folded into a delta.
    zero = stopwatch(lambda: torch.zeros((B, H, L, 64), device=dev,
                                         dtype=torch.float32))
    empt = stopwatch(lambda: torch.empty((B, H, L, 64), device=dev,
                                         dtype=torch.bfloat16))
    print(f"\n-- the accumulator's allocation: zeroed fp32 {zero:.4f} ms  vs rung 1's "
          f"uninitialised bf16 {empt:.4f} ms")

    print("\n-- THE CONTROL: N=64, one block, the same work as rung 1")
    print(f"{'arm':18s}{'rung1 ms':>10s}{'rung2 ms':>10s}{'delta':>9s}"
          f"{'rung2 - zeroing':>17s}{'delta':>9s}")
    for c, w in sorted(r1t):
        a, b = r1t[(c, w)], r2t[(c, w, 64, 8, 8)]
        n = b - zero + empt
        print(f"{f'C={c} SWZ={int(w)}':18s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.1f}%"
              f"{n:17.4f}{100*(n-a)/a:+8.1f}%")
    bb = min(r1t.values())
    bc_ = min(r2t[k] for k in r2t if k[3:] == (8, 8))
    print(f"{'FLOOR (best arm)':18s}{bb:10.4f}{bc_:10.4f}{100*(bc_-bb)/bb:+8.1f}%"
          f"{bc_-zero+empt:17.4f}{100*(bc_-zero+empt-bb)/bb:+8.1f}%")

    print("\n-- EVERY ARM, and the figures that survive a change of problem size")
    print(f"{'arm':30s}{'ms':>9s}{'ns/mma':>9s}{'ns/USEFUL mma':>15s}"
          f"{'state/(b,h)':>13s}{'atomic MB':>11s}")
    for c, w, bc, w0, w1, nb, *_ in r2arms:
        n = w0 * w1
        t = r2t[(c, w, bc, w0, w1)]
        mma = mma_count(B, H, L, c, bc, n)
        uf, _, _ = useful_fraction(c, bc, n)
        atomic = B * H * nb * L * 64 * 4 / 1e6
        print(f"{f'C={c} SWZ={int(w)} BC={bc} N={n}':30s}{t:9.4f}"
              f"{1e6*t/mma:9.3f}{1e6*t/(mma*uf):15.3f}"
              f"{n*64*4/1024:11.0f} KiB{atomic:11.1f}")

    print("\n-- the BC axis: does the optimum MOVE?")
    for c in (16, 32, 64):
        row = [(bc, r2t[k]) for k in r2t
               for _, _, bc, w0, w1 in [k] if k[0] == c and not k[1] and (w0, w1) == (64, 64)]
        if not row:
            continue
        best = min(row, key=lambda kv: kv[1])
        print(f"   C={c:2d}  "
              + "   ".join(f"BC={bc}: {t:.4f} ms" for bc, t in sorted(row))
              + f"   -> BEST BC={best[0]}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for nm in (CELLS if which == "both" else [which]):
        cell(nm, *CELLS[nm])
