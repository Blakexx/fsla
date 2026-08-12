"""The stop gate's arms only, timed against rung 5A AND rung 5B in one session.

    python floor.py [A|B|both]

`bench.py headline` sweeps every arm; this times the arms that carry the verdict,
so the cost of the representation can be priced without a twenty-arm session.
Same stopwatch, same interleaving, same warm-up discard, same VRAM guard.

WHY THREE KERNELS AND NOT TWO. Rung 5A is the FLOOR -- the same work with a group
schedule and no gather -- and the delta against it is the standing stop-gate
number. Rung 5B is the PREVIOUS shape of the same compaction, so the delta
against IT is what forming the schedule instead of fusing the AND into the walk
is worth, isolated from everything else compaction costs.

THE VERDICT IS RENDERED AT BC = 128, the dense optimum. At that rectangle with
the flagship's (64, 64) widths the block spans the trailing level ENTIRELY, so
there is exactly ONE clause per side and the form phase is a COPY: what it buys
there is only that the straddling word is not re-loaded, re-ANDed and
re-popcounted. `BC = 32` is the rectangle where the block spans the trailing
level partially, TWO clauses survive per side and the form does a real AND -- so
the two exercise different halves of the fix and are reported apart.

EVERY PAIR IN BOTH ORDERS. Clocks cannot be pinned on this box, so each pair is
timed A-B-B-A and the minimum of each kernel's two readings is taken; the clock
and temperature are printed before and after so drift is on the record.
"""
from __future__ import annotations

import statistics
import subprocess
import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5 import build as r5a  # noqa: E402
from rung5b import build as r5b  # noqa: E402
from rung5b import reference  # noqa: E402
from rung5d import build as r5d  # noqa: E402
from rung5e import build as r5e  # noqa: E402

BLOCKS, REPS, ROUNDS = 12, 20, 3
VRAM_DISCARD_GIB = 10.0
CELLS = {"A": (2, 8, 4096), "B": (20, 8, 1024)}
FLAG = (64, 64)

#: (C, BC, role). The verdict is the BC = 128 rows; BC = 32 is the `A > 1`
#: rectangle, where the form phase performs a real AND rather than a copy.
ARMS = [(16, 128, "verdict"), (32, 128, "verdict"), (64, 128, "verdict"),
        (32, 32, "A>1"), (64, 32, "A>1")]
REGIMES = ("dense", "lead4", "d4")


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


def clocks():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()


def run(name, B, H, L):
    dev = "cuda"
    torch.manual_seed(0)
    v = torch.randn(B, H, L, 64, device=dev).bfloat16()
    print(f"\n=== CELL {name}: B={B} H={H} L={L}", flush=True)
    for regime in REGIMES:
        if regime == "dense":
            pr = reference.amplitudes(B, H, L, FLAG, dev, seed=2)
            pw = reference.amplitudes(B, H, L, FLAG, dev, seed=3)
        else:
            pr, pw = reference.regime_amplitudes(B, H, L, regime, dev, seed=5)
        # THE DISCARDED WARM-UP: the first arm of a cold process bears the clock
        # ramp from 210 MHz and reads ~14% slow.
        tbl0 = r5e.union_table(pr, pw, v, 32, 128, *FLAG)
        stopwatch(lambda: r5e.forward_raw(pr, pw, v, tbl0, 32, False, 128, *FLAG))
        print(f"  -- {regime} -- clocks before: {clocks()}", flush=True)
        print(f"  {'arm':18s}{'role':8s}{'5A ms':>10s}{'5B ms':>10s}{'5D ms':>10s}{'5E ms':>10s}"
              f"{'5E-5A':>9s}{'5D-5A':>9s}{'5E-5D':>9s}", flush=True)
        for c, bc, role in ARMS:
            if L % c:
                continue
            r5a.assert_build_identity(c, False, bc, *FLAG)
            r5b.assert_build_identity(c, False, bc, *FLAG)
            r5d.assert_build_identity(c, False, bc, *FLAG)
            r5e.assert_build_identity(c, False, bc, *FLAG)
            tbl = r5e.union_table(pr, pw, v, c, bc, *FLAG)

            def fa():
                return r5a.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

            def fb():
                return r5b.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

            def fd():
                return r5d.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

            def fe():
                return r5e.forward_raw(pr, pw, v, tbl, c, False, bc, *FLAG)

            a1, b1, d1, e1 = stopwatch(fa), stopwatch(fb), stopwatch(fd), stopwatch(fe)
            e2, d2, b2, a2 = stopwatch(fe), stopwatch(fd), stopwatch(fb), stopwatch(fa)
            a, b, d, e = min(a1, a2), min(b1, b2), min(d1, d2), min(e1, e2)
            print(f"  {f'C={c} BC={bc}':18s}{role:8s}{a:10.4f}{b:10.4f}{d:10.4f}{e:10.4f}"
                  f"{100*(e-a)/a:+8.1f}%{100*(d-a)/a:+8.1f}%{100*(e-d)/d:+8.1f}%",
                  flush=True)
            tbl = None
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1 << 30)
        print(f"  -- clocks after: {clocks()}   device_used={used:.3f} GiB  "
              f"peak_alloc={torch.cuda.max_memory_allocated()/(1<<30):.3f} GiB  "
              f"{'VALID' if used < VRAM_DISCARD_GIB else 'DISCARDED (>10 GiB)'}",
              flush=True)
        pr = pw = tbl0 = None
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for nm in (CELLS if which == "both" else [which]):
        run(nm, *CELLS[nm])
